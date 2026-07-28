#!/usr/bin/env python3
"""Validate and summarize overlap-voxel-v3 block timing artifacts."""

import argparse
import csv
import json
import math
import os
import statistics


STAGE_FIELDS = (
    "da3_ms",
    "registration_ms",
    "pointcloud_build_ms",
    "voxel_fusion_ms",
    "memory_update_ms",
    "pointcloud_total_ms",
    "hist_render_ms",
    "condition_encode_ms",
    "dit_ms",
    "decode_ms",
)

CSV_FIELDS = (
    "block_index",
    "pixel_start",
    "pixel_end",
    "pixel_frames",
    "update_frame_indices",
    "anchor_count",
    "anchor_frame_index_input",
    "anchor_frame_index_output",
    "da3_window_frames",
    "registration_source",
    "registration_accepted",
    "registration_scale",
    "registration_scale_jump",
    "registration_normalized_rmse",
    "registration_inliers",
    "points_before",
    "points_after",
    *STAGE_FIELDS,
)


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def stage_stats(blocks, field):
    values = [float(block.get(field, 0.0)) for block in blocks]
    return {
        "total_ms": float(sum(values)),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p95_ms": percentile(values, 0.95),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def validate_contract(payload):
    summary = payload["summary"]
    blocks = payload["blocks"]
    if summary.get("memory_map_mode") != "overlap_voxel_v3":
        raise AssertionError("Timing artifact is not overlap_voxel_v3")
    if len(blocks) != int(summary["num_blocks"]):
        raise AssertionError("Block count disagrees with summary")
    if not blocks:
        raise AssertionError("No block timing records found")

    previous_anchor = None
    for index, block in enumerate(blocks):
        if int(block["block_index"]) != index:
            raise AssertionError(f"Non-contiguous block index at {index}")
        expected_anchor_count = 0 if index == 0 else 1
        expected_window = 3 if index == 0 else 4
        if int(block["anchor_count"]) != expected_anchor_count:
            raise AssertionError(f"Unexpected anchor count at block {index}")
        if int(block["da3_window_frames"]) != expected_window:
            raise AssertionError(f"Unexpected DA3 window at block {index}")
        if block.get("multi_anchor_retry_implemented") is not False:
            raise AssertionError(f"Multi-anchor retry is active at block {index}")
        if not block.get("registration_accepted"):
            raise AssertionError(f"Registration rejected at block {index}")
        if len(block.get("update_frame_indices", [])) != 3:
            raise AssertionError(f"Expected three new keyframes at block {index}")
        if index and block.get("anchor_frame_index_input") != previous_anchor:
            raise AssertionError(f"Anchor chain is broken at block {index}")
        previous_anchor = block.get("anchor_frame_index_output")

    output_frames = sum(int(block.get("pixel_frames", 0)) for block in blocks)
    if output_frames != int(summary["output_frames"]):
        raise AssertionError("Output frame count disagrees with summary")


def write_csv(path, blocks):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for block in blocks:
            row = {field: block.get(field) for field in CSV_FIELDS}
            row["update_frame_indices"] = ";".join(
                str(value) for value in block.get("update_frame_indices", [])
            )
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timing_json")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary-json")
    args = parser.parse_args()

    with open(args.timing_json) as handle:
        payload = json.load(handle)
    validate_contract(payload)
    blocks = payload["blocks"]
    write_csv(args.csv, blocks)

    report = {
        "contract_valid": True,
        "timing_json": os.path.abspath(args.timing_json),
        "block_csv": os.path.abspath(args.csv),
        "summary": payload["summary"],
        "stages": {field: stage_stats(blocks, field) for field in STAGE_FIELDS},
        "registration": {
            "normalized_rmse_max": max(
                float(block["registration_normalized_rmse"]) for block in blocks
            ),
            "scale_jump_max": max(
                float(block.get("registration_scale_jump") or 0.0) for block in blocks
            ),
            "accepted_blocks": sum(
                bool(block.get("registration_accepted")) for block in blocks
            ),
        },
    }
    if args.summary_json:
        with open(args.summary_json, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
