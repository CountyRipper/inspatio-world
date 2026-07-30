#!/usr/bin/env python3
"""Validate and summarize overlap-voxel-v4 per-block artifacts."""

import argparse
import csv
import json
import os


ARTIFACTS = (
    "pre_reference.mp4",
    "pre_historical.mp4",
    "pre_fused.mp4",
    "pre_reference_mask.mp4",
    "pre_historical_mask.mp4",
    "pre_fused_mask.mp4",
    "post_keyframes.mp4",
    "post_point_map.ply",
    "post_da3_cameras.npz",
    "metrics.json",
)

REQUIRED_METRICS = (
    "historical_keyframe_count",
    "da3_valid_points",
    "reference_uncovered_points",
    "reference_covered_points",
    "reference_geometry_consistent_points",
    "reference_geometry_rejected_points",
    "sim3_correspondence_count",
    "sim3_inlier_count",
    "sim3_inlier_ratio",
    "sim3_normalized_rmse",
    "voxel_size",
    "voxel_count",
    "pointcloud_rebuild_ms",
    "da3_ms",
)


def load_and_validate(root):
    block_names = sorted(
        name for name in os.listdir(root)
        if name.startswith("block_") and os.path.isdir(os.path.join(root, name))
    )
    if not block_names:
        raise AssertionError("No V4 block directories found")
    blocks = []
    voxel_size = None
    for index, name in enumerate(block_names):
        if name != f"block_{index:03d}":
            raise AssertionError(f"Non-contiguous block directory: {name}")
        block_dir = os.path.join(root, name)
        missing = [item for item in ARTIFACTS if not os.path.exists(os.path.join(block_dir, item))]
        if missing:
            raise AssertionError(f"{name} is missing artifacts: {missing}")
        with open(os.path.join(block_dir, "metrics.json")) as handle:
            metric = json.load(handle)
        missing_metrics = [item for item in REQUIRED_METRICS if item not in metric]
        if missing_metrics:
            raise AssertionError(f"{name} is missing metrics: {missing_metrics}")
        if metric.get("memory_map_mode") != "overlap_voxel_v4":
            raise AssertionError(f"{name} has the wrong map mode")
        if int(metric["historical_keyframe_count"]) != 3 * (index + 1):
            raise AssertionError(f"{name} does not contain the full keyframe history")
        if int(metric["reference_covered_points"]) != (
            int(metric["reference_geometry_consistent_points"])
            + int(metric["reference_geometry_rejected_points"])
        ):
            raise AssertionError(f"{name} has inconsistent covered-point counts")
        current_voxel_size = float(metric["voxel_size"])
        if voxel_size is None:
            voxel_size = current_voxel_size
        elif current_voxel_size != voxel_size:
            raise AssertionError("V4 voxel size changed between rebuilds")
        blocks.append(metric)
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary-json")
    args = parser.parse_args()

    blocks = load_and_validate(args.artifact_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    fields = sorted({key for block in blocks for key in block})
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(blocks)

    report = {
        "contract_valid": True,
        "artifact_root": os.path.abspath(args.artifact_root),
        "num_blocks": len(blocks),
        "final_voxel_count": int(blocks[-1]["voxel_count"]),
        "voxel_size": float(blocks[-1]["voxel_size"]),
        "da3_ms": sum(float(block["da3_ms"]) for block in blocks),
        "pointcloud_rebuild_ms": sum(
            float(block["pointcloud_rebuild_ms"]) for block in blocks
        ),
    }
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
