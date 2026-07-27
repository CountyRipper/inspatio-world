#!/usr/bin/env python3
"""Audit dense-two-layer artifacts without loading the full map at once."""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def video_contract(path):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return result


def mask_depth_mismatch(mask_path, depth):
    capture = cv2.VideoCapture(mask_path)
    mismatched = 0
    pixels = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_mask = frame[:, :, 0] > 127
        depth_mask = depth[frame_index] > 0
        mismatched += int(np.count_nonzero(decoded_mask != depth_mask))
        pixels += int(decoded_mask.size)
        frame_index += 1
    capture.release()
    if frame_index != depth.shape[0]:
        raise AssertionError(f"Mask/depth frame mismatch: {frame_index} vs {depth.shape[0]}")
    return mismatched, pixels


def audit_map(output_dir, prefix, expected_output_frames, expected_point_count=None,
              expected_depth_backend=None):
    timing_path = os.path.join(output_dir, f"{prefix}-memory_timing_rank0.json")
    map_manifest_path = os.path.join(
        output_dir, f"{prefix}-historical_memory_final_rank0_manifest.json"
    )
    with open(timing_path) as handle:
        timing = json.load(handle)
    with open(map_manifest_path) as handle:
        map_manifest = json.load(handle)

    total_points = 0
    confidence_min = 1.0
    confidence_max = 0.0
    for chunk in map_manifest["chunks"]:
        with np.load(chunk["path"]) as arrays:
            points = arrays["points"]
            colors = arrays["colors"]
            confidence = arrays["confidence"]
            if points.dtype != np.float32 or colors.dtype != np.float32:
                raise AssertionError("points/colors must be float32")
            if confidence.dtype != np.float16:
                raise AssertionError("confidence must be float16")
            if points.shape != colors.shape or points.shape[1] != 3:
                raise AssertionError("points/colors shapes are not aligned")
            if confidence.shape != (points.shape[0],):
                raise AssertionError("confidence shape is not aligned")
            if not np.isfinite(points).all() or not np.isfinite(colors).all():
                raise AssertionError("non-finite point/color found")
            if not np.isfinite(confidence).all():
                raise AssertionError("non-finite confidence found")
            total_points += int(points.shape[0])
            confidence_min = min(confidence_min, float(confidence.min()))
            confidence_max = max(confidence_max, float(confidence.max()))

    summary = timing["summary"]
    if total_points != summary["final_point_count"] or total_points != map_manifest["point_count"]:
        raise AssertionError("map point counts disagree")
    if not (1e-3 - 1e-5 <= confidence_min <= confidence_max <= 1.0):
        raise AssertionError("confidence is outside [1e-3, 1]")
    if expected_point_count is not None and total_points != expected_point_count:
        raise AssertionError(
            f"Unexpected point count: {total_points} vs {expected_point_count}"
        )
    if expected_depth_backend is not None and summary.get("depth_backend") != expected_depth_backend:
        raise AssertionError(
            f"Unexpected depth backend: {summary.get('depth_backend')}"
        )
    if sum(int(block.get("update_frames", 0)) for block in timing["blocks"]) != expected_output_frames:
        raise AssertionError("Memory updates do not cover every generated RGB frame")
    last_block = timing["blocks"][-1]
    if last_block["historical_coverage"] <= 0:
        raise AssertionError("historical coverage is zero at the second 45-degree visit")
    if last_block["fused_coverage"] < last_block["reference_coverage"]:
        raise AssertionError("fused coverage is below reference coverage")

    pred_contract = video_contract(os.path.join(output_dir, f"{prefix}-pred_video_rank0.mp4"))
    if pred_contract != {
        "width": 832, "height": 480, "fps": 24.0, "frames": expected_output_frames
    }:
        raise AssertionError(f"Unexpected prediction contract: {pred_contract}")
    return {
        "summary": summary,
        "prediction": pred_contract,
        "chunk_count": len(map_manifest["chunks"]),
        "finite_point_count": total_points,
        "confidence_min": confidence_min,
        "confidence_max": confidence_max,
        "revisit_historical_coverage": last_block["historical_coverage"],
        "revisit_fused_coverage": last_block["fused_coverage"],
    }


def audit_reference(render_dir, expected_reference_frames, yaw_indices, yaw_values):
    depth = np.load(os.path.join(render_dir, "depth_offline.npy"), mmap_mode="r")
    if depth.shape != (expected_reference_frames, 480, 832) or depth.dtype != np.float32:
        raise AssertionError(f"Unexpected reference depth contract: {depth.shape}, {depth.dtype}")
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise AssertionError("reference depth is non-finite or negative")
    mismatch, pixels = mask_depth_mismatch(
        os.path.join(render_dir, "mask_offline.mp4"), depth
    )
    poses = np.load(os.path.join(render_dir, "target_c2w.npy"))
    relative = np.swapaxes(poses[0, :3, :3], 0, 1) @ poses[:, :3, :3]
    yaw = np.degrees(np.arctan2(relative[:, 0, 2], relative[:, 0, 0]))
    landmark_yaw = [float(yaw[index]) for index in yaw_indices]
    expected = np.asarray(yaw_values, dtype=np.float32)
    if not np.allclose(np.abs(landmark_yaw), expected, atol=2e-3):
        raise AssertionError(f"Unexpected yaw landmarks: {landmark_yaw}")
    return {
        "depth_shape": list(depth.shape),
        "finite_depth": True,
        "mask_depth_mismatch_pixels": mismatch,
        "mask_depth_mismatch_fraction": mismatch / pixels,
        "yaw_landmarks_degrees": landmark_yaw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_render_dirs", nargs="+", required=True)
    parser.add_argument("--expected_output_frames", type=int, default=117)
    parser.add_argument("--expected_reference_frames", type=int, default=120)
    parser.add_argument("--expected_point_count", type=int)
    parser.add_argument("--expected_depth_backend", choices=["da3", "align3r"])
    parser.add_argument(
        "--yaw_indices", nargs="+", type=int,
        default=[0, 39, 78, 116, 117, 118, 119],
    )
    parser.add_argument(
        "--yaw_values", nargs="+", type=float,
        default=[0, 45, 0, 45, 45, 45, 45],
    )
    args = parser.parse_args()
    if len(args.yaw_indices) != len(args.yaw_values):
        raise ValueError("--yaw_indices and --yaw_values must have equal length")

    timing_paths = sorted(glob.glob(os.path.join(args.output_dir, "*-memory_timing_rank0.json")))
    prefixes = [os.path.basename(path).split("-", 1)[0] for path in timing_paths]
    if len(prefixes) != len(args.reference_render_dirs):
        raise AssertionError("output prefixes and reference render directories differ")
    report = {
        "maps": [
            audit_map(
                args.output_dir,
                prefix,
                args.expected_output_frames,
                args.expected_point_count,
                args.expected_depth_backend,
            )
            for prefix in prefixes
        ],
        "references": [
            audit_reference(
                path,
                args.expected_reference_frames,
                args.yaw_indices,
                args.yaw_values,
            )
            for path in args.reference_render_dirs
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
