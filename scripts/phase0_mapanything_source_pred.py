#!/usr/bin/env python3
"""Phase-0 MapAnything source/pred comparison on existing InSpatio outputs."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.historical_point_memory import IncrementalVoxelSurfelMemory
from utils.mapanything_estimator import MapAnythingPointEstimator
from utils.overlap_da3_registration import backproject_world_grid, pose_residual


def read_video_frames(path: str, indices: list[int]) -> torch.Tensor:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {path}")
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {index} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0


def read_w2c_lines(path: str) -> torch.Tensor:
    matrices = []
    with open(path) as handle:
        for line in handle:
            matrix = np.asarray(ast.literal_eval(line.strip()), dtype=np.float32)
            if matrix.shape == (3, 4):
                matrix = np.concatenate(
                    (matrix, np.asarray([[0, 0, 0, 1]], dtype=np.float32)), axis=0
                )
            matrices.append(matrix)
    w2c = torch.from_numpy(np.stack(matrices))
    c2w = torch.linalg.inv(w2c)
    return torch.linalg.inv(c2w[0]) @ c2w


def read_mask_frames(path: str, indices: list[int]) -> torch.Tensor:
    return read_video_frames(path, indices)[:, 0] > 0.5


def parse_windows(value: str) -> list[list[int]]:
    windows = []
    for group in value.split(";"):
        indices = [int(item) for item in group.split(",") if item]
        if not indices:
            raise ValueError("Each window must contain at least one frame")
        windows.append(indices)
    return windows


def interleave(source: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.stack((source, pred), dim=1).flatten(0, 1)


def evaluate_and_save(
    *,
    name: str,
    batch,
    pred_slots: list[int],
    frame_indices: list[int],
    target_c2w: torch.Tensor,
    reference_depth: np.ndarray,
    reference_mask: torch.Tensor,
    voxel_size: float,
    output_dir: Path,
    apply_gate: bool,
) -> dict:
    selected_points = batch.points[pred_slots]
    selected_colors = batch.colors[pred_slots]
    selected_valid = batch.valid[pred_slots]
    selected_confidence = batch.confidence[pred_slots]
    selected_intrinsics = batch.intrinsics[pred_slots]
    native_height, native_width = batch.processed_size

    depth = torch.from_numpy(
        np.ascontiguousarray(reference_depth[frame_indices])
    ).float()
    depth = F.interpolate(
        depth.unsqueeze(1), size=(native_height, native_width), mode="nearest"
    ).squeeze(1).to(selected_points.device)
    mask = F.interpolate(
        reference_mask.float().unsqueeze(1),
        size=(native_height, native_width),
        mode="nearest",
    ).squeeze(1).bool().to(selected_points.device)

    reference_points, reference_valid = [], []
    for local_index, frame_index in enumerate(frame_indices):
        points, valid = backproject_world_grid(
            depth[local_index],
            selected_intrinsics[local_index],
            c2w=target_c2w[frame_index].to(selected_points.device),
        )
        reference_points.append(points)
        reference_valid.append(valid)
    reference_points = torch.stack(reference_points)
    reference_valid = torch.stack(reference_valid)

    correspondence = selected_valid & reference_valid & mask
    error = torch.linalg.norm(selected_points - reference_points, dim=-1)
    threshold = torch.maximum(
        torch.full_like(depth, 2.0 * voxel_size), 0.03 * depth
    )
    consistent = correspondence & torch.isfinite(error) & (error <= threshold)
    novel = selected_valid & ~mask
    keep = selected_valid
    if apply_gate:
        keep = novel | consistent

    valid_error = error[correspondence]
    pose_metrics = [
        pose_residual(
            target_c2w[frame_index].to(batch.camera_c2w.device),
            batch.camera_c2w[pred_slot],
        )
        for frame_index, pred_slot in zip(frame_indices, pred_slots)
    ]

    memory = IncrementalVoxelSurfelMemory(
        height=native_height,
        width=native_width,
        device=selected_points.device,
        K=selected_intrinsics[0],
        voxel_size=voxel_size,
        max_points=3_000_000,
        point_size=3,
    )
    update = memory.update_points(selected_points, selected_colors, keep)
    ply_path = memory.save_ply(str(output_dir / f"{name}.ply"))
    metric = {
        "name": name,
        "frame_indices": frame_indices,
        "view_count": int(batch.points.shape[0]),
        "pred_view_count": len(pred_slots),
        "processed_size": [native_height, native_width],
        "inference_ms": batch.inference_ms,
        "raw_valid_points": int(selected_valid.sum().item()),
        "reference_correspondences": int(correspondence.sum().item()),
        "reference_consistent_points": int(consistent.sum().item()),
        "reference_consistent_ratio": float(
            consistent.sum().item() / max(1, correspondence.sum().item())
        ),
        "reference_error_median": (
            None if valid_error.numel() == 0 else float(valid_error.median().item())
        ),
        "reference_error_p90": (
            None
            if valid_error.numel() == 0
            else float(torch.quantile(valid_error, 0.9).item())
        ),
        "novel_points": int(novel.sum().item()),
        "kept_points": int(keep.sum().item()),
        "confidence_mean": float(selected_confidence[selected_valid].mean().item()),
        "pose_rotation_degrees_mean": float(
            np.mean([metric["rotation_degrees"] for metric in pose_metrics])
        ),
        "pose_translation_mean": float(
            np.mean([metric["translation"] for metric in pose_metrics])
        ),
        "voxel_count": memory.point_count,
        "ply_path": ply_path,
        **update,
    }
    with open(output_dir / f"{name}.json", "w") as handle:
        json.dump(metric, handle, indent=2)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--pred-video", required=True)
    parser.add_argument("--source-extrinsics", required=True)
    parser.add_argument("--target-c2w", required=True)
    parser.add_argument("--intrinsic", required=True)
    parser.add_argument("--reference-depth", required=True)
    parser.add_argument("--reference-mask-video", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--windows", default="40,44,48;112,116,120;188,192,196"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.0037405583595979293)
    parser.add_argument("--confidence-percentile", type=float, default=10.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_c2w = torch.from_numpy(np.load(args.target_c2w).astype(np.float32))
    source_c2w = read_w2c_lines(args.source_extrinsics)
    intrinsic = torch.from_numpy(np.load(args.intrinsic).astype(np.float32))
    reference_depth = np.load(args.reference_depth, mmap_mode="r")

    estimator = MapAnythingPointEstimator(
        args.model_path,
        torch.device(args.device),
        confidence_percentile=args.confidence_percentile,
    )
    all_metrics = []
    for window in parse_windows(args.windows):
        window_dir = output_dir / ("frames_" + "_".join(map(str, window)))
        window_dir.mkdir(parents=True, exist_ok=True)
        source = read_video_frames(args.source_video, window)
        pred = read_video_frames(args.pred_video, window)
        reference_mask = read_mask_frames(args.reference_mask_video, window)
        repeated_k = intrinsic.unsqueeze(0).repeat(len(window), 1, 1)

        pred_batch = estimator.estimate_views(
            pred,
            intrinsics_t33=repeated_k,
            camera_c2w_t44=target_c2w[window],
        )
        all_metrics.append(evaluate_and_save(
            name="pred_only",
            batch=pred_batch,
            pred_slots=list(range(len(window))),
            frame_indices=window,
            target_c2w=target_c2w,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
            voxel_size=args.voxel_size,
            output_dir=window_dir,
            apply_gate=False,
        ))

        paired_rgb = interleave(source, pred)
        paired_pose = interleave(source_c2w[window], target_c2w[window])
        paired_k = intrinsic.unsqueeze(0).repeat(paired_rgb.shape[0], 1, 1)
        paired_batch = estimator.estimate_views(
            paired_rgb,
            intrinsics_t33=paired_k,
            camera_c2w_t44=paired_pose,
        )
        pred_slots = list(range(1, paired_rgb.shape[0], 2))
        for name, apply_gate in (("paired", False), ("paired_gate", True)):
            all_metrics.append(evaluate_and_save(
                name=name,
                batch=paired_batch,
                pred_slots=pred_slots,
                frame_indices=window,
                target_c2w=target_c2w,
                reference_depth=reference_depth,
                reference_mask=reference_mask,
                voxel_size=args.voxel_size,
                output_dir=window_dir,
                apply_gate=apply_gate,
            ))

    summary = {
        "model_path": args.model_path,
        "device": args.device,
        "confidence_percentile": args.confidence_percentile,
        "peak_memory_gb": estimator.last_peak_memory_gb,
        "metrics": all_metrics,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
