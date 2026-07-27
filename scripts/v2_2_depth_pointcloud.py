#!/usr/bin/env python3
"""Prepare and build the controlled DA3-vs-Align3R V2.2 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


PLY_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
])


def selected_frame_indices(expected_frames: int, frame_step: int) -> list[int]:
    if expected_frames < 1:
        raise ValueError("expected_frames must be positive")
    if frame_step < 1:
        raise ValueError("frame_step must be positive")
    indices = list(range(0, expected_frames, frame_step))
    if indices[-1] != expected_frames - 1:
        raise ValueError("The frame schedule must include the final generated frame")
    return indices


def require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video, image, and resize operations")
    return cv2


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_depths(depths: np.ndarray, height: int, width: int) -> np.ndarray:
    depths = np.asarray(depths, dtype=np.float32)
    if depths.ndim != 3:
        raise ValueError(f"Expected depth [T,H,W], got {depths.shape}")
    if depths.shape[1:] == (height, width):
        return np.ascontiguousarray(depths)
    cv = require_cv2()
    return np.stack([
        cv.resize(depth, (width, height), interpolation=cv.INTER_NEAREST)
        for depth in depths
    ]).astype(np.float32, copy=False)


def log_depth_gradient(depths: np.ndarray, min_depth: float = 0.1) -> np.ndarray:
    depths = np.asarray(depths, dtype=np.float32)
    valid = np.isfinite(depths) & (depths > min_depth)
    safe = np.where(valid, depths, 1.0).clip(min=min_depth)
    log_depth = np.log(safe)
    gradient = np.zeros_like(log_depth)
    vertical = np.abs(log_depth[:, 1:] - log_depth[:, :-1])
    horizontal = np.abs(log_depth[:, :, 1:] - log_depth[:, :, :-1])
    gradient[:, 1:] = np.maximum(gradient[:, 1:], vertical)
    gradient[:, :-1] = np.maximum(gradient[:, :-1], vertical)
    gradient[:, :, 1:] = np.maximum(gradient[:, :, 1:], horizontal)
    gradient[:, :, :-1] = np.maximum(gradient[:, :, :-1], horizontal)
    gradient[~valid] = np.inf
    return gradient


def global_depth_scale(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
    calibration_mask: np.ndarray,
    trim_fraction: float = 0.05,
) -> tuple[float, dict]:
    if predicted_depth.shape != reference_depth.shape:
        raise ValueError(
            f"Depth shapes differ: {predicted_depth.shape} vs {reference_depth.shape}"
        )
    if calibration_mask.shape != predicted_depth.shape:
        raise ValueError("Calibration mask must match the depth arrays")
    ratios = np.log(reference_depth[calibration_mask] / predicted_depth[calibration_mask])
    if ratios.size < 4096:
        raise RuntimeError(f"Only {ratios.size} global scale anchor pixels are available")
    lower, upper = np.quantile(ratios, [trim_fraction, 1.0 - trim_fraction])
    trimmed = ratios[(ratios >= lower) & (ratios <= upper)]
    median = float(np.median(trimmed))
    mad = float(np.median(np.abs(trimmed - median)))
    scale = float(np.exp(median))
    return scale, {
        "scale": scale,
        "overlap_pixels": int(ratios.size),
        "trimmed_overlap_pixels": int(trimmed.size),
        "log_mad": mad,
        "median_abs_log_residual": float(np.median(np.abs(trimmed - median))),
        "p90_abs_log_residual": float(np.quantile(np.abs(trimmed - median), 0.90)),
    }


def backproject_world(depth: np.ndarray, intrinsic: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    z = depth.astype(np.float32, copy=False)
    camera = np.stack([
        (u - intrinsic[0, 2]) * z / intrinsic[0, 0],
        (v - intrinsic[1, 2]) * z / intrinsic[1, 1],
        z,
    ], axis=-1)
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


def write_binary_ply(
    path: Path,
    depths: np.ndarray,
    frame_paths: list[Path],
    intrinsic: np.ndarray,
    c2w: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[int, str]:
    cv = require_cv2()
    point_count = int(valid_mask.sum())
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    rgb_digest = hashlib.sha256()
    with path.open("wb") as handle:
        handle.write(header)
        for frame_index, frame_path in enumerate(frame_paths):
            bgr = cv.imread(str(frame_path), cv.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Could not read {frame_path}")
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            mask = valid_mask[frame_index]
            world = backproject_world(depths[frame_index], intrinsic, c2w[frame_index])[mask]
            colors = rgb[mask]
            if not np.isfinite(world).all():
                raise RuntimeError(f"Non-finite points in frame {frame_index}")
            records = np.empty(world.shape[0], dtype=PLY_DTYPE)
            records["x"], records["y"], records["z"] = world.T
            records["red"], records["green"], records["blue"] = colors.T
            rgb_digest.update(np.ascontiguousarray(colors).tobytes())
            handle.write(records.tobytes())
    return point_count, rgb_digest.hexdigest()


def prepare(args: argparse.Namespace) -> None:
    cv = require_cv2()
    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    keyframe_indices = selected_frame_indices(args.expected_frames, args.frame_step)

    capture = cv.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    selected = []
    frame_index = 0
    keyframe_set = set(keyframe_indices)
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if frame_index in keyframe_set:
            path = frame_dir / f"frame_{frame_index:04d}.png"
            if not cv.imwrite(str(path), bgr):
                raise RuntimeError(f"Could not write {path}")
            selected.append(path)
        frame_index += 1
    capture.release()
    if frame_index != args.expected_frames or len(selected) != len(keyframe_indices):
        raise RuntimeError(
            f"Video contract failed: frames={frame_index}, keyframes={len(selected)}"
        )

    sample = cv.imread(str(selected[0]), cv.IMREAD_COLOR)
    height, width = sample.shape[:2]
    c2w = np.load(args.target_c2w)
    reference_depth = np.load(args.reference_depth, mmap_mode="r")
    intrinsic = np.load(args.intrinsic)
    if c2w.shape != (args.expected_frames, 4, 4):
        raise ValueError(f"Unexpected c2w shape: {c2w.shape}")
    if reference_depth.shape != (args.expected_frames, height, width):
        raise ValueError(f"Unexpected reference depth shape: {reference_depth.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Unexpected intrinsic shape: {intrinsic.shape}")
    if not np.isfinite(c2w).all() or not np.isfinite(intrinsic).all():
        raise ValueError("Camera arrays contain non-finite values")

    c2w_output = output_dir / "target_c2w_keyframes.npy"
    intrinsic_output = output_dir / "intrinsic.npy"
    reference_output = output_dir / "reference_depth_keyframes.npy"
    np.save(c2w_output, c2w[keyframe_indices].astype(np.float32))
    np.save(intrinsic_output, intrinsic.astype(np.float32))
    np.save(
        reference_output,
        np.asarray(reference_depth[keyframe_indices], dtype=np.float32),
    )
    manifest = {
        "generated_video": str(Path(args.video).resolve()),
        "generated_video_sha256": sha256_file(Path(args.video)),
        "expected_frames": args.expected_frames,
        "frame_step": args.frame_step,
        "keyframe_indices": keyframe_indices,
        "keyframe_count": len(keyframe_indices),
        "height": height,
        "width": width,
        "target_c2w_source": str(Path(args.target_c2w).resolve()),
        "target_c2w_keyframes_sha256": sha256_file(c2w_output),
        "intrinsic_source": str(Path(args.intrinsic).resolve()),
        "intrinsic_sha256": sha256_file(intrinsic_output),
        "reference_depth_source": str(Path(args.reference_depth).resolve()),
        "reference_depth_keyframes_sha256": sha256_file(reference_output),
    }
    (output_dir / "prepared_manifest.json").write_text(json.dumps(manifest, indent=2))


def load_align3r_depths(depth_dir: Path) -> np.ndarray:
    pattern = re.compile(r"^frame_\d{4}\.npy$")
    paths = [path for path in sorted(depth_dir.iterdir()) if pattern.match(path.name)]
    if not paths:
        raise RuntimeError(f"No Align3R frame depth arrays found in {depth_dir}")
    depths = [np.asarray(np.load(path), dtype=np.float32) for path in paths]
    if len({depth.shape for depth in depths}) != 1:
        raise RuntimeError("Align3R depth shapes are inconsistent")
    return np.stack(depths)


def build_align3r_full_frames(args: argparse.Namespace) -> None:
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((prepared_dir / "prepared_manifest.json").read_text())
    frame_indices = prepared["keyframe_indices"]
    if frame_indices != list(range(prepared["expected_frames"])):
        raise RuntimeError("Align3R full-frame build requires every consecutive frame")
    frame_paths = [
        prepared_dir / "frames" / f"frame_{index:04d}.png"
        for index in frame_indices
    ]
    c2w = np.load(prepared_dir / "target_c2w_keyframes.npy")
    intrinsic = np.load(prepared_dir / "intrinsic.npy")
    reference = np.load(prepared_dir / "reference_depth_keyframes.npy")
    height, width = reference.shape[1:]

    align3r_source = load_align3r_depths(Path(args.align3r_depth_dir))
    align3r_raw = resize_depths(align3r_source, height, width)
    if align3r_raw.shape != reference.shape:
        raise RuntimeError(
            f"Depth count mismatch: ref={reference.shape}, Align3R={align3r_raw.shape}"
        )

    min_depth = float(args.min_depth)
    valid = np.isfinite(align3r_raw) & (align3r_raw > min_depth)
    calibration_mask = (
        valid
        & np.isfinite(reference) & (reference > min_depth)
        & (log_depth_gradient(reference, min_depth) <= args.max_log_gradient)
        & (log_depth_gradient(align3r_raw, min_depth) <= args.max_log_gradient)
    )
    scale, scale_stats = global_depth_scale(
        align3r_raw, reference, calibration_mask
    )
    depth = np.ascontiguousarray(align3r_raw * scale, dtype=np.float32)
    depth_path = output_dir / "align3r_full_frame_global_scaled_depth.npy"
    mask_path = output_dir / "align3r_full_frame_valid_mask.npy"
    pointcloud_path = output_dir / "align3r_full_frame_pointcloud.ply"
    np.save(depth_path, depth)
    np.save(mask_path, valid)
    point_count, rgb_sha = write_binary_ply(
        pointcloud_path, depth, frame_paths, intrinsic, c2w, valid
    )

    manifest = {
        "protocol": "Align3R on every consecutive generated RGB frame with known target c2w and K",
        "generated_video": prepared["generated_video"],
        "generated_video_sha256": prepared["generated_video_sha256"],
        "frame_indices": frame_indices,
        "frame_count": len(frame_indices),
        "source_depth_shape": list(align3r_source.shape),
        "depth_shape": list(depth.shape),
        "target_c2w_sha256": prepared["target_c2w_keyframes_sha256"],
        "intrinsic_sha256": prepared["intrinsic_sha256"],
        "reference_depth_sha256": prepared["reference_depth_keyframes_sha256"],
        "calibration_pixel_count": int(calibration_mask.sum()),
        "point_count": point_count,
        "rgb_stream_sha256": rgb_sha,
        "global_scale": scale_stats,
        "raw_depth_dir": str(Path(args.align3r_depth_dir).resolve()),
        "scaled_depth": str(depth_path.resolve()),
        "valid_mask": str(mask_path.resolve()),
        "pointcloud": str(pointcloud_path.resolve()),
        "pointcloud_sha256": sha256_file(pointcloud_path),
    }
    (output_dir / "align3r_full_frame_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


def build(args: argparse.Namespace) -> None:
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((prepared_dir / "prepared_manifest.json").read_text())
    frame_paths = [
        prepared_dir / "frames" / f"frame_{index:04d}.png"
        for index in prepared["keyframe_indices"]
    ]
    c2w = np.load(prepared_dir / "target_c2w_keyframes.npy")
    intrinsic = np.load(prepared_dir / "intrinsic.npy")
    reference = np.load(prepared_dir / "reference_depth_keyframes.npy")
    height, width = reference.shape[1:]

    da3_source = np.load(args.da3_depth)
    align3r_source = load_align3r_depths(Path(args.align3r_depth_dir))
    da3_raw = resize_depths(da3_source, height, width)
    align3r_raw = resize_depths(align3r_source, height, width)
    expected_shape = reference.shape
    if da3_raw.shape != expected_shape or align3r_raw.shape != expected_shape:
        raise RuntimeError(
            f"Depth count mismatch: ref={expected_shape}, DA3={da3_raw.shape}, "
            f"Align3R={align3r_raw.shape}"
        )

    min_depth = float(args.min_depth)
    common_valid = (
        np.isfinite(da3_raw) & (da3_raw > min_depth)
        & np.isfinite(align3r_raw) & (align3r_raw > min_depth)
    )
    calibration_mask = (
        common_valid
        & np.isfinite(reference) & (reference > min_depth)
        & (log_depth_gradient(reference, min_depth) <= args.max_log_gradient)
        & (log_depth_gradient(da3_raw, min_depth) <= args.max_log_gradient)
        & (log_depth_gradient(align3r_raw, min_depth) <= args.max_log_gradient)
    )
    da3_scale, da3_stats = global_depth_scale(da3_raw, reference, calibration_mask)
    align3r_scale, align3r_stats = global_depth_scale(
        align3r_raw, reference, calibration_mask
    )
    da3_depth = np.ascontiguousarray(da3_raw * da3_scale, dtype=np.float32)
    align3r_depth = np.ascontiguousarray(align3r_raw * align3r_scale, dtype=np.float32)
    np.save(output_dir / "da3_global_scaled_depth.npy", da3_depth)
    np.save(output_dir / "align3r_global_scaled_depth.npy", align3r_depth)
    np.save(output_dir / "common_valid_mask.npy", common_valid)

    da3_ply = output_dir / "da3_global_scale_pointcloud.ply"
    align3r_ply = output_dir / "align3r_global_depth_pointcloud.ply"
    da3_count, da3_rgb_sha = write_binary_ply(
        da3_ply, da3_depth, frame_paths, intrinsic, c2w, common_valid
    )
    align3r_count, align3r_rgb_sha = write_binary_ply(
        align3r_ply, align3r_depth, frame_paths, intrinsic, c2w, common_valid
    )
    if da3_count != align3r_count or da3_rgb_sha != align3r_rgb_sha:
        raise AssertionError("The two point clouds do not use identical pixels/colors")

    comparison = {
        "protocol": "same generated RGB, keyframes, valid pixels, RGB, K, and known c2w",
        "generated_video": prepared["generated_video"],
        "generated_video_sha256": prepared["generated_video_sha256"],
        "keyframe_indices": prepared["keyframe_indices"],
        "keyframe_count": prepared["keyframe_count"],
        "source_depth_shapes": {
            "da3": list(da3_source.shape),
            "align3r": list(align3r_source.shape),
        },
        "target_c2w_keyframes_sha256": prepared["target_c2w_keyframes_sha256"],
        "intrinsic_sha256": prepared["intrinsic_sha256"],
        "reference_depth_keyframes_sha256": prepared[
            "reference_depth_keyframes_sha256"
        ],
        "depth_shape": list(expected_shape),
        "calibration_pixel_count": int(calibration_mask.sum()),
        "common_point_count": da3_count,
        "rgb_stream_sha256": da3_rgb_sha,
        "da3_global_scale": {
            **da3_stats,
            "raw_depth": str(Path(args.da3_depth).resolve()),
            "scaled_depth": str((output_dir / "da3_global_scaled_depth.npy").resolve()),
            "pointcloud": str(da3_ply.resolve()),
            "pointcloud_sha256": sha256_file(da3_ply),
        },
        "align3r_global_depth": {
            **align3r_stats,
            "raw_depth_dir": str(Path(args.align3r_depth_dir).resolve()),
            "scaled_depth": str((output_dir / "align3r_global_scaled_depth.npy").resolve()),
            "pointcloud": str(align3r_ply.resolve()),
            "pointcloud_sha256": sha256_file(align3r_ply),
        },
    }
    (output_dir / "comparison_manifest.json").write_text(json.dumps(comparison, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--video", required=True)
    prepare_parser.add_argument("--target-c2w", required=True)
    prepare_parser.add_argument("--intrinsic", required=True)
    prepare_parser.add_argument("--reference-depth", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--expected-frames", type=int, default=237)
    prepare_parser.add_argument("--frame-step", type=int, default=4)
    prepare_parser.set_defaults(func=prepare)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--prepared-dir", required=True)
    build_parser.add_argument("--da3-depth", required=True)
    build_parser.add_argument("--align3r-depth-dir", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--min-depth", type=float, default=0.1)
    build_parser.add_argument("--max-log-gradient", type=float, default=0.05)
    build_parser.set_defaults(func=build)

    align3r_parser = subparsers.add_parser("build-align3r-full-frames")
    align3r_parser.add_argument("--prepared-dir", required=True)
    align3r_parser.add_argument("--align3r-depth-dir", required=True)
    align3r_parser.add_argument("--output-dir", required=True)
    align3r_parser.add_argument("--min-depth", type=float, default=0.1)
    align3r_parser.add_argument("--max-log-gradient", type=float, default=0.05)
    align3r_parser.set_defaults(func=build_align3r_full_frames)
    return parser


if __name__ == "__main__":
    parsed = make_parser().parse_args()
    parsed.func(parsed)
