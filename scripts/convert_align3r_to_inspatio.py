#!/usr/bin/env python3
"""Adapt an official Align3R reconstruction to InSpatio's RGB-D interface.

Align3R writes camera-to-world poses as ``timestamp tx ty tz qw qx qy qz``.
InSpatio's renderer consumes DA3-compatible float32-RGBA depths, per-frame RGB,
intrinsics, and 3x4 world-to-camera matrices.  The camera gauge is normalized
so the first Align3R pose is identity, matching InSpatio's trajectory renderer.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


DEPTH_PATTERN = re.compile(r"^frame_(\d{4})\.npy$")


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion [w,x,y,z], got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion is not finite and non-zero")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_normalized_c2w(path: Path, expected_frames: int) -> np.ndarray:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.shape != (expected_frames, 8):
        raise ValueError(
            f"Expected {expected_frames} Align3R pose rows with 8 fields, got {rows.shape}"
        )
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], expected_frames, axis=0)
    for index, row in enumerate(rows):
        c2w[index, :3, :3] = quaternion_wxyz_to_matrix(row[4:8])
        c2w[index, :3, 3] = row[1:4]
    first_inverse = np.linalg.inv(c2w[0])
    normalized = first_inverse[None] @ c2w
    if not np.isfinite(normalized).all():
        raise ValueError("Align3R poses contain non-finite values")
    return normalized.astype(np.float32)


def load_intrinsics(path: Path, expected_frames: int) -> np.ndarray:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.shape == (1, 9):
        rows = np.repeat(rows, expected_frames, axis=0)
    if rows.shape != (expected_frames, 9):
        raise ValueError(
            f"Expected one or {expected_frames} flattened intrinsics, got {rows.shape}"
        )
    intrinsics = rows.reshape(expected_frames, 3, 3)
    if not np.isfinite(intrinsics).all():
        raise ValueError("Align3R intrinsics contain non-finite values")
    return intrinsics.astype(np.float32)


def indexed_depths(align3r_dir: Path, expected_frames: int) -> list[Path]:
    indexed = {}
    for path in align3r_dir.iterdir():
        match = DEPTH_PATTERN.match(path.name)
        if match:
            indexed[int(match.group(1))] = path
    expected = list(range(expected_frames))
    if sorted(indexed) != expected:
        raise ValueError(
            f"Align3R depth indices must be consecutive 0..{expected_frames - 1}; "
            f"found {len(indexed)} files"
        )
    return [indexed[index] for index in expected]


def write_float32_rgba_depth(depth: np.ndarray, path: Path) -> None:
    depth = np.ascontiguousarray(depth, dtype="<f4")
    encoded = depth.view(np.uint8).reshape(depth.shape[0], depth.shape[1], 4)
    Image.fromarray(encoded, mode="RGBA").save(path)


def convert(align3r_dir: Path, output_dir: Path, expected_frames: int) -> dict:
    depth_paths = indexed_depths(align3r_dir, expected_frames)
    c2w = load_normalized_c2w(align3r_dir / "pred_traj.txt", expected_frames)
    intrinsics = load_intrinsics(
        align3r_dir / "pred_intrinsics.txt", expected_frames
    )

    frame_dir = output_dir / "frames"
    depth_dir = output_dir / "depth"
    for directory in (frame_dir, depth_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    depth_min = float("inf")
    depth_max = float("-inf")
    depth_shape = None
    for index, depth_path in enumerate(depth_paths):
        depth = np.asarray(np.load(depth_path), dtype=np.float32)
        if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"Invalid positive finite depth map: {depth_path}")
        if depth_shape is None:
            depth_shape = depth.shape
        elif depth.shape != depth_shape:
            raise ValueError(f"Inconsistent depth shape at {depth_path}: {depth.shape}")
        depth_min = min(depth_min, float(depth.min()))
        depth_max = max(depth_max, float(depth.max()))
        write_float32_rgba_depth(depth, depth_dir / f"{index:04d}.png")

        rgb_path = align3r_dir / f"frame_{index:04d}_rgb.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing Align3R RGB frame: {rgb_path}")
        with Image.open(rgb_path) as image:
            image.convert("RGB").save(frame_dir / f"{index:04d}.png")

    w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
    np.savetxt(output_dir / "extrinsic.txt", w2c[:, :3, :].reshape(-1, 4), fmt="%.9g")
    np.savetxt(output_dir / "intrinsic.txt", intrinsics.reshape(-1, 3), fmt="%.9g")

    manifest = {
        "backend": "align3r",
        "frame_count": expected_frames,
        "depth_shape": list(depth_shape),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "pose_input_format": "timestamp tx ty tz qw qx qy qz",
        "pose_gauge": "first_camera_identity",
        "first_c2w_identity_max_error": float(
            np.max(np.abs(c2w[0] - np.eye(4, dtype=np.float32)))
        ),
        "source": str(align3r_dir.resolve()),
    }
    (output_dir / "align3r_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--align3r_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_frames", type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = convert(args.align3r_dir, args.output_dir, args.expected_frames)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
