#!/usr/bin/env python3
"""Stream all valid Align3R RGB-D pixels into one world-space dense PLY.

The output coordinate gauge is identical to ``convert_align3r_to_inspatio.py``:
Align3R's first camera-to-world pose is normalized to identity.  Points are
written frame by frame, so export memory is bounded by one native RGB-D frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from convert_align3r_to_inspatio import (
    indexed_depths,
    load_intrinsics,
    load_normalized_c2w,
)


VERTEX_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
])


def _rgb_path(align3r_dir: Path, index: int) -> Path:
    return align3r_dir / f"frame_{index:04d}_rgb.png"


def _load_depth(path: Path, expected_shape: tuple[int, int] | None) -> np.ndarray:
    depth = np.asarray(np.load(path), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got {depth.shape}: {path}")
    if expected_shape is not None and depth.shape != expected_shape:
        raise ValueError(
            f"Inconsistent depth shape {depth.shape}, expected {expected_shape}: {path}"
        )
    return depth


def _load_rgb(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Align3R RGB frame: {path}")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != expected_shape:
        raise ValueError(
            f"RGB shape {rgb.shape[:2]} does not match depth {expected_shape}: {path}"
        )
    return rgb


def _ply_header(point_count: int) -> bytes:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment all valid native Align3R RGB-D pixels, first camera is world origin\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")


def _pose_statistics(c2w: np.ndarray) -> dict:
    translations = c2w[:, :3, 3].astype(np.float64)
    steps = np.diff(translations, axis=0)
    trace = np.trace(c2w[:, :3, :3].astype(np.float64), axis1=1, axis2=2)
    rotation_degrees = np.degrees(
        np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    )
    return {
        "translation_axis_min": translations.min(axis=0).tolist(),
        "translation_axis_max": translations.max(axis=0).tolist(),
        "translation_axis_span": np.ptp(translations, axis=0).tolist(),
        "translation_max_norm": float(np.linalg.norm(translations, axis=1).max()),
        "translation_path_length": float(np.linalg.norm(steps, axis=1).sum()),
        "rotation_angle_degrees_min": float(rotation_degrees.min()),
        "rotation_angle_degrees_max": float(rotation_degrees.max()),
    }


def export_dense_ply(
    align3r_dir: Path,
    output_path: Path,
    expected_frames: int,
    overwrite: bool = False,
) -> dict:
    """Export all positive finite depth pixels, without sampling or truncation."""
    align3r_dir = align3r_dir.resolve()
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    depth_paths = indexed_depths(align3r_dir, expected_frames)
    c2w = load_normalized_c2w(align3r_dir / "pred_traj.txt", expected_frames)
    intrinsics = load_intrinsics(
        align3r_dir / "pred_intrinsics.txt", expected_frames
    )

    depth_shape = None
    point_count = 0
    depth_min = float("inf")
    depth_max = float("-inf")
    per_frame_counts = []
    for index, depth_path in enumerate(depth_paths):
        depth = _load_depth(depth_path, depth_shape)
        if depth_shape is None:
            depth_shape = depth.shape
        _load_rgb(_rgb_path(align3r_dir, index), depth_shape)
        valid = np.isfinite(depth) & (depth > 0)
        count = int(valid.sum())
        if count:
            depth_min = min(depth_min, float(depth[valid].min()))
            depth_max = max(depth_max, float(depth[valid].max()))
        per_frame_counts.append(count)
        point_count += count

    if depth_shape is None or point_count == 0:
        raise ValueError("No positive finite Align3R depth pixels were found")

    height, width = depth_shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    digest = hashlib.sha256()
    header = _ply_header(point_count)

    try:
        with temporary_path.open("wb") as stream:
            stream.write(header)
            digest.update(header)
            for index, depth_path in enumerate(depth_paths):
                depth = _load_depth(depth_path, depth_shape)
                rgb = _load_rgb(_rgb_path(align3r_dir, index), depth_shape)
                valid = np.isfinite(depth) & (depth > 0)
                z = depth[valid]
                intrinsic = intrinsics[index]
                fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
                cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
                if not (fx > 0 and fy > 0):
                    raise ValueError(f"Invalid focal length at frame {index}: {fx}, {fy}")

                camera_points = np.empty((len(z), 3), dtype=np.float32)
                camera_points[:, 0] = (xx[valid] - cx) * z / fx
                camera_points[:, 1] = (yy[valid] - cy) * z / fy
                camera_points[:, 2] = z
                world_points = (
                    camera_points @ c2w[index, :3, :3].T
                    + c2w[index, :3, 3]
                )
                if not np.isfinite(world_points).all():
                    raise ValueError(f"Non-finite world point at frame {index}")

                vertices = np.empty(len(z), dtype=VERTEX_DTYPE)
                vertices["x"] = world_points[:, 0]
                vertices["y"] = world_points[:, 1]
                vertices["z"] = world_points[:, 2]
                vertices["red"] = rgb[..., 0][valid]
                vertices["green"] = rgb[..., 1][valid]
                vertices["blue"] = rgb[..., 2][valid]
                payload = vertices.tobytes()
                stream.write(payload)
                digest.update(payload)
                if (index + 1) % 25 == 0 or index + 1 == expected_frames:
                    print(
                        f"wrote {index + 1}/{expected_frames} frames "
                        f"({sum(per_frame_counts[:index + 1]):,}/{point_count:,} points)",
                        flush=True,
                    )
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    full_pixel_upper_bound = expected_frames * height * width
    manifest = {
        "format": "binary_little_endian_ply_xyz_float32_rgb_uint8",
        "source": str(align3r_dir),
        "output": str(output_path),
        "coordinate_frame": "Align3R world gauge normalized so frame-0 c2w is identity",
        "frame_count": expected_frames,
        "native_resolution": [width, height],
        "point_count": point_count,
        "full_pixel_upper_bound": full_pixel_upper_bound,
        "invalid_or_nonpositive_depth_pixels": full_pixel_upper_bound - point_count,
        "sampling": "all positive finite native depth pixels; no keyframes, voxelization, or truncation",
        "depth_min": depth_min,
        "depth_max": depth_max,
        "vertex_bytes": VERTEX_DTYPE.itemsize,
        "file_bytes": output_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "source_camera_pose": _pose_statistics(c2w),
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--align3r_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected_frames", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = export_dense_ply(
        args.align3r_dir,
        args.output,
        args.expected_frames,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
