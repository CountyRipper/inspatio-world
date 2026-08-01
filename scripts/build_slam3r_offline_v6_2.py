#!/usr/bin/env python3
"""Build and render one fixed canonical map from official SLAM3R predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.historical_point_memory import RGBPointMemory, VideoStreamWriter
from utils.slam3r_incremental import CenterCropTransform, prepare_reference_geometry


def apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray,
               translation: np.ndarray) -> np.ndarray:
    """Apply y = scale * R * x + t to row-vector points."""
    return scale * (points @ rotation.T) + translation


def weighted_umeyama(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate the Sim(3) mapping source points onto target points."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matched [N,3] arrays, got {source.shape}, {target.shape}")
    if source.shape[0] < 4:
        raise ValueError("At least four correspondences are required")
    if weights is None:
        weights = np.ones(source.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(weights)
        & (weights > 0)
    )
    source, target, weights = source[valid], target[valid], weights[valid]
    if source.shape[0] < 4:
        raise ValueError("Too few finite, positive-weight correspondences")
    weights = weights / weights.sum()
    source_mean = np.sum(source * weights[:, None], axis=0)
    target_mean = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered * weights[:, None]).T @ source_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1] = -1.0
    rotation = u @ np.diag(correction) @ vt
    variance = np.sum(weights * np.sum(source_centered ** 2, axis=1))
    if not np.isfinite(variance) or variance <= 1e-12:
        raise ValueError("Degenerate source geometry for Sim(3)")
    scale = float(np.sum(singular_values * correction) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid Sim(3) scale: {scale}")
    return scale, rotation.astype(np.float64), translation.astype(np.float64)


def robust_sim3(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold: float,
    seed: int = 42,
    trials: int = 256,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """RANSAC followed by confidence-weighted Huber Sim(3) refinement."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if source.shape[0] < 32:
        raise ValueError(f"Only {source.shape[0]} Sim(3) correspondences")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        indices = rng.choice(source.shape[0], size=4, replace=False)
        try:
            scale, rotation, translation = weighted_umeyama(
                source[indices], target[indices], confidence[indices]
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = np.linalg.norm(
            apply_sim3(source, scale, rotation, translation) - target, axis=1
        )
        inliers = residual < threshold
        score = (int(inliers.sum()), -float(np.median(residual[inliers])) if inliers.any() else -np.inf)
        if best is None or score > best[0]:
            best = (score, scale, rotation, translation, inliers)
    if best is None or best[0][0] < 16:
        raise RuntimeError("Robust Sim(3) initialization failed")

    scale, rotation, translation, inliers = best[1:]
    base_weight = np.sqrt(np.maximum(confidence, 1e-6))
    for _ in range(6):
        residual = np.linalg.norm(
            apply_sim3(source, scale, rotation, translation) - target, axis=1
        )
        huber = np.ones_like(residual)
        large = residual > threshold
        huber[large] = threshold / np.maximum(residual[large], 1e-12)
        # Keep all points softly, while preserving the RANSAC consensus preference.
        weights = base_weight * huber * np.where(inliers, 1.0, 0.1)
        scale, rotation, translation = weighted_umeyama(source, target, weights)
        residual = np.linalg.norm(
            apply_sim3(source, scale, rotation, translation) - target, axis=1
        )
        inliers = residual < threshold

    residual = np.linalg.norm(
        apply_sim3(source, scale, rotation, translation) - target, axis=1
    )
    diagnostics = {
        "correspondences": int(source.shape[0]),
        "threshold": float(threshold),
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "residual_p50": float(np.percentile(residual, 50)),
        "residual_p90": float(np.percentile(residual, 90)),
        "residual_p95": float(np.percentile(residual, 95)),
    }
    return scale, rotation, translation, diagnostics


def best_confidence_voxels(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    source_ids: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep exactly the highest-confidence original observation in each voxel."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    source_ids = np.asarray(source_ids, dtype=np.int32).reshape(-1)
    if not (points.shape == colors.shape and points.ndim == 2 and points.shape[1] == 3):
        raise ValueError("points and colors must both be [N,3]")
    if confidence.shape[0] != points.shape[0] or source_ids.shape[0] != points.shape[0]:
        raise ValueError("Point metadata lengths differ")
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(confidence)
    )
    points, colors = points[valid], np.clip(colors[valid], 0, 1)
    confidence, source_ids = confidence[valid], source_ids[valid]
    if points.shape[0] == 0:
        return points, colors, confidence, source_ids
    voxels = np.floor(points / voxel_size).astype(np.int64)
    # Primary key is voxel xyz; final key is descending confidence.
    order = np.lexsort((-confidence, voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    sorted_voxels = voxels[order]
    first = np.ones(order.shape[0], dtype=bool)
    first[1:] = np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1)
    keep = order[first]
    return points[keep], colors[keep], confidence[keep], source_ids[keep]


def strict_reference_priority_fusion(
    reference_rgb: torch.Tensor,
    reference_mask: torch.Tensor,
    historical_rgb: torch.Tensor,
    historical_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse [0,1] tensors with black invalid pixels and immutable reference priority."""
    reference_mask = reference_mask.bool()
    historical_mask = historical_mask.bool()
    historical_add = historical_mask & ~reference_mask
    fused_mask = reference_mask | historical_add
    fused_rgb = torch.zeros_like(reference_rgb)
    fused_rgb = torch.where(reference_mask.expand_as(fused_rgb), reference_rgb, fused_rgb)
    fused_rgb = torch.where(historical_add.expand_as(fused_rgb), historical_rgb, fused_rgb)
    return fused_rgb, fused_mask, historical_add


def _selected_video_masks(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(indices)
    result: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_index = 0
    try:
        while wanted:
            success, frame = capture.read()
            if not success:
                break
            if frame_index in wanted:
                result[frame_index] = np.any(frame > 127, axis=2)
                wanted.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if wanted:
        raise RuntimeError(f"Missing mask frames in {path}: {sorted(wanted)}")
    return result


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    values = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        float(capture.get(cv2.CAP_PROP_FPS)),
    )
    capture.release()
    return values


def _write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    source_ids: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = (np.clip(colors, 0, 1) * 255).round().astype(np.uint8)
    vertices = np.empty(points.shape[0], dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("confidence", "<f4"), ("source_frame", "<i4"),
    ])
    if points.shape[0]:
        vertices["x"], vertices["y"], vertices["z"] = points.astype(np.float32).T
        vertices["red"], vertices["green"], vertices["blue"] = colors_u8.T
        vertices["confidence"] = confidence.astype(np.float32)
        vertices["source_frame"] = source_ids.astype(np.int32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float confidence\nproperty int source_frame\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def _safe_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _reference_correspondences(
    registered_pcds: np.ndarray,
    registered_confs: np.ndarray,
    manifest: dict,
    reference_depth: np.ndarray,
    target_c2w: np.ndarray,
    intrinsic: np.ndarray,
    reference_masks: dict[int, np.ndarray],
    *,
    frame_count: int,
    confidence_threshold: float,
    per_frame_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    fit_source, fit_target, fit_confidence = [], [], []
    validation_source, validation_target, validation_confidence = [], [], []
    frame_details = []
    rng = np.random.default_rng(42)
    for keyframe in manifest["keyframes"][:frame_count]:
        keyframe_index = int(keyframe["keyframe_index"])
        planned_index = int(keyframe["planned_c2w_index"])
        crop = CenterCropTransform(**keyframe["crop"])
        geometry = prepare_reference_geometry(
            torch.from_numpy(np.asarray(reference_depth[planned_index])),
            torch.from_numpy(reference_masks[planned_index]),
            torch.from_numpy(np.asarray(intrinsic)),
            torch.from_numpy(np.asarray(target_c2w[planned_index])),
            crop,
            device=device,
        )
        source = np.asarray(registered_pcds[keyframe_index], dtype=np.float32)
        confidence = np.asarray(registered_confs[keyframe_index], dtype=np.float32)
        target = geometry.points.detach().cpu().numpy().astype(np.float32)
        valid_reference = (
            geometry.valid & geometry.mask & torch.isfinite(geometry.points).all(dim=-1)
        ).detach().cpu().numpy()
        valid = (
            valid_reference
            & np.isfinite(source).all(axis=-1)
            & (np.linalg.norm(source, axis=-1) > 1e-8)
            & np.isfinite(confidence)
            & (confidence >= confidence_threshold)
        )
        yy, xx = np.indices(valid.shape)
        fit_valid = valid & (((xx + yy) & 1) == 0)
        validation_valid = valid & ~fit_valid
        frame_record = {"keyframe_index": keyframe_index, "planned_c2w_index": planned_index}
        for split, split_valid, out_source, out_target, out_conf in (
            ("fit", fit_valid, fit_source, fit_target, fit_confidence),
            ("validation", validation_valid, validation_source, validation_target, validation_confidence),
        ):
            flat_indices = np.flatnonzero(split_valid)
            if flat_indices.size > per_frame_samples:
                flat_indices = rng.choice(flat_indices, per_frame_samples, replace=False)
            out_source.append(source.reshape(-1, 3)[flat_indices])
            out_target.append(target.reshape(-1, 3)[flat_indices])
            out_conf.append(confidence.reshape(-1)[flat_indices])
            frame_record[f"{split}_correspondences"] = int(flat_indices.size)
        frame_details.append(frame_record)

    def concatenate(values: list[np.ndarray], width: int | None = None) -> np.ndarray:
        if not values:
            shape = (0, width) if width is not None else (0,)
            return np.empty(shape, dtype=np.float32)
        return np.concatenate(values, axis=0)

    return (
        concatenate(fit_source, 3), concatenate(fit_target, 3), concatenate(fit_confidence),
        concatenate(validation_source, 3), concatenate(validation_target, 3),
        concatenate(validation_confidence), {"frames": frame_details},
    )


def run(args: argparse.Namespace) -> dict:
    preds_dir = args.preds_dir.resolve()
    output_dir = args.output_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.resolve().read_text())
    registered_pcds = np.load(preds_dir / "registered_pcds.npy", mmap_mode="r")
    registered_confs = np.load(preds_dir / "registered_confs.npy", mmap_mode="r")
    input_imgs = np.load(preds_dir / "input_imgs.npy", mmap_mode="r")
    keyframe_count = int(manifest["keyframe_count"])
    if registered_pcds.shape[:3] != registered_confs.shape:
        raise ValueError("registered_pcds and registered_confs shapes differ")
    if registered_pcds.shape[0] != keyframe_count or input_imgs.shape[0] != keyframe_count:
        raise ValueError("Official prediction count differs from keyframe manifest")

    rgb_path = reference_dir / "render_offline.mp4"
    mask_path = reference_dir / "mask_offline.mp4"
    pose_path = reference_dir / "target_c2w.npy"
    intrinsic_path = reference_dir / "intrinsic.npy"
    depth_path = reference_dir / "depth_offline.npy"
    for path in (rgb_path, mask_path, pose_path, intrinsic_path, depth_path):
        if not path.exists():
            raise FileNotFoundError(path)
    video_frames, width, height, fps = _video_metadata(rgb_path)
    mask_frames, mask_width, mask_height, _ = _video_metadata(mask_path)
    if (mask_width, mask_height) != (width, height):
        raise ValueError("Reference RGB/mask resolutions differ")
    target_c2w = np.load(pose_path, mmap_mode="r")
    intrinsic = np.load(intrinsic_path).astype(np.float32)
    reference_depth = np.load(depth_path, mmap_mode="r")
    if intrinsic.ndim == 3:
        intrinsic = intrinsic[0]
    render_frame_count = min(
        video_frames, mask_frames, target_c2w.shape[0], reference_depth.shape[0]
    )
    if (height, width) != tuple(reference_depth.shape[-2:]):
        raise ValueError("Reference video and depth resolutions differ")
    sim3_frame_count = min(args.sim3_frames, keyframe_count)
    planned_indices = [
        int(item["planned_c2w_index"])
        for item in manifest["keyframes"][:sim3_frame_count]
    ]
    if max(planned_indices) >= render_frame_count:
        raise ValueError("Initialization keyframe exceeds reference trajectory")
    selected_masks = _selected_video_masks(mask_path, planned_indices)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v6_2 canonical construction and rendering require CUDA")
    torch.cuda.set_device(device)

    depth_samples = []
    for planned_index in planned_indices:
        frame_depth = np.asarray(reference_depth[planned_index])
        valid = np.isfinite(frame_depth) & (frame_depth > 0) & selected_masks[planned_index]
        if valid.any():
            depth_samples.append(frame_depth[valid])
    if not depth_samples:
        raise RuntimeError("No valid initialization reference depths")
    median_reference_depth = float(np.median(np.concatenate(depth_samples)))
    focal = float(max(abs(intrinsic[0, 0]), abs(intrinsic[1, 1])))
    raw_voxel_size = args.target_pixel_spacing * median_reference_depth / focal
    voxel_size = float(np.clip(raw_voxel_size, args.min_voxel_size, args.max_voxel_size))
    sim3_threshold = max(2.0 * voxel_size, 0.03 * median_reference_depth)

    (fit_source, fit_target, fit_confidence, validation_source, validation_target,
     validation_confidence, correspondence_details) = _reference_correspondences(
        registered_pcds,
        registered_confs,
        manifest,
        reference_depth,
        target_c2w,
        intrinsic,
        selected_masks,
        frame_count=sim3_frame_count,
        confidence_threshold=args.sim3_confidence,
        per_frame_samples=args.sim3_samples_per_frame,
        device=device,
    )
    scale, rotation, translation, fit_diagnostics = robust_sim3(
        fit_source,
        fit_target,
        fit_confidence,
        threshold=sim3_threshold,
        seed=args.seed,
        trials=args.sim3_ransac_trials,
    )
    validation_residual = np.linalg.norm(
        apply_sim3(validation_source, scale, rotation, translation) - validation_target,
        axis=1,
    )
    validation_diagnostics = {
        "correspondences": int(validation_residual.size),
        "residual_p50": float(np.percentile(validation_residual, 50)),
        "residual_p90": float(np.percentile(validation_residual, 90)),
        "residual_p95": float(np.percentile(validation_residual, 95)),
        "inlier_ratio": float(np.mean(validation_residual < sim3_threshold)),
    }
    if validation_diagnostics["residual_p90"] > args.max_sim3_validation_p90:
        raise RuntimeError(
            "Frozen Sim(3) validation failed: "
            f"p90={validation_diagnostics['residual_p90']:.4f} > "
            f"{args.max_sim3_validation_p90:.4f}"
        )

    sim3 = {
        "definition": "canonical = scale * rotation @ slam3r + translation",
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "frozen_after_keyframes": sim3_frame_count,
        "fit": fit_diagnostics,
        "validation": validation_diagnostics,
        "correspondence_details": correspondence_details,
    }
    (output_dir / "sim3.json").write_text(json.dumps(sim3, indent=2))

    raw_points, raw_colors, raw_confidence, raw_source_ids = [], [], [], []
    colors_all = np.asarray(input_imgs, dtype=np.float32)
    if float(np.nanmax(colors_all)) > 1.5:
        colors_all = colors_all / 255.0
    for frame_index in range(keyframe_count):
        frame_points = np.asarray(registered_pcds[frame_index], dtype=np.float32).reshape(-1, 3)
        frame_confidence = np.asarray(registered_confs[frame_index], dtype=np.float32).reshape(-1)
        frame_colors = np.asarray(colors_all[frame_index], dtype=np.float32).reshape(-1, 3)
        valid = (
            np.isfinite(frame_points).all(axis=1)
            & (np.linalg.norm(frame_points, axis=1) > 1e-8)
            & np.isfinite(frame_confidence)
            & (frame_confidence >= args.map_confidence)
            & np.isfinite(frame_colors).all(axis=1)
        )
        if not valid.any():
            continue
        raw_points.append(apply_sim3(frame_points[valid], scale, rotation, translation).astype(np.float32))
        raw_colors.append(np.clip(frame_colors[valid], 0, 1))
        raw_confidence.append(frame_confidence[valid])
        raw_source_ids.append(np.full(int(valid.sum()), frame_index, dtype=np.int32))
    if not raw_points:
        raise RuntimeError(f"No SLAM3R points pass map confidence {args.map_confidence}")
    raw_points_np = np.concatenate(raw_points)
    raw_colors_np = np.concatenate(raw_colors)
    raw_confidence_np = np.concatenate(raw_confidence)
    raw_source_ids_np = np.concatenate(raw_source_ids)
    _write_ply(
        output_dir / "slam3r_raw_observations.ply",
        raw_points_np, raw_colors_np, raw_confidence_np, raw_source_ids_np,
    )
    map_points, map_colors, map_confidence, map_source_ids = best_confidence_voxels(
        raw_points_np, raw_colors_np, raw_confidence_np, raw_source_ids_np, voxel_size
    )
    if map_points.shape[0] > args.max_map_points:
        keep = np.argpartition(map_confidence, -args.max_map_points)[-args.max_map_points:]
        map_points, map_colors = map_points[keep], map_colors[keep]
        map_confidence, map_source_ids = map_confidence[keep], map_source_ids[keep]
    _write_ply(
        output_dir / "slam3r_memory_map.ply",
        map_points, map_colors, map_confidence, map_source_ids,
    )
    memory_npz_path = output_dir / "memory_map.npz"
    np.savez_compressed(
        memory_npz_path,
        positions=map_points.astype(np.float32),
        colors=map_colors.astype(np.float32),
        confidence=map_confidence.astype(np.float32),
        source_ids=map_source_ids.astype(np.int32),
        voxel_size=np.float32(voxel_size),
        sim3_scale=np.float64(scale),
        sim3_rotation=rotation.astype(np.float64),
        sim3_translation=translation.astype(np.float64),
    )

    pass2_root = output_dir / "pass2_input" / "vggt_depth"
    pass2_render_dir = pass2_root / "render"
    pass2_render_dir.mkdir(parents=True, exist_ok=True)
    _safe_symlink(reference_dir.parent / "depth", pass2_root / "depth")
    _safe_symlink(reference_dir.parent / "metadata.txt", pass2_root / "metadata.txt")
    for name, source in (
        ("target_c2w.npy", pose_path),
        ("intrinsic.npy", intrinsic_path),
        ("depth_offline.npy", depth_path),
    ):
        _safe_symlink(source, pass2_render_dir / name)

    memory = RGBPointMemory(
        height=height,
        width=width,
        device=device,
        K=torch.from_numpy(intrinsic),
        voxel_size=0.0,
        max_points=max(1, map_points.shape[0]),
        point_size=1,
    )
    memory.points = torch.from_numpy(map_points).to(device=device, dtype=torch.float32)
    memory.colors = torch.from_numpy(map_colors).to(device=device, dtype=torch.float32)
    writer_specs = {
        "historical_render": diagnostics_dir / "historical_render.mp4",
        "historical_mask": diagnostics_dir / "historical_mask.mp4",
        "reference_render": diagnostics_dir / "reference_render.mp4",
        "reference_mask": diagnostics_dir / "reference_mask.mp4",
        "historical_add_mask": diagnostics_dir / "historical_add_mask.mp4",
        "fused_render": pass2_render_dir / "render_offline.mp4",
        "fused_mask": pass2_render_dir / "mask_offline.mp4",
    }
    writers = {
        name: VideoStreamWriter(str(path), width, height, fps=max(1, round(fps)))
        for name, path in writer_specs.items()
    }
    rgb_capture = cv2.VideoCapture(str(rgb_path))
    mask_capture = cv2.VideoCapture(str(mask_path))
    if not rgb_capture.isOpened() or not mask_capture.isOpened():
        raise RuntimeError("Could not reopen reference videos for rendering")
    coverage = {"reference_pixels": 0, "historical_pixels": 0, "historical_add_pixels": 0,
                "fused_pixels": 0, "total_pixels": render_frame_count * height * width}
    try:
        for start in range(0, render_frame_count, args.render_batch_size):
            end = min(start + args.render_batch_size, render_frame_count)
            historical_rgb, historical_mask = memory.render(
                torch.from_numpy(np.asarray(target_c2w[start:end])),
                torch.from_numpy(np.broadcast_to(intrinsic, (end - start, 3, 3)).copy()),
            )
            reference_rgbs, reference_masks = [], []
            for _ in range(start, end):
                rgb_success, rgb_bgr = rgb_capture.read()
                mask_success, mask_bgr = mask_capture.read()
                if not rgb_success or not mask_success:
                    raise RuntimeError("Reference video ended before geometry arrays")
                reference_rgbs.append(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
                reference_masks.append(np.any(mask_bgr > 127, axis=2))
            reference_rgb = torch.from_numpy(np.stack(reference_rgbs)).to(
                device=device, dtype=torch.float32
            ).permute(0, 3, 1, 2).div(255.0)
            reference_mask = torch.from_numpy(np.stack(reference_masks)).to(
                device=device
            ).unsqueeze(1)
            fused_rgb, fused_mask, historical_add = strict_reference_priority_fusion(
                reference_rgb, reference_mask, historical_rgb, historical_mask
            )
            coverage["reference_pixels"] += int(reference_mask.sum().item())
            coverage["historical_pixels"] += int(historical_mask.sum().item())
            coverage["historical_add_pixels"] += int(historical_add.sum().item())
            coverage["fused_pixels"] += int(fused_mask.sum().item())
            writers["historical_render"].write(historical_rgb)
            writers["historical_mask"].write(historical_mask.float())
            writers["reference_render"].write(reference_rgb * reference_mask)
            writers["reference_mask"].write(reference_mask.float())
            writers["historical_add_mask"].write(historical_add.float())
            writers["fused_render"].write(fused_rgb)
            writers["fused_mask"].write(fused_mask.float())
            print(f"Rendered fixed map frames {start}:{end}/{render_frame_count}", flush=True)
    finally:
        rgb_capture.release()
        mask_capture.release()
        for writer in writers.values():
            writer.close()
    for key in ("reference_pixels", "historical_pixels", "historical_add_pixels", "fused_pixels"):
        coverage[key.replace("pixels", "ratio")] = coverage[key] / coverage["total_pixels"]

    original_json = json.loads(args.original_json.resolve().read_text())
    if not isinstance(original_json, list) or len(original_json) != 1:
        raise ValueError("v6_2 demo builder expects exactly one JSON sample")
    pass2_json = [dict(original_json[0])]
    pass2_json[0]["vggt_depth_path"] = str(pass2_root)
    pass2_json_path = output_dir / "pass2_input" / "new_v6_2.json"
    pass2_json_path.write_text(json.dumps(pass2_json, indent=2))

    with memory_npz_path.open("rb") as handle:
        memory_sha256 = hashlib.sha256(handle.read()).hexdigest()
    point_count = {
        "raw_observations": int(raw_points_np.shape[0]),
        "memory_voxels": int(map_points.shape[0]),
        "compression_ratio": float(map_points.shape[0] / raw_points_np.shape[0]),
        "map_confidence_threshold": float(args.map_confidence),
        "voxel_size": voxel_size,
        "voxel_formula": {
            "target_pixel_spacing": float(args.target_pixel_spacing),
            "median_reference_depth": median_reference_depth,
            "focal": focal,
            "raw_voxel_size": raw_voxel_size,
            "min": float(args.min_voxel_size),
            "max": float(args.max_voxel_size),
        },
    }
    (diagnostics_dir / "point_count.json").write_text(json.dumps(point_count, indent=2))
    histogram, edges = np.histogram(raw_confidence_np, bins=50)
    (diagnostics_dir / "confidence_histogram.json").write_text(json.dumps({
        "counts": histogram.tolist(), "bin_edges": edges.tolist()
    }, indent=2))
    summary = {
        "status": "complete",
        "format": "slam3r_offline_v6_2",
        "fixed_map": True,
        "causal": False,
        "keyframe_count": keyframe_count,
        "render_frame_count": render_frame_count,
        "point_size": 1,
        "map_npz": str(memory_npz_path),
        "map_sha256": memory_sha256,
        "pass2_json": str(pass2_json_path),
        "pass2_render": str(pass2_render_dir / "render_offline.mp4"),
        "pass2_mask": str(pass2_render_dir / "mask_offline.mp4"),
        "sim3": sim3,
        "point_count": point_count,
        "coverage": coverage,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--original-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim3-frames", type=int, default=5)
    parser.add_argument("--sim3-confidence", type=float, default=1.5)
    parser.add_argument("--sim3-samples-per-frame", type=int, default=12000)
    parser.add_argument("--sim3-ransac-trials", type=int, default=256)
    parser.add_argument("--max-sim3-validation-p90", type=float, default=0.20)
    parser.add_argument("--map-confidence", type=float, default=12.0)
    parser.add_argument("--target-pixel-spacing", type=float, default=6.0)
    parser.add_argument("--min-voxel-size", type=float, default=0.003)
    parser.add_argument("--max-voxel-size", type=float, default=0.03)
    parser.add_argument("--max-map-points", type=int, default=3_000_000)
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "status": summary["status"],
        "keyframe_count": summary["keyframe_count"],
        "render_frame_count": summary["render_frame_count"],
        "memory_voxels": summary["point_count"]["memory_voxels"],
        "coverage": summary["coverage"],
        "pass2_json": summary["pass2_json"],
    }, indent=2))


if __name__ == "__main__":
    main()
