"""Training-free historical RGB point-cloud memory utilities.

The memory in this module is deliberately external to the DiT and KV cache. It
stores only generated RGB-D observations and exposes a small render interface so
that a future feature-point backend can replace it without changing the causal
pipeline hooks.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional, Protocol, Tuple

import numpy as np
import torch


class HistoricalPointCloudMemory(Protocol):
    def update(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose_c2w: torch.Tensor,
        mask: torch.Tensor,
        K: Optional[torch.Tensor] = None,
    ) -> dict:
        ...

    def render(
        self,
        target_poses: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


def latent_block_to_pixel_span(latent_start: int, latent_count: int) -> Tuple[int, int]:
    """Return the Wan-VAE RGB interval [start, end) for a latent block."""
    if latent_start < 0 or latent_count <= 0:
        raise ValueError(f"Invalid latent span: start={latent_start}, count={latent_count}")
    pixel_start = 0 if latent_start == 0 else 4 * latent_start - 3
    last_latent = latent_start + latent_count - 1
    pixel_end = 4 * last_latent + 1
    return pixel_start, pixel_end


def latent_keyframe_indices(latent_start: int, latent_count: int) -> Tuple[int, ...]:
    """Return one causal RGB keyframe per Wan latent in a latent block.

    Latent zero is represented by RGB frame zero. Every later latent consumes
    four RGB frames and uses the last frame of that causal group.
    """
    if latent_start < 0 or latent_count <= 0:
        raise ValueError(f"Invalid latent span: start={latent_start}, count={latent_count}")
    return tuple(
        0 if latent_index == 0 else 4 * latent_index
        for latent_index in range(latent_start, latent_start + latent_count)
    )


def fuse_reference_and_history(
    reference_rgb: torch.Tensor,
    reference_mask: torch.Tensor,
    historical_rgb: torch.Tensor,
    historical_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse pixel-space conditions with strict reference priority."""
    reference_mask = reference_mask.bool()
    historical_mask = historical_mask.bool()
    hist_only = historical_mask & ~reference_mask
    fused_rgb = torch.where(hist_only.expand_as(reference_rgb), historical_rgb, reference_rgb)
    fused_mask = reference_mask | hist_only
    return fused_rgb, fused_mask, hist_only


def dense_point_count(frame_count: int, height: int, width: int) -> int:
    """Return the exact upper-bound point count for an all-valid dense map."""
    if frame_count < 0 or height <= 0 or width <= 0:
        raise ValueError("frame_count must be non-negative and image size must be positive")
    return int(frame_count) * int(height) * int(width)


def scale_adaptive_voxel_size(
    reference_depth: torch.Tensor,
    K: torch.Tensor,
    reference_mask: Optional[torch.Tensor] = None,
    *,
    target_pixel_spacing: float = 3.0,
    min_voxel_size: float = 0.003,
    max_voxel_size: float = 0.012,
) -> Tuple[float, dict]:
    """Choose one canonical voxel size from the reference scene scale."""
    if target_pixel_spacing <= 0:
        raise ValueError("target_pixel_spacing must be positive")
    if min_voxel_size <= 0 or max_voxel_size < min_voxel_size:
        raise ValueError("Invalid adaptive voxel bounds")

    depth = reference_depth.detach().float()
    while depth.ndim > 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 3:
        raise ValueError(f"Expected reference depth [T,H,W], got {tuple(depth.shape)}")

    valid = torch.isfinite(depth) & (depth > 0)
    if reference_mask is not None:
        mask = reference_mask.detach()
        while mask.ndim > 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 4 and mask.shape[0] in (1, 3):
            mask = mask[0]
        if mask.shape != depth.shape:
            raise ValueError(
                f"Reference mask {tuple(mask.shape)} does not match depth {tuple(depth.shape)}"
            )
        valid &= mask.to(depth.device) > 0

    valid_depth = depth[valid]
    if valid_depth.numel() == 0:
        raise ValueError("Reference depth has no valid pixels for adaptive voxel sizing")

    intrinsic = K.detach().float()
    while intrinsic.ndim > 2 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected K [3,3], got {tuple(intrinsic.shape)}")
    focal = float(torch.maximum(intrinsic[0, 0].abs(), intrinsic[1, 1].abs()).item())
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError(f"Invalid focal length: {focal}")

    median_depth = float(torch.median(valid_depth).item())
    world_pixel_size = median_depth / focal
    raw_voxel_size = target_pixel_spacing * world_pixel_size
    voxel_size = float(np.clip(raw_voxel_size, min_voxel_size, max_voxel_size))
    projected_pixel_spacing = voxel_size / world_pixel_size
    return voxel_size, {
        "adaptive_voxel": True,
        "target_pixel_spacing": float(target_pixel_spacing),
        "median_reference_depth": median_depth,
        "reference_focal": focal,
        "world_pixel_size_at_median_depth": world_pixel_size,
        "raw_voxel_size": raw_voxel_size,
        "voxel_size": voxel_size,
        "min_voxel_size": float(min_voxel_size),
        "max_voxel_size": float(max_voxel_size),
        "projected_pixel_spacing": projected_pixel_spacing,
        "voxel_size_clamped": not np.isclose(voxel_size, raw_voxel_size),
        "valid_reference_pixels": int(valid_depth.numel()),
    }


def _log_depth_gradient(depth: torch.Tensor, min_depth: float) -> torch.Tensor:
    """Conservative four-neighbour log-depth gradient for edge rejection/weighting."""
    valid = torch.isfinite(depth) & (depth > min_depth)
    safe = torch.where(valid, depth, torch.ones_like(depth)).clamp_min(min_depth)
    log_depth = torch.log(safe)
    gradient = torch.zeros_like(log_depth)
    gradient[..., 1:, :] = torch.maximum(
        gradient[..., 1:, :], (log_depth[..., 1:, :] - log_depth[..., :-1, :]).abs()
    )
    gradient[..., :-1, :] = torch.maximum(
        gradient[..., :-1, :], (log_depth[..., 1:, :] - log_depth[..., :-1, :]).abs()
    )
    gradient[..., :, 1:] = torch.maximum(
        gradient[..., :, 1:], (log_depth[..., :, 1:] - log_depth[..., :, :-1]).abs()
    )
    gradient[..., :, :-1] = torch.maximum(
        gradient[..., :, :-1], (log_depth[..., :, 1:] - log_depth[..., :, :-1]).abs()
    )
    return torch.where(valid, gradient, torch.full_like(gradient, float("inf")))


def calibrate_depth_scale(
    generated_depth: torch.Tensor,
    reference_depth: torch.Tensor,
    reference_mask: torch.Tensor,
    previous_scale: Optional[float] = None,
    *,
    min_depth: float = 0.1,
    min_overlap: int = 4096,
    trim_fraction: float = 0.05,
    max_log_mad: float = 0.15,
    max_reference_log_gradient: float = 0.05,
    ema_alpha: float = 0.5,
) -> Tuple[float, dict]:
    """Robustly align generated depth to the immutable reference depth layer."""
    if generated_depth.shape != reference_depth.shape:
        raise ValueError(
            f"Depth shapes differ: {tuple(generated_depth.shape)} vs "
            f"{tuple(reference_depth.shape)}"
        )
    if reference_mask.ndim == generated_depth.ndim + 1 and reference_mask.shape[-3] == 1:
        reference_mask = reference_mask.squeeze(-3)
    if reference_mask.shape != generated_depth.shape:
        raise ValueError(
            f"Reference mask shape {tuple(reference_mask.shape)} does not match depth "
            f"{tuple(generated_depth.shape)}"
        )

    generated_depth = generated_depth.float()
    reference_depth = reference_depth.to(generated_depth.device, dtype=torch.float32)
    reference_mask = reference_mask.to(generated_depth.device).bool()
    reference_gradient = _log_depth_gradient(reference_depth, min_depth)
    generated_gradient = _log_depth_gradient(generated_depth, min_depth)
    valid = (
        reference_mask
        & torch.isfinite(generated_depth)
        & (generated_depth > min_depth)
        & torch.isfinite(reference_depth)
        & (reference_depth > min_depth)
        & (reference_gradient <= max_reference_log_gradient)
        & (generated_gradient <= max_reference_log_gradient)
    )
    log_ratios = torch.log(reference_depth[valid] / generated_depth[valid])
    overlap = int(log_ratios.numel())
    trimmed_count = 0
    raw_scale = None
    log_mad = None
    reliable = False

    if overlap >= min_overlap:
        log_ratios = torch.sort(log_ratios).values
        trim = int(log_ratios.numel() * trim_fraction)
        if trim > 0 and 2 * trim < log_ratios.numel():
            log_ratios = log_ratios[trim:-trim]
        trimmed_count = int(log_ratios.numel())
        median_log_ratio = torch.median(log_ratios)
        log_mad_tensor = torch.median((log_ratios - median_log_ratio).abs())
        log_mad = float(log_mad_tensor.item())
        raw_scale = float(torch.exp(median_log_ratio).item())
        reliable = bool(np.isfinite(raw_scale) and np.isfinite(log_mad) and log_mad <= max_log_mad)

    fallback_scale = 1.0 if previous_scale is None else float(previous_scale)
    if reliable:
        if previous_scale is None:
            scale = float(raw_scale)
        else:
            scale = float(np.exp(
                (1.0 - ema_alpha) * np.log(max(float(previous_scale), 1e-12))
                + ema_alpha * np.log(max(float(raw_scale), 1e-12))
            ))
    else:
        scale = fallback_scale

    return scale, {
        "scale": scale,
        "raw_scale": raw_scale,
        "overlap_pixels": overlap,
        "trimmed_overlap_pixels": trimmed_count,
        "log_mad": log_mad,
        "scale_reliable": reliable,
        "used_previous_scale": not reliable and previous_scale is not None,
    }


def compute_depth_confidence(
    scaled_depth: torch.Tensor,
    *,
    scale_reliable: bool,
    log_mad: Optional[float],
    historical_depth: Optional[torch.Tensor] = None,
    min_depth: float = 0.1,
) -> torch.Tensor:
    """Compute non-destructive confidence from scale, edges, and map residuals."""
    depth = scaled_depth.float()
    gradient = _log_depth_gradient(depth, min_depth)
    edge_confidence = torch.exp(-torch.square(gradient / 0.10))
    if scale_reliable:
        mad = 0.0 if log_mad is None else float(log_mad)
        scale_confidence = float(np.exp(-((mad / 0.15) ** 2)))
    else:
        scale_confidence = 0.25
    confidence = edge_confidence * scale_confidence

    if historical_depth is not None:
        historical_depth = historical_depth.to(depth.device, dtype=torch.float32)
        if historical_depth.shape != depth.shape:
            raise ValueError(
                f"Historical depth shape {tuple(historical_depth.shape)} does not match "
                f"generated depth {tuple(depth.shape)}"
            )
        historical_valid = torch.isfinite(historical_depth) & (historical_depth > min_depth)
        finite_depth = depth[torch.isfinite(depth) & (depth > min_depth)]
        scene_median = (
            torch.median(finite_depth)
            if finite_depth.numel() > 0
            else torch.tensor(1.0, device=depth.device)
        )
        tolerance = torch.maximum(
            0.02 * scene_median,
            0.05 * historical_depth.clamp_min(min_depth),
        )
        residual_confidence = torch.exp(
            -torch.square((depth - historical_depth).abs() / tolerance.clamp_min(1e-6))
        )
        confidence = confidence * torch.where(
            historical_valid, residual_confidence, torch.ones_like(residual_confidence)
        )

    valid = torch.isfinite(depth) & (depth > min_depth)
    confidence = torch.where(valid, confidence, torch.full_like(confidence, 1e-3))
    return confidence.nan_to_num(1e-3, posinf=1e-3, neginf=1e-3).clamp(1e-3, 1.0)


class RGBPointMemory:
    """Bounded GPU/CPU RGB point-cloud memory with z-buffer rendering."""

    def __init__(
        self,
        height: int,
        width: int,
        device: torch.device,
        K: torch.Tensor,
        voxel_size: float = 0.02,
        max_points: int = 500_000,
        point_size: int = 1,
        min_depth: float = 0.1,
    ):
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        if point_size <= 0:
            raise ValueError("point_size must be positive")

        self.height = int(height)
        self.width = int(width)
        self.device = torch.device(device)
        self.voxel_size = float(voxel_size)
        self.max_points = int(max_points)
        self.point_size = int(point_size)
        self.min_depth = float(min_depth)
        self.K = K.detach().to(self.device, dtype=torch.float32)
        if self.K.shape != (3, 3):
            raise ValueError(f"Expected K (3, 3), got {tuple(self.K.shape)}")
        self.points = torch.empty((0, 3), device=self.device, dtype=torch.float32)
        self.colors = torch.empty((0, 3), device=self.device, dtype=torch.float32)
        self.num_updates = 0

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    def _normalize_inputs(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose_c2w: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = rgb.detach().to(self.device, dtype=torch.float32)
        depth = depth.detach().to(self.device, dtype=torch.float32)
        pose_c2w = pose_c2w.detach().to(self.device, dtype=torch.float32)
        mask = mask.detach().to(self.device).bool()

        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.permute(1, 2, 0)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected RGB {(self.height, self.width, 3)}, got {tuple(rgb.shape)}")
        if depth.shape != (self.height, self.width):
            raise ValueError(f"Expected depth {(self.height, self.width)}, got {tuple(depth.shape)}")
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.shape != (self.height, self.width):
            raise ValueError(f"Expected mask {(self.height, self.width)}, got {tuple(mask.shape)}")
        if pose_c2w.shape != (4, 4):
            raise ValueError(f"Expected pose (4, 4), got {tuple(pose_c2w.shape)}")
        return rgb.clamp(0, 1), depth, pose_c2w, mask

    def update(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose_c2w: torch.Tensor,
        mask: torch.Tensor,
        K: Optional[torch.Tensor] = None,
    ) -> dict:
        rgb, depth, pose_c2w, mask = self._normalize_inputs(rgb, depth, pose_c2w, mask)
        K = self.K if K is None else K.detach().to(self.device, dtype=torch.float32)
        if K.shape != (3, 3):
            raise ValueError(f"Expected K (3, 3), got {tuple(K.shape)}")

        valid = mask & torch.isfinite(depth) & (depth > self.min_depth)
        added_pixels = int(valid.sum().item())
        if added_pixels == 0:
            self.num_updates += 1
            return {
                "added_pixels": 0,
                "points_before": self.point_count,
                "points_after": self.point_count,
            }

        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        z = depth[valid]
        x = (xx[valid] - K[0, 2]) * z / K[0, 0]
        y = (yy[valid] - K[1, 2]) * z / K[1, 1]
        camera_points = torch.stack([x, y, z], dim=-1)
        world_points = camera_points @ pose_c2w[:3, :3].T + pose_c2w[:3, 3]
        new_colors = rgb[valid]

        points_before = self.point_count
        self.points = torch.cat([self.points, world_points], dim=0)
        self.colors = torch.cat([self.colors, new_colors], dim=0)
        self._compress()
        self.num_updates += 1
        return {
            "added_pixels": added_pixels,
            "points_before": points_before,
            "points_after": self.point_count,
        }

    def update_block(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose_c2w: torch.Tensor,
        mask: torch.Tensor,
        K: Optional[torch.Tensor] = None,
    ) -> dict:
        """Add a full RGB-D block and voxel-compress it once."""
        if rgb.ndim != 4 or rgb.shape[0] != depth.shape[0]:
            raise ValueError(
                f"Expected aligned RGB/depth blocks, got {tuple(rgb.shape)} and "
                f"{tuple(depth.shape)}"
            )
        frame_count = rgb.shape[0]
        if pose_c2w.shape != (frame_count, 4, 4):
            raise ValueError(
                f"Expected poses {(frame_count, 4, 4)}, got {tuple(pose_c2w.shape)}"
            )
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.shape != (frame_count, self.height, self.width):
            raise ValueError(
                f"Expected masks {(frame_count, self.height, self.width)}, "
                f"got {tuple(mask.shape)}"
            )
        if K is None:
            K = self.K.unsqueeze(0).expand(frame_count, -1, -1)
        if K.shape != (frame_count, 3, 3):
            raise ValueError(f"Expected K {(frame_count, 3, 3)}, got {tuple(K.shape)}")

        points_before = self.point_count
        new_points = []
        new_colors = []
        added_pixels = 0
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        for frame_index in range(frame_count):
            frame_rgb, frame_depth, frame_pose, frame_mask = self._normalize_inputs(
                rgb[frame_index],
                depth[frame_index],
                pose_c2w[frame_index],
                mask[frame_index],
            )
            intrinsic = K[frame_index].detach().to(self.device, dtype=torch.float32)
            valid = (
                frame_mask
                & torch.isfinite(frame_depth)
                & (frame_depth > self.min_depth)
            )
            frame_added = int(valid.sum().item())
            added_pixels += frame_added
            if frame_added == 0:
                continue

            z = frame_depth[valid]
            x = (xx[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
            y = (yy[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
            camera_points = torch.stack([x, y, z], dim=-1)
            world_points = (
                camera_points @ frame_pose[:3, :3].T + frame_pose[:3, 3]
            )
            new_points.append(world_points)
            new_colors.append(frame_rgb[valid])

        if new_points:
            self.points = torch.cat([self.points, *new_points], dim=0)
            self.colors = torch.cat([self.colors, *new_colors], dim=0)
            self._compress()
        self.num_updates += frame_count
        return {
            "added_pixels": added_pixels,
            "points_before": points_before,
            "points_after": self.point_count,
        }

    def _compress(self) -> None:
        if self.points.numel() == 0:
            return

        if self.voxel_size > 0:
            voxel_coords = torch.floor(self.points / self.voxel_size).to(torch.int64)
            _, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
            num_voxels = int(inverse.max().item()) + 1
            point_sums = torch.zeros((num_voxels, 3), device=self.device, dtype=torch.float32)
            color_sums = torch.zeros_like(point_sums)
            counts = torch.zeros((num_voxels, 1), device=self.device, dtype=torch.float32)
            point_sums.index_add_(0, inverse, self.points)
            color_sums.index_add_(0, inverse, self.colors)
            counts.index_add_(0, inverse, torch.ones((inverse.shape[0], 1), device=self.device))
            self.points = point_sums / counts.clamp_min(1)
            self.colors = color_sums / counts.clamp_min(1)

        if self.point_count > self.max_points:
            keep = torch.linspace(
                0,
                self.point_count - 1,
                self.max_points,
                device=self.device,
            ).round().long()
            self.points = self.points[keep]
            self.colors = self.colors[keep]

    def render(
        self,
        target_poses: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_poses = target_poses.detach().to(self.device, dtype=torch.float32)
        K = K.detach().to(self.device, dtype=torch.float32)
        if target_poses.ndim == 2:
            target_poses = target_poses.unsqueeze(0)
        if target_poses.ndim != 3 or target_poses.shape[1:] != (4, 4):
            raise ValueError(f"Expected target poses [T,4,4], got {tuple(target_poses.shape)}")
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(target_poses.shape[0], -1, -1)
        if K.shape != (target_poses.shape[0], 3, 3):
            raise ValueError(f"Expected K [T,3,3], got {tuple(K.shape)}")

        rgbs, masks = [], []
        for pose, intrinsic in zip(target_poses, K):
            rgb, mask = self._render_one(pose, intrinsic)
            rgbs.append(rgb)
            masks.append(mask)
        return torch.stack(rgbs, dim=0), torch.stack(masks, dim=0)

    def _render_one(self, pose_c2w: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        canvas = torch.zeros((3, self.height, self.width), device=self.device, dtype=torch.float32)
        mask_canvas = torch.zeros((1, self.height, self.width), device=self.device, dtype=torch.bool)
        if self.point_count == 0:
            return canvas, mask_canvas

        w2c = torch.linalg.inv(pose_c2w)
        camera_points = self.points @ w2c[:3, :3].T + w2c[:3, 3]
        z = camera_points[:, 2]
        valid_z = torch.isfinite(z) & (z > self.min_depth)
        if not valid_z.any():
            return canvas, mask_canvas

        camera_points = camera_points[valid_z]
        colors = self.colors[valid_z]
        z = z[valid_z]
        u = K[0, 0] * camera_points[:, 0] / z + K[0, 2]
        v = K[1, 1] * camera_points[:, 1] / z + K[1, 2]

        if self.point_size > 1:
            radius = self.point_size // 2
            offsets = torch.arange(-radius, radius + 1, device=self.device)
            dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
            dx = dx.flatten()
            dy = dy.flatten()
            u = (torch.round(u).long()[:, None] + dx[None, :]).flatten()
            v = (torch.round(v).long()[:, None] + dy[None, :]).flatten()
            z = z[:, None].expand(-1, dx.numel()).flatten()
            colors = colors[:, None, :].expand(-1, dx.numel(), -1).reshape(-1, 3)
        else:
            u = torch.round(u).long()
            v = torch.round(v).long()

        valid = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        if not valid.any():
            return canvas, mask_canvas
        u, v, z, colors = u[valid], v[valid], z[valid], colors[valid]

        flat_indices = v * self.width + u
        depth_buffer = torch.full(
            (self.height * self.width,),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        depth_buffer.scatter_reduce_(0, flat_indices, z, reduce="amin", include_self=True)
        closest = z <= depth_buffer[flat_indices] + 1e-4
        u, v, colors = u[closest], v[closest], colors[closest]
        canvas[:, v, u] = colors.T
        mask_canvas[:, v, u] = True
        return canvas, mask_canvas

    def save(self, path_prefix: str) -> Tuple[str, str]:
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        points = self.points.detach().cpu().numpy().astype(np.float32)
        colors = (self.colors.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

        npz_path = path_prefix + ".npz"
        np.savez_compressed(npz_path, points=points, colors=colors)

        ply_path = path_prefix + ".ply"
        vertices = np.empty(
            points.shape[0],
            dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ],
        )
        if points.shape[0] > 0:
            vertices["x"], vertices["y"], vertices["z"] = points.T
            vertices["red"], vertices["green"], vertices["blue"] = colors.T
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {points.shape[0]}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        with open(ply_path, "wb") as handle:
            handle.write(header.encode("ascii"))
            vertices.tofile(handle)
        return npz_path, ply_path


class IncrementalVoxelSurfelMemory(RGBPointMemory):
    """Bounded voxel surfels with persistent observation counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.voxel_size <= 0:
            raise ValueError("Incremental voxel surfels require voxel_size > 0")
        self.voxel_coords = torch.empty(
            (0, 3), device=self.device, dtype=torch.int64
        )
        self.observation_counts = torch.empty(
            (0,), device=self.device, dtype=torch.int64
        )

    @property
    def splat_diameter(self) -> int:
        return 1 if self.point_size <= 1 else 2 * (self.point_size // 2) + 1

    def update_points(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> dict:
        """Fuse canonical world points into one counted surfel per voxel."""
        points = points.detach().to(self.device, dtype=torch.float32).reshape(-1, 3)
        colors = colors.detach().to(self.device, dtype=torch.float32).reshape(-1, 3)
        if points.shape != colors.shape:
            raise ValueError(
                f"Point/color shapes differ: {tuple(points.shape)} vs {tuple(colors.shape)}"
            )
        if valid is None:
            valid = torch.ones(points.shape[0], device=self.device, dtype=torch.bool)
        else:
            valid = valid.detach().to(self.device).bool().reshape(-1)
        if valid.shape[0] != points.shape[0]:
            raise ValueError("Validity mask length does not match point count")
        valid &= torch.isfinite(points).all(dim=1) & torch.isfinite(colors).all(dim=1)
        points = points[valid]
        colors = colors[valid].clamp(0, 1)
        raw_valid_points = int(points.shape[0])
        points_before = self.point_count
        if raw_valid_points == 0:
            return {
                "raw_valid_points": 0,
                "batch_voxels": 0,
                "points_before": points_before,
                "points_after": points_before,
                "evicted_voxels": 0,
            }

        new_coords = torch.floor(points / self.voxel_size).to(torch.int64)
        batch_coords, batch_inverse = torch.unique(
            new_coords, dim=0, return_inverse=True
        )
        batch_count = int(batch_coords.shape[0])
        batch_point_sums = torch.zeros(
            (batch_count, 3), device=self.device, dtype=torch.float32
        )
        batch_color_sums = torch.zeros_like(batch_point_sums)
        batch_counts = torch.zeros(
            (batch_count,), device=self.device, dtype=torch.int64
        )
        batch_point_sums.index_add_(0, batch_inverse, points)
        batch_color_sums.index_add_(0, batch_inverse, colors)
        batch_counts.index_add_(
            0,
            batch_inverse,
            torch.ones(batch_inverse.shape[0], device=self.device, dtype=torch.int64),
        )

        if self.point_count:
            old_counts_float = self.observation_counts.float().unsqueeze(1)
            all_coords = torch.cat((self.voxel_coords, batch_coords), dim=0)
            all_point_sums = torch.cat(
                (self.points * old_counts_float, batch_point_sums), dim=0
            )
            all_color_sums = torch.cat(
                (self.colors * old_counts_float, batch_color_sums), dim=0
            )
            all_counts = torch.cat((self.observation_counts, batch_counts), dim=0)
        else:
            all_coords = batch_coords
            all_point_sums = batch_point_sums
            all_color_sums = batch_color_sums
            all_counts = batch_counts

        merged_coords, merged_inverse = torch.unique(
            all_coords, dim=0, return_inverse=True
        )
        merged_count = int(merged_coords.shape[0])
        merged_point_sums = torch.zeros(
            (merged_count, 3), device=self.device, dtype=torch.float32
        )
        merged_color_sums = torch.zeros_like(merged_point_sums)
        merged_counts = torch.zeros(
            (merged_count,), device=self.device, dtype=torch.int64
        )
        merged_point_sums.index_add_(0, merged_inverse, all_point_sums)
        merged_color_sums.index_add_(0, merged_inverse, all_color_sums)
        merged_counts.index_add_(0, merged_inverse, all_counts)

        evicted = max(0, merged_count - self.max_points)
        if evicted:
            keep = torch.linspace(
                0,
                merged_count - 1,
                self.max_points,
                device=self.device,
            ).round().long()
            merged_coords = merged_coords[keep]
            merged_point_sums = merged_point_sums[keep]
            merged_color_sums = merged_color_sums[keep]
            merged_counts = merged_counts[keep]

        counts_float = merged_counts.float().unsqueeze(1).clamp_min(1)
        self.voxel_coords = merged_coords
        self.observation_counts = merged_counts
        self.points = merged_point_sums / counts_float
        self.colors = merged_color_sums / counts_float
        self.num_updates += 1
        return {
            "raw_valid_points": raw_valid_points,
            "batch_voxels": batch_count,
            "points_before": points_before,
            "points_after": self.point_count,
            "evicted_voxels": evicted,
            "observation_count_total": int(self.observation_counts.sum().item()),
        }

    def save(self, path_prefix: str) -> Tuple[str, str]:
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        points = self.points.detach().cpu().numpy().astype(np.float32)
        colors_float = self.colors.detach().cpu().numpy().astype(np.float32)
        colors = (colors_float.clip(0, 1) * 255).round().astype(np.uint8)
        counts = self.observation_counts.detach().cpu().numpy().astype(np.int64)
        coords = self.voxel_coords.detach().cpu().numpy().astype(np.int64)

        npz_path = path_prefix + ".npz"
        np.savez_compressed(
            npz_path,
            points=points,
            colors=colors_float,
            observation_counts=counts,
            voxel_coords=coords,
            voxel_size=np.float32(self.voxel_size),
            splat_diameter=np.int32(self.splat_diameter),
        )

        ply_path = path_prefix + ".ply"
        vertices = np.empty(
            points.shape[0],
            dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ],
        )
        if points.shape[0] > 0:
            vertices["x"], vertices["y"], vertices["z"] = points.T
            vertices["red"], vertices["green"], vertices["blue"] = colors.T
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {points.shape[0]}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        with open(ply_path, "wb") as handle:
            handle.write(header.encode("ascii"))
            vertices.tofile(handle)
        return npz_path, ply_path


class DenseGeneratedPointMemory:
    """GPU-resident append-only generated RGB-D memory stored by STAR chunk."""

    def __init__(
        self,
        height: int,
        width: int,
        device: torch.device,
        K: torch.Tensor,
        point_size: int = 1,
        min_depth: float = 0.1,
        anchor_confidence: float = 0.1,
    ):
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if point_size != 1:
            raise ValueError("dense_two_layer currently requires point_size=1")
        self.height = int(height)
        self.width = int(width)
        self.device = torch.device(device)
        self.K = K.detach().to(self.device, dtype=torch.float32)
        if self.K.shape != (3, 3):
            raise ValueError(f"Expected K (3, 3), got {tuple(self.K.shape)}")
        self.point_size = int(point_size)
        self.min_depth = float(min_depth)
        self.anchor_confidence = float(anchor_confidence)
        self.point_chunks = []
        self.color_chunks = []
        self.confidence_chunks = []
        self.chunk_metadata = []
        self.num_updates = 0

    @property
    def point_count(self) -> int:
        return int(sum(chunk.shape[0] for chunk in self.point_chunks))

    @property
    def chunk_count(self) -> int:
        return len(self.point_chunks)

    @staticmethod
    def storage_bytes_for_points(point_count: int) -> int:
        """xyz/color float32 plus confidence float16."""
        return int(point_count) * (3 * 4 + 3 * 4 + 2)

    def update_block(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose_c2w: torch.Tensor,
        mask: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Backproject every valid pixel into one immutable GPU chunk."""
        rgb = rgb.detach().to(self.device, dtype=torch.float32)
        depth = depth.detach().to(self.device, dtype=torch.float32)
        pose_c2w = pose_c2w.detach().to(self.device, dtype=torch.float32)
        mask = mask.detach().to(self.device).bool()
        if rgb.ndim != 4 or rgb.shape[1:] != (3, self.height, self.width):
            raise ValueError(
                f"Expected RGB [T,3,{self.height},{self.width}], got {tuple(rgb.shape)}"
            )
        frame_count = rgb.shape[0]
        if depth.shape != (frame_count, self.height, self.width):
            raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.shape != depth.shape:
            raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
        if pose_c2w.shape != (frame_count, 4, 4):
            raise ValueError(f"Unexpected pose shape: {tuple(pose_c2w.shape)}")
        if K is None:
            K = self.K.unsqueeze(0).expand(frame_count, -1, -1)
        else:
            K = K.detach().to(self.device, dtype=torch.float32)
            if K.ndim == 2:
                K = K.unsqueeze(0).expand(frame_count, -1, -1)
        if K.shape != (frame_count, 3, 3):
            raise ValueError(f"Unexpected intrinsic shape: {tuple(K.shape)}")
        if confidence is None:
            confidence = torch.ones_like(depth)
        else:
            confidence = confidence.detach().to(self.device, dtype=torch.float32)
        if confidence.shape != depth.shape:
            raise ValueError(f"Unexpected confidence shape: {tuple(confidence.shape)}")

        valid = mask & torch.isfinite(depth) & (depth > self.min_depth)
        valid_counts = valid.flatten(1).sum(dim=1, dtype=torch.int64)
        added_pixels = int(valid_counts.sum().item())
        points_before = self.point_count
        if added_pixels == 0:
            self.num_updates += frame_count
            return {
                "added_pixels": 0,
                "points_before": points_before,
                "points_after": points_before,
                "stored_chunk_index": None,
            }

        points = torch.empty((added_pixels, 3), device=self.device, dtype=torch.float32)
        colors = torch.empty_like(points)
        confidences = torch.empty((added_pixels,), device=self.device, dtype=torch.float16)
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        offset = 0
        for frame_index in range(frame_count):
            count = int(valid_counts[frame_index].item())
            if count == 0:
                continue
            frame_valid = valid[frame_index]
            z = depth[frame_index][frame_valid]
            intrinsic = K[frame_index]
            x = (xx[frame_valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
            y = (yy[frame_valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
            camera_points = torch.stack((x, y, z), dim=-1)
            pose = pose_c2w[frame_index]
            points[offset:offset + count] = (
                camera_points @ pose[:3, :3].T + pose[:3, 3]
            )
            colors[offset:offset + count] = rgb[frame_index, :, frame_valid].T.clamp(0, 1)
            confidences[offset:offset + count] = confidence[frame_index][frame_valid].to(torch.float16)
            offset += count

        if not torch.isfinite(points).all():
            raise RuntimeError("Non-finite world point produced from valid dense depth")
        if not torch.isfinite(colors).all():
            raise RuntimeError("Non-finite generated RGB encountered in dense memory update")
        if not torch.isfinite(confidences).all():
            raise RuntimeError("Non-finite confidence encountered in dense memory update")

        chunk_index = self.chunk_count
        self.point_chunks.append(points)
        self.color_chunks.append(colors)
        self.confidence_chunks.append(confidences)
        chunk_metadata = dict(metadata or {})
        chunk_metadata.update({
            "chunk_index": chunk_index,
            "point_count": added_pixels,
            "frame_count": frame_count,
        })
        self.chunk_metadata.append(chunk_metadata)
        self.num_updates += frame_count
        return {
            "added_pixels": added_pixels,
            "points_before": points_before,
            "points_after": self.point_count,
            "stored_chunk_index": chunk_index,
        }

    def _project_chunk(
        self,
        chunk_index: int,
        w2c: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points = self.point_chunks[chunk_index]
        camera_points = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = camera_points[:, 2]
        valid_z = torch.isfinite(z) & (z > self.min_depth)
        if not valid_z.any():
            empty_index = torch.empty((0,), device=self.device, dtype=torch.int64)
            empty_float = torch.empty((0,), device=self.device, dtype=torch.float32)
            return empty_index, empty_float, empty_float, torch.empty(
                (0, 3), device=self.device, dtype=torch.float32
            )
        camera_points = camera_points[valid_z]
        z = z[valid_z]
        confidence = self.confidence_chunks[chunk_index][valid_z].float()
        colors = self.color_chunks[chunk_index][valid_z]
        u = torch.round(K[0, 0] * camera_points[:, 0] / z + K[0, 2]).long()
        v = torch.round(K[1, 1] * camera_points[:, 1] / z + K[1, 2]).long()
        visible = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return v[visible] * self.width + u[visible], z[visible], confidence[visible], colors[visible]

    def render_with_depth(
        self,
        target_poses: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_poses = target_poses.detach().to(self.device, dtype=torch.float32)
        K = K.detach().to(self.device, dtype=torch.float32)
        if target_poses.ndim == 2:
            target_poses = target_poses.unsqueeze(0)
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(target_poses.shape[0], -1, -1)
        if target_poses.ndim != 3 or target_poses.shape[1:] != (4, 4):
            raise ValueError(f"Expected target poses [T,4,4], got {tuple(target_poses.shape)}")
        if K.shape != (target_poses.shape[0], 3, 3):
            raise ValueError(f"Expected K [T,3,3], got {tuple(K.shape)}")

        rgbs, masks, depths = [], [], []
        for pose, intrinsic in zip(target_poses, K):
            rgb, mask, depth = self._render_one(pose, intrinsic)
            rgbs.append(rgb)
            masks.append(mask)
            depths.append(depth)
        return torch.stack(rgbs), torch.stack(masks), torch.stack(depths)

    def render(self, target_poses: torch.Tensor, K: torch.Tensor):
        rgb, mask, _ = self.render_with_depth(target_poses, K)
        return rgb, mask

    def _render_one(self, pose_c2w: torch.Tensor, K: torch.Tensor):
        pixel_count = self.height * self.width
        if self.point_count == 0:
            return (
                torch.zeros((3, self.height, self.width), device=self.device),
                torch.zeros((1, self.height, self.width), device=self.device, dtype=torch.bool),
                torch.zeros((self.height, self.width), device=self.device),
            )
        w2c = torch.linalg.inv(pose_c2w)
        anchor_high = torch.full((pixel_count,), float("inf"), device=self.device)
        anchor_all = torch.full_like(anchor_high, float("inf"))

        # Pass 1: establish a high-confidence surface anchor, with per-pixel fallback.
        for chunk_index in range(self.chunk_count):
            indices, z, confidence, _ = self._project_chunk(chunk_index, w2c, K)
            if indices.numel() == 0:
                continue
            anchor_all.scatter_reduce_(0, indices, z, reduce="amin", include_self=True)
            high = confidence >= self.anchor_confidence
            if high.any():
                anchor_high.scatter_reduce_(
                    0, indices[high], z[high], reduce="amin", include_self=True
                )
        anchor = torch.where(torch.isfinite(anchor_high), anchor_high, anchor_all)
        anchor_valid = torch.isfinite(anchor)
        finite_anchor = anchor[anchor_valid]
        scene_median_depth = (
            torch.median(finite_anchor)
            if finite_anchor.numel() > 0
            else torch.tensor(1.0, device=self.device)
        )

        # Pass 2: confidence-weight all hits supported by the anchored surface.
        color_sum = torch.zeros((pixel_count, 3), device=self.device, dtype=torch.float32)
        depth_sum = torch.zeros((pixel_count,), device=self.device, dtype=torch.float32)
        weight_sum = torch.zeros((pixel_count,), device=self.device, dtype=torch.float32)
        for chunk_index in range(self.chunk_count):
            indices, z, confidence, colors = self._project_chunk(chunk_index, w2c, K)
            if indices.numel() == 0:
                continue
            per_hit_anchor = anchor[indices]
            tolerance = torch.maximum(
                0.02 * scene_median_depth,
                0.05 * per_hit_anchor,
            )
            supported = torch.isfinite(per_hit_anchor) & ((z - per_hit_anchor).abs() <= tolerance)
            if not supported.any():
                continue
            indices = indices[supported]
            weights = confidence[supported].clamp_min(1e-3)
            weight_sum.index_add_(0, indices, weights)
            depth_sum.index_add_(0, indices, z[supported] * weights)
            color_sum.index_add_(0, indices, colors[supported] * weights[:, None])

        mask = weight_sum > 0
        safe_weight = weight_sum.clamp_min(1e-12)
        rgb = (color_sum / safe_weight[:, None]).reshape(self.height, self.width, 3)
        depth = (depth_sum / safe_weight).reshape(self.height, self.width)
        return (
            rgb.permute(2, 0, 1),
            mask.reshape(1, self.height, self.width),
            torch.where(mask.reshape(self.height, self.width), depth, torch.zeros_like(depth)),
        )

    def save(self, path_prefix: str) -> Tuple[str, str]:
        """Save one NPZ per stored chunk, a JSON manifest, and one streamed PLY."""
        output_dir = os.path.dirname(path_prefix)
        os.makedirs(output_dir, exist_ok=True)
        chunk_dir = path_prefix + "_chunks"
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_entries = []
        for chunk_index, (points, colors, confidence) in enumerate(zip(
            self.point_chunks, self.color_chunks, self.confidence_chunks
        )):
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:04d}.npz")
            np.savez_compressed(
                chunk_path,
                points=points.detach().cpu().numpy().astype(np.float32),
                colors=colors.detach().cpu().numpy().astype(np.float32),
                confidence=confidence.detach().cpu().numpy().astype(np.float16),
            )
            entry = dict(self.chunk_metadata[chunk_index])
            entry["path"] = chunk_path
            chunk_entries.append(entry)

        ply_path = path_prefix + ".ply"
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {self.point_count}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property float confidence\nend_header\n"
        )
        vertex_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("confidence", "<f4"),
        ])
        with open(ply_path, "wb") as handle:
            handle.write(header.encode("ascii"))
            for points, colors, confidence in zip(
                self.point_chunks, self.color_chunks, self.confidence_chunks
            ):
                points_np = points.detach().cpu().numpy().astype(np.float32)
                colors_np = (colors.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
                vertices = np.empty(points_np.shape[0], dtype=vertex_dtype)
                vertices["x"], vertices["y"], vertices["z"] = points_np.T
                vertices["red"], vertices["green"], vertices["blue"] = colors_np.T
                vertices["confidence"] = confidence_np
                vertices.tofile(handle)

        manifest_path = path_prefix + "_manifest.json"
        with open(manifest_path, "w") as handle:
            json.dump({
                "format": "dense_generated_point_memory_v1",
                "point_count": self.point_count,
                "chunk_count": self.chunk_count,
                "resident_storage_bytes": self.storage_bytes_for_points(self.point_count),
                "chunks": chunk_entries,
                "ply_path": ply_path,
            }, handle, indent=2)
        return manifest_path, ply_path


class VideoStreamWriter:
    """Small ffmpeg RGB24 writer used for block-wise diagnostics."""

    def __init__(self, path: str, width: int, height: int, fps: int = 24):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        command = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-loglevel", "warning", path,
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, video_tchw: torch.Tensor) -> None:
        video = video_tchw.detach().float().clamp(0, 1)
        if video.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W], got {tuple(video.shape)}")
        if video.shape[1] == 1:
            video = video.expand(-1, 3, -1, -1)
        frames = (video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
        assert self.process.stdin is not None
        self.process.stdin.write(frames.tobytes())

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg diagnostic writer failed with code {return_code}")
