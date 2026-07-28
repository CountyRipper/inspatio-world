"""Geometry utilities for single-anchor overlapping DA3 windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SimilarityRegistration:
    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    rmse: float
    normalized_rmse: float


def _as_c2w(w2c: torch.Tensor) -> torch.Tensor:
    if w2c.shape == (3, 4):
        bottom = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], device=w2c.device, dtype=w2c.dtype
        )
        w2c = torch.cat((w2c, bottom), dim=0)
    if w2c.shape != (4, 4):
        raise ValueError(f"Expected w2c [3,4] or [4,4], got {tuple(w2c.shape)}")
    return torch.linalg.inv(w2c)


def backproject_world_grid(
    depth: torch.Tensor,
    K: torch.Tensor,
    *,
    c2w: Optional[torch.Tensor] = None,
    w2c: Optional[torch.Tensor] = None,
    min_depth: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backproject one depth map into a world-space point grid."""
    if depth.ndim != 2:
        raise ValueError(f"Expected depth [H,W], got {tuple(depth.shape)}")
    if K.shape != (3, 3):
        raise ValueError(f"Expected K [3,3], got {tuple(K.shape)}")
    if (c2w is None) == (w2c is None):
        raise ValueError("Exactly one of c2w or w2c must be supplied")
    if c2w is None:
        c2w = _as_c2w(w2c)
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected c2w [4,4], got {tuple(c2w.shape)}")

    depth = depth.float()
    K = K.to(depth.device, dtype=torch.float32)
    c2w = c2w.to(depth.device, dtype=torch.float32)
    height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=torch.float32),
        torch.arange(width, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    valid = torch.isfinite(depth) & (depth > min_depth)
    z = torch.where(valid, depth, torch.zeros_like(depth))
    x = (xx - K[0, 2]) * z / K[0, 0]
    y = (yy - K[1, 2]) * z / K[1, 1]
    camera = torch.stack((x, y, z), dim=-1)
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    valid &= torch.isfinite(world).all(dim=-1)
    return world, valid


def apply_similarity(
    points: torch.Tensor,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return float(scale) * (points @ rotation.T) + translation


def transform_da3_c2w(
    w2c: torch.Tensor,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Map a DA3 local camera into the canonical world under a Sim(3)."""
    local_c2w = _as_c2w(w2c).to(rotation.device, dtype=torch.float32)
    result = torch.eye(4, device=rotation.device, dtype=torch.float32)
    result[:3, :3] = rotation @ local_c2w[:3, :3]
    result[:3, 3] = apply_similarity(
        local_c2w[:3, 3].unsqueeze(0), scale, rotation, translation
    )[0]
    return result


def pose_residual(plan_c2w: torch.Tensor, observed_c2w: torch.Tensor) -> dict:
    plan_c2w = plan_c2w.to(observed_c2w.device, dtype=torch.float32)
    relative_rotation = plan_c2w[:3, :3].T @ observed_c2w[:3, :3]
    cosine = ((torch.trace(relative_rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_degrees = torch.rad2deg(torch.acos(cosine))
    translation = torch.linalg.norm(observed_c2w[:3, 3] - plan_c2w[:3, 3])
    return {
        "rotation_degrees": float(rotation_degrees.item()),
        "translation": float(translation.item()),
    }


def _solve_similarity(source: torch.Tensor, target: torch.Tensor):
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = source_centered.square().sum() / source.shape[0]
    if not torch.isfinite(variance) or variance <= 1e-12:
        raise ValueError("Degenerate source points for Sim(3)")

    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones(3, device=source.device, dtype=source.dtype)
    correction[-1] = torch.sign(torch.det(u @ vh))
    rotation = u @ torch.diag(correction) @ vh
    scale_tensor = (singular * correction).sum() / variance
    if not torch.isfinite(scale_tensor) or scale_tensor <= 0:
        raise ValueError("Sim(3) produced a non-positive scale")
    translation = target_mean - scale_tensor * (rotation @ source_mean)
    return float(scale_tensor.item()), rotation, translation


def estimate_similarity_registration(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_correspondences: int = 4096,
    max_correspondences: int = 60_000,
    trim_quantile: float = 0.80,
    iterations: int = 3,
) -> SimilarityRegistration:
    """Robustly align same-pixel 3D correspondences with iterative trimming."""
    if source_points.shape != target_points.shape or source_points.shape[-1] != 3:
        raise ValueError("Source and target points must have matching [...,3] shapes")
    if valid.shape != source_points.shape[:-1]:
        raise ValueError("Validity mask must match the point-grid dimensions")
    if not 0.5 <= trim_quantile <= 1.0:
        raise ValueError("trim_quantile must be in [0.5, 1.0]")

    flat_valid = (
        valid.bool().reshape(-1)
        & torch.isfinite(source_points.reshape(-1, 3)).all(dim=1)
        & torch.isfinite(target_points.reshape(-1, 3)).all(dim=1)
    )
    source = source_points.reshape(-1, 3)[flat_valid].float()
    target = target_points.reshape(-1, 3)[flat_valid].float()
    correspondence_count = int(source.shape[0])
    if correspondence_count < min_correspondences:
        raise ValueError(
            f"Only {correspondence_count} valid correspondences; "
            f"need {min_correspondences}"
        )
    if correspondence_count > max_correspondences:
        stride = max(1, correspondence_count // max_correspondences)
        source = source[::stride][:max_correspondences]
        target = target[::stride][:max_correspondences]

    keep = torch.ones(source.shape[0], device=source.device, dtype=torch.bool)
    for _ in range(iterations):
        scale, rotation, translation = _solve_similarity(source[keep], target[keep])
        residual = torch.linalg.norm(
            apply_similarity(source, scale, rotation, translation) - target, dim=1
        )
        if trim_quantile < 1.0:
            cutoff = torch.quantile(residual, trim_quantile)
            keep = residual <= cutoff

    scale, rotation, translation = _solve_similarity(source[keep], target[keep])
    residual = torch.linalg.norm(
        apply_similarity(source[keep], scale, rotation, translation) - target[keep], dim=1
    )
    rmse = torch.sqrt(torch.mean(residual.square()))
    target_radius = torch.median(
        torch.linalg.norm(target[keep] - target[keep].mean(dim=0), dim=1)
    ).clamp_min(1e-6)
    return SimilarityRegistration(
        scale=scale,
        rotation=rotation,
        translation=translation,
        correspondence_count=correspondence_count,
        inlier_count=int(keep.sum().item()),
        inlier_ratio=float(keep.float().mean().item()),
        rmse=float(rmse.item()),
        normalized_rmse=float((rmse / target_radius).item()),
    )
