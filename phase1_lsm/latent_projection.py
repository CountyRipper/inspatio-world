"""Minimal LSM-style latent backprojection/project for the Phase 1 experiment.

Rewritten from the geometry math in LatentSpatialMemory
``src/mirage/latent_point_cloud.py`` at commit
``b2cd7383140ca0e11cbb0bc9594a7c8286b6a427``. This module has no runtime
dependency on that repository and intentionally contains none of mirage/spatia.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _backproject_slot(
    latent_chw: torch.Tensor,
    depth_hw: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latent_chw.ndim != 3:
        raise ValueError(f"latent must be [C,h,w], got {tuple(latent_chw.shape)}")
    if depth_hw.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {tuple(depth_hw.shape)}")
    if tuple(intrinsics.shape) != (3, 3) or tuple(cam2world.shape) != (4, 4):
        raise ValueError("intrinsics/cam2world must be [3,3]/[4,4]")

    device = latent_chw.device
    _, h_lat, w_lat = latent_chw.shape
    depth = depth_hw.to(device=device, dtype=torch.float32)
    K = intrinsics.to(device=device, dtype=torch.float32).clone()
    c2w = cam2world.to(device=device, dtype=torch.float32)
    depth_latent = F.interpolate(
        depth[None, None], size=(h_lat, w_lat), mode="bilinear", align_corners=False
    )[0, 0]

    depth_h, depth_w = depth.shape
    K[0, 0] *= w_lat / depth_w
    K[0, 2] *= w_lat / depth_w
    K[1, 1] *= h_lat / depth_h
    K[1, 2] *= h_lat / depth_h

    v, u = torch.meshgrid(
        torch.arange(h_lat, device=device),
        torch.arange(w_lat, device=device),
        indexing="ij",
    )
    x = (u - K[0, 2]) * depth_latent / K[0, 0]
    y = (v - K[1, 2]) * depth_latent / K[1, 1]
    points_camera = torch.stack((x, y, depth_latent), dim=-1)
    points_world = points_camera @ c2w[:3, :3].T + c2w[:3, 3]
    features = latent_chw.float().permute(1, 2, 0).reshape(-1, latent_chw.shape[0])
    return points_world.reshape(-1, 3), features, depth_latent.reshape(-1) > 0


def _project_slot(
    points_world: torch.Tensor,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    target_cam2world: torch.Tensor,
    intrinsics_latent: torch.Tensor,
    latent_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = points_world.device
    c2w = target_cam2world.to(device=device, dtype=torch.float32)
    K = intrinsics_latent.to(device=device, dtype=torch.float32)
    h, w = latent_hw
    world2cam = torch.inverse(c2w)
    points_camera = points_world @ world2cam[:3, :3].T + world2cam[:3, 3]
    z = points_camera[:, 2]
    u = torch.round(points_camera[:, 0] * K[0, 0] / z + K[0, 2]).long()
    v = torch.round(points_camera[:, 1] * K[1, 1] / z + K[1, 2]).long()
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    projected_valid = valid_mask & in_bounds & (z > 0)

    u = u[projected_valid]
    v = v[projected_valid]
    z = z[projected_valid]
    features = features[projected_valid]
    order = torch.argsort(z, descending=True)
    flat_index = v[order] * w + u[order]

    projected = torch.zeros(features.shape[1], h * w, device=device, dtype=torch.float32)
    zbuffer = torch.full((h * w,), float("inf"), device=device)
    zbuffer[flat_index] = z[order]
    projected[:, flat_index] = features[order].T
    occupancy = zbuffer.reshape(h, w).isfinite()
    return projected.reshape(features.shape[1], h, w), occupancy


def project_memory_sequence(
    latent_bfchw: torch.Tensor,
    depth_fhw: torch.Tensor,
    intrinsics: torch.Tensor,
    source_c2w_f44: torch.Tensor,
    target_c2w_f44: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project each of the three temporal slots independently (never average)."""
    if latent_bfchw.ndim != 5 or latent_bfchw.shape[:3] != (1, 3, 16):
        raise ValueError(
            "Phase 1 latent must be [1,3,16,h,w], got "
            f"{tuple(latent_bfchw.shape)}"
        )
    if depth_fhw.ndim != 3 or depth_fhw.shape[0] != 3:
        raise ValueError(f"depth must be [3,H,W], got {tuple(depth_fhw.shape)}")
    if tuple(source_c2w_f44.shape) != (3, 4, 4):
        raise ValueError(f"source c2w must be [3,4,4], got {source_c2w_f44.shape}")
    if tuple(target_c2w_f44.shape) != (3, 4, 4):
        raise ValueError(f"target c2w must be [3,4,4], got {target_c2w_f44.shape}")

    _, _, _, h, w = latent_bfchw.shape
    depth_h, depth_w = depth_fhw.shape[-2:]
    K_latent = intrinsics.to(latent_bfchw.device, torch.float32).clone()
    K_latent[0, 0] *= w / depth_w
    K_latent[0, 2] *= w / depth_w
    K_latent[1, 1] *= h / depth_h
    K_latent[1, 2] *= h / depth_h

    projected_slots = []
    occupancy_slots = []
    for slot in range(3):
        points, features, valid = _backproject_slot(
            latent_bfchw[0, slot],
            depth_fhw[slot],
            intrinsics,
            source_c2w_f44[slot],
        )
        projected, occupancy = _project_slot(
            points,
            features,
            valid,
            target_c2w_f44[slot],
            K_latent,
            (h, w),
        )
        projected_slots.append(projected.to(latent_bfchw.dtype))
        occupancy_slots.append(occupancy)

    projected_bfchw = torch.stack(projected_slots, dim=0).unsqueeze(0)
    occupancy_bf1hw = torch.stack(occupancy_slots, dim=0)[None, :, None]
    memory_mask4 = occupancy_bf1hw.expand(-1, -1, 4, -1, -1)
    memory_mask4 = memory_mask4.to(latent_bfchw.dtype).mul(2).sub(1)
    return projected_bfchw, memory_mask4, occupancy_bf1hw


def identity_reprojection_error(
    direct: torch.Tensor,
    projected: torch.Tensor,
    occupancy: torch.Tensor,
) -> dict[str, float]:
    valid = occupancy.expand_as(direct)
    if not valid.any():
        raise AssertionError("identity reprojection produced no valid latent pixels")
    difference = (direct.float() - projected.float()).abs()[valid]
    return {
        "valid_fraction": float(occupancy.float().mean()),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }
