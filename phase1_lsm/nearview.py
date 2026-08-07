from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phase1_lsm.trajectory import A_KEYFRAMES, APRIME_KEYFRAMES, NUM_RGB_FRAMES


A_YAW_DEGREES = 45.0
SUPPORTED_VIEW_OFFSETS = (0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0)
MIN_WIDE_OCCUPANCY = 0.05


def nearview_controls(offset_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if offset_degrees not in SUPPORTED_VIEW_OFFSETS:
        raise ValueError(f"unsupported near-view offset: {offset_degrees}")
    pitch = np.zeros(NUM_RGB_FRAMES, dtype=np.float64)
    yaw = np.zeros(NUM_RGB_FRAMES, dtype=np.float64)
    radius = np.zeros(NUM_RGB_FRAMES, dtype=np.float64)
    yaw[0:58] = np.linspace(0.0, A_YAW_DEGREES, 58)
    yaw[57:69] = A_YAW_DEGREES
    yaw[68:154] = np.linspace(A_YAW_DEGREES, 0.0, 86)
    yaw[153:165] = 0.0
    target_yaw = A_YAW_DEGREES + offset_degrees
    yaw[164:226] = np.linspace(0.0, target_yaw, 62)
    yaw[225:240] = target_yaw
    if not np.all(yaw[57:69] == A_YAW_DEGREES):
        raise AssertionError("near-view trajectory changed the block-5 A pose")
    if not np.all(yaw[225:237] == target_yaw):
        raise AssertionError("block-19 query does not hold the requested near-view pose")
    return pitch, yaw, radius


def write_nearview_trajectory(path: str | Path, offset_degrees: float) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    controls = nearview_controls(offset_degrees)
    with path.open("w", encoding="utf-8") as handle:
        for values in controls:
            handle.write(" ".join(f"{value:.10f}" for value in values) + "\n")
    loaded = np.loadtxt(path)
    if loaded.shape != (3, NUM_RGB_FRAMES):
        raise AssertionError(f"trajectory must be [3,240], got {loaded.shape}")
    return path


def _signed_yaw_delta_degrees(source: np.ndarray, target: np.ndarray) -> float:
    relative = source[:3, :3].T @ target[:3, :3]
    return float(-np.degrees(np.arctan2(relative[2, 0], relative[0, 0])))


def _rotation_angle_degrees(source: np.ndarray, target: np.ndarray) -> float:
    relative = source[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def validate_nearview_c2w(
    target_c2w: np.ndarray,
    offset_degrees: float,
) -> dict[str, object]:
    if target_c2w.shape != (NUM_RGB_FRAMES, 4, 4):
        raise AssertionError(f"target_c2w must be [240,4,4], got {target_c2w.shape}")
    centers = target_c2w[:, :3, 3]
    center_drift = float(np.linalg.norm(centers - centers[0], axis=1).max())
    if center_drift > 1e-6:
        raise AssertionError(f"camera center moved: {center_drift}")
    if not np.allclose(target_c2w[57:69], target_c2w[57], atol=1e-6, rtol=0):
        raise AssertionError("block-5 A pose is not held constant")
    if not np.allclose(target_c2w[225:237], target_c2w[225], atol=1e-6, rtol=0):
        raise AssertionError("block-19 A-prime pose is not held constant")
    signed_deltas = [
        _signed_yaw_delta_degrees(target_c2w[source], target_c2w[target])
        for source, target in zip(A_KEYFRAMES, APRIME_KEYFRAMES)
    ]
    rotation_angles = [
        _rotation_angle_degrees(target_c2w[source], target_c2w[target])
        for source, target in zip(A_KEYFRAMES, APRIME_KEYFRAMES)
    ]
    max_error = max(abs(value - offset_degrees) for value in signed_deltas)
    if max_error > 0.1:
        raise AssertionError(
            f"actual A-to-A-prime yaw delta differs from {offset_degrees}: "
            f"{signed_deltas}"
        )
    return {
        "requested_signed_yaw_delta_degrees": offset_degrees,
        "actual_signed_yaw_delta_degrees": signed_deltas,
        "rotation_angle_degrees": rotation_angles,
        "max_signed_yaw_error_degrees": max_error,
        "max_camera_center_drift": center_drift,
    }


def projection_displacement_statistics(
    depth_fhw: torch.Tensor,
    intrinsics: torch.Tensor,
    source_c2w_f44: torch.Tensor,
    target_c2w_f44: torch.Tensor,
    latent_hw: tuple[int, int],
) -> dict[str, object]:
    if depth_fhw.shape[0] != 3:
        raise ValueError("near-view displacement expects three temporal slots")
    h_latent, w_latent = latent_hw
    h_depth, w_depth = depth_fhw.shape[-2:]
    K = intrinsics.float().clone()
    K[0, 0] *= w_latent / w_depth
    K[0, 2] *= w_latent / w_depth
    K[1, 1] *= h_latent / h_depth
    K[1, 2] *= h_latent / h_depth
    v, u = torch.meshgrid(
        torch.arange(h_latent, device=depth_fhw.device, dtype=torch.float32),
        torch.arange(w_latent, device=depth_fhw.device, dtype=torch.float32),
        indexing="ij",
    )
    per_slot = []
    all_displacements = []
    for slot in range(3):
        depth = F.interpolate(
            depth_fhw[slot][None, None].float(),
            size=latent_hw,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        x = (u - K[0, 2]) * depth / K[0, 0]
        y = (v - K[1, 2]) * depth / K[1, 1]
        points_camera = torch.stack((x, y, depth), dim=-1).reshape(-1, 3)
        source = source_c2w_f44[slot].float()
        target_w2c = torch.inverse(target_c2w_f44[slot].float())
        points_world = points_camera @ source[:3, :3].T + source[:3, 3]
        points_target = points_world @ target_w2c[:3, :3].T + target_w2c[:3, 3]
        z = points_target[:, 2]
        projected_u = points_target[:, 0] * K[0, 0] / z + K[0, 2]
        projected_v = points_target[:, 1] * K[1, 1] / z + K[1, 2]
        valid = (
            (depth.reshape(-1) > 0)
            & (z > 0)
            & (projected_u >= 0)
            & (projected_u < w_latent)
            & (projected_v >= 0)
            & (projected_v < h_latent)
        )
        displacement = torch.sqrt(
            (projected_u[valid] - u.reshape(-1)[valid]).square()
            + (projected_v[valid] - v.reshape(-1)[valid]).square()
        )
        if displacement.numel():
            all_displacements.append(displacement)
        per_slot.append({
            "slot": slot,
            "valid_source_points": int(displacement.numel()),
            "mean_latent_pixel_displacement": (
                float(displacement.mean()) if displacement.numel() else None
            ),
            "max_latent_pixel_displacement": (
                float(displacement.max()) if displacement.numel() else None
            ),
        })
    if not all_displacements:
        return {
            "units": "latent pixels",
            "mean_pixel_displacement": 0.0,
            "max_pixel_displacement": 0.0,
            "no_overlap": True,
            "per_slot": per_slot,
        }
    combined = torch.cat(all_displacements)
    return {
        "units": "latent pixels",
        "mean_pixel_displacement": float(combined.mean()),
        "max_pixel_displacement": float(combined.max()),
        "no_overlap": False,
        "per_slot": per_slot,
    }


def preservation_invalid_mask(occupancy: torch.Tensor) -> torch.Tensor:
    if occupancy.ndim != 5 or occupancy.shape[2] != 1:
        raise ValueError("occupancy must be [B,F,1,H,W]")
    return ~occupancy.bool()


def invalid_raw_l1(
    prediction: torch.Tensor,
    no_memory: torch.Tensor,
    occupancy: torch.Tensor,
) -> torch.Tensor:
    """Strict invalid-region raw L1 used by both training and acceptance."""
    invalid = preservation_invalid_mask(occupancy).expand_as(prediction)
    if not invalid.any():
        # Exact identity reprojection can occupy the complete latent grid.  Its
        # strict invalid set is empty, so it contributes no preservation term.
        return prediction.float().sum() * 0.0
    return (
        prediction.float() - no_memory.detach().float()
    ).abs()[invalid].mean()


def choose_wide_offset(
    direction: int,
    occupancy_at_20: float,
    occupancy_at_15: float,
) -> tuple[float, bool]:
    """Apply the fixed pre-training 20-to-15 degree coverage rule."""
    if direction not in (-1, 1):
        raise ValueError("wide direction must be -1 or +1")
    for value in (occupancy_at_20, occupancy_at_15):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid occupancy fraction: {value}")
    if occupancy_at_20 >= MIN_WIDE_OCCUPANCY:
        return float(20 * direction), False
    return float(15 * direction), occupancy_at_15 < MIN_WIDE_OCCUPANCY
