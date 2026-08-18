from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mapkv_proto.memory_context import MemoryContext, make_memory_context
from mapkv_proto.pose_utils import (
    rotation_geodesic,
    scale_intrinsics,
    to_cut3r_c2w,
)

from .surfel_index import SurfelIndex


def load_intrinsics(path: str | Path) -> np.ndarray:
    """Load either np.array2string or whitespace-delimited 3x3 intrinsics."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()
    try:
        matrix = np.asarray(ast.literal_eval(raw), dtype=np.float64)
    except (SyntaxError, ValueError):
        matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.size < 9:
        raise ValueError(f"Invalid intrinsic matrix in {path}: {matrix.shape}")
    return matrix.reshape(-1, 3)[:3, :3]


def infer_intrinsic_image_hw(intrinsics: np.ndarray) -> tuple[int, int]:
    """Infer the image extent used by centered upstream intrinsics."""
    matrix = np.asarray(intrinsics, dtype=np.float64)
    return (
        max(int(round(2.0 * float(matrix[1, 2]))), 1),
        max(int(round(2.0 * float(matrix[0, 2]))), 1),
    )


def latent_to_rgb_index(
    latent_index: int, latent_length: int, rgb_length: int
) -> int:
    if latent_length < 1 or rgb_length < 1:
        raise ValueError("latent_length and rgb_length must be positive")
    return int(
        round(
            int(latent_index)
            * (rgb_length - 1)
            / max(latent_length - 1, 1)
        )
    )


def build_rotation_target_to_source_grid(
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    source_intrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    image_hw: tuple[int, int],
    *,
    translation_tolerance: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Build an align_corners=False target-pixel -> source-pixel sampling grid.

    The implementation is exact for a shared camera center. Poses are absolute
    c2w matrices. The source-to-target homography is returned for audit and
    image-space visualization.
    """
    source = np.asarray(source_c2w, dtype=np.float64)
    target = np.asarray(target_c2w, dtype=np.float64)
    if source.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("source_c2w and target_c2w must both be 4x4")
    translation = float(np.linalg.norm(source[:3, 3] - target[:3, 3]))
    if translation > translation_tolerance:
        raise ValueError(
            "Rotation homography requires a shared camera center; "
            f"translation distance is {translation}"
        )
    source_k = np.asarray(source_intrinsics, dtype=np.float64)
    target_k = np.asarray(target_intrinsics, dtype=np.float64)
    source_to_target = (
        target_k
        @ target[:3, :3].T
        @ source[:3, :3]
        @ np.linalg.inv(source_k)
    )
    target_to_source = (
        source_k
        @ source[:3, :3].T
        @ target[:3, :3]
        @ np.linalg.inv(target_k)
    )
    height, width = (int(image_hw[0]), int(image_hw[1]))
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    target_pixels = np.stack(
        [xx, yy, np.ones_like(xx)], axis=0
    ).reshape(3, -1)
    source_pixels_h = target_to_source @ target_pixels
    depth = source_pixels_h[2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, 1.0)
    source_x = (source_pixels_h[0] / safe_depth).reshape(height, width)
    source_y = (source_pixels_h[1] / safe_depth).reshape(height, width)
    positive = depth.reshape(height, width) > 1e-12
    valid = (
        positive
        & (source_x >= -0.5)
        & (source_x <= width - 0.5)
        & (source_y >= -0.5)
        & (source_y <= height - 0.5)
    )
    # grid_sample(align_corners=False) maps pixel center x to
    # 2 * (x + 0.5) / width - 1.
    grid_x = 2.0 * (source_x + 0.5) / width - 1.0
    grid_y = 2.0 * (source_y + 0.5) / height - 1.0
    grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    return (
        torch.from_numpy(grid),
        torch.from_numpy(valid.astype(np.float32)),
        source_to_target,
    )


def feather_coverage(
    coverage: torch.Tensor, kernel_size: int = 3
) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("feather kernel must be a positive odd integer")
    if kernel_size == 1:
        return coverage.float().clamp(0, 1)
    if coverage.ndim != 4:
        raise ValueError(
            f"coverage must be [B,F,H,W], got {tuple(coverage.shape)}"
        )
    batch, frames, height, width = coverage.shape
    flat = coverage.float().reshape(batch * frames, 1, height, width)
    smooth = F.avg_pool2d(
        flat, kernel_size, stride=1, padding=kernel_size // 2
    )
    return smooth.reshape(batch, frames, height, width).clamp(0, 1)


def warp_latent(
    historical_latent: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Warp BFCHW latent frames with one target->source grid per frame."""
    if historical_latent.ndim != 5:
        raise ValueError(
            "historical_latent must be [B,F,C,H,W], got "
            f"{tuple(historical_latent.shape)}"
        )
    batch, frames, channels, height, width = historical_latent.shape
    grid = torch.as_tensor(
        grid,
        dtype=torch.float32,
        device=historical_latent.device,
    )
    if grid.ndim == 4:
        grid = grid.unsqueeze(0)
    if grid.shape[0] == 1 and batch > 1:
        grid = grid.expand(batch, -1, -1, -1, -1)
    if tuple(grid.shape) != (batch, frames, height, width, 2):
        raise ValueError(
            f"Warp grid {tuple(grid.shape)} does not match latent "
            f"{(batch, frames, channels, height, width)}"
        )
    # CUDA grid_sample does not accept bf16 input with an fp32 grid. The
    # reprojection is cheap relative to DiT inference, so compute it in fp32
    # and restore the native latent dtype afterwards.
    source_dtype = historical_latent.dtype
    warped = F.grid_sample(
        historical_latent.float().reshape(
            batch * frames, channels, height, width
        ),
        grid.reshape(batch * frames, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return warped.reshape(batch, frames, channels, height, width).to(
        dtype=source_dtype
    )


@dataclass
class WarpReencodePlan:
    target_block: int
    source_chunk: int
    historical_latent: torch.Tensor
    target_to_source_grid: torch.Tensor
    coverage: torch.Tensor
    selected_layers: tuple[int, ...]
    selected_step_indices: tuple[int, ...]
    alpha: float = 1.0
    source_rgb_indices: tuple[int, ...] = ()
    target_rgb_indices: tuple[int, ...] = ()
    source_to_target_homographies: tuple[np.ndarray, ...] = ()
    source_target_rotation_degrees: tuple[float, ...] = ()
    source_target_translation: tuple[float, ...] = ()
    recent_target_to_source_grid: torch.Tensor | None = None
    recent_coverage: torch.Tensor | None = None
    mode: str = "block_on_warp_reencode"
    geometry_audit: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    artifacts: dict[str, torch.Tensor] = field(default_factory=dict)
    memory_context: MemoryContext | None = None

    def compose(self, current_recent: torch.Tensor) -> torch.Tensor:
        if current_recent.shape != self.historical_latent.shape:
            raise ValueError(
                f"Current recent latent {tuple(current_recent.shape)} != "
                f"historical {tuple(self.historical_latent.shape)}"
            )
        historical = self.historical_latent.to(
            device=current_recent.device, dtype=current_recent.dtype
        )
        warped = warp_latent(historical, self.target_to_source_grid)
        if self.recent_target_to_source_grid is None:
            warped_recent = current_recent
        else:
            warped_recent = warp_latent(
                current_recent, self.recent_target_to_source_grid
            )
        coverage = self.coverage.to(
            device=current_recent.device, dtype=torch.float32
        )
        if coverage.ndim == 4:
            coverage = coverage.unsqueeze(2)
        if coverage.shape[:2] != current_recent.shape[:2] or coverage.shape[2] != 1:
            raise ValueError(
                f"Coverage {tuple(coverage.shape)} is incompatible with "
                f"recent {tuple(current_recent.shape)}"
            )
        blend = coverage.to(dtype=current_recent.dtype)
        virtual = blend * warped + (1.0 - blend) * warped_recent
        self.artifacts = {
            "historical": historical.detach().cpu(),
            "warped": warped.detach().cpu(),
            "current_recent": current_recent.detach().cpu(),
            "warped_recent": warped_recent.detach().cpu(),
            "virtual_recent": virtual.detach().cpu(),
            "coverage": coverage.detach().cpu(),
            "target_to_source_grid": self.target_to_source_grid.detach().cpu(),
        }
        if self.recent_target_to_source_grid is not None:
            self.artifacts["recent_target_to_source_grid"] = (
                self.recent_target_to_source_grid.detach().cpu()
            )
        if self.recent_coverage is not None:
            self.artifacts["recent_coverage"] = (
                self.recent_coverage.detach().cpu()
            )
        weighted_denominator = float(coverage.sum().item()) * current_recent.shape[2]
        historical_delta = float(
            (
                (warped.float() - current_recent.float()).abs()
                * coverage
            ).sum().item()
            / max(weighted_denominator, 1e-8)
        )
        self.audit.update(
            {
                "mode": self.mode,
                "target_block": int(self.target_block),
                "source_chunk": int(self.source_chunk),
                "coverage_fraction": float(coverage.mean().item()),
                "historical_vs_current_recent_overlap_l1": historical_delta,
                "virtual_vs_current_recent_l1": float(
                    (virtual.float() - current_recent.float()).abs().mean().item()
                ),
                "short_term_recent_reprojected": (
                    self.recent_target_to_source_grid is not None
                ),
                "warped_recent_vs_raw_recent_l1": float(
                    (
                        warped_recent.float() - current_recent.float()
                    ).abs().mean().item()
                ),
                "recent_warp_coverage_fraction": (
                    None
                    if self.recent_coverage is None
                    else float(self.recent_coverage.float().mean().item())
                ),
                "source_rgb_indices": list(self.source_rgb_indices),
                "target_rgb_indices": list(self.target_rgb_indices),
                "source_target_rotation_degrees": list(
                    self.source_target_rotation_degrees
                ),
                "source_target_translation": list(
                    self.source_target_translation
                ),
                "homography_source_to_target": [
                    matrix.tolist()
                    for matrix in self.source_to_target_homographies
                ],
                "geometry": dict(self.geometry_audit),
            }
        )
        return virtual

    def make_memory_context(
        self,
        layer_payloads: dict[int, tuple[torch.Tensor, torch.Tensor]],
        writer_audit: dict,
    ) -> MemoryContext:
        self.audit["writer"] = writer_audit
        context = make_memory_context(
            target_block=self.target_block,
            source_chunk=self.source_chunk,
            layer_payloads=layer_payloads,
            selected_layers=self.selected_layers,
            selected_step_indices=self.selected_step_indices,
            alpha=self.alpha,
            injection_mode="replace_recent_delta",
            gate_mode="global",
            smooth_kernel=1,
        )
        if context is None:
            raise RuntimeError("Global virtual recent context unexpectedly disabled")
        self.memory_context = context
        return context


def build_warp_reencode_plans(
    *,
    source_latents_path: str | Path,
    source_chunk: int,
    target_chunks: Iterable[int],
    target_pose_path: str | Path,
    intrinsics_path: str | Path,
    latent_length: int,
    rgb_length: int,
    frames_per_block: int,
    latent_hw: tuple[int, int],
    image_hw: tuple[int, int],
    selected_layers: Iterable[int],
    selected_step_indices: Iterable[int],
    alpha: float,
    feather_kernel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, WarpReencodePlan], list[dict]]:
    source_path = Path(source_latents_path).resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        payload = payload["pred_latents"]
    if payload.ndim != 5 or int(payload.shape[1]) != int(latent_length):
        raise ValueError(
            f"Source latent shape {tuple(payload.shape)} is incompatible with "
            f"latent length {latent_length}"
        )
    source_start = int(source_chunk) * int(frames_per_block)
    source_stop = source_start + int(frames_per_block)
    if source_start < 0 or source_stop > latent_length:
        raise IndexError(f"Source chunk {source_chunk} is outside latent sequence")
    historical = payload[:, source_start:source_stop].to(device=device, dtype=dtype)
    poses = np.load(Path(target_pose_path).resolve()).astype(np.float64)
    if poses.shape != (rgb_length, 4, 4):
        raise ValueError(
            f"Pose artifact {poses.shape} != expected {(rgb_length, 4, 4)}"
        )
    raw_intrinsics = load_intrinsics(intrinsics_path)
    intrinsic_source_hw = infer_intrinsic_image_hw(raw_intrinsics)
    image_intrinsics = scale_intrinsics(
        raw_intrinsics,
        source_hw=intrinsic_source_hw,
        target_hw=image_hw,
    )
    latent_intrinsics = scale_intrinsics(
        image_intrinsics,
        source_hw=image_hw,
        target_hw=latent_hw,
    )
    source_rgb = tuple(
        latent_to_rgb_index(index, latent_length, rgb_length)
        for index in range(source_start, source_stop)
    )
    source_poses = poses[np.asarray(source_rgb, dtype=np.int64)]
    plans: dict[int, WarpReencodePlan] = {}
    selections: list[dict] = []
    selected_layers = tuple(int(layer) for layer in selected_layers)
    selected_steps = tuple(int(step) for step in selected_step_indices)
    for target_chunk_raw in target_chunks:
        target_chunk = int(target_chunk_raw)
        if int(source_chunk) >= target_chunk - 1:
            raise ValueError(
                f"Source chunk {source_chunk} is not causally valid for {target_chunk}"
            )
        target_start = target_chunk * frames_per_block
        target_stop = target_start + frames_per_block
        if target_stop > latent_length:
            raise IndexError(f"Target chunk {target_chunk} is outside latent sequence")
        target_rgb = tuple(
            latent_to_rgb_index(index, latent_length, rgb_length)
            for index in range(target_start, target_stop)
        )
        target_poses = poses[np.asarray(target_rgb, dtype=np.int64)]
        grids = []
        masks = []
        homographies = []
        rotations = []
        translations = []
        for source_pose, target_pose in zip(source_poses, target_poses):
            grid, mask, homography = build_rotation_target_to_source_grid(
                source_pose,
                target_pose,
                latent_intrinsics,
                latent_intrinsics,
                latent_hw,
            )
            grids.append(grid)
            masks.append(mask)
            homographies.append(homography)
            rotations.append(
                float(
                    np.degrees(
                        rotation_geodesic(
                            source_pose[:3, :3], target_pose[:3, :3]
                        )
                    )
                )
            )
            translations.append(
                float(
                    np.linalg.norm(source_pose[:3, 3] - target_pose[:3, 3])
                )
            )
        grid_tensor = torch.stack(grids).to(device=device)
        hard_coverage = torch.stack(masks).unsqueeze(0)
        coverage = feather_coverage(hard_coverage, feather_kernel).to(device=device)
        plan = WarpReencodePlan(
            target_block=target_chunk,
            source_chunk=int(source_chunk),
            historical_latent=historical,
            target_to_source_grid=grid_tensor,
            coverage=coverage,
            selected_layers=selected_layers,
            selected_step_indices=selected_steps,
            alpha=float(alpha),
            source_rgb_indices=source_rgb,
            target_rgb_indices=target_rgb,
            source_to_target_homographies=tuple(homographies),
            source_target_rotation_degrees=tuple(rotations),
            source_target_translation=tuple(translations),
        )
        plans[target_chunk] = plan
        selections.append(
            {
                "target_chunk": target_chunk,
                "source_chunk": int(source_chunk),
                "mode": "manual_warp_reencode",
                "payload_kind": "target_aligned_virtual_recent_native_kv",
                "coverage_fraction": float(coverage.mean().item()),
                "source_rgb_indices": list(source_rgb),
                "target_rgb_indices": list(target_rgb),
                "source_target_rotation_degrees": rotations,
                "source_latents_path": str(source_path),
                "status": "scheduled",
            }
        )
    return plans, selections


def _surfel_coverage_for_pose(
    *,
    surfel_index: SurfelIndex,
    source_chunk: int,
    target_chunk: int,
    query_pose: np.ndarray,
    intrinsics: np.ndarray,
    source_image_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> tuple[torch.Tensor, dict]:
    """Project source-chunk surfels after eligibility filtering."""
    eligible = [
        cell
        for cell in surfel_index.cells
        if int(source_chunk) in {int(chunk) for chunk in cell.observing_chunks}
    ]
    if any(int(cell.first_seen_chunk) > int(source_chunk) for cell in eligible):
        raise RuntimeError(
            "A source-observed surfel was created after the source chunk"
        )
    visible = surfel_index.visible_cells(
        query_pose,
        intrinsics,
        target_hw,
        source_image_size=source_image_hw,
        eligible_max_chunk=int(target_chunk) - 2,
        eligible_chunks={int(source_chunk)},
        use_occlusion=True,
    )
    coverage = np.zeros(target_hw, dtype=np.float32)
    pixels = np.asarray(visible["pixels"], dtype=np.int32)
    if len(pixels):
        coverage[pixels[:, 0], pixels[:, 1]] = 1.0
    return torch.from_numpy(coverage), {
        "source_chunk": int(source_chunk),
        "target_chunk": int(target_chunk),
        "eligibility_before_zbuffer": True,
        "eligible_source_observed_surfels": len(eligible),
        "num_eligible_surfels": int(visible["num_eligible_cells"]),
        "num_visible_surfels": int(visible["num_visible_cells"]),
        "num_visible_pixels": int(len(pixels)),
        "raw_coverage_fraction": float(coverage.mean()),
        "future_geometry_used": False,
    }


def build_continuous_virtual_recent_plans(
    *,
    source_latents_path: str | Path,
    source_chunk: int,
    target_pose_path: str | Path,
    intrinsics_path: str | Path,
    surfel_index_path: str | Path,
    surfel_sequence_path: str | Path,
    latent_length: int,
    rgb_length: int,
    frames_per_block: int,
    latent_hw: tuple[int, int],
    image_hw: tuple[int, int],
    selected_layers: Iterable[int],
    selected_step_indices: Iterable[int],
    alpha: float,
    feather_kernel: int,
    device: torch.device,
    dtype: torch.dtype,
    min_history_gap_chunks: int = 2,
) -> tuple[dict[int, WarpReencodePlan], list[dict]]:
    """Build visibility-driven Virtual Recent plans for every causal block.

    Historical B1 and the runtime short-term Recent are both reprojected into
    each target block's camera layout. Only projected source-chunk surfel
    support selects the historical branch; no target-block schedule or alpha
    ramp is used.
    """
    if min_history_gap_chunks < 2:
        raise ValueError(
            "Long-term memory must exclude the immediate recent chunk"
        )
    source_path = Path(source_latents_path).resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        payload = payload["pred_latents"]
    if payload.ndim != 5 or int(payload.shape[1]) != int(latent_length):
        raise ValueError(
            f"Source latent shape {tuple(payload.shape)} is incompatible with "
            f"latent length {latent_length}"
        )
    num_blocks = int(latent_length) // int(frames_per_block)
    source_start = int(source_chunk) * int(frames_per_block)
    source_stop = source_start + int(frames_per_block)
    if source_start < 0 or source_stop > latent_length:
        raise IndexError(
            f"Source chunk {source_chunk} is outside latent sequence"
        )
    historical = payload[:, source_start:source_stop].to(
        device=device, dtype=dtype
    )
    poses = np.load(Path(target_pose_path).resolve()).astype(np.float64)
    if poses.shape != (rgb_length, 4, 4):
        raise ValueError(
            f"Pose artifact {poses.shape} != expected {(rgb_length, 4, 4)}"
        )
    raw_intrinsics = load_intrinsics(intrinsics_path)
    intrinsic_source_hw = infer_intrinsic_image_hw(raw_intrinsics)
    image_intrinsics = scale_intrinsics(
        raw_intrinsics,
        source_hw=intrinsic_source_hw,
        target_hw=image_hw,
    )
    latent_intrinsics = scale_intrinsics(
        image_intrinsics,
        source_hw=image_hw,
        target_hw=latent_hw,
    )
    source_rgb = tuple(
        latent_to_rgb_index(index, latent_length, rgb_length)
        for index in range(source_start, source_stop)
    )
    source_poses = poses[np.asarray(source_rgb, dtype=np.int64)]

    sequence_path = Path(surfel_sequence_path).resolve()
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if sequence.get("cut3r_predicted_pose_used_for_map", True):
        raise ValueError(
            "Continuous CAVR requires known-pose CUT3R geometry"
        )
    source_frame = next(
        (
            item
            for item in sequence["frames"]
            if int(item["chunk_id"]) == int(source_chunk)
        ),
        None,
    )
    if source_frame is None:
        raise ValueError(
            f"Source chunk {source_chunk} is absent from CUT3R prefix"
        )
    surfel_intrinsics = np.asarray(
        source_frame["intrinsics"], dtype=np.float64
    )
    surfel_source_hw = tuple(
        int(value) for value in source_frame["shape"]
    )
    surfel_index = SurfelIndex.load(Path(surfel_index_path).resolve())

    selected_layers = tuple(int(layer) for layer in selected_layers)
    selected_steps = tuple(int(step) for step in selected_step_indices)
    plans: dict[int, WarpReencodePlan] = {}
    selections: list[dict] = []
    first_eligible = int(source_chunk) + int(min_history_gap_chunks)
    for target_chunk in range(first_eligible, num_blocks):
        target_start = target_chunk * frames_per_block
        target_stop = target_start + frames_per_block
        recent_start = (target_chunk - 1) * frames_per_block
        target_rgb = tuple(
            latent_to_rgb_index(index, latent_length, rgb_length)
            for index in range(target_start, target_stop)
        )
        recent_rgb = tuple(
            latent_to_rgb_index(index, latent_length, rgb_length)
            for index in range(recent_start, target_start)
        )
        target_poses = poses[np.asarray(target_rgb, dtype=np.int64)]
        recent_poses = poses[np.asarray(recent_rgb, dtype=np.int64)]
        historical_grids = []
        recent_grids = []
        recent_valid = []
        homographies = []
        rotations = []
        translations = []
        surfel_masks = []
        per_frame_geometry = []
        for source_pose, recent_pose, target_pose in zip(
            source_poses, recent_poses, target_poses
        ):
            history_grid, history_mask, homography = (
                build_rotation_target_to_source_grid(
                    source_pose,
                    target_pose,
                    latent_intrinsics,
                    latent_intrinsics,
                    latent_hw,
                )
            )
            recent_grid, recent_mask, _ = (
                build_rotation_target_to_source_grid(
                    recent_pose,
                    target_pose,
                    latent_intrinsics,
                    latent_intrinsics,
                    latent_hw,
                )
            )
            surfel_mask, geometry = _surfel_coverage_for_pose(
                surfel_index=surfel_index,
                source_chunk=int(source_chunk),
                target_chunk=target_chunk,
                query_pose=to_cut3r_c2w(target_pose),
                intrinsics=surfel_intrinsics,
                source_image_hw=surfel_source_hw,
                target_hw=latent_hw,
            )
            historical_grids.append(history_grid)
            recent_grids.append(recent_grid)
            recent_valid.append(recent_mask)
            surfel_masks.append(surfel_mask * history_mask)
            homographies.append(homography)
            rotations.append(
                float(
                    np.degrees(
                        rotation_geodesic(
                            source_pose[:3, :3], target_pose[:3, :3]
                        )
                    )
                )
            )
            translations.append(
                float(
                    np.linalg.norm(
                        source_pose[:3, 3] - target_pose[:3, 3]
                    )
                )
            )
            per_frame_geometry.append(geometry)
        hard_memory_coverage = torch.stack(surfel_masks).unsqueeze(0)
        coverage = feather_coverage(
            hard_memory_coverage, feather_kernel
        ).to(device=device)
        selection = {
            "target_chunk": target_chunk,
            "source_chunk": int(source_chunk),
            "mode": "continuous_geometry_reprojected_virtual_recent",
            "payload_kind": "target_aligned_virtual_recent_native_kv",
            "coverage_fraction": float(coverage.mean().item()),
            "hard_coverage_fraction": float(
                hard_memory_coverage.mean().item()
            ),
            "source_rgb_indices": list(source_rgb),
            "recent_rgb_indices": list(recent_rgb),
            "target_rgb_indices": list(target_rgb),
            "source_target_rotation_degrees": rotations,
            "source_latents_path": str(source_path),
            "activation_policy": "visible_source_surfel_support",
            "geometry_frames": per_frame_geometry,
        }
        if not bool(hard_memory_coverage.any()):
            selection["status"] = "memory_off_no_visible_support"
            selections.append(selection)
            continue
        recent_coverage = torch.stack(recent_valid).unsqueeze(0).to(
            device=device
        )
        plan = WarpReencodePlan(
            target_block=target_chunk,
            source_chunk=int(source_chunk),
            historical_latent=historical,
            target_to_source_grid=torch.stack(historical_grids).to(
                device=device
            ),
            coverage=coverage,
            selected_layers=selected_layers,
            selected_step_indices=selected_steps,
            alpha=float(alpha),
            source_rgb_indices=source_rgb,
            target_rgb_indices=target_rgb,
            source_to_target_homographies=tuple(homographies),
            source_target_rotation_degrees=tuple(rotations),
            source_target_translation=tuple(translations),
            recent_target_to_source_grid=torch.stack(recent_grids).to(
                device=device
            ),
            recent_coverage=recent_coverage,
            mode="continuous_geometry_reprojected_virtual_recent",
            geometry_audit={
                "coordinate_frame": sequence.get("coordinate_frame"),
                "pose_source": "known_control_c2w_to_cut3r_c2w",
                "surfel_index": str(Path(surfel_index_path).resolve()),
                "surfel_sequence": str(sequence_path),
                "source_candidate_only": True,
                "per_frame": per_frame_geometry,
            },
        )
        plans[target_chunk] = plan
        selection["status"] = "scheduled_visible_support"
        selections.append(selection)
    return plans, selections


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _save_rgb_tensor(frame: torch.Tensor, path: Path) -> None:
    array = (
        frame.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(array).save(path)


@torch.no_grad()
def save_warp_reencode_artifacts(
    *,
    plans: dict[int, WarpReencodePlan],
    vae,
    output_root: str | Path,
    device: torch.device,
) -> dict:
    root = Path(output_root).resolve() / "warp"
    root.mkdir(parents=True, exist_ok=True)
    plan_modes = {plan.mode for plan in plans.values()}
    manifest = {
        "mode": (
            next(iter(plan_modes))
            if len(plan_modes) == 1
            else "mixed_virtual_recent"
        ),
        "targets": [],
    }
    names = (
        "historical",
        "warped",
        "current_recent",
        "warped_recent",
        "virtual_recent",
    )
    for target, plan in sorted(plans.items()):
        if not plan.artifacts:
            raise RuntimeError(f"Warp plan for target {target} was not materialized")
        target_root = root / f"target_{target:04d}"
        target_root.mkdir(parents=True, exist_ok=True)
        latent_batch = torch.cat(
            [plan.artifacts[name] for name in names], dim=0
        ).to(device=device, dtype=plan.historical_latent.dtype)
        decoded = (
            vae.decode_to_pixel(latent_batch, use_cache=False) * 0.5 + 0.5
        ).clamp(0, 1)
        center = decoded.shape[1] // 2
        for index, name in enumerate(names):
            _save_rgb_tensor(decoded[index, center], target_root / f"{name}.png")
        coverage = plan.artifacts["coverage"].float().mean(dim=(0, 1, 2))
        coverage_image = Image.fromarray(
            (coverage.numpy().clip(0, 1) * 255).round().astype(np.uint8)
        ).resize(
            (int(decoded.shape[-1]), int(decoded.shape[-2])),
            Image.Resampling.BILINEAR,
        )
        coverage_image.save(target_root / "coverage.png")
        if "recent_coverage" in plan.artifacts:
            recent_coverage = plan.artifacts["recent_coverage"].float().mean(
                dim=(0, 1)
            )
            if recent_coverage.ndim == 3:
                recent_coverage = recent_coverage.mean(dim=0)
            Image.fromarray(
                (recent_coverage.numpy().clip(0, 1) * 255)
                .round()
                .astype(np.uint8)
            ).resize(
                (int(decoded.shape[-1]), int(decoded.shape[-2])),
                Image.Resampling.BILINEAR,
            ).save(target_root / "recent_coverage.png")
        torch.save(
            {
                key: value
                for key, value in plan.artifacts.items()
            },
            target_root / "warp_state.pt",
        )
        entry = dict(plan.audit)
        entry.update(
            {
                "artifact_root": str(target_root.relative_to(root.parent)),
                "historical_latent_sha256": _tensor_sha256(
                    plan.artifacts["historical"]
                ),
                "virtual_recent_latent_sha256": _tensor_sha256(
                    plan.artifacts["virtual_recent"]
                ),
            }
        )
        (target_root / "audit.json").write_text(
            json.dumps(entry, indent=2), encoding="utf-8"
        )
        manifest["targets"].append(entry)
        del latent_batch, decoded
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


__all__ = [
    "WarpReencodePlan",
    "build_continuous_virtual_recent_plans",
    "build_rotation_target_to_source_grid",
    "build_warp_reencode_plans",
    "feather_coverage",
    "infer_intrinsic_image_hw",
    "latent_to_rgb_index",
    "load_intrinsics",
    "save_warp_reencode_artifacts",
    "warp_latent",
]
