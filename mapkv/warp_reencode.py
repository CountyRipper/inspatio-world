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
from .reentry_memory import (
    ReentryMemoryLifecycle,
    erode_binary_coverage,
)


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


def strong_memory_coverage(
    hard_coverage: torch.Tensor,
    dilation_kernel: int = 3,
) -> torch.Tensor:
    """Build a binary, support-preserving memory-composition mask.

    The old WRE path averaged the projected surfel mask before both latent
    composition and attention gating.  This path keeps the historical core at
    one and only expands it by a small, explicit morphology operation.
    """
    if dilation_kernel < 1 or dilation_kernel % 2 == 0:
        raise ValueError("memory dilation kernel must be a positive odd integer")
    if hard_coverage.ndim != 4:
        raise ValueError(
            "hard_coverage must be [B,F,H,W], got "
            f"{tuple(hard_coverage.shape)}"
        )
    binary = (hard_coverage.float() > 0).to(dtype=torch.float32)
    if dilation_kernel == 1:
        return binary
    batch, frames, height, width = binary.shape
    dilated = F.max_pool2d(
        binary.reshape(batch * frames, 1, height, width),
        dilation_kernel,
        stride=1,
        padding=dilation_kernel // 2,
    )
    return dilated.reshape(batch, frames, height, width)


def reference_protected_coverage(
    mask_block: torch.Tensor,
    *,
    dilation_kernel: int = 3,
) -> torch.Tensor:
    """Return [B,F,H,W] source-valid support protected from MapKV."""
    if dilation_kernel < 1 or dilation_kernel % 2 == 0:
        raise ValueError(
            "reference protection kernel must be a positive odd integer"
        )
    if mask_block.ndim != 5:
        raise ValueError(
            f"mask_block must be [B,F,C,H,W], got {tuple(mask_block.shape)}"
        )
    valid = ((mask_block.float() + 1.0) * 0.5).clamp(0, 1).mean(dim=2)
    if dilation_kernel == 1:
        return valid
    batch, frames, height, width = valid.shape
    protected = F.max_pool2d(
        valid.reshape(batch * frames, 1, height, width),
        dilation_kernel,
        stride=1,
        padding=dilation_kernel // 2,
    )
    return protected.reshape(batch, frames, height, width).clamp(0, 1)


def warp_latent(
    historical_latent: torch.Tensor,
    grid: torch.Tensor,
    *,
    padding_mode: str = "zeros",
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
    if padding_mode not in {"zeros", "border", "reflection"}:
        raise ValueError(f"Unsupported grid-sample padding mode: {padding_mode}")
    source_dtype = historical_latent.dtype
    warped = F.grid_sample(
        historical_latent.float().reshape(
            batch * frames, channels, height, width
        ),
        grid.reshape(batch * frames, height, width, 2),
        mode="bilinear",
        padding_mode=padding_mode,
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
    hard_coverage: torch.Tensor | None = None
    history_coverage: torch.Tensor | None = None
    reference_valid_coverage: torch.Tensor | None = None
    reference_protected_coverage: torch.Tensor | None = None
    need_coverage: torch.Tensor | None = None
    warp_valid_coverage: torch.Tensor | None = None
    reentry_coverage: torch.Tensor | None = None
    safe_coverage: torch.Tensor | None = None
    query_coverage: torch.Tensor | None = None
    query_feather_kernel: int = 1
    reference_protection_kernel: int = 3
    historical_representation: str = "latent_warp"
    historical_is_target_aligned: bool = False
    rgb_padding_mode: str = "zeros"
    rgb_preview_source: torch.Tensor | None = None
    rgb_preview_target: torch.Tensor | None = None
    rgb_warp_coverage_preview: torch.Tensor | None = None
    query_gate_mode: str = "global"
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
        warped = (
            historical
            if self.historical_is_target_aligned
            else warp_latent(
                historical,
                self.target_to_source_grid,
                padding_mode=self.rgb_padding_mode,
            )
        )
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
            "memory_coverage": coverage.detach().cpu(),
            "target_to_source_grid": self.target_to_source_grid.detach().cpu(),
        }
        if self.rgb_preview_source is not None:
            self.artifacts["rgb_preview_source"] = (
                self.rgb_preview_source.detach().cpu()
            )
        if self.rgb_preview_target is not None:
            self.artifacts["rgb_preview_target"] = (
                self.rgb_preview_target.detach().cpu()
            )
        if self.rgb_warp_coverage_preview is not None:
            self.artifacts["rgb_warp_coverage_preview"] = (
                self.rgb_warp_coverage_preview.detach().cpu()
            )
        if self.hard_coverage is not None:
            self.artifacts["hard_coverage"] = self.hard_coverage.detach().cpu()
        if self.history_coverage is not None:
            self.artifacts["history_coverage"] = (
                self.history_coverage.detach().cpu()
            )
        if self.reference_valid_coverage is not None:
            self.artifacts["reference_valid_coverage"] = (
                self.reference_valid_coverage.detach().cpu()
            )
        if self.reference_protected_coverage is not None:
            self.artifacts["reference_protected_coverage"] = (
                self.reference_protected_coverage.detach().cpu()
            )
        if self.need_coverage is not None:
            self.artifacts["need_coverage"] = (
                self.need_coverage.detach().cpu()
            )
        if self.warp_valid_coverage is not None:
            self.artifacts["warp_valid_coverage"] = (
                self.warp_valid_coverage.detach().cpu()
            )
        if self.reentry_coverage is not None:
            self.artifacts["reentry_coverage"] = (
                self.reentry_coverage.detach().cpu()
            )
        if self.safe_coverage is not None:
            self.artifacts["safe_coverage"] = (
                self.safe_coverage.detach().cpu()
            )
        if self.query_coverage is not None:
            self.artifacts["query_coverage_source"] = (
                self.query_coverage.detach().cpu()
            )
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
                "historical_representation": self.historical_representation,
                "historical_is_target_aligned": self.historical_is_target_aligned,
                "rgb_padding_mode": self.rgb_padding_mode,
                "target_block": int(self.target_block),
                "source_chunk": int(self.source_chunk),
                "coverage_fraction": float(coverage.mean().item()),
                "memory_coverage_fraction": float(coverage.mean().item()),
                "hard_coverage_fraction": (
                    None
                    if self.hard_coverage is None
                    else float(self.hard_coverage.float().mean().item())
                ),
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
        if self.query_gate_mode not in {
            "global",
            "surfel_exact",
            "surfel_support_preserving",
            "surfel_source_protected",
            "surfel_edge_safe_source_protected",
        }:
            raise ValueError(
                f"Unsupported Virtual Recent query gate: {self.query_gate_mode}"
            )
        query_coverage = (
            self.coverage
            if self.query_coverage is None
            else self.query_coverage
        )
        context = make_memory_context(
            target_block=self.target_block,
            source_chunk=self.source_chunk,
            layer_payloads=layer_payloads,
            selected_layers=self.selected_layers,
            selected_step_indices=self.selected_step_indices,
            alpha=self.alpha,
            injection_mode="replace_recent_delta",
            gate_mode=self.query_gate_mode,
            smooth_kernel=self.query_feather_kernel,
            coverage=(
                query_coverage
                if self.query_gate_mode != "global"
                else None
            ),
            reference_protection_kernel=self.reference_protection_kernel,
        )
        if context is None:
            raise RuntimeError("Virtual recent context unexpectedly disabled")
        self.audit.update(
            {
                "attention_query_gate_mode": self.query_gate_mode,
                "query_gate_uses_latent_composition_mask": (
                    self.query_gate_mode == "surfel_exact"
                    and query_coverage is self.coverage
                ),
                "memory_and_query_masks_split": (
                    self.query_gate_mode
                    in {
                        "surfel_support_preserving",
                        "surfel_source_protected",
                        "surfel_edge_safe_source_protected",
                    }
                ),
                "query_feather_kernel": int(self.query_feather_kernel),
                "reference_protection_kernel": int(
                    self.reference_protection_kernel
                ),
            }
        )
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
    generated_only: bool = False,
    reference_blind_threshold: float = 0.5,
    eligible_indices: np.ndarray | None = None,
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
    generated_only_indices = None
    if generated_only:
        tagged = [
            cell
            for cell in eligible
            if int(source_chunk) in cell.reference_blind_at_write
        ]
        if not tagged:
            raise RuntimeError(
                "Source-protected WRE requires reference_blind_at_write "
                f"metadata for source chunk {source_chunk}"
            )
        generated_only_indices = surfel_index.generated_only_cell_indices(
            source_chunk,
            reference_blind_threshold=reference_blind_threshold,
        )
    if eligible_indices is not None:
        restricted = np.asarray(eligible_indices, dtype=np.int32).reshape(-1)
        generated_only_indices = (
            restricted
            if generated_only_indices is None
            else np.intersect1d(
                generated_only_indices, restricted, assume_unique=False
            ).astype(np.int32)
        )
    visible = surfel_index.visible_cells(
        query_pose,
        intrinsics,
        target_hw,
        source_image_size=source_image_hw,
        eligible_max_chunk=int(target_chunk) - 2,
        eligible_chunks={int(source_chunk)},
        eligible_indices=generated_only_indices,
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
        "generated_only_filter": bool(generated_only),
        "reference_blind_threshold": (
            float(reference_blind_threshold) if generated_only else None
        ),
        "generated_only_source_surfels": (
            None
            if generated_only_indices is None
            else int(len(generated_only_indices))
        ),
        "surface_group_restricted": eligible_indices is not None,
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
    warp_short_term_recent: bool = True,
    query_gate_mode: str = "global",
    mask_policy: str = "legacy_soft",
    memory_dilation_kernel: int = 3,
    query_feather_kernel: int = 3,
    historical_representation: str = "latent_warp",
    vae=None,
    reference_mask_latent: torch.Tensor | None = None,
    source_protection: bool = False,
    reference_protection_dilation_kernel: int = 3,
    generated_only_threshold: float = 0.5,
) -> tuple[dict[int, WarpReencodePlan], list[dict]]:
    """Build visibility-driven Virtual Recent plans for every causal block.

    Historical B1 is reprojected into each target block's camera layout.
    ``warp_short_term_recent=True`` preserves the original failed CAVR
    ablation; the repaired path keeps raw native last_pred as the fallback.
    Projected source-chunk surfel support controls activation, and optionally
    gates the counterfactual attention delta through ``surfel_exact``.
    """
    if min_history_gap_chunks < 2:
        raise ValueError(
            "Long-term memory must exclude the immediate recent chunk"
        )
    if query_gate_mode not in {
        "global",
        "surfel_exact",
        "surfel_support_preserving",
        "surfel_source_protected",
    }:
        raise ValueError(
            f"Unsupported continuous query gate mode: {query_gate_mode}"
        )
    if mask_policy not in {"legacy_soft", "strong_core"}:
        raise ValueError(f"Unsupported continuous mask policy: {mask_policy}")
    if mask_policy == "strong_core" and query_gate_mode not in {
        "surfel_support_preserving",
        "surfel_source_protected",
    }:
        raise ValueError(
            "strong_core mask policy requires surfel_support_preserving query gate"
        )
    if historical_representation not in {"latent_warp", "rgb_warp_vae"}:
        raise ValueError(
            f"Unsupported historical representation: {historical_representation}"
        )
    if historical_representation == "rgb_warp_vae" and vae is None:
        raise ValueError("rgb_warp_vae requires the native Wan VAE")
    if source_protection:
        if reference_mask_latent is None:
            raise ValueError(
                "source-protected WRE requires the current reference mask"
            )
        if reference_mask_latent.ndim != 5:
            raise ValueError(
                "reference_mask_latent must be [B,F,C,H,W]"
            )
        if int(reference_mask_latent.shape[1]) != int(latent_length):
            raise ValueError(
                "reference mask latent length does not match generation"
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
    historical_rgb = None
    if historical_representation == "rgb_warp_vae":
        with torch.no_grad():
            historical_rgb = (
                vae.decode_to_pixel(historical, use_cache=False) * 0.5 + 0.5
            ).clamp(0, 1)
        if historical_rgb.ndim != 5 or historical_rgb.shape[2] != 3:
            raise RuntimeError(
                "Native VAE decode must return [B,T,3,H,W], got "
                f"{tuple(historical_rgb.shape)}"
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
            if warp_short_term_recent:
                recent_grid, recent_mask, _ = (
                    build_rotation_target_to_source_grid(
                        recent_pose,
                        target_pose,
                        latent_intrinsics,
                        latent_intrinsics,
                        latent_hw,
                    )
                )
                recent_grids.append(recent_grid)
                recent_valid.append(recent_mask)
            surfel_mask, geometry = _surfel_coverage_for_pose(
                surfel_index=surfel_index,
                source_chunk=int(source_chunk),
                target_chunk=target_chunk,
                query_pose=to_cut3r_c2w(target_pose),
                intrinsics=surfel_intrinsics,
                source_image_hw=surfel_source_hw,
                target_hw=latent_hw,
                generated_only=source_protection,
                reference_blind_threshold=generated_only_threshold,
            )
            historical_grids.append(history_grid)
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
        plan_historical = historical
        rgb_preview_source = None
        rgb_preview_target = None
        rgb_warp_coverage_preview = None
        if historical_representation == "rgb_warp_vae":
            rgb_frames = int(historical_rgb.shape[1])
            source_dense_rgb = np.rint(
                np.linspace(source_rgb[0], source_rgb[-1], rgb_frames)
            ).astype(np.int64)
            target_dense_rgb = np.rint(
                np.linspace(target_rgb[0], target_rgb[-1], rgb_frames)
            ).astype(np.int64)
            rgb_grids = []
            rgb_valid = []
            for source_index, target_index in zip(
                source_dense_rgb, target_dense_rgb
            ):
                rgb_grid, rgb_mask, _ = build_rotation_target_to_source_grid(
                    poses[int(source_index)],
                    poses[int(target_index)],
                    image_intrinsics,
                    image_intrinsics,
                    image_hw,
                )
                rgb_grids.append(rgb_grid)
                rgb_valid.append(rgb_mask)
            rgb_grid_tensor = torch.stack(rgb_grids).to(device=device)
            warped_rgb = warp_latent(historical_rgb, rgb_grid_tensor)
            with torch.no_grad():
                plan_historical = vae.encode_to_latent(
                    (warped_rgb * 2.0 - 1.0)
                    .to(dtype=dtype)
                    .permute(0, 2, 1, 3, 4)
                ).to(device=device, dtype=dtype)
            if plan_historical.shape != historical.shape:
                raise RuntimeError(
                    "RGB-warp VAE encode changed historical block shape: "
                    f"{tuple(plan_historical.shape)} != {tuple(historical.shape)}"
                )
            preview_index = rgb_frames // 2
            rgb_preview_source = historical_rgb[:, preview_index]
            rgb_preview_target = warped_rgb[:, preview_index]
            rgb_warp_coverage_preview = torch.stack(rgb_valid)[
                preview_index
            ]
        history_coverage = torch.stack(surfel_masks).unsqueeze(0)
        reference_valid = None
        reference_protected = None
        if source_protection:
            target_reference_mask = reference_mask_latent[
                :, target_start:target_stop
            ].to(device=device)
            reference_valid = (
                ((target_reference_mask.float() + 1.0) * 0.5)
                .clamp(0, 1)
                .mean(dim=2)
            )
            reference_protected = reference_protected_coverage(
                target_reference_mask,
                dilation_kernel=reference_protection_dilation_kernel,
            ).to(device=device)
            hard_memory_coverage = (
                history_coverage.to(device=device)
                * (1.0 - reference_protected)
            )
        else:
            hard_memory_coverage = history_coverage.to(device=device)
        if mask_policy == "strong_core":
            coverage = strong_memory_coverage(
                hard_memory_coverage, memory_dilation_kernel
            ).to(device=device)
        else:
            coverage = feather_coverage(
                hard_memory_coverage, feather_kernel
            ).to(device=device)
        if reference_protected is not None:
            # Dilation/feather must never leak back into protected source.
            coverage = coverage * (1.0 - reference_protected)
        if warp_short_term_recent:
            mode = "continuous_geometry_reprojected_virtual_recent"
        elif query_gate_mode == "global":
            mode = "continuous_raw_recent_warp_reencode"
        elif mask_policy == "strong_core":
            mode = "strong_core_masked_continuous_wre"
        else:
            mode = "masked_continuous_warp_reencode"
        if historical_representation == "rgb_warp_vae":
            mode = "strong_core_rgb_warp_vae_wre"
        if source_protection:
            mode = "source_protected_rgb_warp_vae_wre"
        selection = {
            "target_chunk": target_chunk,
            "source_chunk": int(source_chunk),
            "mode": mode,
            "payload_kind": "target_aligned_virtual_recent_native_kv",
            "coverage_fraction": float(coverage.mean().item()),
            "hard_coverage_fraction": float(
                hard_memory_coverage.mean().item()
            ),
            "memory_mask_policy": mask_policy,
            "memory_dilation_kernel": int(memory_dilation_kernel),
            "query_feather_kernel": int(query_feather_kernel),
            "source_rgb_indices": list(source_rgb),
            "recent_rgb_indices": list(recent_rgb),
            "target_rgb_indices": list(target_rgb),
            "source_target_rotation_degrees": rotations,
            "source_latents_path": str(source_path),
            "activation_policy": "visible_source_surfel_support",
            "source_protection": bool(source_protection),
            "generated_only_history": bool(source_protection),
            "generated_only_threshold": (
                float(generated_only_threshold) if source_protection else None
            ),
            "reference_protection_dilation_kernel": (
                int(reference_protection_dilation_kernel)
                if source_protection
                else None
            ),
            "history_coverage_fraction": float(
                history_coverage.float().mean().item()
            ),
            "reference_valid_fraction": (
                None
                if reference_valid is None
                else float(reference_valid.mean().item())
            ),
            "reference_protected_fraction": (
                None
                if reference_protected is None
                else float(reference_protected.mean().item())
            ),
            "memory_need_fraction": float(
                hard_memory_coverage.float().mean().item()
            ),
            "short_term_recent": (
                "camera_warped" if warp_short_term_recent else "raw_last_pred"
            ),
            "attention_query_gate": query_gate_mode,
            "same_mask_controls_latent_and_attention": (
                query_gate_mode == "surfel_exact"
            ),
            "memory_and_query_masks_split": (
                query_gate_mode == "surfel_support_preserving"
            ),
            "historical_representation": historical_representation,
            "rgb_warp_before_vae": historical_representation == "rgb_warp_vae",
            "geometry_frames": per_frame_geometry,
        }
        if not bool(hard_memory_coverage.any()):
            selection["status"] = "memory_off_no_visible_support"
            selections.append(selection)
            continue
        recent_grid_tensor = None
        recent_coverage = None
        if warp_short_term_recent:
            recent_grid_tensor = torch.stack(recent_grids).to(device=device)
            recent_coverage = torch.stack(recent_valid).unsqueeze(0).to(
                device=device
            )
        plan = WarpReencodePlan(
            target_block=target_chunk,
            source_chunk=int(source_chunk),
            historical_latent=plan_historical,
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
            recent_target_to_source_grid=recent_grid_tensor,
            recent_coverage=recent_coverage,
            hard_coverage=hard_memory_coverage.to(device=device),
            history_coverage=history_coverage.to(device=device),
            reference_valid_coverage=reference_valid,
            reference_protected_coverage=reference_protected,
            need_coverage=hard_memory_coverage.to(device=device),
            query_coverage=coverage,
            query_feather_kernel=(
                int(query_feather_kernel)
                if query_gate_mode == "surfel_support_preserving"
                else 1
            ),
            query_gate_mode=query_gate_mode,
            reference_protection_kernel=int(
                reference_protection_dilation_kernel
            ),
            mode=mode,
            historical_representation=historical_representation,
            historical_is_target_aligned=(
                historical_representation == "rgb_warp_vae"
            ),
            rgb_preview_source=rgb_preview_source,
            rgb_preview_target=rgb_preview_target,
            rgb_warp_coverage_preview=rgb_warp_coverage_preview,
            geometry_audit={
                "coordinate_frame": sequence.get("coordinate_frame"),
                "pose_source": "known_control_c2w_to_cut3r_c2w",
                "surfel_index": str(Path(surfel_index_path).resolve()),
                "surfel_sequence": str(sequence_path),
                "source_candidate_only": True,
                "source_protection": bool(source_protection),
                "generated_only_observations": bool(source_protection),
                "memory_mask_policy": mask_policy,
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
        if "rgb_preview_source" in plan.artifacts:
            _save_rgb_tensor(
                plan.artifacts["rgb_preview_source"][0],
                target_root / "rgb_history_source.png",
            )
        if "rgb_preview_target" in plan.artifacts:
            _save_rgb_tensor(
                plan.artifacts["rgb_preview_target"][0],
                target_root / "rgb_history_warped_to_target.png",
            )
        if "rgb_warp_coverage_preview" in plan.artifacts:
            rgb_mask = plan.artifacts["rgb_warp_coverage_preview"].float()
            while rgb_mask.ndim > 2:
                rgb_mask = rgb_mask.mean(dim=0)
            Image.fromarray(
                (rgb_mask.numpy().clip(0, 1) * 255).round().astype(np.uint8)
            ).save(target_root / "rgb_warp_coverage.png")
        mask_artifacts = {
            "coverage": "coverage.png",
            "hard_coverage": "M_hard.png",
            "history_coverage": "M_history.png",
            "reference_valid_coverage": "M_ref_valid.png",
            "reference_protected_coverage": "M_ref_protected.png",
            "need_coverage": "M_need.png",
            "warp_valid_coverage": "M_warp_valid.png",
            "reentry_coverage": "M_reentry.png",
            "safe_coverage": "M_safe.png",
            "memory_coverage": "M_memory.png",
            "query_coverage_source": "M_query_source.png",
            "query_gate_tokens": "M_query.png",
        }
        for key, filename in mask_artifacts.items():
            if key not in plan.artifacts:
                continue
            mask = plan.artifacts[key].float()
            while mask.ndim > 2:
                mask = mask.mean(dim=0)
            Image.fromarray(
                (mask.numpy().clip(0, 1) * 255).round().astype(np.uint8)
            ).resize(
                (int(decoded.shape[-1]), int(decoded.shape[-2])),
                Image.Resampling.NEAREST if key == "hard_coverage" else Image.Resampling.BILINEAR,
            ).save(target_root / filename)
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
    "strong_memory_coverage",
    "infer_intrinsic_image_hw",
    "latent_to_rgb_index",
    "load_intrinsics",
    "save_warp_reencode_artifacts",
    "warp_latent",
]
