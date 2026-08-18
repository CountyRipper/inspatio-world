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
from mapkv_proto.pose_utils import rotation_geodesic, scale_intrinsics


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
        virtual = blend * warped + (1.0 - blend) * current_recent
        self.artifacts = {
            "historical": historical.detach().cpu(),
            "warped": warped.detach().cpu(),
            "current_recent": current_recent.detach().cpu(),
            "virtual_recent": virtual.detach().cpu(),
            "coverage": coverage.detach().cpu(),
            "target_to_source_grid": self.target_to_source_grid.detach().cpu(),
        }
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
                "mode": "camera_aligned_warp_and_reencode_recent",
                "target_block": int(self.target_block),
                "source_chunk": int(self.source_chunk),
                "coverage_fraction": float(coverage.mean().item()),
                "historical_vs_current_recent_overlap_l1": historical_delta,
                "virtual_vs_current_recent_l1": float(
                    (virtual.float() - current_recent.float()).abs().mean().item()
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
    manifest = {
        "mode": "camera_aligned_warp_and_reencode_recent",
        "targets": [],
    }
    names = ("historical", "warped", "current_recent", "virtual_recent")
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
    "build_rotation_target_to_source_grid",
    "build_warp_reencode_plans",
    "feather_coverage",
    "infer_intrinsic_image_hw",
    "latent_to_rgb_index",
    "load_intrinsics",
    "save_warp_reencode_artifacts",
    "warp_latent",
]
