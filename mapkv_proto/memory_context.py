from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import torch
import torch.nn.functional as F

from .config import resolve_indices


@dataclass(frozen=True)
class ActiveLayerMemory:
    """One layer's native recent-slot payload, ready for attention."""

    k: torch.Tensor
    v: torch.Tensor
    alpha: float
    query_gate: torch.Tensor | None
    source_chunk: int


def _smooth_gate(gate: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size == 1:
        return gate
    batch, frames, height, width = gate.shape
    flat = gate.reshape(batch * frames, 1, height, width)
    flat = F.avg_pool2d(flat, kernel_size, stride=1, padding=kernel_size // 2)
    return flat.reshape(batch, frames, height, width).clamp_(0, 1)


def _dilate_gate(gate: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    batch, frames, height, width = gate.shape
    flat = gate.reshape(batch * frames, 1, height, width)
    flat = F.max_pool2d(flat, kernel_size, stride=1, padding=kernel_size // 2)
    return flat.reshape(batch, frames, height, width)


def reference_blind_gate(
    mask_block: torch.Tensor,
    token_hw: tuple[int, int],
    *,
    smooth_kernel: int = 3,
) -> torch.Tensor:
    """Convert the upstream [-1, 1] warp-valid mask to a token query gate."""
    if mask_block.ndim != 5:
        raise ValueError(f"mask_block must be [B,F,C,H,W], got {tuple(mask_block.shape)}")
    batch, frames = mask_block.shape[:2]
    mask01 = ((mask_block.float() + 1.0) * 0.5).clamp_(0, 1)
    ref_valid_latent = mask01.mean(dim=2)
    pooled = F.adaptive_avg_pool2d(
        ref_valid_latent.reshape(batch * frames, 1, *ref_valid_latent.shape[-2:]),
        token_hw,
    ).reshape(batch, frames, *token_hw)
    return _smooth_gate(1.0 - pooled, smooth_kernel)


def normalize_coverage(
    coverage: torch.Tensor,
    *,
    batch: int,
    frames: int,
    token_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Normalize a center-frame or per-frame surfel mask to [B,F,Htok,Wtok]."""
    coverage = torch.as_tensor(coverage, dtype=torch.float32, device=device)
    if coverage.ndim == 2:
        coverage = coverage[None, None]
    elif coverage.ndim == 3:
        coverage = coverage[None]
    if coverage.ndim != 4:
        raise ValueError(f"coverage must have 2, 3, or 4 dimensions, got {coverage.ndim}")
    if coverage.shape[0] == 1 and batch > 1:
        coverage = coverage.expand(batch, -1, -1, -1)
    if coverage.shape[1] == 1 and frames > 1:
        coverage = coverage.expand(-1, frames, -1, -1)
    if coverage.shape[:2] != (batch, frames):
        raise ValueError(
            f"coverage batch/frame shape {tuple(coverage.shape[:2])} != {(batch, frames)}"
        )
    if tuple(coverage.shape[-2:]) != token_hw:
        coverage = F.interpolate(
            coverage.reshape(batch * frames, 1, *coverage.shape[-2:]),
            size=token_hw,
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, *token_hw)
    return coverage.clamp_(0, 1)


@dataclass(frozen=True)
class MemoryContext:
    """Immutable activation plan for one target latent block."""

    target_block: int
    source_chunk: int
    layer_payloads: dict[int, tuple[torch.Tensor, torch.Tensor]]
    selected_layers: tuple[int, ...]
    selected_step_indices: tuple[int, ...]
    alpha: float
    gate_mode: str = "ref_blind"
    smooth_kernel: int = 3
    coverage: torch.Tensor | None = None
    query_gate: torch.Tensor | None = None
    active_step: int | None = None
    num_steps: int | None = None
    audit_log: list[dict] = field(default_factory=list, compare=False)

    def for_denoising_step(self, step_index: int, num_steps: int) -> "MemoryContext | None":
        selected = resolve_indices(self.selected_step_indices, num_steps, name="denoising step")
        if step_index not in selected or self.alpha == 0.0:
            return None
        return replace(self, active_step=step_index, num_steps=num_steps)

    def with_query_gate(
        self,
        mask_block: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> "MemoryContext":
        batch, frames = mask_block.shape[:2]
        if self.gate_mode == "global":
            gate = torch.ones(
                (batch, frames, *token_hw), device=mask_block.device, dtype=torch.float32
            )
        else:
            gate = reference_blind_gate(
                mask_block, token_hw, smooth_kernel=self.smooth_kernel
            )
            if self.gate_mode == "surfel_ref_blind":
                if self.coverage is None:
                    raise ValueError("surfel_ref_blind gate requires a coverage mask")
                coverage = normalize_coverage(
                    self.coverage,
                    batch=batch,
                    frames=frames,
                    token_hw=token_hw,
                    device=mask_block.device,
                )
                coverage = _smooth_gate(_dilate_gate(coverage), self.smooth_kernel)
                gate = gate * coverage
            elif self.gate_mode != "ref_blind":
                raise ValueError(f"Unsupported gate mode: {self.gate_mode}")
        return replace(self, query_gate=gate.flatten(1).to(dtype=mask_block.dtype))

    def for_layer(self, block_index: int, num_layers: int) -> ActiveLayerMemory | None:
        selected = resolve_indices(self.selected_layers, num_layers, name="transformer layer")
        if block_index not in selected:
            return None
        if block_index not in self.layer_payloads:
            raise KeyError(f"No KV payload for selected transformer layer {block_index}")
        if self.active_step is None:
            raise RuntimeError("MemoryContext must be activated with for_denoising_step first")
        k, v = self.layer_payloads[block_index]
        self.audit_log.append(
            {
                "target_block": self.target_block,
                "source_chunk": self.source_chunk,
                "step_index": self.active_step,
                "layer_index": block_index,
                "alpha": self.alpha,
            }
        )
        return ActiveLayerMemory(
            k=k,
            v=v,
            alpha=self.alpha,
            query_gate=self.query_gate,
            source_chunk=self.source_chunk,
        )

    @property
    def coverage_is_empty(self) -> bool:
        return self.coverage is not None and not bool(torch.as_tensor(self.coverage).any())


def make_memory_context(
    *,
    target_block: int,
    source_chunk: int,
    layer_payloads: dict[int, tuple[torch.Tensor, torch.Tensor]],
    selected_layers: Iterable[int],
    selected_step_indices: Iterable[int],
    alpha: float,
    gate_mode: str,
    smooth_kernel: int,
    coverage: torch.Tensor | None = None,
) -> MemoryContext | None:
    if gate_mode == "surfel_ref_blind" and coverage is not None and not bool(coverage.any()):
        return None
    return MemoryContext(
        target_block=target_block,
        source_chunk=source_chunk,
        layer_payloads=layer_payloads,
        selected_layers=tuple(selected_layers),
        selected_step_indices=tuple(selected_step_indices),
        alpha=float(alpha),
        gate_mode=gate_mode,
        smooth_kernel=smooth_kernel,
        coverage=coverage,
    )
