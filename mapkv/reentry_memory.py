from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class MemoryEpisodeState(str, Enum):
    """Lifecycle of one remembered surface group."""

    VISIBLE_RECENT = "VISIBLE_RECENT"
    ABSENT = "ABSENT"
    REENTERED = "REENTERED"
    SERVED = "SERVED"


@dataclass(frozen=True)
class MemoryEpisodeDecision:
    state_before: MemoryEpisodeState
    state_after: MemoryEpisodeState
    visible: bool
    read_support: bool
    read_long_term: bool
    absence_count: int
    episode_id: int


class ReentryMemoryLifecycle:
    """Group-level write-first/read-on-reentry memory policy.

    The first visibility episode is propagated by native ``last_pred`` only.
    A remembered group becomes readable after ``absent_blocks`` consecutive
    invisible blocks.  One supported re-entry block is served, after which
    native Recent owns propagation until the next genuine absence episode.
    """

    def __init__(self, absent_blocks: int = 2):
        if absent_blocks < 1:
            raise ValueError("absent_blocks must be positive")
        self.absent_blocks = int(absent_blocks)
        self.state = MemoryEpisodeState.VISIBLE_RECENT
        self.absence_count = 0
        self.episode_id = 0

    def step(
        self,
        *,
        visible: bool,
        read_support: bool,
    ) -> MemoryEpisodeDecision:
        before = self.state
        read_long_term = False

        if self.state in {
            MemoryEpisodeState.VISIBLE_RECENT,
            MemoryEpisodeState.SERVED,
        }:
            if visible:
                self.absence_count = 0
            else:
                self.absence_count += 1
                if self.absence_count >= self.absent_blocks:
                    self.state = MemoryEpisodeState.ABSENT
        elif self.state == MemoryEpisodeState.ABSENT:
            if visible:
                self.state = MemoryEpisodeState.REENTERED
                self.absence_count = 0
                self.episode_id += 1
        elif self.state == MemoryEpisodeState.REENTERED:
            if visible:
                self.absence_count = 0
            else:
                self.absence_count += 1
                if self.absence_count >= self.absent_blocks:
                    self.state = MemoryEpisodeState.ABSENT

        if (
            self.state == MemoryEpisodeState.REENTERED
            and visible
            and read_support
        ):
            read_long_term = True
            self.state = MemoryEpisodeState.SERVED
            self.absence_count = 0

        return MemoryEpisodeDecision(
            state_before=before,
            state_after=self.state,
            visible=bool(visible),
            read_support=bool(read_support),
            read_long_term=read_long_term,
            absence_count=int(self.absence_count),
            episode_id=int(self.episode_id),
        )


def erode_binary_coverage(
    coverage: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Erode BFHW support without ever expanding beyond the valid FOV."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("erosion kernel must be a positive odd integer")
    if coverage.ndim != 4:
        raise ValueError(
            f"coverage must be [B,F,H,W], got {tuple(coverage.shape)}"
        )
    hard = (coverage.float() > 0).to(torch.float32)
    if kernel_size == 1:
        return hard
    batch, frames, height, width = hard.shape
    flat = hard.reshape(batch * frames, 1, height, width)
    eroded = 1.0 - F.max_pool2d(
        1.0 - flat,
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return eroded.reshape(batch, frames, height, width).clamp(0, 1)


def inward_feather_token_gate(
    coverage: torch.Tensor,
    *,
    batch: int,
    frames: int,
    token_hw: tuple[int, int],
    device: torch.device,
    feather_kernel: int = 3,
) -> torch.Tensor:
    """Support-preserving token gate with a boundary feather only inward.

    Outside support is mathematically zero.  Interior tokens stay one, while
    only supported boundary tokens receive a soft value.
    """
    if feather_kernel < 1 or feather_kernel % 2 == 0:
        raise ValueError("feather kernel must be a positive odd integer")
    value = torch.as_tensor(coverage, dtype=torch.float32, device=device)
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[None]
    if value.ndim != 4:
        raise ValueError("coverage must have 2, 3, or 4 dimensions")
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, -1, -1, -1)
    if value.shape[1] == 1 and frames > 1:
        value = value.expand(-1, frames, -1, -1)
    if value.shape[:2] != (batch, frames):
        raise ValueError(
            f"coverage batch/frame shape {tuple(value.shape[:2])} != "
            f"{(batch, frames)}"
        )
    flat = value.reshape(batch * frames, 1, *value.shape[-2:]).clamp(0, 1)
    if tuple(flat.shape[-2:]) != tuple(token_hw):
        if flat.shape[-2] >= token_hw[0] and flat.shape[-1] >= token_hw[1]:
            flat = F.adaptive_max_pool2d(flat, token_hw)
        else:
            flat = F.interpolate(flat, size=token_hw, mode="nearest")
    hard = (flat > 0).to(torch.float32)
    if feather_kernel == 1:
        return hard.reshape(batch, frames, *token_hw)
    interior = 1.0 - F.max_pool2d(
        1.0 - hard,
        feather_kernel,
        stride=1,
        padding=feather_kernel // 2,
    )
    local_density = F.avg_pool2d(
        hard,
        feather_kernel,
        stride=1,
        padding=feather_kernel // 2,
    )
    feathered = torch.where(interior > 0, torch.ones_like(hard), local_density)
    feathered = feathered * hard
    return feathered.reshape(batch, frames, *token_hw).clamp(0, 1)


__all__ = [
    "MemoryEpisodeDecision",
    "MemoryEpisodeState",
    "ReentryMemoryLifecycle",
    "erode_binary_coverage",
    "inward_feather_token_gate",
]
