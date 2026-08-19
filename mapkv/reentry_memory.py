from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class MemoryEpisodeState(str, Enum):
    """Lifecycle of one remembered surface group."""

    VISIBLE_RECENT = "VISIBLE_RECENT"
    FIRST_VISIBILITY = "FIRST_VISIBILITY"
    ABSENT = "ABSENT"
    REENTERED = "REENTERED"
    REENTRY_ACTIVE = "REENTRY_ACTIVE"
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


class ReentryEpisodeLifecycle:
    """Write-first, continuously-read-on-reentry episode policy.

    The first visibility episode never reads long-term memory. After a true
    absence, every readable block in the complete continuous visibility
    episode receives memory. A second absence closes that episode and arms
    the next one. This deliberately has no SERVED handoff state.
    """

    def __init__(self, absent_blocks: int = 2):
        if absent_blocks < 1:
            raise ValueError("absent_blocks must be positive")
        self.absent_blocks = int(absent_blocks)
        self.state = MemoryEpisodeState.FIRST_VISIBILITY
        self.absence_count = 0
        self.episode_id = 0

    def step(
        self,
        *,
        visible: bool,
        read_support: bool,
    ) -> MemoryEpisodeDecision:
        before = self.state

        if self.state == MemoryEpisodeState.FIRST_VISIBILITY:
            if visible:
                self.absence_count = 0
            else:
                self.absence_count += 1
                if self.absence_count >= self.absent_blocks:
                    self.state = MemoryEpisodeState.ABSENT
        elif self.state == MemoryEpisodeState.ABSENT:
            if visible:
                self.state = MemoryEpisodeState.REENTRY_ACTIVE
                self.absence_count = 0
                self.episode_id += 1
        elif self.state == MemoryEpisodeState.REENTRY_ACTIVE:
            if visible:
                self.absence_count = 0
            else:
                self.absence_count += 1
                if self.absence_count >= self.absent_blocks:
                    self.state = MemoryEpisodeState.ABSENT

        read_long_term = bool(
            self.state == MemoryEpisodeState.REENTRY_ACTIVE
            and visible
            and read_support
        )
        return MemoryEpisodeDecision(
            state_before=before,
            state_after=self.state,
            visible=bool(visible),
            read_support=bool(read_support),
            read_long_term=read_long_term,
            absence_count=int(self.absence_count),
            episode_id=int(self.episode_id),
        )


@dataclass(frozen=True)
class SurfaceRefreshDecision:
    """Per-block state for independently refreshed remembered surfels."""

    visible_surface_ids: tuple[int, ...]
    readable_surface_ids: tuple[int, ...]
    newly_reentered_surface_ids: tuple[int, ...]
    active_surface_ids: tuple[int, ...]
    armed_surface_count: int
    ttl_histogram_before_decrement: dict[int, int]


class PerSurfaceRefreshLifecycle:
    """Give each truly re-entered surfel an independent fixed refresh TTL."""

    def __init__(
        self,
        surface_ids,
        *,
        absent_blocks: int = 2,
        refresh_ttl_blocks: int = 2,
    ):
        if absent_blocks < 1:
            raise ValueError("absent_blocks must be positive")
        if refresh_ttl_blocks < 1:
            raise ValueError("refresh_ttl_blocks must be positive")
        ids = tuple(sorted({int(value) for value in surface_ids}))
        if not ids:
            raise ValueError("surface_ids must be non-empty")
        self.surface_ids = ids
        self.absent_blocks = int(absent_blocks)
        self.refresh_ttl_blocks = int(refresh_ttl_blocks)
        self._absence = {surface_id: 0 for surface_id in ids}
        self._armed = {surface_id: False for surface_id in ids}
        self._ttl = {surface_id: 0 for surface_id in ids}

    def step(
        self,
        *,
        visible_surface_ids,
        readable_surface_ids,
    ) -> SurfaceRefreshDecision:
        known = set(self.surface_ids)
        visible = {int(value) for value in visible_surface_ids} & known
        readable = {int(value) for value in readable_surface_ids} & visible
        newly_reentered: set[int] = set()

        for surface_id in self.surface_ids:
            if surface_id in visible:
                self._absence[surface_id] = 0
                if self._armed[surface_id] and surface_id in readable:
                    self._ttl[surface_id] = self.refresh_ttl_blocks
                    self._armed[surface_id] = False
                    newly_reentered.add(surface_id)
            else:
                self._absence[surface_id] += 1
                if self._absence[surface_id] >= self.absent_blocks:
                    self._armed[surface_id] = True
                    self._ttl[surface_id] = 0

        active = {
            surface_id
            for surface_id in readable
            if self._ttl[surface_id] > 0
        }
        ttl_histogram: dict[int, int] = {}
        for surface_id in active:
            ttl = int(self._ttl[surface_id])
            ttl_histogram[ttl] = ttl_histogram.get(ttl, 0) + 1
        decision = SurfaceRefreshDecision(
            visible_surface_ids=tuple(sorted(visible)),
            readable_surface_ids=tuple(sorted(readable)),
            newly_reentered_surface_ids=tuple(sorted(newly_reentered)),
            active_surface_ids=tuple(sorted(active)),
            armed_surface_count=sum(bool(value) for value in self._armed.values()),
            ttl_histogram_before_decrement=ttl_histogram,
        )
        for surface_id in active:
            self._ttl[surface_id] -= 1
        return decision


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
    "PerSurfaceRefreshLifecycle",
    "ReentryEpisodeLifecycle",
    "ReentryMemoryLifecycle",
    "SurfaceRefreshDecision",
    "erode_binary_coverage",
    "inward_feather_token_gate",
]
