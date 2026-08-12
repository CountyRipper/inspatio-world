"""Tensor contracts for the immutable read-only World State."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch


class Provenance(IntEnum):
    SOURCE = 0
    GENERATED = 1


class Authority(IntEnum):
    UNKNOWN = 0
    GENERATED = 1
    SOURCE = 2


@dataclass(frozen=True)
class CameraBatch:
    """Actual per-latent-frame camera calibration in the W0 frame."""

    K: torch.Tensor  # [B,F,3,3]
    c2w_W0: torch.Tensor  # [B,F,4,4]

    def __post_init__(self) -> None:
        if self.K.ndim != 4 or self.K.shape[-2:] != (3, 3):
            raise ValueError("K must have shape [B,F,3,3]")
        if self.c2w_W0.ndim != 4 or self.c2w_W0.shape[-2:] != (4, 4):
            raise ValueError("c2w_W0 must have shape [B,F,4,4]")
        if self.K.shape[:2] != self.c2w_W0.shape[:2]:
            raise ValueError("K and c2w_W0 must share B,F")

    @property
    def batch_size(self) -> int:
        return self.K.shape[0]

    @property
    def frames(self) -> int:
        return self.K.shape[1]

    def slice(self, start: int, size: int) -> "CameraBatch":
        return CameraBatch(
            K=self.K[:, start:start + size],
            c2w_W0=self.c2w_W0[:, start:start + size],
        )

    def to(self, *args, **kwargs) -> "CameraBatch":
        return CameraBatch(
            K=self.K.to(*args, **kwargs),
            c2w_W0=self.c2w_W0.to(*args, **kwargs),
        )


@dataclass(frozen=True)
class WorldObservation:
    scene_id: str
    world_id: str
    observation_id: str
    provenance: int

    clean_latent: torch.Tensor  # [F=3,16,H,W]
    K: torch.Tensor  # [F,3,3]
    c2w_W0: torch.Tensor  # [F,4,4]
    depth: Optional[torch.Tensor]

    valid: torch.Tensor  # [F,1,H,W]
    static_confidence: torch.Tensor  # [F,1,H,W]
    geometry_confidence: torch.Tensor  # [F,1,H,W]

    def __post_init__(self) -> None:
        if self.clean_latent.ndim != 4 or self.clean_latent.shape[1] != 16:
            raise ValueError("clean_latent must have shape [F,16,H,W]")
        frames, _, height, width = self.clean_latent.shape
        if tuple(self.K.shape) != (frames, 3, 3):
            raise ValueError("observation K must have shape [F,3,3]")
        if tuple(self.c2w_W0.shape) != (frames, 4, 4):
            raise ValueError("observation c2w_W0 must have shape [F,4,4]")
        expected_mask = (frames, 1, height, width)
        if self.depth is not None and tuple(self.depth.shape) != expected_mask:
            raise ValueError(f"depth must have shape {expected_mask} when present")
        for name in ("valid", "static_confidence", "geometry_confidence"):
            if tuple(getattr(self, name).shape) != expected_mask:
                raise ValueError(f"{name} must have shape {expected_mask}")
        if int(self.provenance) not in (int(Provenance.SOURCE), int(Provenance.GENERATED)):
            raise ValueError("unsupported observation provenance")

    @property
    def authority(self) -> int:
        return (
            int(Authority.SOURCE)
            if int(self.provenance) == int(Provenance.SOURCE)
            else int(Authority.GENERATED)
        )


@dataclass(frozen=True)
class WorldReadPacket:
    """Independent projected candidates before patch-neighborhood expansion."""

    candidate_20ch: torch.Tensor  # [B,Kobs,F,20,H,W]
    valid: torch.Tensor  # [B,Kobs,F,1,H,W], bool
    authority: torch.Tensor  # [B,Kobs,F,1,H,W], integer
    confidence: torch.Tensor  # [B,Kobs,F,1,H,W]
    relative_pose: torch.Tensor  # [B,Kobs,F,6]
    view_angle: torch.Tensor  # [B,Kobs,F,1]
    subpixel_offset: torch.Tensor  # [B,Kobs,F,2,H,W]
    provenance: torch.Tensor  # [B,Kobs], integer
    observation_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.candidate_20ch.ndim != 6 or self.candidate_20ch.shape[3] != 20:
            raise ValueError("candidate_20ch must have shape [B,K,F,20,H,W]")
        batch, candidates, frames, _, height, width = self.candidate_20ch.shape
        mask_shape = (batch, candidates, frames, 1, height, width)
        if tuple(self.valid.shape) != mask_shape or self.valid.dtype != torch.bool:
            raise ValueError("valid must be bool [B,K,F,1,H,W]")
        for name in ("authority", "confidence"):
            if tuple(getattr(self, name).shape) != mask_shape:
                raise ValueError(f"{name} must have shape [B,K,F,1,H,W]")
        if tuple(self.relative_pose.shape) != (batch, candidates, frames, 6):
            raise ValueError("relative_pose must have shape [B,K,F,6]")
        if tuple(self.view_angle.shape) != (batch, candidates, frames, 1):
            raise ValueError("view_angle must have shape [B,K,F,1]")
        if tuple(self.subpixel_offset.shape) != (batch, candidates, frames, 2, height, width):
            raise ValueError("subpixel_offset must have shape [B,K,F,2,H,W]")
        if tuple(self.provenance.shape) != (batch, candidates):
            raise ValueError("provenance must have shape [B,K]")
        if len(self.observation_ids) != candidates:
            raise ValueError("observation_ids must match the candidate dimension")

    def has_any_valid_candidate(self) -> bool:
        return bool(self.valid.any().item())

    @property
    def coverage(self) -> torch.Tensor:
        return self.valid.float().mean(dim=(2, 3, 4, 5))


@dataclass(frozen=True)
class EncodedWorldTokens:
    tokens: torch.Tensor  # [B,L,Klocal,W]
    valid: torch.Tensor  # [B,L,Klocal]
    attention_bias: torch.Tensor  # [B,L,Klocal]
    is_null: torch.Tensor  # [Klocal], bool


@dataclass(frozen=True)
class WorldLayerContext:
    key: torch.Tensor  # [B,L,K,H,D]
    value: torch.Tensor  # [B,L,K,H,D]
    valid: torch.Tensor  # [B,L,K]
    attention_bias: torch.Tensor  # [B,L,K]
    enable_lora: bool = False


@dataclass(frozen=True)
class WorldBlockContext:
    layers: Dict[int, WorldLayerContext]
    coverage: torch.Tensor
    observation_ids: Tuple[str, ...]
