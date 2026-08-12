"""Mutually exclusive source, memory, and unknown latent domains."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .types import Provenance, WorldReadPacket


@dataclass(frozen=True)
class ThreeDomainMasks:
    source: torch.Tensor  # [B,F,1,H,W], bool
    source_core: torch.Tensor  # [B,F,1,H,W], bool
    memory: torch.Tensor  # [B,F,1,H,W], bool
    unknown: torch.Tensor  # [B,F,1,H,W], bool

    def __post_init__(self) -> None:
        expected = self.source.shape
        for name in ("source_core", "memory", "unknown"):
            value = getattr(self, name)
            if value.shape != expected or value.dtype != torch.bool:
                raise ValueError(f"{name} must be bool with shape {expected}")
        if self.source.dtype != torch.bool:
            raise ValueError("source must be bool")
        if (self.source & self.memory).any():
            raise ValueError("source and memory domains overlap")
        if (self.source & self.unknown).any() or (self.memory & self.unknown).any():
            raise ValueError("three-domain masks must be mutually exclusive")
        if not torch.all(self.source | self.memory | self.unknown):
            raise ValueError("three-domain masks must cover the latent grid")
        if (self.source_core & ~self.source).any():
            raise ValueError("source_core must be contained in source")


def strict_source_mask(mask4: torch.Tensor) -> torch.Tensor:
    """Recover binary source ownership from the four encoded render-mask frames.

    The stored mask is an interpolated conditioning tensor in ``[-1, 1]``.
    Requiring every channel to be strictly positive is deliberately conservative:
    boundary values and mixed four-frame groups remain in the native render collar.
    """
    if mask4.ndim != 5 or mask4.shape[2] != 4:
        raise ValueError("render mask must have external shape [B,F,4,H,W]")
    return (mask4 > 0).all(dim=2, keepdim=True)


def erode_source_mask(source: torch.Tensor, collar: int = 1) -> torch.Tensor:
    """Remove a narrow spatial collar while preserving frame independence."""
    if source.ndim != 5 or source.shape[2] != 1 or source.dtype != torch.bool:
        raise ValueError("source must be bool [B,F,1,H,W]")
    if collar < 0:
        raise ValueError("collar must be non-negative")
    if collar == 0:
        return source
    internal = source.permute(0, 2, 1, 3, 4)
    invalid = F.max_pool3d(
        (~internal).float(),
        kernel_size=(1, 2 * collar + 1, 2 * collar + 1),
        stride=1,
        padding=(0, collar, collar),
    )
    return (invalid == 0).permute(0, 2, 1, 3, 4)


def generated_projection(
    packet: WorldReadPacket,
    *,
    confidence_threshold: float,
) -> tuple[int, torch.Tensor]:
    """Select the single generated center candidate used by Reader v1."""
    generated = (packet.provenance[0] == int(Provenance.GENERATED)).nonzero(
        as_tuple=False
    ).flatten()
    if generated.numel() != 1:
        raise ValueError(
            "Reader v1 expects exactly one retrieved generated observation; "
            f"got {generated.numel()}"
        )
    index = int(generated.item())
    trusted = packet.valid[:, index] & (
        packet.confidence[:, index] >= float(confidence_threshold)
    )
    return index, trusted


def build_three_domains(
    render_mask4: torch.Tensor,
    packet: WorldReadPacket,
    *,
    confidence_threshold: float,
    source_collar: int = 1,
) -> ThreeDomainMasks:
    """Build exhaustive ``S``, ``M=(~S)&valid``, and ``U`` domains."""
    source = strict_source_mask(render_mask4)
    _, projected_memory_valid = generated_projection(
        packet, confidence_threshold=confidence_threshold
    )
    if projected_memory_valid.shape != source.shape:
        raise ValueError("render mask and projected memory must share B,F,H,W")
    memory = (~source) & projected_memory_valid
    unknown = (~source) & (~projected_memory_valid)
    return ThreeDomainMasks(
        source=source,
        source_core=erode_source_mask(source, collar=source_collar),
        memory=memory,
        unknown=unknown,
    )


def patchify_domains(domains: ThreeDomainMasks) -> tuple[torch.Tensor, ...]:
    """Patch ownership with source authority over every touched 2x2 patch."""

    def any_patch(value: torch.Tensor) -> torch.Tensor:
        internal = value.permute(0, 2, 1, 3, 4).float()
        pooled = F.max_pool3d(
            internal, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        return (pooled > 0).permute(0, 2, 1, 3, 4)

    source_patch = any_patch(domains.source)
    memory_patch = any_patch(domains.memory) & ~source_patch
    unknown_patch = ~(source_patch | memory_patch)
    return source_patch, memory_patch, unknown_patch
