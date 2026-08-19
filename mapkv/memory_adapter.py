from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn


ADAPTER_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class MemoryAdapterConfig:
    """Configuration for the parallel, zero-init MapKV patch residual."""

    latent_channels: int = 16
    hidden_channels: int = 32
    model_dim: int = 1536
    patch_size: tuple[int, int, int] = (1, 2, 2)
    inject_middle: bool = False
    middle_start: int | None = None
    middle_stop: int | None = None

    @property
    def input_channels(self) -> int:
        return 2 * int(self.latent_channels) + 1

    def validate(self) -> None:
        if self.latent_channels <= 0 or self.hidden_channels <= 0 or self.model_dim <= 0:
            raise ValueError("Adapter channel dimensions must be positive")
        if len(self.patch_size) != 3 or any(int(value) <= 0 for value in self.patch_size):
            raise ValueError("patch_size must contain three positive integers")
        if self.inject_middle:
            if self.middle_start is None or self.middle_stop is None:
                raise ValueError("Middle injection requires an explicit half-open block range")
            if not 0 <= int(self.middle_start) < int(self.middle_stop):
                raise ValueError("Invalid middle adapter block range")


class MemoryPatchAdapter(nn.Module):
    """Tiny Conv3D memory encoder injected in parallel with native patch embed.

    Inputs stay in VAE-latent layout. The final 1x1 projection is zero initialized,
    so installing the adapter is exactly baseline-equivalent before training.
    """

    def __init__(self, config: MemoryAdapterConfig):
        super().__init__()
        config.validate()
        self.config = config
        hidden = int(config.hidden_channels)
        self.encoder = nn.Sequential(
            nn.Conv3d(config.input_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.patch_projection = nn.Conv3d(hidden, config.model_dim, kernel_size=1)
        nn.init.zeros_(self.patch_projection.weight)
        nn.init.zeros_(self.patch_projection.bias)
        self.middle_projection = (
            nn.Conv3d(hidden, config.model_dim, kernel_size=1)
            if config.inject_middle
            else None
        )
        if self.middle_projection is not None:
            nn.init.zeros_(self.middle_projection.weight)
            nn.init.zeros_(self.middle_projection.bias)

    def _validate_inputs(
        self,
        memory_latent: torch.Tensor,
        recent_latent: torch.Tensor,
        need_mask: torch.Tensor,
    ) -> torch.Tensor:
        if memory_latent.ndim != 5 or recent_latent.ndim != 5:
            raise ValueError("Adapter latents must be BCFHW")
        if memory_latent.shape != recent_latent.shape:
            raise ValueError(
                f"Memory/recent shape mismatch: {tuple(memory_latent.shape)} vs "
                f"{tuple(recent_latent.shape)}"
            )
        if memory_latent.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"Expected {self.config.latent_channels} latent channels, got "
                f"{memory_latent.shape[1]}"
            )
        mask = need_mask
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)
        if mask.ndim != 5 or mask.shape[1] != 1:
            raise ValueError("M_need must be BFHW or B1FHW")
        if mask.shape[0] != memory_latent.shape[0] or mask.shape[2:] != memory_latent.shape[2:]:
            raise ValueError(
                f"M_need shape {tuple(mask.shape)} does not match latent spatial/temporal "
                f"shape {tuple(memory_latent.shape)}"
            )
        return mask.to(device=memory_latent.device, dtype=memory_latent.dtype).clamp(0, 1)

    def encode(
        self,
        memory_latent: torch.Tensor,
        recent_latent: torch.Tensor,
        need_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self._validate_inputs(memory_latent, recent_latent, need_mask)
        # Mask both latent inputs at the branch entrance: unsupported pixels cannot
        # leak into the learned memory feature through Conv3D receptive fields.
        value = torch.cat([memory_latent * mask, recent_latent * mask, mask], dim=1)
        return self.encoder(value), mask

    def _tokenize(
        self, feature: torch.Tensor, mask: torch.Tensor, projection: nn.Conv3d
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch = tuple(int(value) for value in self.config.patch_size)
        residual = projection(feature)
        residual = F.avg_pool3d(residual, kernel_size=patch, stride=patch)
        token_mask = F.max_pool3d(mask.float(), kernel_size=patch, stride=patch)
        token_mask = token_mask.to(dtype=residual.dtype)
        return residual * token_mask, token_mask

    def forward(
        self,
        memory_latent: torch.Tensor,
        recent_latent: torch.Tensor,
        need_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        feature, mask = self.encode(memory_latent, recent_latent, need_mask)
        patch_residual, token_mask = self._tokenize(
            feature, mask, self.patch_projection
        )
        middle_residual = None
        if self.middle_projection is not None:
            middle_residual, _ = self._tokenize(
                feature, mask, self.middle_projection
            )
        return patch_residual, token_mask, middle_residual

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def zero_init_is_exact(self) -> bool:
        projections = [self.patch_projection]
        if self.middle_projection is not None:
            projections.append(self.middle_projection)
        return all(
            bool(torch.count_nonzero(module.weight).item() == 0)
            and (
                module.bias is None
                or bool(torch.count_nonzero(module.bias).item() == 0)
            )
            for module in projections
        )


@dataclass
class MemoryAdapterContext:
    """Per-block target-aligned adapter input; tensors use BFCHW externally."""

    target_block: int
    source_chunk: int
    memory_latent: torch.Tensor
    recent_latent: torch.Tensor
    need_mask: torch.Tensor
    audit: dict = field(default_factory=dict)
    _middle_tokens: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.memory_latent.ndim != 5 or self.recent_latent.ndim != 5:
            raise ValueError("Memory adapter latents must be BFCHW")
        if self.memory_latent.shape != self.recent_latent.shape:
            raise ValueError("Memory adapter memory/recent shapes must match")
        if int(self.source_chunk) >= int(self.target_block) - 1:
            raise ValueError("Adapter history must be older than runtime Recent")

    @staticmethod
    def _bcfhw(
        value: torch.Tensor, reference: torch.Tensor, latent_channels: int
    ) -> torch.Tensor:
        expected_bfchw = (
            reference.shape[0], reference.shape[2], int(latent_channels),
            reference.shape[3], reference.shape[4],
        )
        if tuple(value.shape) != expected_bfchw:
            raise ValueError(
                f"Adapter BFCHW value {tuple(value.shape)} != expected {expected_bfchw}"
            )
        return value.permute(0, 2, 1, 3, 4).contiguous().to(
            device=reference.device, dtype=reference.dtype
        )

    def patch_residual(
        self,
        adapter: MemoryPatchAdapter,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        memory = self._bcfhw(
            self.memory_latent, model_input, adapter.config.latent_channels
        )
        recent = self._bcfhw(
            self.recent_latent, model_input, adapter.config.latent_channels
        )
        mask = self.need_mask
        if mask.ndim == 4:
            mask = mask.unsqueeze(2)
        if mask.ndim != 5 or mask.shape[1] != memory.shape[2] or mask.shape[2] != 1:
            raise ValueError("Adapter mask must use BF1HW layout")
        mask = mask.permute(0, 2, 1, 3, 4).contiguous().to(
            device=model_input.device, dtype=model_input.dtype
        )
        patch, token_mask, middle = adapter(memory, recent, mask)
        self._middle_tokens = (
            None
            if middle is None
            else middle.flatten(2).transpose(1, 2).contiguous()
        )
        self.audit.update(
            {
                "target_block": int(self.target_block),
                "source_chunk": int(self.source_chunk),
                "need_fraction": float(mask.float().mean().item()),
                "active_token_fraction": float(token_mask.float().mean().item()),
                "patch_residual_abs_mean": float(patch.detach().float().abs().mean().item()),
                "patch_residual_max_abs": float(patch.detach().float().abs().max().item()),
            }
        )
        return patch

    def middle_residual(self, block_index: int) -> torch.Tensor | None:
        if self._middle_tokens is None:
            return None
        start = int(self.audit["middle_start"])
        stop = int(self.audit["middle_stop"])
        return self._middle_tokens if start <= int(block_index) < stop else None


@dataclass
class MemoryAdapterPlan:
    target_block: int
    source_chunk: int
    memory_latent: torch.Tensor
    need_mask: torch.Tensor
    metadata: dict = field(default_factory=dict)

    def bind_recent(self, recent_latent: torch.Tensor) -> MemoryAdapterContext:
        return MemoryAdapterContext(
            target_block=self.target_block,
            source_chunk=self.source_chunk,
            memory_latent=self.memory_latent,
            recent_latent=recent_latent,
            need_mask=self.need_mask,
            audit=dict(self.metadata),
        )


def plans_from_warp_reencode(plans: Mapping[int, object]) -> dict[int, MemoryAdapterPlan]:
    result: dict[int, MemoryAdapterPlan] = {}
    for target, source in plans.items():
        need = getattr(source, "need_coverage", None)
        if need is None:
            need = source.coverage
        result[int(target)] = MemoryAdapterPlan(
            target_block=int(source.target_block),
            source_chunk=int(source.source_chunk),
            memory_latent=source.historical_latent,
            need_mask=torch.as_tensor(need),
            metadata={
                "memory_representation": "rgb_warp_vae",
                "lifecycle": "episode_continuous",
                "source_protected": True,
            },
        )
    return result


def save_adapter_checkpoint(
    path: str | Path,
    adapter: MemoryPatchAdapter,
    *,
    training: Mapping | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": ADAPTER_CHECKPOINT_VERSION,
            "config": asdict(adapter.config),
            "state_dict": {
                key: value.detach().cpu() for key, value in adapter.state_dict().items()
            },
            "training": dict(training or {}),
        },
        destination,
    )
    return destination


def load_adapter_checkpoint(
    path: str | Path,
    *,
    model_dim: int | None = None,
    patch_size: tuple[int, int, int] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[MemoryPatchAdapter, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if int(payload.get("version", -1)) != ADAPTER_CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported adapter checkpoint version: {payload.get('version')}")
    raw = dict(payload["config"])
    raw["patch_size"] = tuple(raw["patch_size"])
    config = MemoryAdapterConfig(**raw)
    if model_dim is not None and int(config.model_dim) != int(model_dim):
        raise ValueError("Adapter/model hidden dimension mismatch")
    if patch_size is not None and tuple(config.patch_size) != tuple(patch_size):
        raise ValueError("Adapter/model patch size mismatch")
    adapter = MemoryPatchAdapter(config)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter, dict(payload.get("training", {}))


def freeze_backbone_for_adapter(model: nn.Module, adapter: MemoryPatchAdapter) -> dict:
    model.requires_grad_(False)
    adapter.requires_grad_(True)
    adapter_names = {id(parameter) for parameter in adapter.parameters()}
    leaked = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in adapter_names
    ]
    if leaked:
        raise RuntimeError(f"Frozen-backbone audit failed: {leaked[:8]}")
    return {
        "backbone_trainable_parameters": 0,
        "adapter_trainable_parameters": int(
            sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        ),
        "adapter_total_parameters": int(sum(p.numel() for p in adapter.parameters())),
    }


__all__ = [
    "ADAPTER_CHECKPOINT_VERSION",
    "MemoryAdapterConfig",
    "MemoryAdapterContext",
    "MemoryAdapterPlan",
    "MemoryPatchAdapter",
    "freeze_backbone_for_adapter",
    "load_adapter_checkpoint",
    "plans_from_warp_reencode",
    "save_adapter_checkpoint",
]
