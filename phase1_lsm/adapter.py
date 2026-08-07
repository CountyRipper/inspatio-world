from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn


ADAPTER_IN_CHANNELS = 20
ADAPTER_OUT_CHANNELS = 1536
ADAPTER_PATCH_SIZE = (1, 2, 2)
ADAPTER_PARAMETER_COUNT = 122_880


class MemoryPatchAdapter(nn.Module):
    """Zero-initialized patch projection for [mask4 | memory latent16]."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            ADAPTER_IN_CHANNELS,
            ADAPTER_OUT_CHANNELS,
            kernel_size=ADAPTER_PATCH_SIZE,
            stride=ADAPTER_PATCH_SIZE,
            bias=False,
        )
        self.reset_parameters()
        assert self.parameter_count == ADAPTER_PARAMETER_COUNT

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)

    def forward(self, memory_condition: torch.Tensor) -> torch.Tensor:
        if memory_condition.ndim != 5:
            raise ValueError(
                "memory_condition must be [B,20,F,H,W], got "
                f"{tuple(memory_condition.shape)}"
            )
        if memory_condition.shape[1] != ADAPTER_IN_CHANNELS:
            raise ValueError(
                f"memory_condition must have 20 channels, got {memory_condition.shape[1]}"
            )
        return self.proj(memory_condition)


def patch_occupancy_gate(occupancy: torch.Tensor) -> torch.Tensor:
    """Max-pool a strict latent 0/1 occupancy to the adapter patch grid."""
    if occupancy.ndim != 5 or occupancy.shape[1] != 1:
        raise ValueError(
            "occupancy must be [B,1,F,H,W], got "
            f"{tuple(occupancy.shape)}"
        )
    if not torch.all((occupancy == 0) | (occupancy == 1)):
        raise AssertionError("memory occupancy must contain only 0/1 values")
    gate = F.max_pool3d(
        occupancy.float(),
        kernel_size=ADAPTER_PATCH_SIZE,
        stride=ADAPTER_PATCH_SIZE,
    )
    if not torch.all((gate == 0) | (gate == 1)):
        raise AssertionError("patch occupancy gate is not strictly binary")
    return gate


def gated_adapter_residual(
    adapter: MemoryPatchAdapter,
    memory_condition: torch.Tensor,
    occupancy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the parameter-free hard gate after the adapter projection."""
    residual = adapter(memory_condition)
    gate = patch_occupancy_gate(occupancy).to(
        device=residual.device, dtype=residual.dtype
    )
    if gate.shape[0] != residual.shape[0] or gate.shape[2:] != residual.shape[2:]:
        raise ValueError("patch gate shape does not match adapter residual")
    gated = residual * gate
    invalid = (gate == 0).expand_as(gated)
    invalid_values = gated[invalid]
    invalid_max_abs = (
        float(invalid_values.detach().abs().max())
        if invalid_values.numel()
        else 0.0
    )
    invalid_nonzero = int(torch.count_nonzero(invalid_values.detach()))
    if invalid_max_abs != 0.0 or invalid_nonzero != 0:
        raise AssertionError("hard-gated adapter residual is nonzero outside G_patch")
    return gated, gate


def adapter_config() -> dict[str, object]:
    return {
        "architecture": "MemoryPatchAdapter",
        "input_order": ["memory_mask4", "projected_memory_latent16"],
        "in_channels": ADAPTER_IN_CHANNELS,
        "out_channels": ADAPTER_OUT_CHANNELS,
        "kernel_size": list(ADAPTER_PATCH_SIZE),
        "stride": list(ADAPTER_PATCH_SIZE),
        "bias": False,
        "zero_initialized": True,
        "parameter_count": ADAPTER_PARAMETER_COUNT,
        "output_hard_gate": {
            "source": "strict_binary_projected_occupancy1",
            "pool": "max_pool3d",
            "kernel_size": list(ADAPTER_PATCH_SIZE),
            "stride": list(ADAPTER_PATCH_SIZE),
            "learned_parameters": 0,
        },
    }


def save_adapter(adapter: MemoryPatchAdapter, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().contiguous().cpu()
        for name, tensor in adapter.state_dict().items()
    }
    save_file(tensors, output_dir / "memory_adapter.safetensors")
    (output_dir / "memory_adapter_config.json").write_text(
        json.dumps(adapter_config(), indent=2) + "\n",
        encoding="utf-8",
    )


def load_adapter(
    adapter: MemoryPatchAdapter,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> None:
    state = load_file(str(checkpoint_path), device=str(device))
    adapter.load_state_dict(state, strict=True)
