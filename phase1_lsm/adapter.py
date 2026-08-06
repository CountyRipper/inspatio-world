from __future__ import annotations

import json
from pathlib import Path

import torch
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
