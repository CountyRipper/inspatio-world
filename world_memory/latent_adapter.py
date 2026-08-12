"""A minimal sidecar adapter for exact-pose clean-latent memory."""

from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn


PathLike = Union[str, Path]


class LatentMemoryAdapter(nn.Module):
    """Project a 20-channel memory condition into DiT patch space."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=20,
            out_channels=model_dim,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            bias=False,
        )

    @property
    def model_dim(self) -> int:
        return self.proj.out_channels

    def forward(self, memory_condition: torch.Tensor) -> torch.Tensor:
        if memory_condition.ndim != 5 or memory_condition.shape[1] != 20:
            raise ValueError(
                "memory_condition must have internal shape [B,20,F,H,W], "
                f"got {tuple(memory_condition.shape)}"
            )
        return self.proj(memory_condition)


def patchify_binary_occupancy(occupancy: torch.Tensor) -> torch.Tensor:
    """Patchify internal [B,1,F,H,W] binary occupancy with a hard max gate."""
    if occupancy.ndim != 5 or occupancy.shape[1] != 1:
        raise ValueError(
            "memory_occupancy must have internal shape [B,1,F,H,W], "
            f"got {tuple(occupancy.shape)}"
        )
    patch_gate = F.max_pool3d(
        occupancy.float(),
        kernel_size=(1, 2, 2),
        stride=(1, 2, 2),
    )
    return patch_gate > 0


def gated_memory_residual(
    adapter: LatentMemoryAdapter,
    memory_condition: torch.Tensor,
    occupancy: torch.Tensor,
) -> torch.Tensor:
    """Compute the adapter residual, then apply the required hard patch gate."""
    same_batch = memory_condition.shape[0] == occupancy.shape[0]
    same_spatiotemporal_shape = memory_condition.shape[2:] == occupancy.shape[2:]
    if not same_batch or not same_spatiotemporal_shape:
        raise ValueError("memory_condition and memory_occupancy must share B,F,H,W")

    residual = adapter(memory_condition)
    patch_gate = patchify_binary_occupancy(occupancy).to(residual.dtype)
    same_batch = patch_gate.shape[0] == residual.shape[0]
    same_spatiotemporal_shape = patch_gate.shape[2:] == residual.shape[2:]
    if not same_batch or not same_spatiotemporal_shape:
        raise ValueError("patchified memory occupancy does not match adapter output")
    return residual * patch_gate


def add_gated_memory_residual(
    base_embeddings: List[torch.Tensor],
    adapter: Optional[LatentMemoryAdapter],
    memory_condition: Optional[torch.Tensor] = None,
    occupancy: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Add memory after patch embedding, with an exact structural off path."""
    if memory_condition is None:
        if occupancy is not None:
            raise ValueError("memory_condition is required when memory_occupancy is provided")
        return base_embeddings
    if occupancy is None:
        raise ValueError("memory_occupancy is required when memory_condition is provided")
    if adapter is None:
        raise RuntimeError("attach a LatentMemoryAdapter before enabling memory")

    residual = gated_memory_residual(adapter, memory_condition, occupancy)
    if residual.shape[0] != len(base_embeddings):
        raise ValueError("memory batch size does not match the DiT input")
    return [
        base + residual[index:index + 1]
        for index, base in enumerate(base_embeddings)
    ]


def attach_latent_memory_adapter(
    model: nn.Module,
    adapter: Optional[LatentMemoryAdapter] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> LatentMemoryAdapter:
    """Attach an independent adapter after the base checkpoint has been loaded."""
    if getattr(model, "memory_adapter", None) is not None:
        raise ValueError("model already has a latent memory adapter")

    model_dim = int(getattr(model, "dim"))
    adapter = LatentMemoryAdapter(model_dim) if adapter is None else adapter
    if adapter.model_dim != model_dim:
        raise ValueError(
            f"adapter model_dim {adapter.model_dim} does not match model dim {model_dim}"
        )

    patch_weight = getattr(model, "patch_embedding").weight
    target_device = patch_weight.device if device is None else torch.device(device)
    target_dtype = patch_weight.dtype if dtype is None else dtype
    adapter = adapter.to(device=target_device, dtype=target_dtype)
    model.add_module("memory_adapter", adapter)
    return adapter


def save_latent_memory_adapter(adapter: LatentMemoryAdapter, path: PathLike) -> None:
    """Save only sidecar weights, never the frozen base model."""
    from safetensors.torch import save_file

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in adapter.state_dict().items()
    }
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "format": "inspatio_exact_pose_latent_memory_adapter_v1",
            "model_dim": str(adapter.model_dim),
        },
    )


def load_latent_memory_adapter(
    path: PathLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> LatentMemoryAdapter:
    """Load an adapter sidecar and infer its model dimension from the weight."""
    from safetensors.torch import load_file

    state_dict = load_file(str(path), device="cpu")
    weight = state_dict.get("proj.weight")
    if weight is None or weight.ndim != 5 or tuple(weight.shape[1:]) != (20, 1, 2, 2):
        raise ValueError("invalid latent memory adapter sidecar")

    adapter = LatentMemoryAdapter(model_dim=weight.shape[0])
    adapter.load_state_dict(state_dict, strict=True)
    target_device = torch.device("cpu") if device is None else torch.device(device)
    if dtype is None:
        return adapter.to(device=target_device)
    return adapter.to(device=target_device, dtype=dtype)
