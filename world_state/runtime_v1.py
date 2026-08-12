"""Attachment, block precompute, and sidecar IO for WorldStateReader v1."""

import json
import weakref
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from world_memory.latent_adapter import LatentMemoryAdapter, load_latent_memory_adapter

from .domains import ThreeDomainMasks
from .encoder_v1 import IdentityPreservingWorldEncoder
from .reader_v1 import IdentityPreservingWorldReader, PatchGatedLoRA
from .types import WorldBlockContextV1, WorldReadPacket


class WorldStateRuntimeV1(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        content_adapter: LatentMemoryAdapter,
        *,
        selected_layers: Sequence[int] = (8, 14, 20),
        selector_width: int = 256,
        confidence_threshold: float = 0.35,
        lora_rank: int = 0,
    ):
        super().__init__()
        self.selected_layers = tuple(int(index) for index in selected_layers)
        self.selector_width = int(selector_width)
        self.confidence_threshold = float(confidence_threshold)
        self.lora_rank = int(lora_rank)
        self.lora_enabled = False
        self._model_ref = weakref.ref(model)
        self.encoder = IdentityPreservingWorldEncoder(
            content_adapter,
            selector_width=selector_width,
            confidence_threshold=confidence_threshold,
        )
        for index in self.selected_layers:
            if index < 0 or index >= len(model.blocks):
                raise ValueError(f"selected layer {index} is outside the DiT")
            block = model.blocks[index]
            block.add_module(
                "world_reader",
                IdentityPreservingWorldReader(
                    model.dim, selector_width=selector_width
                ),
            )
            if lora_rank > 0:
                block.self_attn.add_module(
                    "world_q_lora", PatchGatedLoRA(model.dim, lora_rank)
                )
                block.self_attn.add_module(
                    "world_o_lora", PatchGatedLoRA(model.dim, lora_rank)
                )

    def precompute(
        self,
        packet: WorldReadPacket,
        domains: ThreeDomainMasks,
    ) -> Optional[WorldBlockContextV1]:
        if not domains.memory.any():
            return None
        model = self._model_ref()
        if model is None:
            raise RuntimeError("the attached DiT model no longer exists")
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
            if packet.candidate_20ch.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            encoded = self.encoder(packet, domains)
            layers = {
                index: model.blocks[index].world_reader.precompute(
                    encoded,
                    enable_lora=self.lora_enabled and self.lora_rank > 0,
                )
                for index in self.selected_layers
            }
        return WorldBlockContextV1(
            layers=layers,
            coverage=domains.memory.float().mean(dim=(1, 2, 3, 4)).detach(),
            observation_ids=packet.observation_ids,
        )

    def set_lora_enabled(self, enabled: bool) -> None:
        if enabled and self.lora_rank <= 0:
            raise ValueError("cannot enable World LoRA when lora_rank is zero")
        self.lora_enabled = bool(enabled)


def attach_world_state_reader_v1(
    model: nn.Module,
    direct_adapter_path,
    *,
    selected_layers: Sequence[int] = (8, 14, 20),
    selector_width: int = 256,
    confidence_threshold: float = 0.35,
    lora_rank: int = 0,
) -> WorldStateRuntimeV1:
    if getattr(model, "world_state_runtime", None) is not None:
        raise ValueError("model already has a WorldStateRuntime")
    if getattr(model, "memory_adapter", None) is not None:
        raise ValueError("formal Reader must not attach the direct residual adapter")
    adapter = load_latent_memory_adapter(direct_adapter_path, dtype=torch.float32)
    adapter.eval().requires_grad_(False)
    runtime = WorldStateRuntimeV1(
        model,
        adapter,
        selected_layers=selected_layers,
        selector_width=selector_width,
        confidence_threshold=confidence_threshold,
        lora_rank=lora_rank,
    )
    model.add_module("world_state_runtime", runtime)
    target_device = model.patch_embedding.weight.device
    runtime.to(device=target_device)
    for index in runtime.selected_layers:
        block = model.blocks[index]
        block.world_reader.to(device=target_device)
        if hasattr(block.self_attn, "world_q_lora"):
            block.self_attn.world_q_lora.to(device=target_device)
            block.self_attn.world_o_lora.to(device=target_device)
    return runtime


def world_state_v1_trainable_parameters(
    model: nn.Module, *, include_lora: bool = False
):
    parameters = []
    for name, parameter in model.named_parameters():
        trainable = (
            "world_state_runtime.encoder" in name
            and ".content_adapter." not in name
        ) or ".world_reader." in name or (
            include_lora
            and (".world_q_lora." in name or ".world_o_lora." in name)
        )
        parameter.requires_grad_(trainable)
        if trainable:
            parameters.append(parameter)
    return parameters


def _reader_v1_state(model: nn.Module):
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if (
            "world_state_runtime.encoder" in name
            and ".content_adapter." not in name
        )
        or ".world_reader." in name
        or ".world_q_lora." in name
        or ".world_o_lora." in name
    }


def save_world_state_reader_v1(
    model: nn.Module, path, *, metadata: Optional[dict] = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = model.world_state_runtime
    values = {
        "format": "inspatio_worldstate_reader_v1",
        "selected_layers": json.dumps(runtime.selected_layers),
        "selector_width": str(runtime.selector_width),
        "confidence_threshold": str(runtime.confidence_threshold),
        "lora_rank": str(runtime.lora_rank),
    }
    values.update({key: str(value) for key, value in (metadata or {}).items()})
    save_file(_reader_v1_state(model), str(path), metadata=values)


def load_world_state_reader_v1(model: nn.Module, path) -> None:
    state = load_file(str(path), device="cpu")
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    unexpected = sorted(set(state) - set(targets))
    if unexpected:
        raise ValueError(f"unexpected Reader v1 sidecar keys: {unexpected[:3]}")
    missing = [key for key in _reader_v1_state(model) if key not in state]
    if missing:
        raise ValueError(f"missing Reader v1 sidecar keys: {missing[:3]}")
    with torch.no_grad():
        for key, value in state.items():
            targets[key].copy_(
                value.to(device=targets[key].device, dtype=targets[key].dtype)
            )
