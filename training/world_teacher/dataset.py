"""Paired immutable-bank records and block-local training examples."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from scripts.world_memory.common import pad_clean_latent
from world_state import CameraBatch, FixedWorldBank, Provenance, WorldObservation
from world_state.source_truth import conservative_static_confidence


def load_scene_record(path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def query_camera(record: dict, block_index: int, device) -> CameraBatch:
    start = block_index * int(record["block_size"])
    end = start + int(record["block_size"])
    return CameraBatch(
        K=record["camera_K"][:, start:end].to(device=device, dtype=torch.float32),
        c2w_W0=record["target_c2w_W0"][:, start:end].to(
            device=device, dtype=torch.float32
        ),
    )


def _observation_from_dict(values: dict, device, dtype) -> WorldObservation:
    return WorldObservation(
        scene_id=values["scene_id"],
        world_id=values["world_id"],
        observation_id=values["observation_id"],
        provenance=int(values["provenance"]),
        clean_latent=values["clean_latent"].to(device=device, dtype=dtype),
        K=values["K"].to(device=device, dtype=torch.float32),
        c2w_W0=values["c2w_W0"].to(device=device, dtype=torch.float32),
        depth=(
            None
            if values.get("depth") is None
            else values["depth"].to(device=device, dtype=torch.float32)
        ),
        valid=values["valid"].to(device=device, dtype=torch.bool),
        static_confidence=values["static_confidence"].to(
            device=device, dtype=dtype
        ),
        geometry_confidence=values["geometry_confidence"].to(
            device=device, dtype=dtype
        ),
    )


def make_bank(record: dict, identity: str, device, dtype) -> FixedWorldBank:
    source_values = dict(record["source_observation"])
    source_values["world_id"] = identity
    source = _observation_from_dict(source_values, device, dtype)

    write_block = int(record["write_block"])
    start = write_block * int(record["block_size"])
    end = start + int(record["block_size"])
    memory = record[f"output_{identity}"][:, start:end][0]
    valid = torch.ones(
        memory.shape[0], 1, *memory.shape[-2:], dtype=torch.bool
    )
    confidence = torch.ones_like(valid, dtype=torch.float32)
    static_confidence = conservative_static_confidence(memory.float())
    generated = WorldObservation(
        scene_id=record["scene_id"],
        world_id=identity,
        observation_id=f"{record['scene_id']}:M40_{identity}",
        provenance=int(Provenance.GENERATED),
        clean_latent=memory.to(device=device, dtype=dtype),
        K=record["camera_K"][0, start:end].to(device=device, dtype=torch.float32),
        c2w_W0=record["target_c2w_W0"][0, start:end].to(
            device=device, dtype=torch.float32
        ),
        depth=None,
        valid=valid.to(device),
        static_confidence=static_confidence.to(device=device, dtype=dtype),
        geometry_confidence=confidence.to(device=device, dtype=dtype),
    )
    return FixedWorldBank((source, generated))


@dataclass(frozen=True)
class BlockExample:
    noisy_input: torch.Tensor
    context_frames: torch.Tensor
    render_block: torch.Tensor
    target: torch.Tensor
    no_memory_target: torch.Tensor
    camera: CameraBatch
    denoising_steps: torch.Tensor
    transition_seed: int


def make_block_example(
    record: dict,
    *,
    identity: str,
    query_block: int,
    target_block: int,
    device,
    dtype=torch.bfloat16,
) -> BlockExample:
    block_size = int(record["block_size"])
    start = query_block * block_size
    end = start + block_size
    target_start = target_block * block_size
    target_end = target_start + block_size
    previous_start = start - block_size
    if previous_start < 0:
        raise ValueError("block-local Teacher examples require a previous STAR block")

    ref = record["ref_latent"][:, start:end].to(device=device, dtype=dtype)
    previous = record["output_A"][:, previous_start:start].to(
        device=device, dtype=dtype
    )
    context = torch.cat((pad_clean_latent(ref), pad_clean_latent(previous)), dim=1)
    render = torch.cat(
        (
            record["mask_latent"][:, start:end],
            record["render_latent"][:, start:end],
        ),
        dim=2,
    ).to(device=device, dtype=dtype)
    return BlockExample(
        noisy_input=record["noise_A"][:, start:end].to(device=device, dtype=dtype),
        context_frames=context,
        render_block=render,
        target=record[f"output_{identity}"][:, target_start:target_end].to(
            device=device, dtype=dtype
        ),
        no_memory_target=record["output_A"][:, start:end].to(
            device=device, dtype=dtype
        ),
        camera=query_camera(record, query_block, device),
        denoising_steps=record["denoising_steps"].to(device=device),
        transition_seed=int(record["transition_seed"]) + query_block,
    )


def ownership_masks(
    packet,
    *,
    generated_static_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.zeros_like(packet.valid[:, 0])
    generated = torch.zeros_like(source)
    for index in range(packet.valid.shape[1]):
        provenance = int(packet.provenance[0, index])
        if provenance == int(Provenance.SOURCE):
            source |= packet.valid[:, index]
        elif provenance == int(Provenance.GENERATED):
            generated |= packet.valid[:, index] & (
                packet.confidence[:, index] >= generated_static_threshold
            )
    generated &= ~source
    unknown = ~(source | generated)
    return source, generated, unknown
