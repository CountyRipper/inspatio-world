from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from phase1_lsm.runtime import allocate_kv_cache
from phase2_memory.anchoring import masked_latent_anchor
from phase2_memory.bank import LatentMemoryBank, MemoryProjection, MemoryRecord
from phase2_memory.data import PreparedTrajectory
from phase2_memory.trajectory import TrajectoryStation, block_keyframes
from pipeline.causal_inference import denoise_block
from scripts.phase1_lsm.run_sharedA_hardgate_5deg import padded_context


ROLLOUT_VARIANTS = ("no_memory", "correct", "wrong", "mask_only")


@dataclass
class ReturnRecord:
    block: int
    memory_id: str
    memory_version: int
    retrieved_ids: list[str]
    retrieved_scores: list[float]
    previous_latent: torch.Tensor
    ref_latent: torch.Tensor
    render_condition: torch.Tensor
    step_inputs: torch.Tensor
    transition_noises: torch.Tensor
    projected_memory: torch.Tensor
    memory_mask4: torch.Tensor
    occupancy: torch.Tensor
    anchor_noise: torch.Tensor
    prediction: torch.Tensor
    no_memory_full: torch.Tensor | None = None
    no_memory_truncated: torch.Tensor | None = None


@dataclass
class ReturnObservation:
    block: int
    memory_id: str
    retrieved_ids: list[str]
    retrieved_versions: list[int]
    retrieved_scores: list[float]
    selected_id: str | None
    selected_version: int | None
    projection_non_identity: bool
    occupancy_fraction: float
    output: torch.Tensor
    correct_target: torch.Tensor
    correct_occupancy: torch.Tensor


@dataclass
class RolloutResult:
    output: torch.Tensor
    observations: list[ReturnObservation]
    records: list[ReturnRecord]
    bank_events: list[dict[str, object]]


def clear_kv_cache(kv_cache: list[dict[str, torch.Tensor]]) -> None:
    for item in kv_cache:
        item["k"].detach_().zero_()
        item["v"].detach_().zero_()


def source_occupancy(
    depth: torch.Tensor,
    latent_hw: tuple[int, int],
) -> torch.Tensor:
    valid = (depth > 0).float()[:, None]
    resized = F.interpolate(valid, size=latent_hw, mode="nearest").bool()
    return resized[None].contiguous()


def make_record(
    station: TrajectoryStation,
    prediction: torch.Tensor,
    prepared: PreparedTrajectory,
) -> MemoryRecord:
    keys = torch.as_tensor(
        block_keyframes(station.block),
        device=prepared.target_depth.device,
        dtype=torch.long,
    )
    depth = prepared.target_depth[keys]
    occupancy = source_occupancy(depth, prediction.shape[-2:])
    return MemoryRecord(
        memory_id=station.memory_id,
        clean_latent=prediction.detach(),
        c2w=prepared.target_c2w[keys].detach(),
        intrinsics=prepared.intrinsics.detach(),
        depth=depth.detach(),
        occupancy=occupancy,
        confidence=occupancy.float(),
        fov_degrees=prepared.fov_degrees,
    )


def fixed_noise(
    seed: int,
    device: torch.device,
    shape: tuple[int, ...] = (1, 60, 16, 60, 104),
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    initial = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    transitions = torch.randn(
        (20, 3, 1, 3, 16, 60, 104),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    return initial, transitions


def _anchor_transform(
    scheduler,
    projection: MemoryProjection | None,
    anchor_noise: torch.Tensor,
    strength: float,
) -> Callable[[int, torch.Tensor | float, torch.Tensor], torch.Tensor] | None:
    if projection is None or strength == 0.0:
        return None

    def transform(
        _step_index: int,
        timestep: torch.Tensor | float,
        noisy_input: torch.Tensor,
    ) -> torch.Tensor:
        return masked_latent_anchor(
            scheduler,
            noisy_input,
            projection.latent,
            projection.occupancy,
            anchor_noise,
            timestep,
            strength,
        )

    return transform


@torch.inference_mode()
def run_online_rollout(
    pipeline,
    prepared: PreparedTrajectory,
    *,
    variant: str,
    anchoring_strength: float,
    capture_records: bool = False,
) -> RolloutResult:
    if variant not in ROLLOUT_VARIANTS:
        raise ValueError(f"unknown rollout variant: {variant}")
    device = prepared.ref_latent.device
    noise, transition_noises = fixed_noise(prepared.group.seed, device)
    output = torch.zeros_like(noise)
    bank = LatentMemoryBank()
    kv_cache = allocate_kv_cache(pipeline.generator, device)
    station_by_block = {station.block: station for station in prepared.group.stations}
    observations = []
    records = []
    last_pred = None

    for block in range(20):
        start = block * 3
        noisy_input = noise[:, start:start + 3]
        ref_block = prepared.ref_latent[:, start:start + 3]
        render_block = torch.cat((
            prepared.mask_latent[:, start:start + 3],
            prepared.render_latent[:, start:start + 3],
        ), dim=2)
        context, kv_size = padded_context(ref_block, last_pred)
        station = station_by_block.get(block)
        correct_projection = None
        selected_projection = None
        matches = []
        if station is not None and station.action != "write":
            keys = torch.as_tensor(
                block_keyframes(block), device=device, dtype=torch.long
            )
            query_c2w = prepared.target_c2w[keys]
            matches = bank.retrieve(
                query_c2w,
                prepared.fov_degrees,
                top_k=max(2, len(bank)),
            )
            if not matches or matches[0].record.memory_id != station.memory_id:
                got = None if not matches else matches[0].record.memory_id
                raise AssertionError(
                    f"{prepared.group.group_id}/block{block}: "
                    f"retrieved {got}, expected {station.memory_id}"
                )
            correct_projection = bank.project(matches[0], query_c2w)
            if torch.equal(
                correct_projection.latent,
                matches[0].record.clean_latent,
            ):
                raise AssertionError(
                    f"{prepared.group.group_id}/block{block}: projection is identity"
                )
            if variant == "correct":
                selected_projection = correct_projection
            elif variant == "wrong":
                wrong = next(
                    (match for match in matches if match.record.memory_id != station.memory_id),
                    None,
                )
                if wrong is None:
                    raise AssertionError("wrong-memory control has no alternative record")
                selected_projection = bank.project(wrong, query_c2w)
            elif variant == "mask_only":
                selected_projection = MemoryProjection(
                    match=correct_projection.match,
                    latent=torch.zeros_like(correct_projection.latent),
                    mask4=correct_projection.mask4,
                    occupancy=correct_projection.occupancy,
                    confidence=correct_projection.confidence,
                )

        condition = (
            None if selected_projection is None else selected_projection.condition
        )
        gate = (
            None if selected_projection is None else selected_projection.occupancy.float()
        )
        captured_inputs: dict[int, torch.Tensor] = {}
        captured_transitions: dict[int, torch.Tensor] = {}

        def callback(
            *,
            step_index: int,
            timestep: float,
            noisy_input: torch.Tensor,
            transition_noise: torch.Tensor | None,
        ) -> None:
            del timestep
            captured_inputs[step_index] = noisy_input.detach().clone()
            if transition_noise is not None:
                captured_transitions[step_index] = transition_noise.detach().clone()

        anchor_noise = noise[:, start:start + 3]
        prediction, _ = denoise_block(
            pipeline.generator,
            pipeline.scheduler,
            noisy_input,
            prepared.conditional,
            kv_cache,
            context_frames=context,
            context_no_grad=True,
            render_block=render_block,
            denoising_kv_size=kv_size,
            denoising_steps=pipeline.denoising_step_list,
            memory_condition=condition,
            memory_gate=gate,
            transition_noises=transition_noises[block],
            step_callback=callback if capture_records and correct_projection is not None else None,
            step_input_transform=_anchor_transform(
                pipeline.scheduler,
                selected_projection,
                anchor_noise,
                anchoring_strength,
            ),
        )
        output[:, start:start + 3] = prediction
        previous = last_pred
        last_pred = prediction.detach().clone()

        if correct_projection is not None:
            selected = None if selected_projection is None else selected_projection.match.record
            observations.append(ReturnObservation(
                block=block,
                memory_id=station.memory_id,
                retrieved_ids=[match.record.memory_id for match in matches],
                retrieved_versions=[match.record.version for match in matches],
                retrieved_scores=[match.score for match in matches],
                selected_id=None if selected is None else selected.memory_id,
                selected_version=None if selected is None else selected.version,
                projection_non_identity=True,
                occupancy_fraction=float(correct_projection.occupancy.float().mean()),
                output=last_pred,
                correct_target=correct_projection.latent.detach(),
                correct_occupancy=correct_projection.occupancy.detach(),
            ))
            if capture_records:
                if variant != "correct":
                    raise ValueError("training records must come from the correct rollout")
                if set(captured_inputs) != {0, 1, 2, 3}:
                    raise AssertionError("incomplete four-step inputs")
                if set(captured_transitions) != {0, 1, 2}:
                    raise AssertionError("incomplete transition noise")
                if previous is None:
                    raise AssertionError("return block has no historical predecessor")
                recorded_step_inputs = torch.stack([
                    captured_inputs[index] for index in range(4)
                ])
                recorded_step_inputs[0] = noisy_input
                records.append(ReturnRecord(
                    block=block,
                    memory_id=station.memory_id,
                    memory_version=correct_projection.match.record.version,
                    retrieved_ids=[match.record.memory_id for match in matches],
                    retrieved_scores=[match.score for match in matches],
                    previous_latent=previous.detach(),
                    ref_latent=ref_block.detach(),
                    render_condition=render_block.detach(),
                    step_inputs=recorded_step_inputs,
                    transition_noises=torch.stack([
                        captured_transitions[index] for index in range(3)
                    ]),
                    projected_memory=correct_projection.latent.detach(),
                    memory_mask4=correct_projection.mask4.detach(),
                    occupancy=correct_projection.occupancy.detach(),
                    anchor_noise=anchor_noise.detach(),
                    prediction=last_pred,
                ))

        if station is not None and station.action in ("write", "return_write"):
            bank.write(
                make_record(station, last_pred, prepared),
                replace_existing=station.action == "return_write",
            )

    return RolloutResult(
        output=output,
        observations=observations,
        records=records,
        bank_events=bank.events,
    )


def replay_return_record(
    generator,
    scheduler,
    denoising_steps: torch.Tensor,
    conditional: dict[str, torch.Tensor],
    record: ReturnRecord,
    *,
    memory_condition: torch.Tensor | None,
    memory_gate: torch.Tensor | None,
    anchoring_strength: float,
    truncate_history: bool,
    backpropagate_all_steps: bool = False,
    kv_cache: list[dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor:
    device = record.step_inputs.device
    kv_cache = allocate_kv_cache(generator, device) if kv_cache is None else kv_cache
    clear_kv_cache(kv_cache)
    context, kv_size = padded_context(
        record.ref_latent,
        None if truncate_history else record.previous_latent,
    )
    projection = None
    if memory_condition is not None:
        projection = MemoryProjection(
            match=None,  # type: ignore[arg-type]
            latent=record.projected_memory,
            mask4=record.memory_mask4,
            occupancy=record.occupancy,
            confidence=record.occupancy.float(),
        )
    prediction, _ = denoise_block(
        generator,
        scheduler,
        record.step_inputs[0],
        conditional,
        kv_cache,
        context_frames=context,
        context_no_grad=True,
        render_block=record.render_condition,
        denoising_kv_size=kv_size,
        denoising_steps=denoising_steps,
        memory_condition=memory_condition,
        memory_gate=memory_gate,
        transition_noises=record.transition_noises,
        backpropagate_all_steps=backpropagate_all_steps,
        step_input_transform=_anchor_transform(
            scheduler,
            projection,
            record.anchor_noise,
            anchoring_strength,
        ),
    )
    return prediction
