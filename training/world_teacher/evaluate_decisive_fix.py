#!/usr/bin/env python3
"""Exact-only shared-domain ablation for the innovation Reader."""

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.causal_inference import denoise_block
from scripts.world_memory.common import (
    init_single_gpu_distributed,
    load_configs,
    load_frozen_generator,
    pad_clean_latent,
    resolve_repo_path,
)
from training.world_teacher.dataset import (
    load_scene_record,
    make_bank,
    query_camera,
)
from training.world_teacher.visualize import block_frame, write_grid
from utils.wan_wrapper import WanVAEWrapper
from world_memory import attach_latent_memory_adapter, load_latent_memory_adapter
from world_state import (
    Provenance,
    RotationProjector,
    build_exact_shared_domains,
    erode_source_mask,
    strict_source_mask,
)
from world_state.runtime import (
    attach_world_state_reader,
    load_world_state_reader,
)
from world_state.runtime_v1 import (
    attach_world_state_reader_v1,
    load_world_state_reader_v1,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/world_teacher/decisive_fix.yaml"
    )
    parser.add_argument("--scene", choices=("S0", "S1"), required=True)
    parser.add_argument("--reader-checkpoint")
    parser.add_argument("--output-name")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_cache(cache, device, dtype):
    return [
        {
            "k": values["k"].to(device=device, dtype=dtype),
            "v": values["v"].to(device=device, dtype=dtype),
        }
        for values in cache
    ]


def cache_to_cpu(cache):
    return [
        {
            "k": values["k"].detach().cpu().contiguous(),
            "v": values["v"].detach().cpu().contiguous(),
        }
        for values in cache
    ]


def generated_observation(bank):
    values = [
        observation
        for observation in bank.observations
        if int(observation.provenance) == int(Provenance.GENERATED)
    ]
    if len(values) != 1:
        raise ValueError("decisive evaluation expects exactly one generated M40")
    return values[0]


def generated_packet(bank, projector, camera):
    return projector.project((generated_observation(bank),), camera)


def swap_packet_content(base_packet, other_packet):
    """Keep all A geometry/metadata and replace only the projected latent16."""
    condition = base_packet.candidate_20ch.clone()
    condition[:, :, :, 4:] = other_packet.candidate_20ch[:, :, :, 4:]
    return replace(base_packet, candidate_20ch=condition)


def build_shared_exact_inputs(record, banks, projector, config, device, dtype):
    """Create one A-geometry domain; B changes only projected latent content."""
    shared = {}
    for block_index in (
        int(config.evaluation.exact_block),
        int(config.evaluation.hold_block),
    ):
        values = block_inputs(record, block_index, device, dtype)
        camera = query_camera(record, block_index, device)
        packet_A = generated_packet(banks["A"], projector, camera)
        raw_packet_B = generated_packet(banks["B"], projector, camera)
        packet_B = swap_packet_content(packet_A, raw_packet_B)
        domains = build_exact_shared_domains(
            values["mask4"],
            packet_A,
            source_collar=int(config.reader.source_collar),
        )
        for name in (
            "valid",
            "authority",
            "confidence",
            "relative_pose",
            "view_angle",
            "subpixel_offset",
            "provenance",
        ):
            if not torch.equal(getattr(packet_A, name), getattr(packet_B, name)):
                raise AssertionError(f"A/B must share exact packet metadata: {name}")
        shared[block_index] = {
            "camera": camera,
            "domains": domains,
            "packets": {"A": packet_A, "B": packet_B},
        }
    return shared


def block_inputs(record, block_index, device, dtype):
    block_size = int(record["block_size"])
    start = block_index * block_size
    end = start + block_size
    mask4 = record["mask_latent"][:, start:end].to(
        device=device, dtype=dtype
    )
    source_clean = record["render_latent"][:, start:end].to(
        device=device, dtype=dtype
    )
    return {
        "noisy": record["noise_A"][:, start:end].to(device=device, dtype=dtype),
        "ref": record["ref_latent"][:, start:end].to(device=device, dtype=dtype),
        "mask4": mask4,
        "source_clean": source_clean,
        "fixed_source_noise": record["noise_A"][:, start:end].to(
            device=device, dtype=dtype
        ),
        "render": torch.cat((mask4, source_clean), dim=2),
    }


def source_core(values, collar):
    return erode_source_mask(
        strict_source_mask(values["mask4"]), collar=int(collar)
    )


def context_frames(values, last_pred):
    return torch.cat(
        (pad_clean_latent(values["ref"]), pad_clean_latent(last_pred)), dim=1
    )


def run_denoise(
    generator,
    scheduler,
    conditional,
    kv_cache,
    values,
    core,
    denoising_steps,
    transition_seed,
    *,
    memory_condition=None,
    memory_occupancy=None,
    world_context=None,
    last_pred,
    with_grad=False,
):
    set_seed(transition_seed)
    gradient_context = nullcontext() if with_grad else torch.no_grad()
    with gradient_context, torch.autocast(
        "cuda", dtype=torch.bfloat16, cache_enabled=False
    ):
        prediction, _ = denoise_block(
            generator,
            scheduler,
            values["noisy"],
            conditional,
            kv_cache,
            context_frames=context_frames(values, last_pred),
            context_no_grad=True,
            context_freqs_offset=0,
            render_block=values["render"],
            denoising_kv_size=1560 * 6,
            denoising_steps=denoising_steps,
            memory_condition=memory_condition,
            memory_occupancy=memory_occupancy,
            world_context=world_context,
            full_denoise_grad=with_grad,
            source_clean=values["source_clean"],
            source_core=core,
            fixed_source_noise=values["fixed_source_noise"],
        )
    return prediction


def advance_shared_prefix(
    generator,
    scheduler,
    conditional,
    record,
    snapshot,
    device,
    dtype,
    start_block,
    exact_block,
    source_collar,
):
    cache = clone_cache(snapshot["kv_cache"], device, dtype)
    last_pred = snapshot["last_pred"].to(device=device, dtype=dtype)
    denoising_steps = record["denoising_steps"].to(device)
    for block_index in range(start_block, exact_block):
        values = block_inputs(record, block_index, device, dtype)
        last_pred = run_denoise(
            generator,
            scheduler,
            conditional,
            cache,
            values,
            source_core(values, source_collar),
            denoising_steps,
            int(record["transition_seed"]) + block_index,
            last_pred=last_pred,
        ).detach()
    return {
        "kv_cache": cache_to_cpu(cache),
        "last_pred": last_pred.cpu().contiguous(),
    }


def restrict_v0_packet(packet, memory):
    generated = (packet.provenance[0] == int(Provenance.GENERATED)).nonzero(
        as_tuple=False
    ).flatten()
    if generated.numel() != 1:
        raise ValueError("Teacher v0 control expects one generated candidate")
    index = int(generated.item())
    packet.valid[:, index] &= memory
    packet.candidate_20ch[:, index, :, :4] = memory.to(
        packet.candidate_20ch.dtype
    ).expand(-1, -1, 4, -1, -1)
    packet.candidate_20ch[:, index, :, 4:] *= memory.to(
        packet.candidate_20ch.dtype
    )
    packet.confidence[:, index] *= memory.to(packet.confidence.dtype)


def direct_inputs(packet, domains, dtype):
    memory = domains.memory.to(dtype)
    latent = packet.candidate_20ch[:, 0, :, 4:] * memory
    condition = torch.cat(
        (memory.expand(-1, -1, 4, -1, -1), latent), dim=2
    )
    return condition, domains.memory.float()


def run_branch(
    generator,
    scheduler,
    runtime,
    conditional,
    record,
    exact_start,
    bank,
    projector,
    shared,
    config,
    device,
    dtype,
    *,
    identity="A",
    active_layers=None,
    direct=False,
    teacher_v0=False,
):
    exact_block = int(config.evaluation.exact_block)
    hold_block = int(config.evaluation.hold_block)
    cache = clone_cache(exact_start["kv_cache"], device, dtype)
    last_pred = exact_start["last_pred"].to(device=device, dtype=dtype)
    denoising_steps = record["denoising_steps"].to(device)
    outputs = []
    domains_by_block = {}
    bank_A = bank["A"]
    for block_index in (exact_block, hold_block):
        values = block_inputs(record, block_index, device, dtype)
        shared_block = shared[block_index]
        camera = shared_block["camera"]
        packet_A = shared_block["packets"]["A"]
        packet = shared_block["packets"][identity]
        domains = shared_block["domains"]
        domains_by_block[block_index] = domains
        memory_condition = memory_occupancy = world_context = None
        if block_index == exact_block:
            if direct:
                memory_condition, memory_occupancy = direct_inputs(
                    packet_A, domains, dtype
                )
            elif teacher_v0:
                full_packet = bank_A.retrieve_and_project(
                    projector,
                    camera,
                    top_observations=2,
                )
                restrict_v0_packet(full_packet, domains.memory)
                world_context = runtime.precompute(full_packet)
            elif runtime is not None:
                world_context = runtime.precompute(
                    packet,
                    domains,
                    active_layers=active_layers,
                    force_memory_gate=True,
                )
        last_pred = run_denoise(
            generator,
            scheduler,
            conditional,
            cache,
            values,
            domains.source_core,
            denoising_steps,
            int(record["transition_seed"]) + block_index,
            memory_condition=memory_condition,
            memory_occupancy=memory_occupancy,
            world_context=world_context,
            last_pred=last_pred,
        ).detach()
        outputs.append(last_pred.cpu().contiguous())
    del cache
    torch.cuda.empty_cache()
    return torch.cat(outputs, dim=1), domains_by_block


def target_for(record, identity, block_index, domains):
    block_size = int(record["block_size"])
    query_start = block_index * block_size
    query_end = query_start + block_size
    write_start = int(record["write_block"]) * block_size
    write_end = write_start + block_size
    source_clean = record["render_latent"][:, query_start:query_end]
    first_g = record[f"output_{identity}"][:, write_start:write_end]
    return torch.where(domains.source.cpu(), source_clean, first_g)


def masked_l1(prediction, target, mask):
    error = (prediction.float() - target.float()).abs().mean(dim=2, keepdim=True)
    mask = mask.float().cpu()
    return float((error * mask).sum() / mask.sum().clamp_min(1.0))


def exact_hold_tile(video):
    exact = block_frame(video, 0)
    hold = block_frame(video, 1)
    height = max(exact.height, hold.height)
    tile = Image.new("RGB", (exact.width + hold.width, height + 28), "white")
    tile.paste(exact.convert("RGB"), (0, 28))
    tile.paste(hold.convert("RGB"), (exact.width, 28))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 7), "block 16 exact", fill="black")
    draw.text((exact.width + 8, 7), "block 17 HOLD", fill="black")
    return tile


def main():
    args = parse_args()
    config, base_config = load_configs(args.config)
    data_dir = resolve_repo_path(config.experiment.data_root) / args.scene
    artifact_dir = resolve_repo_path(config.experiment.artifact_root) / args.scene
    if args.output_name:
        artifact_dir = artifact_dir / args.output_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = load_scene_record(data_dir / "paired_record.pt")
    snapshot = torch.load(
        data_dir / "common_snapshot.pt", map_location="cpu", weights_only=True
    )
    exact_block = int(config.evaluation.exact_block)
    hold_block = int(config.evaluation.hold_block)
    if exact_block != int(record["first_exact_block"]):
        raise ValueError("configured exact block does not match the paired record")
    if hold_block != exact_block + 1:
        raise ValueError("decisive evaluation expects exactly one carryover block")

    device = init_single_gpu_distributed()
    dtype = torch.bfloat16
    projector = RotationProjector()
    banks = {
        identity: make_bank(record, identity, device, dtype)
        for identity in ("A", "B")
    }
    conditional = {
        "prompt_embeds": record["prompt_embeds"].to(device=device, dtype=dtype)
    }
    shared = build_shared_exact_inputs(
        record, banks, projector, config, device, dtype
    )

    generator = load_frozen_generator(
        base_config, config.experiment.checkpoint, device
    )
    runtime_v0 = attach_world_state_reader(
        generator.model,
        resolve_repo_path(config.experiment.direct_adapter),
        selected_layers=tuple(config.reader.ablation_layers),
        world_width=512,
        heads=8,
        neighborhood=3,
        lora_rank=8,
    )
    load_world_state_reader(
        generator.model,
        resolve_repo_path(config.experiment.teacher_v0_checkpoint),
    )
    runtime_v0.set_lora_enabled(False)
    direct_adapter = load_latent_memory_adapter(
        resolve_repo_path(config.experiment.direct_adapter),
        device=device,
        dtype=dtype,
    )
    attach_latent_memory_adapter(
        generator.model, direct_adapter, device=device, dtype=dtype
    )
    generator.eval().requires_grad_(False)
    scheduler = generator.get_scheduler()
    exact_start = advance_shared_prefix(
        generator,
        scheduler,
        conditional,
        record,
        snapshot,
        device,
        dtype,
        int(record["second_traversal_first_block"]),
        exact_block,
        int(config.reader.source_collar),
    )
    outputs = {}
    domains = None
    for branch, direct, teacher_v0 in (
        ("no_memory", False, False),
        ("frozen_direct", True, False),
        ("teacher_v0", False, True),
    ):
        print(f"Evaluating {args.scene} {branch}")
        output, branch_domains = run_branch(
            generator,
            scheduler,
            runtime_v0 if teacher_v0 else None,
            conditional,
            record,
            exact_start,
            banks,
            projector,
            shared,
            config,
            device,
            dtype,
            direct=direct,
            teacher_v0=teacher_v0,
        )
        outputs[branch] = output
        domains = branch_domains

    del runtime_v0, direct_adapter, scheduler, generator
    gc.collect()
    torch.cuda.empty_cache()

    generator = load_frozen_generator(
        base_config, config.experiment.checkpoint, device
    )
    runtime_v1 = attach_world_state_reader_v1(
        generator.model,
        resolve_repo_path(config.experiment.direct_adapter),
        selected_layers=tuple(config.reader.ablation_layers),
        selector_width=int(config.reader.selector_width),
        confidence_threshold=0.35,
        lora_rank=int(config.reader.lora_rank),
    )
    load_world_state_reader_v1(
        generator.model,
        resolve_repo_path(
            args.reader_checkpoint or config.experiment.reader_v1_checkpoint
        ),
    )
    runtime_v1.set_lora_enabled(False)
    generator.eval().requires_grad_(False)
    scheduler = generator.get_scheduler()
    for setting, layer_values in config.evaluation.layer_sets.items():
        layers = tuple(int(value) for value in layer_values)
        for identity in ("A", "B"):
            branch = f"reader_{setting}_{identity}"
            print(f"Evaluating {args.scene} {branch}")
            outputs[branch], _ = run_branch(
                generator,
                scheduler,
                runtime_v1,
                conditional,
                record,
                exact_start,
                banks,
                projector,
                shared,
                config,
                device,
                dtype,
                identity=identity,
                active_layers=layers,
            )

    del runtime_v1, scheduler, generator
    gc.collect()
    torch.cuda.empty_cache()

    targets = {}
    for identity in ("A", "B"):
        targets[f"first_G_reference_{identity}"] = torch.cat(
            [
                target_for(record, identity, block, domains[block])
                for block in (exact_block, hold_block)
            ],
            dim=1,
        ).contiguous()
    save_file(
        {name: value.contiguous() for name, value in outputs.items()},
        str(artifact_dir / "branch_latents.safetensors"),
    )

    metrics = {
        "scene": args.scene,
        "shared_domain": True,
        "confidence_hard_threshold": False,
        "forced_exact_memory_gate": True,
        "reader_lora": False,
        "blocks": {},
    }
    for offset, block_index in enumerate((exact_block, hold_block)):
        start = offset * int(record["block_size"])
        end = start + int(record["block_size"])
        domain = domains[block_index]
        target_A = targets["first_G_reference_A"][:, start:end]
        target_B = targets["first_G_reference_B"][:, start:end]
        block_metrics = {"branches": {}}
        for branch, output in outputs.items():
            prediction = output[:, start:end]
            own_target, cross_target = target_A, target_B
            if branch.endswith("_B"):
                own_target, cross_target = target_B, target_A
            values = {
                "own_target_M_L1": masked_l1(
                    prediction, own_target, domain.memory
                ),
                "cross_target_M_L1": masked_l1(
                    prediction, cross_target, domain.memory
                ),
                "U_to_no_memory_L1": masked_l1(
                    prediction,
                    outputs["no_memory"][:, start:end],
                    domain.unknown,
                ),
            }
            block_metrics["branches"][branch] = values
        metrics["blocks"][str(block_index)] = block_metrics
    with (artifact_dir / "simple_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    vae = WanVAEWrapper(str(resolve_repo_path(base_config.wan_model_folder))).to(
        device=device, dtype=dtype
    )
    vae.eval().requires_grad_(False)
    decoded = {}
    for name, latent in {**targets, **outputs}.items():
        with torch.no_grad():
            decoded[name] = (
                vae.decode_to_pixel(
                    latent.to(device=device, dtype=dtype), use_cache=False
                )
                * 0.5
                + 0.5
            ).clamp(0, 1).cpu()
        vae.model.clear_cache()
        torch.cuda.empty_cache()

    order = (
        "first_G_reference_A",
        "first_G_reference_B",
        "no_memory",
        "frozen_direct",
        "teacher_v0",
        "reader_layer8_A",
        "reader_layer8_B",
        "reader_layer14_A",
        "reader_layer14_B",
        "reader_layer20_A",
        "reader_layer20_B",
        "reader_all_layers_A",
        "reader_all_layers_B",
    )
    write_grid(
        artifact_dir / "montage.png",
        [(name, exact_hold_tile(decoded[name])) for name in order],
        columns=2,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
