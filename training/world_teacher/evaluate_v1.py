#!/usr/bin/env python3
"""Fair six-way native evaluation for Three-Domain WorldStateReader v1."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
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
    write_video_tensor,
)
from training.world_teacher.dataset import (
    load_scene_record,
    make_bank,
    query_camera,
)
from training.world_teacher.visualize import (
    block_frame,
    write_grid,
    write_sync_comparison,
)
from utils.wan_wrapper import WanVAEWrapper
from world_memory import attach_latent_memory_adapter, load_latent_memory_adapter
from world_state import RotationProjector, build_three_domains
from world_state.runtime import (
    attach_world_state_reader,
    load_world_state_reader,
)
from world_state.runtime_v1 import (
    attach_world_state_reader_v1,
    load_world_state_reader_v1,
)


BRANCHES = (
    "first_G_reference",
    "no_memory",
    "frozen_direct",
    "teacher_v0",
    "reader_v1_A",
    "reader_v1_B",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_teacher/teacher_v1.yaml")
    parser.add_argument("--scene", choices=("S0", "S1"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-name", default="evaluation")
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--end-block", type=int)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_snapshot_cache(snapshot, device, dtype):
    return [
        {
            "k": values["k"].to(device=device, dtype=dtype),
            "v": values["v"].to(device=device, dtype=dtype),
        }
        for values in snapshot["kv_cache"]
    ]


def generated_index(packet):
    indices = (packet.provenance[0] == 1).nonzero(as_tuple=False).flatten()
    if indices.numel() != 1:
        raise ValueError("evaluation expects one generated M40 observation")
    return int(indices.item())


def restrict_v0_packet(packet, domains):
    """Keep old Reader behavior but prevent generated memory outside v1 M."""
    index = generated_index(packet)
    allowed = domains.memory
    packet.valid[:, index] &= allowed
    packet.candidate_20ch[:, index, :, :4] = allowed.to(
        packet.candidate_20ch.dtype
    ).expand(-1, -1, 4, -1, -1)
    packet.candidate_20ch[:, index, :, 4:] *= allowed.to(
        packet.candidate_20ch.dtype
    )
    packet.confidence[:, index] *= allowed.to(packet.confidence.dtype)


def block_inputs(record, block_index, device, dtype):
    block_size = int(record["block_size"])
    start, end = block_index * block_size, (block_index + 1) * block_size
    mask4 = record["mask_latent"][:, start:end].to(device=device, dtype=dtype)
    source_clean = record["render_latent"][:, start:end].to(
        device=device, dtype=dtype
    )
    return {
        "start": start,
        "end": end,
        "noisy": record["noise_A"][:, start:end].to(device=device, dtype=dtype),
        "ref": record["ref_latent"][:, start:end].to(device=device, dtype=dtype),
        "mask4": mask4,
        "source_clean": source_clean,
        "fixed_source_noise": record["noise_A"][:, start:end].to(
            device=device, dtype=dtype
        ),
        "render": torch.cat((mask4, source_clean), dim=2),
    }


def paired_reference(record, end_block):
    block_size = int(record["block_size"])
    start_block = int(record["second_traversal_first_block"])
    prefix = record["output_A"][:, : start_block * block_size]
    targets = []
    paired = [2, 3, 4, 5, 6, 6]
    for block_index in range(start_block, end_block + 1):
        target_block = paired[block_index - start_block]
        start = target_block * block_size
        targets.append(record["output_A"][:, start:start + block_size])
    return torch.cat((prefix, *targets), dim=1).contiguous()


def run_branch(
    *,
    branch,
    generator,
    scheduler,
    runtime,
    record,
    snapshot,
    bank,
    projector,
    config,
    device,
    dtype,
    start_block,
    exact_block,
    end_block,
    direct=False,
    v0=False,
):
    block_size = int(record["block_size"])
    chunks = [snapshot["prefix_output"].to(device=device, dtype=dtype)]
    last_pred = snapshot["last_pred"].to(device=device, dtype=dtype)
    kv_cache = clone_snapshot_cache(snapshot, device, dtype)
    conditional = {
        "prompt_embeds": record["prompt_embeds"].to(device=device, dtype=dtype)
    }
    timings = []
    domains_by_block = {}
    torch.cuda.reset_peak_memory_stats(device)
    for block_index in range(start_block, end_block + 1):
        values = block_inputs(record, block_index, device, dtype)
        packet = bank.retrieve_and_project(
            projector,
            query_camera(record, block_index, device),
            top_observations=int(config.reader.top_observations),
        )
        domains = build_three_domains(
            values["mask4"],
            packet,
            confidence_threshold=float(config.reader.confidence_threshold),
            source_collar=int(config.reader.source_collar),
        )
        domains_by_block[block_index] = domains
        context_frames = None
        if block_index != start_block:
            context_frames = torch.cat(
                (pad_clean_latent(values["ref"]), pad_clean_latent(last_pred)),
                dim=1,
            )

        memory_condition = memory_occupancy = world_context = None
        if direct and block_index == exact_block:
            index = generated_index(packet)
            memory = domains.memory.to(dtype)
            latent = packet.candidate_20ch[:, index, :, 4:] * memory
            memory_condition = torch.cat(
                (memory.expand(-1, -1, 4, -1, -1), latent), dim=2
            )
            memory_occupancy = memory.float()
        elif runtime is not None:
            if v0:
                restrict_v0_packet(packet, domains)
                world_context = runtime.precompute(packet)
            else:
                world_context = runtime.precompute(packet, domains)

        set_seed(int(record["transition_seed"]) + block_index)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=dtype, cache_enabled=False
        ):
            prediction, _ = denoise_block(
                generator,
                scheduler,
                values["noisy"],
                conditional,
                kv_cache,
                context_frames=context_frames,
                context_no_grad=True,
                context_freqs_offset=0,
                render_block=values["render"],
                denoising_kv_size=1560 * 6,
                denoising_steps=record["denoising_steps"].to(device),
                memory_condition=memory_condition,
                memory_occupancy=memory_occupancy,
                world_context=world_context,
                source_clean=values["source_clean"],
                source_core=domains.source_core,
                fixed_source_noise=values["fixed_source_noise"],
            )
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
        chunks.append(prediction)
        last_pred = prediction.detach()
    output = torch.cat(chunks, dim=1).cpu().contiguous()
    stats = {
        "mean_ms_per_block": sum(timings) / len(timings),
        "peak_vram_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    del kv_cache
    torch.cuda.empty_cache()
    return output, stats, domains_by_block


def masked_l1(prediction, target, mask):
    error = (prediction.float() - target.float()).abs().mean(dim=2, keepdim=True)
    mask = mask.float().cpu()
    return float((error * mask).sum() / mask.sum().clamp_min(1.0))


def frame_stats(video, block, block_size):
    latent_middle = block * block_size + block_size // 2
    frame = video[0, min(4 * latent_middle, video.shape[1] - 1)].float()
    dx = (frame[:, :, 1:] - frame[:, :, :-1]).abs().mean()
    dy = (frame[:, 1:, :] - frame[:, :-1, :]).abs().mean()
    return {"brightness": float(frame.mean()), "sharpness": float(dx + dy)}


def main():
    args = parse_args()
    config, base_config = load_configs(args.config)
    data_dir = resolve_repo_path(config.experiment.data_root) / args.scene
    artifact_dir = (
        resolve_repo_path(config.experiment.artifact_root)
        / args.scene
        / args.output_name
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = load_scene_record(data_dir / "paired_record.pt")
    snapshot = torch.load(
        data_dir / "common_snapshot.pt", map_location="cpu", weights_only=True
    )
    device = init_single_gpu_distributed()
    dtype = torch.bfloat16
    projector = RotationProjector()
    banks = {
        identity: make_bank(record, identity, device, dtype)
        for identity in ("A", "B")
    }
    start_block = int(record["second_traversal_first_block"])
    exact_block = int(record["first_exact_block"])
    configured_end = int(config.evaluation.end_block)
    end_block = configured_end if args.end_block is None else int(args.end_block)
    end_block = min(end_block, record["ref_latent"].shape[1] // int(record["block_size"]) - 1)

    outputs = {"first_G_reference": paired_reference(record, end_block)}
    timing = {"first_G_reference": {"mean_ms_per_block": 0.0, "peak_vram_GiB": 0.0}}
    domains = {}

    # Load v0 once for the no-memory, direct, and frozen Teacher-v0 branches.
    generator = load_frozen_generator(
        base_config, config.experiment.checkpoint, device
    )
    runtime_v0 = attach_world_state_reader(
        generator.model,
        resolve_repo_path(config.experiment.direct_adapter),
        selected_layers=tuple(config.reader.selected_layers),
        world_width=512,
        heads=8,
        neighborhood=3,
        lora_rank=8,
    )
    load_world_state_reader(
        generator.model,
        resolve_repo_path(config.experiment.teacher_v0_checkpoint),
    )
    runtime_v0.set_lora_enabled(True)
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
    for branch, direct, use_v0 in (
        ("no_memory", False, False),
        ("frozen_direct", True, False),
        ("teacher_v0", False, True),
    ):
        print(f"Evaluating {args.scene} {branch}")
        output, stats, branch_domains = run_branch(
            branch=branch,
            generator=generator,
            scheduler=scheduler,
            runtime=runtime_v0 if use_v0 else None,
            record=record,
            snapshot=snapshot,
            bank=banks["A"],
            projector=projector,
            config=config,
            device=device,
            dtype=dtype,
            start_block=start_block,
            exact_block=exact_block,
            end_block=end_block,
            direct=direct,
            v0=use_v0,
        )
        outputs[branch] = output
        timing[branch] = stats
        if branch == "no_memory":
            domains["A"] = branch_domains

    del runtime_v0, direct_adapter, scheduler, generator
    gc.collect()
    torch.cuda.empty_cache()

    # Fresh model: formal Reader v1 never has the direct residual attached.
    generator = load_frozen_generator(
        base_config, config.experiment.checkpoint, device
    )
    runtime_v1 = attach_world_state_reader_v1(
        generator.model,
        resolve_repo_path(config.experiment.direct_adapter),
        selected_layers=tuple(config.reader.selected_layers),
        selector_width=int(config.reader.selector_width),
        confidence_threshold=float(config.reader.confidence_threshold),
        lora_rank=int(config.reader.lora_rank),
    )
    load_world_state_reader_v1(generator.model, resolve_repo_path(args.checkpoint))
    runtime_v1.set_lora_enabled(args.enable_lora)
    generator.eval().requires_grad_(False)
    scheduler = generator.get_scheduler()
    for identity in ("A", "B"):
        branch = f"reader_v1_{identity}"
        print(f"Evaluating {args.scene} {branch}")
        output, stats, branch_domains = run_branch(
            branch=branch,
            generator=generator,
            scheduler=scheduler,
            runtime=runtime_v1,
            record=record,
            snapshot=snapshot,
            bank=banks[identity],
            projector=projector,
            config=config,
            device=device,
            dtype=dtype,
            start_block=start_block,
            exact_block=exact_block,
            end_block=end_block,
        )
        outputs[branch] = output
        timing[branch] = stats
        domains[identity] = branch_domains

    del runtime_v1, scheduler, generator
    gc.collect()
    torch.cuda.empty_cache()
    for branch, output in outputs.items():
        save_file({"latent": output}, str(artifact_dir / f"{branch}.safetensors"))

    vae = WanVAEWrapper(str(resolve_repo_path(base_config.wan_model_folder))).to(
        device=device, dtype=dtype
    )
    vae.eval().requires_grad_(False)
    decoded = {}
    for branch in BRANCHES:
        with torch.no_grad():
            video = (
                vae.decode_to_pixel(
                    outputs[branch].to(device=device, dtype=dtype), use_cache=False
                )
                * 0.5
                + 0.5
            ).clamp(0, 1)
        decoded[branch] = video.cpu()
        write_video_tensor(artifact_dir / f"{branch}.mp4", video)
        vae.model.clear_cache()
        torch.cuda.empty_cache()
    write_sync_comparison(
        artifact_dir / "synchronized_comparison.mp4",
        [
            (branch.replace("_", " "), artifact_dir / f"{branch}.mp4")
            for branch in BRANCHES
        ],
    )

    block_size = int(record["block_size"])
    target_by_block = dict(
        zip(
            [int(value) for value in config.evaluation.continuous_blocks],
            [int(value) for value in config.evaluation.paired_target_blocks],
        )
    )
    metrics = {
        "scene": args.scene,
        "checkpoint": str(resolve_repo_path(args.checkpoint)),
        "source_clamp_all_comparisons": True,
        "direct_residual_in_reader_v1": False,
        "patch_gated_lora": bool(args.enable_lora),
        "timing": timing,
        "blocks": {},
    }
    for block in range(start_block, end_block + 1):
        target_block = target_by_block[block]
        qs, qe = block * block_size, (block + 1) * block_size
        ts, te = target_block * block_size, (target_block + 1) * block_size
        target_A = record["output_A"][:, ts:te]
        target_B = record["output_B"][:, ts:te]
        mask_A = domains["A"][block].memory.cpu()
        mask_B = domains["B"][block].memory.cpu()
        source_core = domains["A"][block].source_core.cpu()
        unknown = domains["A"][block].unknown.cpu()
        values = {
            "source_coverage": float(domains["A"][block].source.float().mean()),
            "memory_coverage_A": float(mask_A.float().mean()),
            "unknown_coverage": float(unknown.float().mean()),
            "no_memory_to_A_memory_L1": masked_l1(
                outputs["no_memory"][:, qs:qe], target_A, mask_A
            ),
            "frozen_direct_to_A_memory_L1": masked_l1(
                outputs["frozen_direct"][:, qs:qe], target_A, mask_A
            ),
            "teacher_v0_to_A_memory_L1": masked_l1(
                outputs["teacher_v0"][:, qs:qe], target_A, mask_A
            ),
            "reader_v1_A_to_A_memory_L1": masked_l1(
                outputs["reader_v1_A"][:, qs:qe], target_A, mask_A
            ),
            "reader_v1_B_to_B_memory_L1": masked_l1(
                outputs["reader_v1_B"][:, qs:qe], target_B, mask_B
            ),
            "reader_v1_B_to_A_memory_L1": masked_l1(
                outputs["reader_v1_B"][:, qs:qe], target_A, mask_A
            ),
            "reader_v1_A_source_core_L1": masked_l1(
                outputs["reader_v1_A"][:, qs:qe],
                record["render_latent"][:, qs:qe],
                source_core,
            ),
            "reader_v1_A_unknown_to_no_memory_L1": masked_l1(
                outputs["reader_v1_A"][:, qs:qe],
                outputs["no_memory"][:, qs:qe],
                unknown,
            ),
            "visual": {
                branch: frame_stats(decoded[branch], block, block_size)
                for branch in BRANCHES
            },
        }
        metrics["blocks"][str(block)] = values

    montage = []
    for block in [int(value) for value in config.evaluation.montage_blocks]:
        if block > end_block:
            continue
        for branch in BRANCHES:
            montage.append(
                (f"{branch} / block {block}", block_frame(decoded[branch], block))
            )
    write_grid(artifact_dir / "montage.png", montage, columns=6)
    with (artifact_dir / "simple_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
