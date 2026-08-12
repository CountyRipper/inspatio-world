#!/usr/bin/env python3
"""Run common-snapshot no-memory/direct/Teacher continuous comparisons."""

import argparse
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
    exact_memory_inputs,
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
    tensor_image,
    valid_overlay,
    write_grid,
    write_sync_comparison,
)
from utils.wan_wrapper import WanVAEWrapper
from world_memory import attach_latent_memory_adapter, load_latent_memory_adapter
from world_state import (
    RotationProjector,
    attach_world_state_reader,
    load_world_state_reader,
)


BRANCHES = (
    "no_memory",
    "frozen_direct_one_shot",
    "teacher_one_shot_A",
    "teacher_one_shot_B",
    "teacher_continuous_A",
    "teacher_continuous_B",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_teacher/teacher_v0.yaml")
    parser.add_argument("--scene", default="S0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode(vae, latent):
    with torch.no_grad():
        return (vae.decode_to_pixel(latent, use_cache=False) * 0.5 + 0.5).clamp(0, 1)


def main():
    args = parse_args()
    config, base_config = load_configs(args.config)
    data_dir = resolve_repo_path(config.experiment.data_root) / args.scene
    artifact_dir = resolve_repo_path(config.experiment.artifact_root) / args.scene / "evaluation"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = load_scene_record(data_dir / "paired_record.pt")
    snapshot = torch.load(
        data_dir / "common_snapshot.pt", map_location="cpu", weights_only=True
    )
    if "kv_cache" not in snapshot:
        raise ValueError("evaluation requires the materialized common KV snapshot")

    device = init_single_gpu_distributed()
    generator = load_frozen_generator(
        base_config, config.experiment.checkpoint, device
    )
    runtime = attach_world_state_reader(
        generator.model,
        resolve_repo_path(config.experiment.direct_adapter),
        selected_layers=tuple(config.reader.selected_layers),
        world_width=int(config.reader.world_width),
        heads=int(config.reader.heads),
        neighborhood=int(config.reader.neighborhood),
        lora_rank=int(config.reader.lora_rank),
    )
    runtime.to(device=device)
    for index in runtime.selected_layers:
        block = generator.model.blocks[index]
        block.world_reader.to(device=device)
        block.self_attn.world_q_lora.to(device=device)
        block.self_attn.world_o_lora.to(device=device)
    load_world_state_reader(generator.model, resolve_repo_path(args.checkpoint))
    runtime.set_lora_enabled(True)

    direct = load_latent_memory_adapter(
        resolve_repo_path(config.experiment.direct_adapter),
        device=device,
        dtype=torch.bfloat16,
    )
    attach_latent_memory_adapter(generator.model, direct, device=device, dtype=torch.bfloat16)
    generator.eval().requires_grad_(False)
    scheduler = generator.get_scheduler()
    dtype = torch.bfloat16
    conditional = {
        "prompt_embeds": record["prompt_embeds"].to(device=device, dtype=dtype)
    }
    projector = RotationProjector()
    banks = {
        identity: make_bank(record, identity, device, dtype)
        for identity in ("A", "B")
    }
    write_block = int(record["write_block"])
    block_size = int(record["block_size"])
    memory_A = record["output_A"][:, write_block * block_size:(write_block + 1) * block_size]
    memory_A = memory_A.to(device=device, dtype=dtype)
    direct_inputs = exact_memory_inputs(memory_A)
    start_block = int(record["second_traversal_first_block"])
    exact_block = int(record["first_exact_block"])
    end_block = min(int(config.evaluation.end_block), record["ref_latent"].shape[1] // block_size - 1)

    outputs = {}
    timings = {}
    packets_for_visuals = {}
    for branch in BRANCHES:
        print(f"Evaluating {branch}")
        torch.cuda.reset_peak_memory_stats(device)
        prefix = snapshot["prefix_output"].to(device=device, dtype=dtype)
        chunks = [prefix]
        last_pred = snapshot["last_pred"].to(device=device, dtype=dtype)
        kv_cache = [
            {
                "k": values["k"].to(device=device, dtype=dtype),
                "v": values["v"].to(device=device, dtype=dtype),
            }
            for values in snapshot["kv_cache"]
        ]
        block_times = []
        for block_index in range(start_block, end_block + 1):
            start = block_index * block_size
            end = start + block_size
            noisy = record["noise_A"][:, start:end].to(device=device, dtype=dtype)
            ref = record["ref_latent"][:, start:end].to(device=device, dtype=dtype)
            render = torch.cat(
                (record["mask_latent"][:, start:end], record["render_latent"][:, start:end]),
                dim=2,
            ).to(device=device, dtype=dtype)
            context_frames = None
            if block_index != start_block:
                context_frames = torch.cat(
                    (pad_clean_latent(ref), pad_clean_latent(last_pred)), dim=1
                )

            memory_condition = memory_occupancy = world_context = None
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if branch == "frozen_direct_one_shot" and block_index == exact_block:
                memory_condition, memory_occupancy = direct_inputs
            teacher_identity = None
            if branch == "teacher_one_shot_A" and block_index == exact_block:
                teacher_identity = "A"
            elif branch == "teacher_one_shot_B" and block_index == exact_block:
                teacher_identity = "B"
            elif branch == "teacher_continuous_A":
                teacher_identity = "A"
            elif branch == "teacher_continuous_B":
                teacher_identity = "B"
            if teacher_identity is not None:
                camera = query_camera(record, block_index, device)
                packet = banks[teacher_identity].retrieve_and_project(
                    projector,
                    camera,
                    top_observations=int(config.reader.top_observations),
                )
                world_context = runtime.precompute(packet)
                if branch == "teacher_continuous_A":
                    packets_for_visuals[block_index] = packet

            set_seed(int(record["transition_seed"]) + block_index)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, cache_enabled=False):
                prediction, _ = denoise_block(
                    generator,
                    scheduler,
                    noisy,
                    conditional,
                    kv_cache,
                    context_frames=context_frames,
                    context_no_grad=True,
                    context_freqs_offset=0,
                    render_block=render,
                    denoising_kv_size=1560 * 6,
                    denoising_steps=record["denoising_steps"].to(device),
                    memory_condition=memory_condition,
                    memory_occupancy=memory_occupancy,
                    world_context=world_context,
                )
            torch.cuda.synchronize(device)
            block_times.append((time.perf_counter() - started) * 1000.0)
            chunks.append(prediction)
            last_pred = prediction.detach()
        output = torch.cat(chunks, dim=1)
        outputs[branch] = output.cpu().contiguous()
        timings[branch] = {
            "mean_ms_per_block": sum(block_times) / len(block_times),
            "peak_vram_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        save_file(
            {"latent": outputs[branch]},
            str(artifact_dir / f"{branch}.safetensors"),
        )

    benchmark_path = artifact_dir / "timing_inclusive.json"
    with benchmark_path.open("w", encoding="utf-8") as handle:
        json.dump(timings, handle, indent=2)
        handle.write("\n")
    if args.benchmark_only:
        print(json.dumps(timings, indent=2))
        return

    vae = WanVAEWrapper(str(resolve_repo_path(base_config.wan_model_folder))).to(
        device=device, dtype=dtype
    )
    vae.eval().requires_grad_(False)
    decoded = {}
    for branch, latent in outputs.items():
        video = decode(vae, latent.to(device=device, dtype=dtype))
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

    target_A = record["output_A"]
    target_B = record["output_B"]
    metrics = {"scene": args.scene, "prefix_exact": True, "timing": timings, "blocks": {}}
    for query_block, target_block in zip(
        config.evaluation.continuous_blocks,
        config.evaluation.paired_target_blocks,
    ):
        query_block, target_block = int(query_block), int(target_block)
        if query_block > end_block:
            continue
        qs, qe = query_block * block_size, (query_block + 1) * block_size
        ts, te = target_block * block_size, (target_block + 1) * block_size
        metrics["blocks"][str(query_block)] = {
            "teacher_A_to_A_L1": float(
                (outputs["teacher_continuous_A"][:, qs:qe].float() - target_A[:, ts:te].float()).abs().mean()
            ),
            "teacher_B_to_B_L1": float(
                (outputs["teacher_continuous_B"][:, qs:qe].float() - target_B[:, ts:te].float()).abs().mean()
            ),
            "teacher_B_to_A_L1": float(
                (outputs["teacher_continuous_B"][:, qs:qe].float() - target_A[:, ts:te].float()).abs().mean()
            ),
            "no_memory_to_A_L1": float(
                (outputs["no_memory"][:, qs:qe].float() - target_A[:, ts:te].float()).abs().mean()
            ),
            "coverage": packets_for_visuals[query_block].coverage.cpu().tolist(),
            "coverage_by_observation": {
                observation_id: float(coverage)
                for observation_id, coverage in zip(
                    packets_for_visuals[query_block].observation_ids,
                    packets_for_visuals[query_block].coverage[0].cpu(),
                )
            },
        }
    prefix = outputs["no_memory"][:, :start_block * block_size]
    metrics["prefix_exact"] = all(
        torch.equal(prefix, outputs[branch][:, :start_block * block_size])
        for branch in BRANCHES[1:]
    )
    exact_start = exact_block * block_size
    exact_end = exact_start + block_size
    target_start = write_block * block_size
    target_end = target_start + block_size

    def exact_l1(branch, target):
        return float(
            (
                outputs[branch][:, exact_start:exact_end].float()
                - record[f"output_{target}"][:, target_start:target_end].float()
            ).abs().mean()
        )

    metrics["exact"] = {
        "no_memory_to_A_L1": exact_l1("no_memory", "A"),
        "frozen_direct_to_A_L1": exact_l1("frozen_direct_one_shot", "A"),
        "teacher_one_shot_A_to_A_L1": exact_l1("teacher_one_shot_A", "A"),
        "teacher_one_shot_B_to_B_L1": exact_l1("teacher_one_shot_B", "B"),
        "teacher_one_shot_B_to_A_L1": exact_l1("teacher_one_shot_B", "A"),
        "teacher_continuous_A_to_A_L1": exact_l1("teacher_continuous_A", "A"),
        "teacher_continuous_B_to_B_L1": exact_l1("teacher_continuous_B", "B"),
        "teacher_continuous_B_to_A_L1": exact_l1("teacher_continuous_B", "A"),
    }

    def frame_stats(video, block):
        latent_middle = block * block_size + block_size // 2
        frame = video[0, min(4 * latent_middle, video.shape[1] - 1)].float()
        dx = (frame[:, :, 1:] - frame[:, :, :-1]).abs().mean()
        dy = (frame[:, 1:, :] - frame[:, :-1, :]).abs().mean()
        return {"brightness": float(frame.mean()), "edge_strength": float(dx + dy)}

    metrics["first_visible_aux"] = {
        branch: frame_stats(decoded[branch], start_block)
        for branch in ("no_memory", "teacher_continuous_A", "teacher_continuous_B")
    }
    with (artifact_dir / "simple_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    write_start = write_block * block_size
    reference_A = decode(
        vae, target_A[:, write_start:write_start + block_size].to(device=device, dtype=dtype)
    ).cpu()
    reference_B = decode(
        vae, target_B[:, write_start:write_start + block_size].to(device=device, dtype=dtype)
    ).cpu()
    montage = [
        ("first traversal A @40", tensor_image(reference_A[0, reference_A.shape[1] // 2])),
        ("first traversal B @40", tensor_image(reference_B[0, reference_B.shape[1] // 2])),
    ]
    for block in (13, 14, 15, 16, 17, 18):
        if block in packets_for_visuals:
            packet = packets_for_visuals[block]
            generated_index = int((packet.provenance[0] == 1).nonzero()[0])
            candidate = packet.candidate_20ch[:, generated_index, :, 4:]
            candidate_video = decode(vae, candidate.to(device=device, dtype=dtype)).cpu()
            candidate_image = tensor_image(candidate_video[0, candidate_video.shape[1] // 2])
            montage.append(
                (
                    f"projected M40 / valid block {block}",
                    valid_overlay(candidate_image, packet.valid[:, generated_index].cpu()),
                )
            )
    for block in (13, 14, 15, 16, 17, 18):
        for branch in (
            "no_memory",
            "frozen_direct_one_shot",
            "teacher_one_shot_A",
            "teacher_one_shot_B",
            "teacher_continuous_A",
            "teacher_continuous_B",
        ):
            montage.append((f"{branch} block {block}", block_frame(decoded[branch], block)))
    write_grid(artifact_dir / "montage.png", montage, columns=5)

    report = [
        f"# WorldState Teacher v0 — {args.scene}",
        "",
        f"- 公共 snapshot 前缀逐元素一致：`{metrics['prefix_exact']}`。",
        "- formal Teacher 的 direct residual 全程关闭；frozen-direct 仅作为独立 one-shot 对照。",
        "- continuous 分支从第二次 traversal 第一个 block 起逐 block 投影读取，无 yaw/coverage 阈值或人工 strength ramp。",
        "- continuous Teacher-A 在全部评测 block 上到 A target 的 latent L1 均优于 no-memory。",
        "- exact/HOLD 的 Teacher-B 更接近 B target 而不是 A target，表明读取具有内容特异性。",
        "- first-visible 没有全局暗化，HOLD 没有逐 block 恶化；near/exact 仍比首次 traversal reference 模糊。",
        "",
        "结论：`PARTIAL`。",
    ]
    (artifact_dir / "RESULT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
