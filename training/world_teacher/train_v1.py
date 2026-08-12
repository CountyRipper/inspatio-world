#!/usr/bin/env python3
"""Train exact and continuous Three-Domain WorldStateReader v1 stages."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.causal_inference import denoise_block
from scripts.world_memory.common import (
    init_single_gpu_distributed,
    initialize_kv_cache,
    load_configs,
    load_frozen_generator,
    pad_clean_latent,
    resolve_repo_path,
)
from training.world_teacher.dataset import (
    load_scene_record,
    make_bank,
    make_block_example,
)
from world_state import RotationProjector, build_three_domains
from world_state.runtime_v1 import (
    attach_world_state_reader_v1,
    load_world_state_reader_v1,
    save_world_state_reader_v1,
    world_state_v1_trainable_parameters,
)


STAGES = ("exact-reader", "continuous-reader", "continuous-lora", "two-block")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_teacher/teacher_v1.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--scenes", nargs="+", default=["S0", "S1"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_cache(cache) -> None:
    for values in cache:
        values["k"].detach_().zero_()
        values["v"].detach_().zero_()


def masked_l1(prediction, target, mask):
    mask = mask.to(prediction.dtype)
    denominator = mask.sum().clamp_min(1.0)
    error = (prediction.float() - target.float()).abs().mean(dim=2, keepdim=True)
    return (error * mask).sum() / denominator


def masked_spatial_losses(prediction, target, mask):
    prediction = prediction.float()
    target = target.float()
    horizontal_mask = mask[..., :, 1:] & mask[..., :, :-1]
    vertical_mask = mask[..., 1:, :] & mask[..., :-1, :]
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    horizontal = masked_l1(pred_dx, target_dx, horizontal_mask)
    vertical = masked_l1(pred_dy, target_dy, vertical_mask)

    batch, frames, channels, height, width = prediction.shape
    kernel = prediction.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    flat_prediction = prediction.reshape(batch * frames * channels, 1, height, width)
    flat_target = target.reshape(batch * frames * channels, 1, height, width)
    pred_lap = F.conv2d(flat_prediction, kernel, padding=1).reshape_as(prediction)
    target_lap = F.conv2d(flat_target, kernel, padding=1).reshape_as(target)
    laplacian = masked_l1(pred_lap, target_lap, mask)
    return 0.5 * (horizontal + vertical), laplacian


def main():
    args = parse_args()
    config, base_config = load_configs(args.config)
    records = {
        scene: load_scene_record(
            resolve_repo_path(config.experiment.data_root) / scene / "paired_record.pt"
        )
        for scene in args.scenes
    }

    device = init_single_gpu_distributed()
    dtype = torch.bfloat16
    generator = load_frozen_generator(
        base_config, config.experiment.checkpoint, device
    )
    runtime = attach_world_state_reader_v1(
        generator.model,
        resolve_repo_path(config.experiment.direct_adapter),
        selected_layers=tuple(config.reader.selected_layers),
        selector_width=int(config.reader.selector_width),
        confidence_threshold=float(config.reader.confidence_threshold),
        lora_rank=int(config.reader.lora_rank),
    )
    runtime.to(device=device)
    include_lora = args.stage in ("continuous-lora", "two-block")
    runtime.set_lora_enabled(include_lora)
    trainable = world_state_v1_trainable_parameters(
        generator.model, include_lora=include_lora
    )
    if bool(config.training.gradient_checkpointing):
        generator.enable_gradient_checkpointing()
    if not trainable or not all(parameter.dtype == torch.float32 for parameter in trainable):
        raise AssertionError("Reader v1 trainable weights must stay FP32")
    trainable_ids = {id(parameter) for parameter in trainable}
    if any(
        parameter.requires_grad
        for parameter in generator.model.parameters()
        if id(parameter) not in trainable_ids
    ):
        raise AssertionError("the backbone and frozen adapter must remain frozen")

    artifact_root = resolve_repo_path(config.experiment.artifact_root)
    checkpoint_dir = artifact_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = checkpoint_dir / f"{args.stage}.safetensors"
    state_path = checkpoint_dir / f"{args.stage}_training_state.pt"
    if args.init_checkpoint:
        load_world_state_reader_v1(
            generator.model, resolve_repo_path(args.init_checkpoint)
        )

    learning_rate = float(config.training.exact_learning_rate)
    if args.stage == "continuous-reader":
        learning_rate = float(config.training.continuous_learning_rate)
    elif args.stage == "continuous-lora":
        learning_rate = float(config.training.lora_learning_rate)
    elif args.stage == "two-block":
        learning_rate = float(config.training.two_block_learning_rate)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=float(config.training.weight_decay),
    )
    history = []
    start_step = 0
    if args.resume and state_path.exists():
        load_world_state_reader_v1(generator.model, sidecar_path)
        saved = torch.load(state_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = saved.get("history", [])
        start_step = int(saved["step"])

    steps_by_stage = {
        "exact-reader": int(config.training.exact_reader_steps),
        "continuous-reader": int(config.training.continuous_reader_steps),
        "continuous-lora": int(config.training.continuous_lora_steps),
        "two-block": int(config.training.two_block_steps),
    }
    total_steps = steps_by_stage[args.stage] if args.steps is None else args.steps
    query_blocks = [int(value) for value in config.evaluation.continuous_blocks]
    target_blocks = [int(value) for value in config.evaluation.paired_target_blocks]
    milestone_steps = {
        int(value) for value in config.training.milestone_steps
    }
    two_block_windows = [
        tuple(int(value) for value in row)
        for row in config.evaluation.two_block_windows
    ]
    conditional = {
        scene: {
            "prompt_embeds": record["prompt_embeds"].to(device=device, dtype=dtype)
        }
        for scene, record in records.items()
    }
    banks = {
        (scene, identity): make_bank(record, identity, device, dtype)
        for scene, record in records.items()
        for identity in ("A", "B")
    }
    projector = RotationProjector()
    scheduler = generator.get_scheduler()
    kv_cache = initialize_kv_cache(generator, 1, dtype, device)
    second_kv_cache = (
        initialize_kv_cache(generator, 1, dtype, device)
        if args.stage == "two-block"
        else None
    )
    no_memory_cache = {}

    def prepare(scene, identity, query_block, target_block):
        example = make_block_example(
            records[scene],
            identity=identity,
            query_block=query_block,
            target_block=target_block,
            device=device,
            dtype=dtype,
        )
        packet = banks[(scene, identity)].retrieve_and_project(
            projector,
            example.camera,
            top_observations=int(config.reader.top_observations),
        )
        domains = build_three_domains(
            example.source_mask4,
            packet,
            confidence_threshold=float(config.reader.confidence_threshold),
            source_collar=int(config.reader.source_collar),
        )
        return example, packet, domains

    def predict(
        scene,
        example,
        packet,
        domains,
        *,
        memory_enabled=True,
        previous_prediction=None,
        cache=None,
    ):
        cache = kv_cache if cache is None else cache
        reset_cache(cache)
        context_frames = example.context_frames
        context_no_grad = True
        if previous_prediction is not None:
            ref = context_frames[:, :3]
            context_frames = torch.cat(
                (ref, pad_clean_latent(previous_prediction)), dim=1
            )
            context_no_grad = False
        set_seed(example.transition_seed)
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            world_context = runtime.precompute(packet, domains) if memory_enabled else None
            prediction, _ = denoise_block(
                generator,
                scheduler,
                example.noisy_input.clone(),
                conditional[scene],
                cache,
                context_frames=context_frames,
                context_no_grad=context_no_grad,
                context_freqs_offset=0,
                render_block=example.render_block,
                denoising_kv_size=1560 * 6,
                denoising_steps=example.denoising_steps,
                world_context=world_context,
                full_denoise_grad=memory_enabled,
                source_clean=example.source_clean,
                source_core=domains.source_core,
                fixed_source_noise=example.fixed_source_noise,
            )
        return prediction

    def frozen_no_memory(scene, query_block, example, packet, domains):
        key = (scene, int(query_block))
        if key not in no_memory_cache:
            with torch.no_grad():
                no_memory_cache[key] = predict(
                    scene,
                    example,
                    packet,
                    domains,
                    memory_enabled=False,
                ).detach()
        return no_memory_cache[key]

    def loss_for(prediction, target, no_memory, domains):
        memory_l1 = masked_l1(prediction, target, domains.memory)
        gradient, laplacian = masked_spatial_losses(
            prediction, target, domains.memory
        )
        if domains.unknown.any():
            unknown = masked_l1(prediction, no_memory, domains.unknown)
        else:
            unknown = prediction.new_zeros((), dtype=torch.float32)
        total = (
            float(config.training.memory_l1_weight) * memory_l1
            + float(config.training.spatial_gradient_weight) * gradient
            + float(config.training.laplacian_weight) * laplacian
            + float(config.training.unknown_preservation_weight) * unknown
        )
        return total, memory_l1, gradient, laplacian, unknown

    def select_single(step):
        scene = args.scenes[(step // 2) % len(args.scenes)]
        identity = "A" if step % 2 == 0 else "B"
        if args.stage == "exact-reader":
            return (
                scene,
                identity,
                int(records[scene]["first_exact_block"]),
                int(records[scene]["write_block"]),
            )
        if args.stage in ("continuous-reader", "continuous-lora"):
            position = (step // (2 * len(args.scenes))) % len(query_blocks)
            return scene, identity, query_blocks[position], target_blocks[position]
        position = (step // (2 * len(args.scenes))) % len(two_block_windows)
        query_block, _, target_block, _ = two_block_windows[position]
        return scene, identity, query_block, target_block

    def save(step):
        metadata = {
            "stage": args.stage,
            "step": step,
            "source_collar": int(config.reader.source_collar),
            "direct_residual": False,
            "lora_enabled": runtime.lora_enabled,
        }
        save_world_state_reader_v1(
            generator.model, sidecar_path, metadata=metadata
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "step": step,
                "history": history,
                "trainable_parameters": sum(p.numel() for p in trainable),
            },
            state_path,
        )
        save_world_state_reader_v1(
            generator.model,
            checkpoint_dir / "reader_v1_latest.safetensors",
            metadata=metadata,
        )
        if step in milestone_steps:
            save_world_state_reader_v1(
                generator.model,
                checkpoint_dir / f"{args.stage}_step{step}.safetensors",
                metadata=metadata,
            )
        selected_continuous_step = int(
            config.training.selected_continuous_step
        )
        if args.stage == "continuous-reader" and step == selected_continuous_step:
            save_world_state_reader_v1(
                generator.model,
                checkpoint_dir / "reader_v1_final.safetensors",
                metadata=metadata,
            )
        if args.stage == "two-block":
            save_world_state_reader_v1(
                generator.model,
                checkpoint_dir / "reader_v1_final.safetensors",
                metadata=metadata,
            )

    trainable_count = sum(parameter.numel() for parameter in trainable)
    print(
        f"stage={args.stage} scenes={args.scenes} trainable={trainable_count:,} "
        f"lr={learning_rate} full_four_step_grad=True direct_residual=False "
        f"patch_gated_lora={runtime.lora_enabled}"
    )
    generator.eval()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        scene, identity, query_block, target_block = select_single(step)
        example, packet, domains = prepare(
            scene, identity, query_block, target_block
        )
        no_memory = frozen_no_memory(
            scene, query_block, example, packet, domains
        )
        prediction = predict(scene, example, packet, domains)
        total, memory_l1, gradient, laplacian, unknown = loss_for(
            prediction, example.target, no_memory, domains
        )

        if args.stage == "two-block":
            window = next(row for row in two_block_windows if row[0] == query_block)
            _, next_query, _, next_target = window
            next_example, next_packet, next_domains = prepare(
                scene, identity, next_query, next_target
            )
            next_no_memory = frozen_no_memory(
                scene, next_query, next_example, next_packet, next_domains
            )
            next_prediction = predict(
                scene,
                next_example,
                next_packet,
                next_domains,
                previous_prediction=prediction,
                cache=second_kv_cache,
            )
            next_values = loss_for(
                next_prediction,
                next_example.target,
                next_no_memory,
                next_domains,
            )
            total = 0.5 * (total + next_values[0])
            memory_l1 = 0.5 * (memory_l1 + next_values[1])

        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        values = {
            "step": step + 1,
            "scene": scene,
            "identity": identity,
            "query_block": query_block,
            "target_block": target_block,
            "loss": float(total.detach()),
            "memory_l1": float(memory_l1.detach()),
            "gradient_loss": float(gradient.detach()),
            "laplacian_loss": float(laplacian.detach()),
            "unknown_l1": float(unknown.detach()),
            "gradient_norm": gradient_norm,
            "source_coverage": float(domains.source.float().mean()),
            "memory_coverage": float(domains.memory.float().mean()),
            "unknown_coverage": float(domains.unknown.float().mean()),
        }
        history.append(values)
        if step == start_step or (step + 1) % 5 == 0:
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            print(
                f"step={step + 1}/{total_steps} scene={scene} id={identity} "
                f"q={query_block} loss={values['loss']:.6f} "
                f"memory={values['memory_l1']:.6f} grad={gradient_norm:.4f} "
                f"peak_GiB={peak:.2f}"
            )
        if (step + 1) % int(config.training.save_every) == 0:
            save(step + 1)

    save(total_steps)
    summary = {
        "stage": args.stage,
        "steps": total_steps,
        "trainable_parameters": trainable_count,
        "peak_vram_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        "final_records": history[-min(24, len(history)):],
    }
    with (artifact_root / f"training_{args.stage}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    (artifact_root / "RUN_STATE.md").write_text(
        f"Stage: {args.stage} complete\n"
        f"Recent checkpoint: {sidecar_path}\n"
        f"Conclusion: final memory L1 {history[-1]['memory_l1']:.6f}.\n"
        "Next: evaluate before advancing.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
