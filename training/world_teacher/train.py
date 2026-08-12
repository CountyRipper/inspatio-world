#!/usr/bin/env python3
"""Train Reader-only exact, conditional LoRA, and continuous read stages."""

import argparse
import json
import sys
from pathlib import Path

import torch


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
    ownership_masks,
)
from world_state import (
    RotationProjector,
    attach_world_state_reader,
    load_world_state_reader,
    save_world_state_reader,
    world_state_trainable_parameters,
)


STAGES = ("exact-reader", "exact-lora", "continuous", "two-block")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_teacher/teacher_v0.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--scenes", nargs="+", default=["S0"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_losses(prediction, target, mask):
    mask = mask.to(prediction.dtype)
    denominator = mask.sum().clamp_min(1.0)
    raw = ((prediction.float() - target.float()).abs().mean(dim=2, keepdim=True) * mask).sum()
    raw = raw / denominator
    cosine = 1.0 - torch.nn.functional.cosine_similarity(
        prediction.float(), target.float(), dim=2
    )
    cosine = (cosine * mask[:, :, 0]).sum() / denominator
    return raw, cosine


def move_reader_modules(model, runtime, device):
    runtime.to(device=device)
    for index in runtime.selected_layers:
        block = model.blocks[index]
        block.world_reader.to(device=device)
        if hasattr(block.self_attn, "world_q_lora"):
            block.self_attn.world_q_lora.to(device=device)
            block.self_attn.world_o_lora.to(device=device)


def main():
    args = parse_args()
    config, base_config = load_configs(args.config)
    records = {}
    for scene in args.scenes:
        path = resolve_repo_path(config.experiment.data_root) / scene / "paired_record.pt"
        if not path.exists():
            raise FileNotFoundError(f"build paired records first: {path}")
        records[scene] = load_scene_record(path)

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
    move_reader_modules(generator.model, runtime, device)
    include_lora = args.stage != "exact-reader"
    runtime.set_lora_enabled(include_lora)
    trainable = world_state_trainable_parameters(
        generator.model, include_lora=include_lora
    )
    if bool(config.training.gradient_checkpointing):
        generator.enable_gradient_checkpointing()
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable or not all(parameter.dtype == torch.float32 for parameter in trainable):
        raise AssertionError("Teacher trainable weights must stay FP32")
    trainable_ids = {id(parameter) for parameter in trainable}
    if any(
        parameter.requires_grad
        for name, parameter in generator.model.named_parameters()
        if id(parameter) not in trainable_ids
    ):
        raise AssertionError("the non-Teacher backbone must remain frozen")

    artifact_root = resolve_repo_path(config.experiment.artifact_root)
    checkpoint_dir = artifact_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = checkpoint_dir / f"{args.stage}.safetensors"
    state_path = checkpoint_dir / f"{args.stage}_training_state.pt"
    if args.init_checkpoint:
        load_world_state_reader(generator.model, resolve_repo_path(args.init_checkpoint))

    learning_rate = float(config.training.learning_rate)
    if args.stage == "continuous":
        learning_rate = float(config.training.continuous_learning_rate)
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
        load_world_state_reader(generator.model, sidecar_path)
        saved = torch.load(state_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = saved.get("history", [])
        start_step = int(saved["step"])

    projector = RotationProjector()
    dtype = torch.bfloat16
    conditional = {
        scene: {"prompt_embeds": record["prompt_embeds"].to(device=device, dtype=dtype)}
        for scene, record in records.items()
    }
    kv_cache = initialize_kv_cache(generator, 1, dtype, device)
    second_kv_cache = (
        initialize_kv_cache(generator, 1, dtype, device)
        if args.stage == "two-block"
        else None
    )
    scheduler = generator.get_scheduler()

    stage_steps = {
        "exact-reader": int(config.training.exact_reader_steps),
        "exact-lora": int(config.training.exact_lora_steps),
        "continuous": int(config.training.continuous_steps),
        "two-block": int(config.training.two_block_steps),
    }
    total_steps = stage_steps[args.stage] if args.steps is None else args.steps
    query_blocks = list(config.evaluation.continuous_blocks)
    target_blocks = list(config.evaluation.paired_target_blocks)
    two_block_windows = [tuple(int(value) for value in row) for row in config.evaluation.two_block_windows]

    def reset_cache(cache):
        for values in cache:
            values["k"].detach_().zero_()
            values["v"].detach_().zero_()

    def prepare(scene, identity, query_block, target_block):
        record = records[scene]
        example = make_block_example(
            record,
            identity=identity,
            query_block=query_block,
            target_block=target_block,
            device=device,
            dtype=dtype,
        )
        bank = make_bank(record, identity, device, dtype)
        packet = bank.retrieve_and_project(
            projector,
            example.camera,
            top_observations=int(config.reader.top_observations),
        )
        source, generated, unknown = ownership_masks(
            packet,
            generated_static_threshold=float(
                config.training.generated_static_threshold
            ),
        )
        return example, packet, source, generated, unknown

    def predict(scene, example, packet, *, previous_prediction=None, cache=None):
        cache = kv_cache if cache is None else cache
        reset_cache(cache)
        context_frames = example.context_frames
        context_no_grad = True
        if previous_prediction is not None:
            ref = context_frames[:, :3]
            context_frames = torch.cat((ref, pad_clean_latent(previous_prediction)), dim=1)
            context_no_grad = False
        set_seed(example.transition_seed)
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            world_context = runtime.precompute(packet)
            if world_context is None:
                raise RuntimeError("training record unexpectedly has no valid candidate")
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
                full_denoise_grad=True,
            )
        return prediction

    def loss_for(prediction, example, source, generated, unknown):
        known = source | generated
        raw, cosine = masked_losses(prediction, example.target, known)
        if unknown.any() and float(config.training.unknown_consistency_weight) > 0:
            unknown_l1, _ = masked_losses(
                prediction, example.no_memory_target, unknown
            )
        else:
            unknown_l1 = prediction.new_zeros((), dtype=torch.float32)
        total = (
            float(config.training.raw_l1_weight) * raw
            + float(config.training.cosine_weight) * cosine
            + float(config.training.unknown_consistency_weight) * unknown_l1
        )
        return total, raw, cosine, unknown_l1

    def select_single(step):
        scene = args.scenes[(step // 2) % len(args.scenes)]
        identity = "A" if step % 2 == 0 else "B"
        if args.stage.startswith("exact"):
            query_block = int(records[scene]["first_exact_block"])
            target_block = int(records[scene]["write_block"])
        elif args.stage == "continuous":
            position = (step // (2 * len(args.scenes))) % len(query_blocks)
            query_block = int(query_blocks[position])
            target_block = int(target_blocks[position])
        else:
            position = (step // (2 * len(args.scenes))) % len(two_block_windows)
            query_block, _, target_block, _ = two_block_windows[position]
        return scene, identity, query_block, target_block

    def save(step):
        save_world_state_reader(
            generator.model,
            sidecar_path,
            metadata={"stage": args.stage, "step": step},
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "step": step,
                "history": history,
                "trainable_parameters": trainable_count,
            },
            state_path,
        )
        save_world_state_reader(
            generator.model,
            checkpoint_dir / "teacher_latest.safetensors",
            metadata={"stage": args.stage, "step": step},
        )
        if args.stage == "two-block":
            save_world_state_reader(
                generator.model,
                checkpoint_dir / "teacher_final.safetensors",
                metadata={"stage": args.stage, "step": step},
            )

    print(
        f"stage={args.stage} scenes={args.scenes} trainable={trainable_count:,} "
        f"lr={learning_rate} full_four_step_grad=True "
        f"direct_residual=False lora={runtime.lora_enabled}"
    )
    generator.eval()
    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        scene, identity, query_block, target_block = select_single(step)
        example, packet, source, generated, unknown = prepare(
            scene, identity, query_block, target_block
        )
        prediction = predict(scene, example, packet)
        total, raw, cosine, unknown_l1 = loss_for(
            prediction, example, source, generated, unknown
        )

        if args.stage == "two-block":
            window = next(row for row in two_block_windows if row[0] == query_block)
            _, next_query, _, next_target = window
            next_example, next_packet, next_source, next_generated, next_unknown = prepare(
                scene, identity, next_query, int(next_target)
            )
            next_prediction = predict(
                scene,
                next_example,
                next_packet,
                previous_prediction=prediction,
                cache=second_kv_cache,
            )
            next_total, next_raw, _, _ = loss_for(
                next_prediction,
                next_example,
                next_source,
                next_generated,
                next_unknown,
            )
            total = 0.5 * (total + next_total)
            raw = 0.5 * (raw + next_raw)

        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        record = {
            "step": step + 1,
            "scene": scene,
            "identity": identity,
            "query_block": query_block,
            "target_block": target_block,
            "loss": float(total.detach()),
            "raw_l1": float(raw.detach()),
            "cosine": float(cosine.detach()),
            "unknown_l1": float(unknown_l1.detach()),
            "gradient_norm": gradient_norm,
            "coverage": packet.coverage.cpu().tolist(),
        }
        history.append(record)
        if step == start_step or (step + 1) % 5 == 0:
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            print(
                f"step={step + 1}/{total_steps} scene={scene} id={identity} "
                f"q={query_block} loss={record['loss']:.6f} raw={record['raw_l1']:.6f} "
                f"grad={gradient_norm:.4f} peak_GiB={peak:.2f}"
            )
        if (step + 1) % int(config.training.save_every) == 0:
            save(step + 1)

    save(total_steps)
    metrics = {
        "stage": args.stage,
        "steps": total_steps,
        "trainable_parameters": trainable_count,
        "final_records": history[-min(20, len(history)):],
    }
    with (artifact_root / f"training_{args.stage}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    (artifact_root / "RUN_STATE.md").write_text(
        f"Stage: {args.stage} complete\n"
        f"Recent checkpoint: {sidecar_path}\n"
        f"Conclusion: final training loss {history[-1]['loss']:.6f}.\n"
        f"Next: evaluate this stage before advancing.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
