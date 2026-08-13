#!/usr/bin/env python3
"""One allowed exact-only single-layer Reader repair."""

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.world_memory.common import (
    init_single_gpu_distributed,
    load_configs,
    load_frozen_generator,
    resolve_repo_path,
)
from training.world_teacher.dataset import load_scene_record, make_bank
from training.world_teacher.evaluate_decisive_fix import (
    advance_shared_prefix,
    block_inputs,
    build_shared_exact_inputs,
    clone_cache,
    run_denoise,
    set_seed,
    target_for,
)
from world_state import RotationProjector
from world_state.runtime_v1 import (
    attach_world_state_reader_v1,
    load_world_state_reader_v1,
    save_world_state_reader_v1,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/world_teacher/decisive_fix.yaml"
    )
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def masked_l1(prediction, target, mask):
    error = (prediction.float() - target.float()).abs().mean(
        dim=2, keepdim=True
    )
    mask = mask.to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def main():
    args = parse_args()
    config, base_config = load_configs(args.config)
    scenes = ("S0", "S1")
    records = {
        scene: load_scene_record(
            resolve_repo_path(config.experiment.data_root)
            / scene
            / "paired_record.pt"
        )
        for scene in scenes
    }
    snapshots = {
        scene: torch.load(
            resolve_repo_path(config.experiment.data_root)
            / scene
            / "common_snapshot.pt",
            map_location="cpu",
            weights_only=True,
        )
        for scene in scenes
    }

    device = init_single_gpu_distributed()
    dtype = torch.bfloat16
    generator = load_frozen_generator(
        base_config, config.experiment.checkpoint, device
    )
    runtime = attach_world_state_reader_v1(
        generator.model,
        resolve_repo_path(config.experiment.direct_adapter),
        selected_layers=tuple(config.reader.ablation_layers),
        selector_width=int(config.reader.selector_width),
        confidence_threshold=0.35,
        lora_rank=int(config.reader.lora_rank),
    )
    load_world_state_reader_v1(
        generator.model,
        resolve_repo_path(config.experiment.reader_v1_checkpoint),
    )
    runtime.set_lora_enabled(False)
    generator.eval()
    generator.enable_gradient_checkpointing()

    selected_layers = tuple(int(value) for value in config.reader.selected_layers)
    if len(selected_layers) != 1:
        raise ValueError("the decisive repair trains exactly one selected layer")
    selected_layer = selected_layers[0]
    if selected_layer not in runtime.selected_layers:
        raise ValueError("selected training layer is not attached")
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    selected_reader = generator.model.blocks[selected_layer].world_reader
    selected_reader.requires_grad_(True)
    trainable = list(selected_reader.parameters())
    if not trainable or not all(value.dtype == torch.float32 for value in trainable):
        raise AssertionError("the selected Reader must be the only FP32 trainable module")
    frozen_readers = (
        generator.model.blocks[index].world_reader
        for index in runtime.selected_layers
        if index != selected_layer
    )
    if any(
        parameter.requires_grad
        for reader in frozen_readers
        for parameter in reader.parameters()
    ):
        raise AssertionError("all non-selected Reader layers must remain frozen")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = generator.get_scheduler()
    projector = RotationProjector()
    banks = {
        (scene, identity): make_bank(records[scene], identity, device, dtype)
        for scene in scenes
        for identity in ("A", "B")
    }
    shared = {
        scene: build_shared_exact_inputs(
            records[scene],
            {identity: banks[(scene, identity)] for identity in ("A", "B")},
            projector,
            config,
            device,
            dtype,
        )
        for scene in scenes
    }
    conditional = {
        scene: {
            "prompt_embeds": records[scene]["prompt_embeds"].to(
                device=device, dtype=dtype
            )
        }
        for scene in scenes
    }

    exact_starts = {}
    with torch.no_grad():
        for scene in scenes:
            record = records[scene]
            exact_starts[scene] = advance_shared_prefix(
                generator,
                scheduler,
                conditional[scene],
                record,
                snapshots[scene],
                device,
                dtype,
                int(record["second_traversal_first_block"]),
                int(config.evaluation.exact_block),
                int(config.reader.source_collar),
            )

    no_memory = {}
    with torch.no_grad():
        for scene in scenes:
            record = records[scene]
            block_index = int(config.evaluation.exact_block)
            values = block_inputs(record, block_index, device, dtype)
            start = exact_starts[scene]
            cache = clone_cache(start["kv_cache"], device, dtype)
            no_memory[scene] = run_denoise(
                generator,
                scheduler,
                conditional[scene],
                cache,
                values,
                shared[scene][block_index]["domains"].source_core,
                record["denoising_steps"].to(device),
                int(record["transition_seed"]) + block_index,
                last_pred=start["last_pred"].to(device=device, dtype=dtype),
            ).detach()
            del cache

    artifact_root = resolve_repo_path(config.experiment.artifact_root)
    checkpoint_dir = artifact_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ablation_path = (
        checkpoint_dir / f"reader_layer{selected_layer}_exact_fix_ablation.safetensors"
    )
    final_path = checkpoint_dir / f"reader_layer{selected_layer}_exact_fix.safetensors"
    state_path = checkpoint_dir / f"reader_layer{selected_layer}_training_state.pt"
    total_steps = int(config.training.steps) if args.steps is None else args.steps
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    pairs = (("S0", "A"), ("S0", "B"), ("S1", "A"), ("S1", "B"))
    for step in range(total_steps):
        scene, identity = pairs[step % len(pairs)]
        record = records[scene]
        block_index = int(config.evaluation.exact_block)
        shared_block = shared[scene][block_index]
        domains = shared_block["domains"]
        packet = shared_block["packets"][identity]
        values = block_inputs(record, block_index, device, dtype)
        target = target_for(record, identity, block_index, domains).to(
            device=device, dtype=dtype
        )
        start = exact_starts[scene]
        cache = clone_cache(start["kv_cache"], device, dtype)
        optimizer.zero_grad(set_to_none=True)
        set_seed(int(record["transition_seed"]) + block_index)
        world_context = runtime.precompute(
            packet,
            domains,
            active_layers=(selected_layer,),
            force_memory_gate=True,
        )
        prediction = run_denoise(
            generator,
            scheduler,
            conditional[scene],
            cache,
            values,
            domains.source_core,
            record["denoising_steps"].to(device),
            int(record["transition_seed"]) + block_index,
            world_context=world_context,
            last_pred=start["last_pred"].to(device=device, dtype=dtype),
            with_grad=True,
        )
        memory_loss = masked_l1(prediction, target, domains.memory)
        unknown_loss = masked_l1(
            prediction, no_memory[scene], domains.unknown
        )
        loss = memory_loss + unknown_loss
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        del cache
        values_log = {
            "step": step + 1,
            "scene": scene,
            "identity": identity,
            "loss": float(loss.detach()),
            "memory_l1": float(memory_loss.detach()),
            "unknown_l1": float(unknown_loss.detach()),
            "gradient_norm": gradient_norm,
        }
        history.append(values_log)
        if step == 0 or (step + 1) % 5 == 0:
            print(json.dumps(values_log))
        if (step + 1) % int(config.training.save_every) == 0:
            save_world_state_reader_v1(
                generator.model,
                checkpoint_dir / f"reader_layer{selected_layer}_step{step + 1}.safetensors",
                metadata={
                    "stage": "decisive-exact-fix",
                    "step": step + 1,
                    "selected_layer": selected_layer,
                    "forced_memory_gate": True,
                    "lora_enabled": False,
                },
            )

    save_world_state_reader_v1(
        generator.model,
        ablation_path,
        metadata={
            "stage": "decisive-exact-fix",
            "step": total_steps,
            "selected_layer": selected_layer,
            "forced_memory_gate": True,
            "lora_enabled": False,
        },
    )
    save_world_state_reader_v1(
        generator.model,
        final_path,
        selected_layers=(selected_layer,),
        include_lora=False,
        metadata={
            "stage": "decisive-exact-fix",
            "step": total_steps,
            "selected_layer": selected_layer,
            "forced_memory_gate": True,
            "lora_enabled": False,
            "deployment_sidecar": True,
        },
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "history": history,
            "step": total_steps,
            "selected_layer": selected_layer,
        },
        state_path,
    )
    with (artifact_root / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        final_by_pair = {}
        for scene, identity in pairs:
            latest = next(
                value
                for value in reversed(history)
                if value["scene"] == scene and value["identity"] == identity
            )
            final_by_pair[f"{scene}_{identity}"] = latest["memory_l1"]
        json.dump(
            {
                "selected_layer": selected_layer,
                "steps": total_steps,
                "final_own_target_M_L1": final_by_pair,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
