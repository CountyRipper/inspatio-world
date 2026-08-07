#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import random
import time

import torch

from phase1_lsm.adapter import ADAPTER_PARAMETER_COUNT, load_adapter, save_adapter
from phase1_lsm.data_prep import sha256_file
from phase1_lsm.losses import exact_memory_loss
from phase1_lsm.nearview import invalid_raw_l1
from phase1_lsm.runtime import allocate_kv_cache, freeze_except_adapter, load_generator
from phase2_memory.manifest import load_manifest
from phase2_memory.rollout import replay_return_record
from scripts.phase2_memory.common import (
    finish_distributed,
    init_distributed,
    json_dump,
    load_return_record,
    load_tensor_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/phase2_memory_manifest.json")
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", default="train", choices=("train", "heldout"))
    parser.add_argument("--group-id", action="append")
    parser.add_argument("--phase2a-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument("--history-truncation-probability", type=float, default=0.25)
    parser.add_argument("--anchoring-strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= args.history_truncation_probability <= 1.0:
        raise ValueError("history truncation probability must be in [0,1]")
    if not 0.0 <= args.anchoring_strength <= 1.0:
        raise ValueError("anchoring strength must be in [0,1]")
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    manifest = load_manifest(args.manifest)
    groups = manifest.select(
        split=args.split,
        group_ids=None if args.group_id is None else set(args.group_id),
        phase2a_only=args.phase2a_only,
    )
    if not groups:
        raise ValueError("no training groups selected")
    records_root = Path(args.records_root)
    cached = {}
    record_keys = []
    for group in groups:
        group_root = records_root / "groups" / group.group_id
        metadata_path = group_root / "rollout.json"
        tensor_path = group_root / "rollout.safetensors"
        if not metadata_path.is_file() or not tensor_path.is_file():
            raise FileNotFoundError(f"missing rollout record for {group.group_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tensors = load_tensor_file(tensor_path)
        cached[group.group_id] = (metadata, tensors)
        record_keys.extend(
            (group.group_id, index)
            for index in range(len(metadata["records"]))
        )
    if not record_keys:
        raise AssertionError("no return records")

    device = init_distributed()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = Path(args.checkpoint).resolve()
    init_adapter = Path(args.init_adapter).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    init_hash = sha256_file(init_adapter)
    generator, config = load_generator(
        Path(args.repo_root).resolve(), checkpoint, device
    )
    load_adapter(generator.model.memory_adapter, init_adapter, device=device)
    adapter = generator.model.memory_adapter
    trainable = freeze_except_adapter(generator)
    if trainable != [adapter.proj.weight] or adapter.parameter_count != ADAPTER_PARAMETER_COUNT:
        raise AssertionError("unexpected trainable parameters")
    generator.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    kv_cache = allocate_kv_cache(generator, device)
    requested = torch.tensor(config.denoising_step_list, dtype=torch.long)
    scheduler_steps = torch.cat((
        generator.scheduler.timesteps.cpu(),
        torch.tensor([0.0]),
    ))
    denoising_steps = scheduler_steps[1000 - requested]
    if requested.tolist() != [1000, 750, 500, 250]:
        raise AssertionError("unexpected denoising steps")

    rng = random.Random(args.seed)
    shuffled = list(record_keys)
    rng.shuffle(shuffled)
    curve = []
    gradient_names = None
    initial_weight = adapter.proj.weight.detach().cpu().clone()
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        if (step - 1) % len(shuffled) == 0 and step > 1:
            rng.shuffle(shuffled)
        group_id, record_index = shuffled[(step - 1) % len(shuffled)]
        metadata, tensors = cached[group_id]
        conditional, record = load_return_record(
            tensors, metadata, record_index, device
        )
        truncate = rng.random() < args.history_truncation_probability
        baseline = (
            record.no_memory_truncated if truncate else record.no_memory_full
        )
        condition = torch.cat((
            record.memory_mask4,
            record.projected_memory,
        ), dim=2).detach()
        gate = record.occupancy.float().detach()
        optimizer.zero_grad(set_to_none=True)
        prediction = replay_return_record(
            generator,
            generator.scheduler,
            denoising_steps,
            conditional,
            record,
            memory_condition=condition,
            memory_gate=gate,
            anchoring_strength=args.anchoring_strength,
            truncate_history=truncate,
            backpropagate_all_steps=True,
            kv_cache=kv_cache,
        )
        memory_loss, components = exact_memory_loss(
            prediction,
            record.projected_memory,
            record.occupancy,
        )
        preserve_loss = invalid_raw_l1(
            prediction,
            baseline,
            record.occupancy,
        )
        loss = memory_loss + args.preservation_weight * preserve_loss
        loss.backward()
        if step == 1:
            gradient_names = [
                name for name, parameter in generator.named_parameters()
                if parameter.grad is not None
            ]
            if gradient_names != ["model.memory_adapter.proj.weight"]:
                raise AssertionError(gradient_names)
            if torch.count_nonzero(adapter.proj.weight.grad).item() == 0:
                raise AssertionError("adapter gradient is zero")
        optimizer.step()
        row = {
            "step": step,
            "group_id": group_id,
            "record_index": record_index,
            "block": record.block,
            "memory_id": record.memory_id,
            "history_truncated": truncate,
            "loss": float(loss.detach()),
            "memory_loss": float(memory_loss.detach()),
            "smooth_l1": float(components["smooth_l1"].detach()),
            "latent_cosine": float(components["latent_cosine"].detach()),
            "invalid_preservation_l1": float(preserve_loss.detach()),
        }
        curve.append(row)
        if step == 1 or step % 10 == 0:
            print(json.dumps(row), flush=True)
        del conditional, record, baseline, condition, gate, prediction, loss

    if torch.equal(initial_weight, adapter.proj.weight.detach().cpu()):
        raise AssertionError("adapter did not update")
    generator.eval().requires_grad_(False)
    save_adapter(adapter, output_root)
    with (output_root / "training_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    json_dump(output_root / "training_curve.json", curve)
    summary = {
        "manifest": str(manifest.path),
        "records_root": str(records_root.resolve()),
        "group_count": len(groups),
        "return_record_count": len(record_keys),
        "split": args.split,
        "phase2a_only": args.phase2a_only,
        "optimizer": "AdamW",
        "lr": args.lr,
        "steps": args.max_steps,
        "preservation_weight": args.preservation_weight,
        "history_truncation_probability": args.history_truncation_probability,
        "anchoring_strength": args.anchoring_strength,
        "historical_latents_and_states_detached": True,
        "backpropagated_denoise_steps": [0, 1, 2, 3],
        "requested_denoise_indices": requested.tolist(),
        "actual_model_timesteps": [float(value) for value in denoising_steps],
        "gradient_parameter_names": gradient_names,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable
        ),
        "backbone_frozen": True,
        "vae_frozen": True,
        "text_encoder_frozen": True,
        "future_return_gt_used_as_memory": False,
        "checkpoint_sha256_before": checkpoint_hash,
        "checkpoint_sha256_after": sha256_file(checkpoint),
        "init_adapter_sha256_before": init_hash,
        "init_adapter_sha256_after": sha256_file(init_adapter),
        "trained_adapter_sha256": sha256_file(
            output_root / "memory_adapter.safetensors"
        ),
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "seconds": time.perf_counter() - started,
    }
    if (
        summary["checkpoint_sha256_after"] != checkpoint_hash
        or summary["init_adapter_sha256_after"] != init_hash
    ):
        raise AssertionError("immutable checkpoint changed")
    json_dump(output_root / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    del generator, optimizer, kv_cache, cached
    gc.collect()
    torch.cuda.empty_cache()
    finish_distributed()


if __name__ == "__main__":
    main()
