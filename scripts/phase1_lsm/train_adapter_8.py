#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from phase1_lsm.adapter import MemoryPatchAdapter, load_adapter, save_adapter
from phase1_lsm.losses import exact_memory_loss
from phase1_lsm.runtime import (
    allocate_kv_cache,
    fill_teacher_cache,
    freeze_except_adapter,
    load_generator,
)
from pipeline.causal_inference import denoise_block


SAMPLE_ORDER = [
    "S0_P_seed0", "S0_P_seed1", "S0_N_seed0", "S0_N_seed1",
    "S1_P_seed0", "S1_P_seed1", "S1_N_seed0", "S1_N_seed1",
]
TENSOR_NAMES = (
    "z_A", "block18_previous", "block19_base_render16", "block19_base_mask4",
    "block19_ref16", "z_Aprime_no_memory", "direct_memory_latent16",
    "direct_memory_mask4", "projected_memory_latent16",
    "projected_memory_mask4", "projected_occupancy1", "denoise_step_inputs",
    "transition_noises", "prompt_embeds",
)


def load_selected(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in TENSOR_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-kind", choices=("direct", "projected"), required=True)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--early-stop-ratio", type=float, default=0.6)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1000:
        raise ValueError("Phase 1 allows at most 1000 optimizer steps")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    generator, config = load_generator(args.repo_root, args.checkpoint, device)
    trainable = freeze_except_adapter(generator)
    adapter = generator.model.memory_adapter
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(optimizer_parameters) != 1 or optimizer_parameters[0] is not adapter.proj.weight:
        raise AssertionError("AdamW must contain only adapter.proj.weight")

    samples_root = Path(args.samples_root)
    samples_cpu = {
        sample_id: load_selected(samples_root / sample_id / "sample.safetensors")
        for sample_id in SAMPLE_ORDER
    }
    kv_cache = allocate_kv_cache(generator, device)
    requested_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
    scheduler_steps = torch.cat(
        (generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    denoising_steps = scheduler_steps[1000 - requested_steps]

    def move_sample(sample_id: str) -> dict[str, torch.Tensor]:
        return {
            name: tensor.to(device=device, non_blocking=False)
            for name, tensor in samples_cpu[sample_id].items()
        }

    def run_sample(
        sample: dict[str, torch.Tensor], condition_enabled: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        render = torch.cat(
            (sample["block19_base_mask4"], sample["block19_base_render16"]), dim=2
        )
        conditional = {"prompt_embeds": sample["prompt_embeds"]}
        fill_teacher_cache(
            generator,
            conditional,
            kv_cache,
            sample["block19_ref16"],
            sample["block18_previous"],
            render,
        )
        if args.memory_kind == "direct":
            memory_latent = sample["direct_memory_latent16"]
            memory_mask = sample["direct_memory_mask4"]
            valid = torch.ones_like(sample["projected_occupancy1"], dtype=torch.bool)
        else:
            memory_latent = sample["projected_memory_latent16"]
            memory_mask = sample["projected_memory_mask4"]
            valid = sample["projected_occupancy1"].bool()
        condition = (
            torch.cat((memory_mask, memory_latent), dim=2)
            if condition_enabled else None
        )
        prediction, _ = denoise_block(
            generator,
            generator.scheduler,
            sample["denoise_step_inputs"][0],
            conditional,
            kv_cache,
            render_block=render,
            denoising_kv_size=1560 * 6,
            denoising_steps=denoising_steps,
            memory_condition=condition,
            transition_noises=sample["transition_noises"],
        )
        return prediction, sample["z_A"], valid

    def evaluate_all(check_bypass: bool = False) -> dict[str, float]:
        losses = {}
        with torch.no_grad():
            for sample_id in SAMPLE_ORDER:
                sample = move_sample(sample_id)
                prediction, target, valid = run_sample(sample, True)
                loss, _ = exact_memory_loss(prediction, target, valid)
                losses[sample_id] = float(loss)
                if check_bypass:
                    bypass, _, _ = run_sample(sample, False)
                    if not torch.equal(bypass, sample["z_Aprime_no_memory"]):
                        raise AssertionError(f"hard bypass changed for {sample_id}")
                del sample, prediction
        return losses

    initial_losses = evaluate_all(check_bypass=True)
    initial_mean = sum(initial_losses.values()) / len(initial_losses)
    history = []
    aggregate_history = [{"step": 0, "mean_loss": initial_mean}]
    gradient_names = None
    training_started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        sample_id = SAMPLE_ORDER[(step - 1) % len(SAMPLE_ORDER)]
        sample = move_sample(sample_id)
        optimizer.zero_grad(set_to_none=True)
        prediction, target, valid = run_sample(sample, True)
        loss, components = exact_memory_loss(prediction, target, valid)
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
        record = {
            "step": step,
            "sample_id": sample_id,
            "loss": float(loss.detach()),
            "smooth_l1": float(components["smooth_l1"].detach()),
            "latent_cosine": float(components["latent_cosine"].detach()),
        }
        history.append(record)
        if step == 1 or step % 8 == 0:
            print(json.dumps(record), flush=True)
        del sample, prediction, loss

        should_evaluate = step % args.eval_every == 0 or step == args.max_steps
        if should_evaluate:
            losses = evaluate_all()
            mean_loss = sum(losses.values()) / len(losses)
            aggregate = {"step": step, "mean_loss": mean_loss}
            aggregate_history.append(aggregate)
            print(json.dumps(aggregate), flush=True)
            if step >= 64 and mean_loss <= initial_mean * args.early_stop_ratio:
                break

    training_seconds = time.perf_counter() - training_started
    final_losses = evaluate_all(check_bypass=True)
    final_mean = sum(final_losses.values()) / len(final_losses)
    passed = final_mean < initial_mean * 0.8

    save_adapter(adapter, output_dir)
    restored = MemoryPatchAdapter().to(dtype=adapter.proj.weight.dtype)
    load_adapter(restored, output_dir / "memory_adapter.safetensors")
    roundtrip = torch.equal(
        restored.proj.weight.cpu(), adapter.proj.weight.detach().cpu()
    )
    if not roundtrip:
        raise AssertionError("adapter roundtrip failed")

    with (output_dir / "loss_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "loss_curve.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "aggregate_curve.json").write_text(
        json.dumps(aggregate_history, indent=2) + "\n", encoding="utf-8"
    )
    per_sample = {
        sample_id: {
            "initial_loss": initial_losses[sample_id],
            "final_loss": final_losses[sample_id],
            "ratio": final_losses[sample_id] / initial_losses[sample_id],
        }
        for sample_id in SAMPLE_ORDER
    }
    summary = {
        "memory_kind": args.memory_kind,
        "num_samples": 8,
        "sample_order": SAMPLE_ORDER,
        "batch_size": 1,
        "shuffle": False,
        "augmentation": False,
        "dropout": False,
        "dtype": "bfloat16",
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer_steps": len(history),
        "requested_step_indices": requested_steps.tolist(),
        "actual_model_timesteps": [float(value) for value in denoising_steps],
        "gradient_parameter_names": gradient_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "initial_mean_loss": initial_mean,
        "final_mean_loss": final_mean,
        "mean_loss_ratio": final_mean / initial_mean,
        "per_sample": per_sample,
        "hard_bypass_all_torch_equal_after_training": True,
        "adapter_roundtrip_torch_equal": roundtrip,
        "trend_passed": passed,
        "training_seconds": training_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise RuntimeError("fixed 8-sample training trend gate failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
