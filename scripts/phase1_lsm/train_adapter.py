#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from phase1_lsm.adapter import MemoryPatchAdapter, load_adapter, save_adapter
from phase1_lsm.losses import exact_memory_loss
from phase1_lsm.runtime import (
    allocate_kv_cache,
    fill_teacher_cache,
    freeze_except_adapter,
    load_generator,
)
from pipeline.causal_inference import denoise_block


TENSOR_NAMES = (
    "z_A", "z_B", "block18_previous", "block19_base_render16",
    "block19_base_mask4", "block19_ref16", "z_Aprime_no_memory",
    "direct_memory_latent16", "direct_memory_mask4",
    "projected_memory_latent16", "projected_memory_mask4",
    "projected_occupancy1", "denoise_step_inputs", "transition_noises",
    "prompt_embeds",
)


def load_sample(path: str | Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in TENSOR_NAMES}


def actual_timesteps(generator, requested: torch.Tensor) -> torch.Tensor:
    scheduler_steps = torch.cat(
        (generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    return scheduler_steps[1000 - requested]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-kind", choices=("direct", "projected"), required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--early-stop-ratio", type=float, default=0.2)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output_dir}")
    output_dir.mkdir(parents=True)
    if args.max_steps < 1 or args.max_steps > 1000:
        raise ValueError("Phase 1 allows 1..1000 optimizer steps")

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
    if [parameter for group in optimizer.param_groups for parameter in group["params"]] != trainable:
        raise AssertionError("AdamW contains parameters outside the adapter")

    sample_cpu = load_sample(args.sample)
    sample = {
        name: tensor.to(device=device)
        for name, tensor in sample_cpu.items()
    }
    render = torch.cat(
        (sample["block19_base_mask4"], sample["block19_base_render16"]), dim=2
    )
    conditional = {"prompt_embeds": sample["prompt_embeds"]}
    kv_cache = allocate_kv_cache(generator, device)
    fill_teacher_cache(
        generator,
        conditional,
        kv_cache,
        sample["block19_ref16"],
        sample["block18_previous"],
        render,
    )
    requested_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
    denoising_steps = actual_timesteps(generator, requested_steps)
    if requested_steps.tolist() != [1000, 750, 500, 250]:
        raise AssertionError(requested_steps)

    if args.memory_kind == "direct":
        memory_latent = sample["direct_memory_latent16"]
        memory_mask = sample["direct_memory_mask4"]
        valid = torch.ones_like(sample["projected_occupancy1"], dtype=torch.bool)
    else:
        memory_latent = sample["projected_memory_latent16"]
        memory_mask = sample["projected_memory_mask4"]
        valid = sample["projected_occupancy1"].bool()
    memory_condition = torch.cat((memory_mask, memory_latent), dim=2)
    wrong_condition = torch.cat(
        (sample["direct_memory_mask4"], sample["z_B"]), dim=2
    )

    def four_step(condition: torch.Tensor | None) -> torch.Tensor:
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
        return prediction

    with torch.no_grad():
        no_memory_before = four_step(None)
        zero_memory_prediction = four_step(memory_condition)
    if not torch.equal(no_memory_before, sample["z_Aprime_no_memory"]):
        max_error = float(
            (no_memory_before.float() - sample["z_Aprime_no_memory"].float()).abs().max()
        )
        raise AssertionError(f"empty-memory replay failed: {max_error}")
    if not torch.equal(zero_memory_prediction, no_memory_before):
        raise AssertionError("zero-init memory path differs from hard bypass")
    initial_loss, _ = exact_memory_loss(no_memory_before, sample["z_A"], valid)
    initial_loss_value = float(initial_loss)

    history = []
    training_started = time.perf_counter()
    gradient_names: list[str] | None = None
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = four_step(memory_condition)
        loss, components = exact_memory_loss(prediction, sample["z_A"], valid)
        loss.backward()
        if step == 1:
            gradient_names = [
                name for name, parameter in generator.named_parameters()
                if parameter.grad is not None
            ]
            if gradient_names != ["model.memory_adapter.proj.weight"]:
                raise AssertionError(f"unexpected gradient parameters: {gradient_names}")
            if not torch.isfinite(adapter.proj.weight.grad).all():
                raise AssertionError("adapter gradient is non-finite")
            if torch.count_nonzero(adapter.proj.weight.grad).item() == 0:
                raise AssertionError("adapter gradient is identically zero")
        grad_norm = float(adapter.proj.weight.grad.float().norm())
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "smooth_l1": float(components["smooth_l1"].detach()),
            "latent_cosine": float(components["latent_cosine"].detach()),
            "grad_norm": grad_norm,
        }
        history.append(record)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(record), flush=True)
        if step >= 10 and record["loss"] <= initial_loss_value * args.early_stop_ratio:
            break

    training_seconds = time.perf_counter() - training_started
    with torch.no_grad():
        trained_prediction = four_step(memory_condition)
        projected_prediction = four_step(torch.cat(
            (sample["projected_memory_mask4"], sample["projected_memory_latent16"]), dim=2
        ))
        direct_prediction = four_step(torch.cat(
            (sample["direct_memory_mask4"], sample["direct_memory_latent16"]), dim=2
        ))
        wrong_prediction = four_step(wrong_condition)
        no_memory_after = four_step(None)
    if not torch.equal(no_memory_after, sample["z_Aprime_no_memory"]):
        raise AssertionError("trained adapter changed the hard-bypass output")
    final_loss, final_parts = exact_memory_loss(trained_prediction, sample["z_A"], valid)
    no_memory_loss, _ = exact_memory_loss(no_memory_after, sample["z_A"], valid)
    wrong_loss, _ = exact_memory_loss(wrong_prediction, sample["z_A"], valid)
    passed = (
        float(final_loss) <= initial_loss_value * args.early_stop_ratio
        and float(final_loss) < float(no_memory_loss)
    )

    timings_without = []
    timings_with = []
    for condition, bucket in ((None, timings_without), (memory_condition, timings_with)):
        for _ in range(5):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.no_grad():
                four_step(condition)
            torch.cuda.synchronize(device)
            bucket.append((time.perf_counter() - tick) * 1000.0)
    base_ms = statistics.median(timings_without)
    memory_ms = statistics.median(timings_with)

    save_adapter(adapter, output_dir)
    restored = MemoryPatchAdapter().to(dtype=adapter.proj.weight.dtype)
    load_adapter(restored, output_dir / "memory_adapter.safetensors")
    roundtrip_equal = torch.equal(
        restored.proj.weight.cpu(), adapter.proj.weight.detach().cpu()
    )
    if not roundtrip_equal:
        raise AssertionError("adapter-only save/load roundtrip failed")
    save_file(
        {
            "A": sample["z_A"].detach().cpu(),
            "no_memory_Aprime": no_memory_after.detach().cpu(),
            "trained_Aprime": trained_prediction.detach().cpu(),
            "direct_memory_Aprime": direct_prediction.detach().cpu(),
            "projected_memory_Aprime": projected_prediction.detach().cpu(),
            "wrong_memory_Aprime": wrong_prediction.detach().cpu(),
        },
        output_dir / "training_outputs.safetensors",
    )
    with (output_dir / "loss_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "loss_curve.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "memory_kind": args.memory_kind,
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
        "initial_loss": initial_loss_value,
        "final_loss": float(final_loss),
        "final_smooth_l1": float(final_parts["smooth_l1"]),
        "final_latent_cosine": float(final_parts["latent_cosine"]),
        "no_memory_loss": float(no_memory_loss),
        "wrong_memory_loss": float(wrong_loss),
        "loss_ratio": float(final_loss) / initial_loss_value,
        "hard_bypass_torch_equal_after_training": True,
        "adapter_roundtrip_torch_equal": roundtrip_equal,
        "overfit_passed": passed,
        "training_seconds": training_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "base_four_step_block_ms_median": base_ms,
        "memory_four_step_block_ms_median": memory_ms,
        "adapter_added_ms_per_block": memory_ms - base_ms,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise RuntimeError("single-sample overfit gate failed; do not expand to 8 samples")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
