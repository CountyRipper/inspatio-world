#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import save_file

from phase1_lsm.adapter import ADAPTER_PARAMETER_COUNT, load_adapter, save_adapter
from phase1_lsm.losses import exact_memory_loss
from phase1_lsm.nearview import preservation_invalid_mask
from phase1_lsm.runtime import (
    allocate_kv_cache,
    fill_teacher_cache,
    freeze_except_adapter,
    load_generator,
)
from pipeline.causal_inference import denoise_block
from utils.wan_wrapper import WanVAEWrapper


TENSOR_NAMES = (
    "z_B",
    "block18_previous",
    "block19_base_render16",
    "block19_base_mask4",
    "block19_ref16",
    "z_Aprime_no_memory",
    "projected_memory_latent16",
    "projected_memory_mask4",
    "projected_occupancy1",
    "denoise_step_inputs",
    "transition_noises",
    "prompt_embeds",
)
CONDITIONS = ("no_memory", "correct", "mask_only", "wrong_same_mask")
CSV_FIELDS = (
    "sample_id",
    "condition",
    "valid_exact_memory_loss",
    "valid_latent_l1",
    "invalid_preservation_loss",
    "full_composite_exact_memory_loss",
    "full_composite_latent_l1",
    "runtime_ms",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in TENSOR_NAMES}


def clear_kv_cache(kv_cache: list[dict[str, torch.Tensor]]) -> None:
    for block_cache in kv_cache:
        block_cache["k"].zero_()
        block_cache["v"].zero_()


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    expanded = mask.bool().expand_as(prediction)
    return float((prediction.float() - target.float()).abs()[expanded].mean())


def preservation_loss(
    prediction: torch.Tensor,
    no_memory: torch.Tensor,
    preserve_mask: torch.Tensor,
) -> torch.Tensor:
    expanded = preserve_mask.bool().expand_as(prediction)
    if not expanded.any():
        raise AssertionError("preservation region is empty")
    return F.smooth_l1_loss(
        prediction.float()[expanded], no_memory.detach().float()[expanded]
    )


def build_conditions(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    occupancy = sample["projected_occupancy1"].bool()
    memory_mask = sample["projected_memory_mask4"]
    projected = sample["projected_memory_latent16"]
    wrong = torch.where(occupancy.expand_as(sample["z_B"]), sample["z_B"], 0)
    if torch.count_nonzero(wrong[~occupancy.expand_as(wrong)]).item() != 0:
        raise AssertionError("wrong content is nonzero outside occupancy")
    conditions = {
        "no_memory": None,
        "correct": torch.cat((memory_mask, projected), dim=2),
        "mask_only": torch.cat((memory_mask, torch.zeros_like(projected)), dim=2),
        "wrong_same_mask": torch.cat((memory_mask, wrong), dim=2),
    }
    for name in CONDITIONS[1:]:
        if not torch.equal(conditions[name][:, :, :4], memory_mask):
            raise AssertionError(f"{name} changed projected_memory_mask4")
    return conditions


def labels_for_root(root: Path) -> tuple[str, str]:
    if (root / "samples/plus5/sample.safetensors").is_file():
        return "plus5", "minus5"
    if (root / "samples/plus10/sample.safetensors").is_file():
        return "plus10", "minus10"
    raise FileNotFoundError("could not identify near-view sample pair")


@torch.inference_mode()
def decode_full_context_montages(
    root: Path,
    labels: tuple[str, str],
    outputs: dict[str, dict[str, torch.Tensor]],
    wan_root: Path,
    device: torch.device,
) -> dict[str, object]:
    vae = WanVAEWrapper(str(wan_root)).to(device=device, dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)
    montage_audit = {}
    for label in labels:
        sample_path = root / "samples" / label / "sample.safetensors"
        with safe_open(str(sample_path), framework="pt", device="cpu") as handle:
            prefix = handle.get_tensor("latent_prefix_0_18")
        decoded_last12 = {}
        for condition in CONDITIONS:
            full_latents = torch.cat((prefix, outputs[label][condition]), dim=1)
            if tuple(full_latents.shape) != (1, 60, 16, 60, 104):
                raise AssertionError(full_latents.shape)
            decoded = vae.decode_to_pixel(
                full_latents.to(device=device, dtype=torch.bfloat16), use_cache=False
            )
            decoded = (decoded[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
            if decoded.shape[0] != 237:
                raise AssertionError(
                    f"full temporal decode must produce 237 frames, got {decoded.shape}"
                )
            decoded_last12[condition] = decoded[-12:]
            vae.model.clear_cache()

        cell_width, cell_height = 208, 120
        title_height, header_height = 30, 28
        canvas = Image.new(
            "RGB",
            (
                cell_width * len(CONDITIONS),
                title_height + header_height + cell_height * 12,
            ),
            color="white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (6, 8),
            "Full temporal decode: shared blocks 0-18 + condition block 19; last 12 frames",
            fill="black",
        )
        for column, condition in enumerate(CONDITIONS):
            x_offset = column * cell_width
            draw.text((x_offset + 6, title_height + 7), condition, fill="black")
            for row in range(12):
                frame = decoded_last12[condition][row].permute(1, 2, 0).numpy()
                image = Image.fromarray((frame * 255).round().astype(np.uint8))
                image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                canvas.paste(
                    image,
                    (x_offset, title_height + header_height + row * cell_height),
                )
        canvas.save(root / f"montage_{label}.png")
        montage_audit[label] = {
            "decode_scope": "full 60-latent temporal decode",
            "decoded_frame_count": 237,
            "montage_source_frames": list(range(225, 237)),
            "shared_prefix_latents": [0, 57],
            "condition_query_latents": [57, 60],
            "last12_pixel_l1_to_no_memory": {
                condition: float(
                    (
                        decoded_last12[condition]
                        - decoded_last12["no_memory"]
                    ).abs().mean()
                )
                for condition in CONDITIONS
            },
        }
    return montage_audit


def write_report(
    root: Path,
    labels: tuple[str, str],
    aggregate: dict[str, object],
    inference_only: bool,
) -> None:
    report_name = (
        "NEARVIEW_10DEG_REPORT_ZH.md"
        if inference_only
        else "NEARVIEW_5DEG_REPORT_ZH.md"
    )
    lines = [
        f"# {'±10° inference-only 诊断' if inference_only else 'Phase 1 near-view ±5° 过拟合验证'}",
        "",
        "## Aggregate",
        "",
        "| 条件 | mean valid loss | mean valid L1 | mean invalid preserve loss | mean composite loss | mean composite L1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = aggregate["conditions"][condition]
        lines.append(
            f"| {condition} | {metrics['mean_valid_exact_memory_loss']:.8f} | "
            f"{metrics['mean_valid_latent_l1']:.8f} | "
            f"{metrics['mean_invalid_preservation_loss']:.8f} | "
            f"{metrics['mean_full_composite_exact_memory_loss']:.8f} | "
            f"{metrics['mean_full_composite_latent_l1']:.8f} |"
        )
    if inference_only:
        lines.extend([
            "",
            "本目录是使用同一个 ±5° adapter 的 ±10° inference-only 诊断，不参与 ±5° PASS 判定。",
        ])
    else:
        gate = aggregate["pass_gate"]
        lines.extend([
            "",
            "## Quantitative gate",
            "",
            f"- 两侧 correct 均胜过全部对照：{gate['both_sides_correct_beats_all_controls']}",
            f"- correct 相对最佳对照平均 valid loss 改善：{gate['mean_valid_loss_improvement_vs_best_control']:.2%}",
            f"- 两侧 composite L1 均不超过 no-memory ×1.02：{gate['both_sides_composite_within_2_percent']}",
            f"- 量化判定：`{aggregate['quantitative_verdict']}`",
            "",
            "## 人工观察与最终判定",
            "",
            "待补充；只有两侧 full-context montage 均通过后才能确定最终 PASS。",
        ])
    lines.extend([
        "",
        "## 边界",
        "",
        "本实验只验证单场景、训练样本上的 near-view overfit，不证明 held-out 泛化或完整 LSM。",
    ])
    (root / report_name).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--wan-root",
        default="/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B",
    )
    args = parser.parse_args()
    root = Path(args.root)
    labels = labels_for_root(root)
    if args.inference_only and labels != ("plus10", "minus10"):
        raise ValueError("inference-only mode is reserved for the ±10 diagnostic")
    if not args.inference_only and labels != ("plus5", "minus5"):
        raise ValueError("training mode requires the ±5 sample pair")
    if not args.inference_only and args.max_steps != 200:
        raise ValueError("the ±5 experiment requires exactly 200 optimizer steps")
    output_names = [
        "metrics_per_sample.csv",
        "aggregate_metrics.json",
        "COMMAND_LOG.md",
        *(f"montage_{label}.png" for label in labels),
    ]
    if not args.inference_only:
        output_names.extend([
            "memory_adapter.safetensors",
            "memory_adapter_config.json",
            "NEARVIEW_5DEG_REPORT_ZH.md",
        ])
    if any((root / name).exists() for name in output_names):
        raise FileExistsError("refusing to overwrite near-view evaluation outputs")

    command_log = root / "COMMAND_LOG.md"
    command_log.write_text(
        "# Near-view command log\n\n"
        f"- Start: `{datetime.now().astimezone().isoformat()}`\n"
        f"- Command: `{shlex.join([sys.executable, *sys.argv])}`\n"
        f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`\n"
        f"- Mode: `{'inference-only' if args.inference_only else 'train-200-steps'}`\n",
        encoding="utf-8",
    )
    checkpoint_hash_before = sha256_file(args.checkpoint)
    init_adapter_hash_before = sha256_file(args.init_adapter)
    started = time.perf_counter()
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    generator, config = load_generator(args.repo_root, args.checkpoint, device)
    load_adapter(generator.model.memory_adapter, args.init_adapter, device=device)
    adapter = generator.model.memory_adapter
    if adapter.parameter_count != ADAPTER_PARAMETER_COUNT:
        raise AssertionError(adapter.parameter_count)
    if args.inference_only:
        generator.eval().requires_grad_(False)
        trainable = []
        optimizer = None
    else:
        trainable = freeze_except_adapter(generator)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
        if trainable != [adapter.proj.weight]:
            raise AssertionError("only adapter.proj.weight may be trainable")
    initial_adapter_weight = adapter.proj.weight.detach().cpu().clone()

    samples_cpu = {
        label: load_selected(root / "samples" / label / "sample.safetensors")
        for label in labels
    }
    samples = {
        label: {
            name: tensor.to(device=device)
            for name, tensor in sample.items()
        }
        for label, sample in samples_cpu.items()
    }
    conditions = {label: build_conditions(samples[label]) for label in labels}
    preserve_masks = {
        label: preservation_invalid_mask(
            samples[label]["projected_occupancy1"]
        )
        for label in labels
    }
    kv_cache = allocate_kv_cache(generator, device)
    requested_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
    if requested_steps.tolist() != [1000, 750, 500, 250]:
        raise AssertionError(requested_steps)
    scheduler_steps = torch.cat(
        (generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    denoising_steps = scheduler_steps[1000 - requested_steps]

    def run_sample(label: str, condition: torch.Tensor | None) -> torch.Tensor:
        sample = samples[label]
        render = torch.cat(
            (sample["block19_base_mask4"], sample["block19_base_render16"]),
            dim=2,
        )
        conditional = {"prompt_embeds": sample["prompt_embeds"]}
        clear_kv_cache(kv_cache)
        fill_teacher_cache(
            generator,
            conditional,
            kv_cache,
            sample["block19_ref16"],
            sample["block18_previous"],
            render,
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
        return prediction

    no_memory = {}
    with torch.inference_mode():
        for label in labels:
            no_memory[label] = run_sample(label, None)
            if not torch.equal(
                no_memory[label], samples[label]["z_Aprime_no_memory"]
            ):
                raise AssertionError(f"{label} no-memory replay changed")

    def objective(label: str, prediction: torch.Tensor) -> tuple[torch.Tensor, dict]:
        sample = samples[label]
        memory, _ = exact_memory_loss(
            prediction,
            sample["projected_memory_latent16"],
            sample["projected_occupancy1"],
        )
        preserve = preservation_loss(
            prediction, no_memory[label], preserve_masks[label]
        )
        return memory + 0.5 * preserve, {
            "memory_loss": memory,
            "preserve_loss": preserve,
        }

    training_curve = []
    gradient_names = None
    with torch.inference_mode():
        initial_objectives = []
        for label in labels:
            prediction = run_sample(label, conditions[label]["correct"])
            initial_objectives.append(float(objective(label, prediction)[0]))
    if not args.inference_only:
        for step in range(1, args.max_steps + 1):
            label = labels[(step - 1) % len(labels)]
            optimizer.zero_grad(set_to_none=True)
            prediction = run_sample(label, conditions[label]["correct"])
            loss, parts = objective(label, prediction)
            loss.backward()
            if step == 1:
                gradient_names = [
                    name
                    for name, parameter in generator.named_parameters()
                    if parameter.grad is not None
                ]
                if gradient_names != ["model.memory_adapter.proj.weight"]:
                    raise AssertionError(gradient_names)
                if torch.count_nonzero(adapter.proj.weight.grad).item() == 0:
                    raise AssertionError("adapter gradient is zero")
            optimizer.step()
            record = {
                "step": step,
                "sample_id": label,
                "loss": float(loss.detach()),
                "memory_loss": float(parts["memory_loss"].detach()),
                "preserve_loss": float(parts["preserve_loss"].detach()),
            }
            training_curve.append(record)
            if step == 1 or step % 10 == 0:
                print(json.dumps(record), flush=True)

    outputs: dict[str, dict[str, torch.Tensor]] = {}
    records = []
    final_objectives = []
    with torch.inference_mode():
        for label in labels:
            sample = samples[label]
            outputs[label] = {}
            for condition_name in CONDITIONS:
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                prediction = run_sample(
                    label, conditions[label][condition_name]
                )
                torch.cuda.synchronize(device)
                runtime_ms = (time.perf_counter() - tick) * 1000.0
                outputs[label][condition_name] = (
                    prediction.cpu().clone().contiguous()
                )
                valid_loss, _ = exact_memory_loss(
                    prediction,
                    sample["projected_memory_latent16"],
                    sample["projected_occupancy1"],
                )
                composite = torch.where(
                    sample["projected_occupancy1"].bool().expand_as(prediction),
                    sample["projected_memory_latent16"],
                    no_memory[label],
                )
                full_valid = torch.ones_like(
                    sample["projected_occupancy1"], dtype=torch.bool
                )
                composite_loss, _ = exact_memory_loss(
                    prediction, composite, full_valid
                )
                records.append({
                    "sample_id": label,
                    "condition": condition_name,
                    "valid_exact_memory_loss": float(valid_loss),
                    "valid_latent_l1": masked_l1(
                        prediction,
                        sample["projected_memory_latent16"],
                        sample["projected_occupancy1"],
                    ),
                    "invalid_preservation_loss": float(preservation_loss(
                        prediction, no_memory[label], preserve_masks[label]
                    )),
                    "full_composite_exact_memory_loss": float(composite_loss),
                    "full_composite_latent_l1": float(
                        (prediction.float() - composite.float()).abs().mean()
                    ),
                    "runtime_ms": runtime_ms,
                })
            final_prediction = outputs[label]["correct"].to(device)
            final_objectives.append(float(objective(label, final_prediction)[0]))

    if args.inference_only:
        if not torch.equal(
            initial_adapter_weight, adapter.proj.weight.detach().cpu()
        ):
            raise AssertionError("inference-only adapter weights changed")
    else:
        if torch.equal(initial_adapter_weight, adapter.proj.weight.detach().cpu()):
            raise AssertionError("near-view training did not update adapter")
        save_adapter(adapter, root)
        (root / "training_curve.json").write_text(
            json.dumps(training_curve, indent=2) + "\n", encoding="utf-8"
        )
        with (root / "training_curve.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(training_curve[0]))
            writer.writeheader()
            writer.writerows(training_curve)

    with (root / "metrics_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    for label in labels:
        save_file(outputs[label], root / f"outputs_{label}.safetensors")

    conditions_aggregate = {}
    for condition_name in CONDITIONS:
        selected = [
            record for record in records if record["condition"] == condition_name
        ]
        conditions_aggregate[condition_name] = {
            f"mean_{metric}": float(np.mean([record[metric] for record in selected]))
            for metric in CSV_FIELDS[2:]
        }
    by_sample = {
        label: {
            record["condition"]: record
            for record in records
            if record["sample_id"] == label
        }
        for label in labels
    }
    valid_key = "valid_exact_memory_loss"
    side_wins = {
        label: all(
            by_sample[label]["correct"][valid_key]
            < by_sample[label][control][valid_key]
            for control in ("no_memory", "mask_only", "wrong_same_mask")
        )
        for label in labels
    }
    best_control_condition = min(
        ("no_memory", "mask_only", "wrong_same_mask"),
        key=lambda condition: conditions_aggregate[condition][f"mean_{valid_key}"],
    )
    correct_mean = conditions_aggregate["correct"][f"mean_{valid_key}"]
    best_control_mean = conditions_aggregate[best_control_condition][
        f"mean_{valid_key}"
    ]
    improvement = 1.0 - correct_mean / best_control_mean
    composite_ratios = {
        label: (
            by_sample[label]["correct"]["full_composite_latent_l1"]
            / by_sample[label]["no_memory"]["full_composite_latent_l1"]
        )
        for label in labels
    }
    both_composite = all(ratio <= 1.02 for ratio in composite_ratios.values())
    quantitative_pass = all(side_wins.values()) and improvement >= 0.20 and both_composite
    aggregate = {
        "mode": "inference_only" if args.inference_only else "train_200_steps",
        "labels": list(labels),
        "conditions": conditions_aggregate,
        "per_side_correct_beats_all_controls": side_wins,
        "pass_gate": {
            "both_sides_correct_beats_all_controls": all(side_wins.values()),
            "best_control_condition_by_mean_valid_loss": best_control_condition,
            "mean_valid_loss_improvement_vs_best_control": improvement,
            "per_side_correct_to_no_memory_composite_l1_ratio": composite_ratios,
            "both_sides_composite_within_2_percent": both_composite,
        },
        "quantitative_verdict": (
            "INFERENCE_ONLY_DIAGNOSTIC"
            if args.inference_only
            else ("PASS_PENDING_VISUAL_REVIEW" if quantitative_pass else "FAIL")
        ),
        "training": {
            "optimizer": None if args.inference_only else "AdamW",
            "optimizer_steps": 0 if args.inference_only else len(training_curve),
            "lr": None if args.inference_only else args.lr,
            "trainable_parameter_count": 0 if args.inference_only else sum(
                parameter.numel() for parameter in trainable
            ),
            "gradient_parameter_names": gradient_names,
            "initial_mean_objective": float(np.mean(initial_objectives)),
            "final_mean_objective": float(np.mean(final_objectives)),
            "target": "projected_memory_latent16 on occupancy-valid region",
            "preservation_weight": 0.5,
            "preservation_boundary_exclusion_latent_patches": 1,
        },
        "requested_step_indices": requested_steps.tolist(),
        "actual_model_timesteps": [float(value) for value in denoising_steps],
        "checkpoint_sha256": checkpoint_hash_before,
        "initial_adapter_sha256": init_adapter_hash_before,
    }

    del generator, kv_cache, samples
    torch.cuda.empty_cache()
    montage_audit = decode_full_context_montages(
        root, labels, outputs, Path(args.wan_root), device
    )
    aggregate["montage_audit"] = montage_audit
    aggregate["checkpoint_sha256_after"] = sha256_file(args.checkpoint)
    aggregate["initial_adapter_sha256_after"] = sha256_file(args.init_adapter)
    if aggregate["checkpoint_sha256_after"] != checkpoint_hash_before:
        raise AssertionError("base checkpoint changed")
    if aggregate["initial_adapter_sha256_after"] != init_adapter_hash_before:
        raise AssertionError("initial adapter checkpoint changed")
    if not args.inference_only:
        aggregate["trained_adapter_sha256"] = sha256_file(
            root / "memory_adapter.safetensors"
        )
    aggregate["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    aggregate["total_seconds"] = time.perf_counter() - started
    (root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    write_report(root, labels, aggregate, args.inference_only)
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- Finish: `{datetime.now().astimezone().isoformat()}`\n"
            f"- Total seconds: `{aggregate['total_seconds']:.3f}`\n"
            f"- Peak VRAM GiB: `{aggregate['peak_vram_gib']:.6f}`\n"
            f"- Quantitative verdict: `{aggregate['quantitative_verdict']}`\n"
        )
    print(json.dumps(aggregate, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
