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
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import save_file

from phase1_lsm.adapter import load_adapter
from phase1_lsm.losses import exact_memory_loss
from phase1_lsm.runtime import allocate_kv_cache, fill_teacher_cache, load_generator
from pipeline.causal_inference import denoise_block
from utils.wan_wrapper import WanVAEWrapper


SAMPLE_ORDER = (
    "S0_P_seed0",
    "S0_P_seed1",
    "S0_N_seed0",
    "S0_N_seed1",
    "S1_P_seed0",
    "S1_P_seed1",
    "S1_N_seed0",
    "S1_N_seed1",
)
MONTAGE_SAMPLES = ("S0_P_seed0", "S1_P_seed0")
CONDITION_ORDER = ("no_memory", "correct", "mask_only", "wrong_same_mask")
TENSOR_NAMES = (
    "z_A",
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
CSV_FIELDS = (
    "sample_id",
    "condition",
    "occupancy_valid_exact_memory_loss",
    "full_frame_exact_memory_loss",
    "valid_latent_l1_to_z_A",
    "full_frame_latent_l1_to_z_A",
    "output_l1_to_no_memory",
    "runtime_ms",
)
MONTAGE_COLUMNS = (
    ("A", "A"),
    ("no_memory", "no-memory A'"),
    ("correct", "correct projected A'"),
    ("mask_only", "mask-only A'"),
    ("wrong_same_mask", "wrong same-mask A'"),
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


def latent_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> float:
    difference = (prediction.float() - target.float()).abs()
    if valid is not None:
        difference = difference[valid.bool().expand_as(difference)]
    return float(difference.mean())


def condition_inputs(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    memory_mask = sample["projected_memory_mask4"]
    projected = sample["projected_memory_latent16"]
    occupancy = sample["projected_occupancy1"].bool()
    wrong = torch.where(occupancy.expand_as(sample["z_B"]), sample["z_B"], 0)
    if torch.count_nonzero(wrong[~occupancy.expand_as(wrong)]).item() != 0:
        raise AssertionError("wrong_same_mask latent is nonzero outside occupancy")

    conditions = {
        "no_memory": None,
        "correct": torch.cat((memory_mask, projected), dim=2),
        "mask_only": torch.cat((memory_mask, torch.zeros_like(projected)), dim=2),
        "wrong_same_mask": torch.cat((memory_mask, wrong), dim=2),
    }
    for name in ("correct", "mask_only", "wrong_same_mask"):
        if not torch.equal(conditions[name][:, :, :4], memory_mask):
            raise AssertionError(f"{name} did not preserve projected_memory_mask4")
    return conditions


@torch.inference_mode()
def evaluate_samples(
    generator,
    config,
    samples_root: Path,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, dict[str, torch.Tensor]], list[float]]:
    kv_cache = allocate_kv_cache(generator, device)
    requested_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
    if requested_steps.tolist() != [1000, 750, 500, 250]:
        raise AssertionError(f"unexpected denoising steps: {requested_steps.tolist()}")
    scheduler_steps = torch.cat(
        (generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    denoising_steps = scheduler_steps[1000 - requested_steps]
    records: list[dict[str, object]] = []
    montage_outputs: dict[str, dict[str, torch.Tensor]] = {}

    for sample_id in SAMPLE_ORDER:
        sample_cpu = load_selected(samples_root / sample_id / "sample.safetensors")
        sample = {name: tensor.to(device=device) for name, tensor in sample_cpu.items()}
        render = torch.cat(
            (sample["block19_base_mask4"], sample["block19_base_render16"]), dim=2
        )
        conditional = {"prompt_embeds": sample["prompt_embeds"]}
        valid = sample["projected_occupancy1"].bool()
        full_valid = torch.ones_like(valid, dtype=torch.bool)
        conditions = condition_inputs(sample)
        predictions: dict[str, torch.Tensor] = {}
        runtimes: dict[str, float] = {}

        for condition_name in CONDITION_ORDER:
            clear_kv_cache(kv_cache)
            fill_teacher_cache(
                generator,
                conditional,
                kv_cache,
                sample["block19_ref16"],
                sample["block18_previous"],
                render,
            )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction, _ = denoise_block(
                generator,
                generator.scheduler,
                sample["denoise_step_inputs"][0],
                conditional,
                kv_cache,
                render_block=render,
                denoising_kv_size=1560 * 6,
                denoising_steps=denoising_steps,
                memory_condition=conditions[condition_name],
                transition_noises=sample["transition_noises"],
            )
            torch.cuda.synchronize(device)
            predictions[condition_name] = prediction
            runtimes[condition_name] = (time.perf_counter() - started) * 1000.0

        if not torch.equal(predictions["no_memory"], sample["z_Aprime_no_memory"]):
            max_error = float(
                (
                    predictions["no_memory"].float()
                    - sample["z_Aprime_no_memory"].float()
                ).abs().max()
            )
            raise AssertionError(
                f"{sample_id} no-memory replay differs from saved output: {max_error}"
            )

        no_memory = predictions["no_memory"]
        for condition_name in CONDITION_ORDER:
            prediction = predictions[condition_name]
            valid_loss, _ = exact_memory_loss(prediction, sample["z_A"], valid)
            full_loss, _ = exact_memory_loss(prediction, sample["z_A"], full_valid)
            records.append({
                "sample_id": sample_id,
                "condition": condition_name,
                "occupancy_valid_exact_memory_loss": float(valid_loss),
                "full_frame_exact_memory_loss": float(full_loss),
                "valid_latent_l1_to_z_A": latent_l1(prediction, sample["z_A"], valid),
                "full_frame_latent_l1_to_z_A": latent_l1(prediction, sample["z_A"]),
                "output_l1_to_no_memory": latent_l1(prediction, no_memory),
                "runtime_ms": runtimes[condition_name],
            })

        if sample_id in MONTAGE_SAMPLES:
            montage_outputs[sample_id] = {
                "A": sample["z_A"].cpu().clone().contiguous(),
                **{
                    name: predictions[name].cpu().clone().contiguous()
                    for name in CONDITION_ORDER
                },
            }
        print(json.dumps({
            "sample_id": sample_id,
            "valid_loss": {
                record["condition"]: record["occupancy_valid_exact_memory_loss"]
                for record in records
                if record["sample_id"] == sample_id
            },
        }), flush=True)
        del sample, sample_cpu, predictions

    return records, montage_outputs, [float(value) for value in denoising_steps]


def aggregate_records(records: list[dict[str, object]]) -> dict[str, object]:
    metric_names = CSV_FIELDS[2:]
    conditions = {}
    for condition_name in CONDITION_ORDER:
        selected = [record for record in records if record["condition"] == condition_name]
        conditions[condition_name] = {
            f"mean_{metric}": float(np.mean([record[metric] for record in selected]))
            for metric in metric_names
        }

    by_sample = {
        sample_id: {
            record["condition"]: record
            for record in records
            if record["sample_id"] == sample_id
        }
        for sample_id in SAMPLE_ORDER
    }
    valid_key = "occupancy_valid_exact_memory_loss"
    wins_mask = sum(
        by_sample[sample_id]["correct"][valid_key]
        < by_sample[sample_id]["mask_only"][valid_key]
        for sample_id in SAMPLE_ORDER
    )
    wins_wrong = sum(
        by_sample[sample_id]["correct"][valid_key]
        < by_sample[sample_id]["wrong_same_mask"][valid_key]
        for sample_id in SAMPLE_ORDER
    )
    wins_both = sum(
        by_sample[sample_id]["correct"][valid_key]
        < by_sample[sample_id]["mask_only"][valid_key]
        and by_sample[sample_id]["correct"][valid_key]
        < by_sample[sample_id]["wrong_same_mask"][valid_key]
        for sample_id in SAMPLE_ORDER
    )
    aggregate_key = f"mean_{valid_key}"
    correct = conditions["correct"][aggregate_key]
    mask_only = conditions["mask_only"][aggregate_key]
    wrong = conditions["wrong_same_mask"][aggregate_key]
    improvement_mask = 1.0 - correct / mask_only
    improvement_wrong = 1.0 - correct / wrong
    if correct >= mask_only or correct >= wrong:
        quantitative_verdict = "MASK-ONLY/FAIL"
    elif improvement_mask >= 0.10 and improvement_wrong >= 0.10 and wins_both >= 7:
        quantitative_verdict = "PASS_PENDING_VISUAL_REVIEW"
    else:
        quantitative_verdict = "WEAK/INCONCLUSIVE"
    return {
        "conditions": conditions,
        "correct_comparison": {
            "valid_loss_improvement_vs_mask_only": improvement_mask,
            "valid_loss_improvement_vs_wrong_same_mask": improvement_wrong,
            "correct_win_count_vs_mask_only": wins_mask,
            "correct_win_count_vs_wrong_same_mask": wins_wrong,
            "correct_win_count_both": wins_both,
            "required_improvement": 0.10,
            "required_win_count_both": 7,
        },
        "quantitative_verdict": quantitative_verdict,
    }


@torch.inference_mode()
def make_montages(
    montage_outputs: dict[str, dict[str, torch.Tensor]],
    output_dir: Path,
    wan_root: Path,
    device: torch.device,
) -> dict[str, object]:
    vae = WanVAEWrapper(str(wan_root)).to(device=device, dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)
    all_metrics = {}
    for sample_id in MONTAGE_SAMPLES:
        decoded = {}
        for key, _ in MONTAGE_COLUMNS:
            video = vae.decode_to_pixel(
                montage_outputs[sample_id][key].to(device), use_cache=False
            )
            decoded[key] = (video[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
            vae.model.clear_cache()

        frame_count = decoded["A"].shape[0]
        frame_indices = np.linspace(0, frame_count - 1, 3).round().astype(int).tolist()
        cell_width, cell_height = 416, 240
        title_height, header_height = 30, 28
        canvas = Image.new(
            "RGB",
            (
                cell_width * len(MONTAGE_COLUMNS),
                title_height + header_height + cell_height * len(frame_indices),
            ),
            color="white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (6, 8),
            "Isolated block-19 diagnostic decode (NOT a full rollout)",
            fill="black",
        )
        for column, (key, label) in enumerate(MONTAGE_COLUMNS):
            x_offset = column * cell_width
            draw.text((x_offset + 6, title_height + 7), label, fill="black")
            for row, frame_index in enumerate(frame_indices):
                frame = decoded[key][frame_index].permute(1, 2, 0).numpy()
                image = Image.fromarray((frame * 255).round().astype(np.uint8))
                image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                canvas.paste(
                    image,
                    (x_offset, title_height + header_height + row * cell_height),
                )
        canvas.save(output_dir / f"montage_{sample_id}.png")
        all_metrics[sample_id] = {
            "diagnostic_scope": "isolated block-19 decode, not full rollout",
            "decoded_frames_per_block": frame_count,
            "montage_frame_indices": frame_indices,
            "pixel_l1_to_A": {
                name: float((decoded[name] - decoded["A"]).abs().mean())
                for name in CONDITION_ORDER
            },
        }
    return all_metrics


def write_initial_report(output_dir: Path, aggregate: dict[str, object]) -> None:
    conditions = aggregate["conditions"]
    comparison = aggregate["correct_comparison"]
    lines = [
        "# Projected memory 内容判别实验",
        "",
        "本报告由 inference-only 评估脚本生成。人工 montage 观察将在执行后补充。",
        "",
        "## Aggregate",
        "",
        "| 条件 | mean valid loss | mean full loss | mean valid L1 | mean full L1 | mean Δ vs no-memory | mean runtime ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONDITION_ORDER:
        metrics = conditions[name]
        lines.append(
            f"| {name} | {metrics['mean_occupancy_valid_exact_memory_loss']:.8f} | "
            f"{metrics['mean_full_frame_exact_memory_loss']:.8f} | "
            f"{metrics['mean_valid_latent_l1_to_z_A']:.8f} | "
            f"{metrics['mean_full_frame_latent_l1_to_z_A']:.8f} | "
            f"{metrics['mean_output_l1_to_no_memory']:.8f} | "
            f"{metrics['mean_runtime_ms']:.3f} |"
        )
    lines.extend([
        "",
        "## Quantitative gate",
        "",
        f"- correct valid loss 相对 mask_only 改善：{comparison['valid_loss_improvement_vs_mask_only']:.2%}",
        f"- correct valid loss 相对 wrong_same_mask 改善：{comparison['valid_loss_improvement_vs_wrong_same_mask']:.2%}",
        f"- correct 同时胜过两者：{comparison['correct_win_count_both']}/8",
        f"- 自动量化判定：`{aggregate['quantitative_verdict']}`",
        "",
        "## 人工观察与最终判定",
        "",
        "待补充。",
        "",
        "## 边界",
        "",
        "本实验仅评估 fixed-8 训练样本上的 exact-pose、isolated block-19 projected latent 内容读取；不证明 held-out 泛化、near-view 或真正 LSM 空间记忆。",
    ])
    (output_dir / "PROJECTED_CONTENT_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-root", default="artifacts/phase1_lsm/samples"
    )
    parser.add_argument(
        "--checkpoint",
        default="/data4/daixiangting/inspatio-world/checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors",
    )
    parser.add_argument(
        "--adapter",
        default="artifacts/phase1_lsm/train/fixed8_projected/memory_adapter.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/phase1_lsm/projected_content_discrimination",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--wan-root",
        default="/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    started_wall = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    checkpoint_hash_before = sha256_file(args.checkpoint)
    adapter_hash_before = sha256_file(args.adapter)
    command = shlex.join([sys.executable, *sys.argv])
    (output_dir / "COMMAND_LOG.md").write_text(
        "# Projected content discrimination command log\n\n"
        f"- Start: `{started_wall}`\n"
        f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`\n"
        f"- Command: `{command}`\n"
        f"- Checkpoint SHA256: `{checkpoint_hash_before}`\n"
        f"- Adapter SHA256: `{adapter_hash_before}`\n",
        encoding="utf-8",
    )

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    generator, config = load_generator(args.repo_root, args.checkpoint, device)
    load_adapter(generator.model.memory_adapter, args.adapter, device=device)
    generator.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in generator.parameters()):
        raise AssertionError("inference evaluation unexpectedly has trainable parameters")
    adapter_weight_before = generator.model.memory_adapter.proj.weight.cpu().clone()

    with torch.inference_mode():
        records, montage_outputs, actual_timesteps = evaluate_samples(
            generator, config, Path(args.samples_root), device
        )
    if not torch.equal(
        adapter_weight_before, generator.model.memory_adapter.proj.weight.cpu()
    ):
        raise AssertionError("adapter weights changed during inference")
    del generator
    torch.cuda.empty_cache()

    with (output_dir / "metrics_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    for sample_id in MONTAGE_SAMPLES:
        save_file(
            montage_outputs[sample_id],
            output_dir / f"outputs_{sample_id}.safetensors",
        )

    with torch.inference_mode():
        montage_metrics = make_montages(
            montage_outputs, output_dir, Path(args.wan_root), device
        )
    aggregate = aggregate_records(records)
    checkpoint_hash_after = sha256_file(args.checkpoint)
    adapter_hash_after = sha256_file(args.adapter)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise AssertionError("base checkpoint hash changed")
    if adapter_hash_after != adapter_hash_before:
        raise AssertionError("adapter checkpoint hash changed")
    aggregate.update({
        "experiment": "projected memory content discrimination, inference-only",
        "sample_order": list(SAMPLE_ORDER),
        "num_samples": len(SAMPLE_ORDER),
        "conditions": aggregate["conditions"],
        "requested_step_indices": [1000, 750, 500, 250],
        "actual_model_timesteps": actual_timesteps,
        "optimizer_created": False,
        "all_parameters_require_grad_false": True,
        "adapter_weights_torch_equal_after_inference": True,
        "checkpoint_sha256_before_after": [
            checkpoint_hash_before,
            checkpoint_hash_after,
        ],
        "adapter_sha256_before_after": [adapter_hash_before, adapter_hash_after],
        "montage_metrics": montage_metrics,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "total_seconds": time.perf_counter() - started,
    })
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    write_initial_report(output_dir, aggregate)
    with (output_dir / "COMMAND_LOG.md").open("a", encoding="utf-8") as handle:
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
