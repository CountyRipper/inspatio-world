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

from phase1_lsm.adapter import (
    ADAPTER_PARAMETER_COUNT,
    MemoryPatchAdapter,
    gated_adapter_residual,
    load_adapter,
    patch_occupancy_gate,
    save_adapter,
)
from phase1_lsm.losses import exact_memory_loss
from phase1_lsm.nearview import invalid_raw_l1
from phase1_lsm.runtime import (
    allocate_kv_cache,
    fill_teacher_cache,
    freeze_except_adapter,
    load_generator,
)
from pipeline.causal_inference import denoise_block
from utils.wan_wrapper import WanVAEWrapper


LABELS = ("plus5", "minus5")
CONDITIONS = ("no_memory", "correct", "mask_only", "wrong_same_mask")
TENSOR_NAMES = (
    "z_A", "z_B", "latent_prefix_0_18", "block18_previous",
    "block19_base_render16", "block19_base_mask4", "block19_ref16",
    "z_Aprime_no_memory", "projected_memory_latent16",
    "projected_memory_mask4", "projected_occupancy1",
    "denoise_step_inputs", "transition_noises", "prompt_embeds",
)
CSV_FIELDS = (
    "sample_id", "condition", "valid_raw_l1", "valid_exact_loss",
    "invalid_raw_l1", "full_composite_raw_l1", "full_composite_loss",
    "runtime_ms",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sample(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in TENSOR_NAMES}


def clear_kv_cache(kv_cache: list[dict[str, torch.Tensor]]) -> None:
    for item in kv_cache:
        item["k"].zero_()
        item["v"].zero_()


def masked_raw_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask.bool().expand_as(prediction)
    if not selected.any():
        raise AssertionError("masked raw L1 region is empty")
    return (prediction.float() - target.float()).abs()[selected].mean()


def build_conditions(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    occupancy = sample["projected_occupancy1"].bool()
    projected = sample["projected_memory_latent16"]
    memory_mask = sample["projected_memory_mask4"]
    wrong = torch.where(occupancy.expand_as(sample["z_B"]), sample["z_B"], 0)
    if torch.count_nonzero(wrong[~occupancy.expand_as(wrong)]).item() != 0:
        raise AssertionError("wrong_same_mask content leaked outside occupancy")
    result = {
        "no_memory": None,
        "correct": torch.cat((memory_mask, projected), dim=2),
        "mask_only": torch.cat((memory_mask, torch.zeros_like(projected)), dim=2),
        "wrong_same_mask": torch.cat((memory_mask, wrong), dim=2),
    }
    for name in CONDITIONS[1:]:
        if not torch.equal(result[name][:, :, :4], memory_mask):
            raise AssertionError(f"{name} changed projected_memory_mask4")
    return result


def hard_gate_audit(
    adapter: MemoryPatchAdapter,
    samples: dict[str, dict[str, torch.Tensor]],
    conditions: dict[str, dict[str, torch.Tensor | None]],
) -> dict[str, object]:
    audit: dict[str, object] = {"passed": True, "parameter_count": adapter.parameter_count, "samples": {}}
    if adapter.parameter_count != ADAPTER_PARAMETER_COUNT:
        raise AssertionError(adapter.parameter_count)
    for label in LABELS:
        occupancy_bf1hw = samples[label]["projected_occupancy1"].float()
        occupancy_b1fhw = occupancy_bf1hw.permute(0, 2, 1, 3, 4).contiguous()
        expected_gate = patch_occupancy_gate(occupancy_b1fhw)
        condition_audits = {}
        gates = []
        for condition_name in CONDITIONS[1:]:
            condition = conditions[label][condition_name]
            residual, gate = gated_adapter_residual(
                adapter,
                condition.permute(0, 2, 1, 3, 4).contiguous(),
                occupancy_b1fhw,
            )
            gates.append(gate)
            invalid = (gate == 0).expand_as(residual)
            invalid_values = residual[invalid]
            condition_audits[condition_name] = {
                "gate_equal_expected": bool(torch.equal(gate, expected_gate)),
                "invalid_residual_max_abs": float(invalid_values.abs().max()) if invalid_values.numel() else 0.0,
                "invalid_residual_count_nonzero": int(torch.count_nonzero(invalid_values)),
            }
        if not all(torch.equal(gates[0], gate) for gate in gates[1:]):
            raise AssertionError(f"{label} conditions do not share G_patch")
        zero_residual, zero_gate = gated_adapter_residual(
            adapter,
            conditions[label]["correct"].permute(0, 2, 1, 3, 4).contiguous(),
            torch.zeros_like(occupancy_b1fhw),
        )
        zero_exact = torch.equal(zero_residual, torch.zeros_like(zero_residual))
        if not zero_exact or torch.count_nonzero(zero_gate).item() != 0:
            raise AssertionError("all-zero occupancy adapter addition is not exact zero")
        if any(
            item["invalid_residual_max_abs"] != 0.0
            or item["invalid_residual_count_nonzero"] != 0
            or not item["gate_equal_expected"]
            for item in condition_audits.values()
        ):
            raise AssertionError(f"{label} hard-gate audit failed")
        audit["samples"][label] = {
            "g_patch_shape": list(expected_gate.shape),
            "g_patch_unique": sorted(float(x) for x in torch.unique(expected_gate)),
            "g_patch_valid_fraction": float(expected_gate.float().mean()),
            "conditions_share_torch_equal_gate": True,
            "conditions": condition_audits,
            "zero_occupancy_addition_torch_equal_zero": zero_exact,
        }
    return audit


def _to_image(frame: torch.Tensor, width: int = 208, height: int = 120) -> Image.Image:
    array = (frame.permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(array).resize((width, height), Image.Resampling.LANCZOS)


@torch.inference_mode()
def decode_montages(
    root: Path,
    samples_cpu: dict[str, dict[str, torch.Tensor]],
    outputs: dict[str, dict[str, torch.Tensor]],
    wan_root: Path,
    device: torch.device,
) -> dict[str, object]:
    vae = WanVAEWrapper(str(wan_root)).to(device=device, dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)
    audit = {}
    columns = ("projected-A composite reference (not GT)", *CONDITIONS)
    for label in LABELS:
        sample = samples_cpu[label]
        occupancy = sample["projected_occupancy1"].bool()
        composite = torch.where(
            occupancy.expand_as(outputs[label]["no_memory"]),
            sample["projected_memory_latent16"],
            outputs[label]["no_memory"],
        )
        query_by_column = {
            columns[0]: composite,
            **outputs[label],
        }
        decoded = {}
        for column in columns:
            full = torch.cat((sample["latent_prefix_0_18"], query_by_column[column]), dim=1)
            if tuple(full.shape) != (1, 60, 16, 60, 104):
                raise AssertionError(full.shape)
            pixels = vae.decode_to_pixel(
                full.to(device=device, dtype=torch.bfloat16), use_cache=False
            )
            pixels = (pixels[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
            if pixels.shape[0] != 237:
                raise AssertionError(pixels.shape)
            decoded[column] = pixels[-12:]
            vae.model.clear_cache()

        cell_w, cell_h = 208, 120
        title_h, header_h = 38, 34
        canvas = Image.new("RGB", (cell_w * len(columns), title_h + header_h + 12 * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 7), "Full-context VAE decode, fixed frames 225:237; reference is not GT", fill="black")
        for col, name in enumerate(columns):
            draw.text((col * cell_w + 5, title_h + 6), name, fill="black")
            for row in range(12):
                canvas.paste(_to_image(decoded[name][row]), (col * cell_w, title_h + header_h + row * cell_h))
        canvas.save(root / f"montage_{label}.png")

        occ_spatial = F.interpolate(
            occupancy[0].float(), size=decoded[columns[0]].shape[-2:], mode="nearest"
        )[:, 0].bool().repeat_interleave(4, dim=0)
        if occ_spatial.shape[0] != 12:
            raise AssertionError(occ_spatial.shape)
        reference = decoded[columns[0]]
        error_maps = {
            name: (decoded[name] - reference).abs().mean(dim=1)
            for name in columns
        }
        vmax = max(
            float(error_maps[name][occ_spatial].max())
            for name in columns[1:]
        )
        vmax = max(vmax, 1e-8)
        overlay = Image.new("RGB", (cell_w * len(columns), title_h + header_h + 12 * cell_h), "white")
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.text((6, 7), f"Same overlap boundary; overlap error heatmap, shared color max={vmax:.6f}", fill="black")
        for col, name in enumerate(columns):
            overlay_draw.text((col * cell_w + 5, title_h + 6), name, fill="black")
            for row in range(12):
                rgb = decoded[name][row].permute(1, 2, 0).numpy().copy()
                mask = occ_spatial[row]
                normalized = (error_maps[name][row] / vmax).clamp(0, 1)
                heat = normalized.numpy()[..., None]
                inside = mask.numpy()[..., None]
                rgb = np.where(inside, rgb * (1 - 0.62 * heat) + np.array([1.0, 0.0, 0.0]) * (0.62 * heat), rgb)
                mask_float = mask.float()[None, None]
                interior = (-F.max_pool2d(-mask_float, kernel_size=3, stride=1, padding=1))[0, 0].bool()
                boundary = (mask & ~interior).numpy()
                rgb[boundary] = np.array([0.0, 1.0, 1.0])
                image = Image.fromarray((rgb.clip(0, 1) * 255).round().astype(np.uint8)).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                overlay.paste(image, (col * cell_w, title_h + header_h + row * cell_h))
        overlay.save(root / f"overlap_montage_{label}.png")
        audit[label] = {
            "decode_scope": "full 60-latent temporal decode",
            "fixed_output_frames": [225, 237],
            "reference": "projected z_A in overlap, no-memory outside; not GT",
            "columns": list(columns),
            "overlap_heatmap_common_vmax": vmax,
            "pixel_error_to_composite_reference": {
                name: {
                    "full_mean_abs": float(error_maps[name].mean()),
                    "overlap_mean_abs": float(error_maps[name][occ_spatial].mean()),
                    "outside_mean_abs": float(error_maps[name][~occ_spatial].mean()),
                    "residual_temporal_flicker": float(
                        (
                            (decoded[name] - reference)[1:]
                            - (decoded[name] - reference)[:-1]
                        ).abs().mean()
                    ),
                }
                for name in columns
            },
        }
    del vae
    torch.cuda.empty_cache()
    return audit


def write_report(root: Path, aggregate: dict[str, object]) -> None:
    lines = [
        "# Phase 1 shared-A hard-gate ±5° 最小修正实验",
        "",
        f"量化判定：`{aggregate['quantitative_verdict']}`。视觉判定需人工查看两套 full-context montage。",
        "",
        "## 每侧验收",
        "",
        "| 方向 | content-read | preservation | correct valid raw L1 / best control | correct/no-memory composite raw L1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, gate in aggregate["per_side_acceptance"].items():
        lines.append(
            f"| {label} | {gate['content_read_pass']} | {gate['preservation_pass']} | "
            f"{gate['correct_to_best_control_valid_raw_l1_ratio']:.6f} | "
            f"{gate['correct_to_no_memory_composite_raw_l1_ratio']:.6f} |"
        )
    lines.extend([
        "",
        "## Audit",
        "",
        f"- true shared-A：{aggregate['shared_A_audit_passed']}",
        f"- non-identity projection：{aggregate['projection_audit_passed']}",
        f"- adapter output hard gate：{aggregate['hard_gate_audit_passed']}",
        f"- no-memory replay torch.equal：{aggregate['no_memory_replay_torch_equal']}",
        f"- adapter save/load torch.equal：{aggregate['adapter_roundtrip_torch_equal']}",
        "",
        "## 视觉结论",
        "",
        "待人工检查固定 225:237 帧主 montage 与 overlap montage 后填写。reference 为 projected-A composite reference，不是 GT。",
        "",
        "## 边界",
        "",
        "本实验仅验证 S0、seed0、训练内 ±5° exact shared-A near-view overfit；不证明 held-out 泛化、near-view 普适性、动态更新或完整 LSM 空间记忆。若 preservation 仍失败，下一步仅建议检查四个 denoise steps 是否完整参与反传，本轮未实施。",
    ])
    (root / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument("--wan-root", default="/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B")
    args = parser.parse_args()
    if (args.max_steps, args.lr, args.preservation_weight) != (200, 1e-3, 0.5):
        raise ValueError("fixed experiment requires steps=200, lr=1e-3, preservation_weight=0.5")
    root = Path(args.root)
    for label in LABELS:
        if not (root / "samples" / label / "sample.safetensors").is_file():
            raise FileNotFoundError(label)
    forbidden_existing = (
        "memory_adapter.safetensors", "metrics_per_sample.csv", "aggregate_metrics.json",
        "training_curve.csv", "training_curve.json", "REPORT_ZH.md",
    )
    if any((root / name).exists() for name in forbidden_existing):
        raise FileExistsError("refusing to overwrite training/evaluation outputs")

    command_log = root / "COMMAND_LOG.md"
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Train/eval\n\n"
            f"- Start: `{datetime.now().astimezone().isoformat()}`\n"
            f"- In-process command: `{shlex.join([sys.executable, *sys.argv])}`\n"
            "- Fixed: AdamW, lr=1e-3, 200 steps, preservation_weight=0.5, last denoise step backprop\n"
        )
    started = time.perf_counter()
    checkpoint_hash = sha256_file(args.checkpoint)
    init_hash = sha256_file(args.init_adapter)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    generator, config = load_generator(args.repo_root, args.checkpoint, device)
    load_adapter(generator.model.memory_adapter, args.init_adapter, device=device)
    adapter = generator.model.memory_adapter
    trainable = freeze_except_adapter(generator)
    if trainable != [adapter.proj.weight] or adapter.parameter_count != ADAPTER_PARAMETER_COUNT:
        raise AssertionError("trainable set or adapter count changed")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    initial_weight = adapter.proj.weight.detach().cpu().clone()

    samples_cpu = {label: load_sample(root / "samples" / label / "sample.safetensors") for label in LABELS}
    samples = {label: {name: tensor.to(device) for name, tensor in sample.items()} for label, sample in samples_cpu.items()}
    conditions = {label: build_conditions(samples[label]) for label in LABELS}
    hard_audit = hard_gate_audit(adapter, samples, conditions)
    (root / "hard_gate_audit.json").write_text(json.dumps(hard_audit, indent=2) + "\n", encoding="utf-8")

    kv_cache = allocate_kv_cache(generator, device)
    requested_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
    if requested_steps.tolist() != [1000, 750, 500, 250]:
        raise AssertionError(requested_steps)
    scheduler_steps = torch.cat((generator.scheduler.timesteps.cpu(), torch.tensor([0.0])))
    denoising_steps = scheduler_steps[1000 - requested_steps]

    def run_sample(label: str, condition: torch.Tensor | None) -> torch.Tensor:
        sample = samples[label]
        render = torch.cat((sample["block19_base_mask4"], sample["block19_base_render16"]), dim=2)
        conditional = {"prompt_embeds": sample["prompt_embeds"]}
        clear_kv_cache(kv_cache)
        fill_teacher_cache(generator, conditional, kv_cache, sample["block19_ref16"], sample["block18_previous"], render)
        prediction, _ = denoise_block(
            generator, generator.scheduler, sample["denoise_step_inputs"][0], conditional, kv_cache,
            render_block=render, denoising_kv_size=1560 * 6,
            denoising_steps=denoising_steps, memory_condition=condition,
            memory_gate=None if condition is None else sample["projected_occupancy1"].float(),
            transition_noises=sample["transition_noises"],
        )
        return prediction

    no_memory = {}
    no_memory_equal = {}
    with torch.inference_mode():
        for label in LABELS:
            no_memory[label] = run_sample(label, None)
            no_memory_equal[label] = bool(torch.equal(no_memory[label], samples[label]["z_Aprime_no_memory"]))
            if not no_memory_equal[label]:
                raise AssertionError(f"{label} no-memory replay changed")

    def objective(label: str, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = samples[label]
        memory, _ = exact_memory_loss(prediction, sample["projected_memory_latent16"], sample["projected_occupancy1"])
        preserve = invalid_raw_l1(prediction, no_memory[label], sample["projected_occupancy1"])
        return memory + args.preservation_weight * preserve, memory, preserve

    curve = []
    gradient_names = None
    for step in range(1, args.max_steps + 1):
        label = LABELS[(step - 1) % 2]
        optimizer.zero_grad(set_to_none=True)
        prediction = run_sample(label, conditions[label]["correct"])
        loss, memory_loss, preserve_loss = objective(label, prediction)
        loss.backward()
        if step == 1:
            gradient_names = [name for name, parameter in generator.named_parameters() if parameter.grad is not None]
            if gradient_names != ["model.memory_adapter.proj.weight"]:
                raise AssertionError(gradient_names)
            if torch.count_nonzero(adapter.proj.weight.grad).item() == 0:
                raise AssertionError("adapter gradient is zero")
        optimizer.step()
        record = {
            "step": step, "sample_id": label, "loss": float(loss.detach()),
            "valid_memory_loss": float(memory_loss.detach()),
            "invalid_raw_l1": float(preserve_loss.detach()),
        }
        curve.append(record)
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
    if torch.equal(initial_weight, adapter.proj.weight.detach().cpu()):
        raise AssertionError("adapter did not update")

    generator.eval().requires_grad_(False)
    save_adapter(adapter, root)
    roundtrip = MemoryPatchAdapter().to(device=device, dtype=torch.bfloat16).eval()
    load_adapter(roundtrip, root / "memory_adapter.safetensors", device=device)
    roundtrip_equal = bool(torch.equal(roundtrip.proj.weight, adapter.proj.weight))
    test_condition = conditions["plus5"]["correct"].permute(0, 2, 1, 3, 4).contiguous()
    test_gate = samples["plus5"]["projected_occupancy1"].permute(0, 2, 1, 3, 4).float().contiguous()
    original_output = gated_adapter_residual(adapter, test_condition, test_gate)[0]
    loaded_output = gated_adapter_residual(roundtrip, test_condition, test_gate)[0]
    roundtrip_equal = roundtrip_equal and bool(torch.equal(original_output, loaded_output))
    if not roundtrip_equal:
        raise AssertionError("adapter save/load roundtrip changed output")
    del roundtrip

    outputs: dict[str, dict[str, torch.Tensor]] = {}
    records = []
    with torch.inference_mode():
        for label in LABELS:
            sample = samples[label]
            outputs[label] = {}
            composite = torch.where(
                sample["projected_occupancy1"].bool().expand_as(no_memory[label]),
                sample["projected_memory_latent16"], no_memory[label],
            )
            for condition_name in CONDITIONS:
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                prediction = run_sample(label, conditions[label][condition_name])
                torch.cuda.synchronize(device)
                runtime_ms = (time.perf_counter() - tick) * 1000
                outputs[label][condition_name] = prediction.detach().cpu().contiguous()
                valid_loss, _ = exact_memory_loss(prediction, sample["projected_memory_latent16"], sample["projected_occupancy1"])
                full_loss, _ = exact_memory_loss(prediction, composite, torch.ones_like(sample["projected_occupancy1"], dtype=torch.bool))
                records.append({
                    "sample_id": label, "condition": condition_name,
                    "valid_raw_l1": float(masked_raw_l1(prediction, sample["projected_memory_latent16"], sample["projected_occupancy1"])),
                    "valid_exact_loss": float(valid_loss),
                    "invalid_raw_l1": float(invalid_raw_l1(prediction, no_memory[label], sample["projected_occupancy1"])),
                    "full_composite_raw_l1": float((prediction.float() - composite.float()).abs().mean()),
                    "full_composite_loss": float(full_loss),
                    "runtime_ms": runtime_ms,
                })

    with (root / "training_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader(); writer.writerows(curve)
    (root / "training_curve.json").write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")
    with (root / "metrics_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(records)
    for label in LABELS:
        save_file(outputs[label], root / f"outputs_{label}.safetensors")

    aggregate_conditions = {}
    for condition_name in CONDITIONS:
        selected = [record for record in records if record["condition"] == condition_name]
        aggregate_conditions[condition_name] = {
            f"mean_{field}": float(np.mean([record[field] for record in selected]))
            for field in CSV_FIELDS[2:]
        }
    by_side = {
        label: {record["condition"]: record for record in records if record["sample_id"] == label}
        for label in LABELS
    }
    side_gates = {}
    for label in LABELS:
        correct = by_side[label]["correct"]
        best_control = min(by_side[label][name]["valid_raw_l1"] for name in ("no_memory", "mask_only", "wrong_same_mask"))
        content_ratio = correct["valid_raw_l1"] / best_control
        composite_ratio = correct["full_composite_raw_l1"] / by_side[label]["no_memory"]["full_composite_raw_l1"]
        side_gates[label] = {
            "content_read_pass": content_ratio <= 0.5,
            "preservation_pass": composite_ratio <= 1.0,
            "correct_to_best_control_valid_raw_l1_ratio": content_ratio,
            "correct_to_no_memory_composite_raw_l1_ratio": composite_ratio,
        }
    shared_audit = json.loads((root / "shared_A_audit.json").read_text())
    projection_audit = json.loads((root / "projection_audit.json").read_text())
    quantitative_pass = (
        bool(shared_audit["passed"]) and bool(projection_audit["passed"])
        and bool(hard_audit["passed"]) and all(no_memory_equal.values())
        and roundtrip_equal
        and all(gate["content_read_pass"] and gate["preservation_pass"] for gate in side_gates.values())
    )
    failures = []
    for label, gate in side_gates.items():
        if not gate["content_read_pass"]:
            failures.append(f"{label}:content-read")
        if not gate["preservation_pass"]:
            failures.append(f"{label}:preservation")
    aggregate = {
        "conditions": aggregate_conditions,
        "per_side_acceptance": side_gates,
        "shared_A_audit_passed": bool(shared_audit["passed"]),
        "projection_audit_passed": bool(projection_audit["passed"]),
        "hard_gate_audit_passed": bool(hard_audit["passed"]),
        "no_memory_replay_torch_equal": no_memory_equal,
        "adapter_roundtrip_torch_equal": roundtrip_equal,
        "quantitative_verdict": "PASS_PENDING_VISUAL_REVIEW" if quantitative_pass else "FAIL",
        "failure_categories": failures,
        "training": {
            "optimizer": "AdamW", "lr": args.lr, "steps": args.max_steps,
            "preservation_weight": args.preservation_weight,
            "preservation": "strict invalid raw L1; I_latent=~M_latent",
            "backpropagated_denoise_steps": "last only (unchanged)",
            "gradient_parameter_names": gradient_names,
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        },
        "requested_step_indices": requested_steps.tolist(),
        "actual_model_timesteps": [float(x) for x in denoising_steps],
        "checkpoint_sha256": checkpoint_hash,
        "initial_adapter_sha256": init_hash,
        "trained_adapter_sha256": sha256_file(root / "memory_adapter.safetensors"),
    }
    del generator, kv_cache, samples, conditions, optimizer
    torch.cuda.empty_cache()
    aggregate["montage_audit"] = decode_montages(root, samples_cpu, outputs, Path(args.wan_root), device)
    aggregate["checkpoint_sha256_after"] = sha256_file(args.checkpoint)
    aggregate["initial_adapter_sha256_after"] = sha256_file(args.init_adapter)
    if aggregate["checkpoint_sha256_after"] != checkpoint_hash or aggregate["initial_adapter_sha256_after"] != init_hash:
        raise AssertionError("immutable checkpoint hash changed")
    aggregate["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    aggregate["train_eval_seconds"] = time.perf_counter() - started
    (root / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    write_report(root, aggregate)
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- Finish: `{datetime.now().astimezone().isoformat()}`\n"
            f"- Train/eval seconds: `{aggregate['train_eval_seconds']:.3f}`\n"
            f"- Peak VRAM GiB: `{aggregate['peak_vram_gib']:.6f}`\n"
            f"- Quantitative verdict: `{aggregate['quantitative_verdict']}`\n"
        )
    print(json.dumps(aggregate, indent=2), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
