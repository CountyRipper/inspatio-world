#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from scripts.render_point_cloud import open_ffmpeg_writer
from utils.wan_wrapper import WanVAEWrapper


CONDITIONS = ("no_memory", "correct", "mask_only", "wrong_same_mask")
MODEL_TENSORS = (
    "z_A", "z_B", "latent_prefix_0_18", "block18_previous",
    "block19_base_render16", "block19_base_mask4", "block19_ref16",
    "z_Aprime_no_memory", "projected_memory_latent16",
    "projected_memory_mask4", "projected_occupancy1",
    "denoise_step_inputs", "transition_noises", "prompt_embeds",
)
CSV_FIELDS = (
    "scene", "query", "sample_id", "condition", "actual_yaw_degrees",
    "overlap_coverage", "latent_displacement_mean_pixels", "eligible",
    "overlap_masked_latent_raw_l1", "overlap_decoded_pixel_l1",
    "invalid_spill_l1", "full_composite_raw_l1", "runtime_ms",
)
ERROR_VMAX = 0.5


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


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
        return prediction.float().sum() * 0.0 + float("nan")
    return (prediction.float() - target.float()).abs()[selected].mean()


def build_conditions(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    occupancy = sample["projected_occupancy1"].bool()
    projected = sample["projected_memory_latent16"]
    memory_mask = sample["projected_memory_mask4"]
    wrong = torch.where(occupancy.expand_as(sample["z_B"]), sample["z_B"], 0)
    result = {
        "no_memory": None,
        "correct": torch.cat((memory_mask, projected), dim=2),
        "mask_only": torch.cat((memory_mask, torch.zeros_like(projected)), dim=2),
        "wrong_same_mask": torch.cat((memory_mask, wrong), dim=2),
    }
    for name in CONDITIONS[1:]:
        if not torch.equal(result[name][:, :, :4], memory_mask):
            raise AssertionError(f"{name} changed the shared memory mask")
    return result


def hard_gate_snapshot(
    adapter: MemoryPatchAdapter,
    labels: list[str],
    samples: dict[str, dict[str, torch.Tensor]],
    conditions: dict[str, dict[str, torch.Tensor | None]],
) -> dict[str, object]:
    if adapter.parameter_count != ADAPTER_PARAMETER_COUNT:
        raise AssertionError(adapter.parameter_count)
    result: dict[str, object] = {"passed": True, "parameter_count": adapter.parameter_count, "queries": {}}
    for label in labels:
        occupancy = samples[label]["projected_occupancy1"].permute(0, 2, 1, 3, 4).float()
        expected = patch_occupancy_gate(occupancy)
        gates = []
        condition_rows = {}
        for name in CONDITIONS[1:]:
            residual, gate = gated_adapter_residual(
                adapter,
                conditions[label][name].permute(0, 2, 1, 3, 4).contiguous(),
                occupancy,
            )
            gates.append(gate)
            invalid = (gate == 0).expand_as(residual)
            values = residual[invalid]
            condition_rows[name] = {
                "gate_equal_expected": bool(torch.equal(gate, expected)),
                "outside_residual_max_abs": float(values.abs().max()) if values.numel() else 0.0,
                "outside_residual_count_nonzero": int(torch.count_nonzero(values)),
            }
        if not all(torch.equal(gates[0], gate) for gate in gates[1:]):
            raise AssertionError(f"{label}: controls do not share G_patch")
        zero, zero_gate = gated_adapter_residual(
            adapter,
            conditions[label]["correct"].permute(0, 2, 1, 3, 4).contiguous(),
            torch.zeros_like(occupancy),
        )
        zero_equal = bool(torch.equal(zero, torch.zeros_like(zero)))
        if not zero_equal or torch.count_nonzero(zero_gate).item() != 0:
            raise AssertionError(f"{label}: zero occupancy addition changed")
        if any(
            row["outside_residual_max_abs"] != 0.0
            or row["outside_residual_count_nonzero"] != 0
            or not row["gate_equal_expected"]
            for row in condition_rows.values()
        ):
            raise AssertionError(f"{label}: hard gate audit failed")
        result["queries"][label] = {
            "g_patch_shape": list(expected.shape),
            "g_patch_unique": sorted(float(x) for x in torch.unique(expected)),
            "g_patch_valid_fraction": float(expected.mean()),
            "controls_share_torch_equal_gate": True,
            "conditions": condition_rows,
            "zero_occupancy_addition_torch_equal_zero": zero_equal,
        }
    return result


def tensor_to_image(frame: torch.Tensor, width: int = 208, height: int = 120) -> Image.Image:
    array = (frame.permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(array).resize((width, height), Image.Resampling.LANCZOS)


def write_video(frames: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[-2:]
    writer = open_ffmpeg_writer(str(path), width, height, 24)
    try:
        array = (frames.permute(0, 2, 3, 1).numpy().clip(0, 1) * 255).round().astype(np.uint8)
        for frame in array:
            writer.stdin.write(frame.tobytes())
    finally:
        writer.stdin.close()
        writer.wait()
    if writer.returncode:
        raise RuntimeError(f"ffmpeg failed for {path}")


def make_main_montage(
    path: Path,
    a_source: torch.Tensor,
    target_render: torch.Tensor,
    decoded: dict[str, torch.Tensor],
) -> None:
    columns = (
        "A/source", "A-prime render", "projected-A reference (not GT)",
        "no_memory", "correct", "mask_only", "wrong_same_mask",
    )
    values = {
        "A/source": a_source,
        "A-prime render": target_render,
        "projected-A reference (not GT)": decoded["reference"],
        **{name: decoded[name] for name in CONDITIONS},
    }
    cell_w, cell_h = 208, 120
    title_h, header_h = 38, 34
    canvas = Image.new("RGB", (cell_w * len(columns), title_h + header_h + 12 * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), "Full-context decode, fixed final 12 frames; projected-A reference is not GT", fill="black")
    for col, name in enumerate(columns):
        draw.text((col * cell_w + 5, title_h + 6), name, fill="black")
        for row in range(12):
            canvas.paste(tensor_to_image(values[name][row]), (col * cell_w, title_h + header_h + row * cell_h))
    canvas.save(path)


def make_error_montage(
    path: Path,
    decoded: dict[str, torch.Tensor],
    occupancy_pixel: torch.Tensor,
) -> dict[str, object]:
    columns = ("reference", *CONDITIONS)
    reference = decoded["reference"]
    errors = {name: (decoded[name] - reference).abs().mean(dim=1) for name in columns}
    cell_w, cell_h = 208, 120
    title_h, header_h = 38, 34
    canvas = Image.new("RGB", (cell_w * len(columns), title_h + header_h + 12 * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), f"Unified cyan occupancy boundary; red error scale 0..{ERROR_VMAX}", fill="black")
    for col, name in enumerate(columns):
        draw.text((col * cell_w + 5, title_h + 6), name, fill="black")
        for row in range(12):
            rgb = decoded[name][row].permute(1, 2, 0).numpy().copy()
            mask = occupancy_pixel[row]
            heat = (errors[name][row] / ERROR_VMAX).clamp(0, 1).numpy()[..., None]
            rgb = np.where(mask.numpy()[..., None], rgb * (1 - 0.62 * heat) + np.array([1.0, 0.0, 0.0]) * (0.62 * heat), rgb)
            mask_float = mask.float()[None, None]
            interior = (-F.max_pool2d(-mask_float, kernel_size=3, stride=1, padding=1))[0, 0].bool()
            rgb[(mask & ~interior).numpy()] = np.array([0.0, 1.0, 1.0])
            image = Image.fromarray((rgb.clip(0, 1) * 255).round().astype(np.uint8))
            canvas.paste(image.resize((cell_w, cell_h), Image.Resampling.LANCZOS), (col * cell_w, title_h + header_h + row * cell_h))
    canvas.save(path)
    return {
        "common_error_vmax": ERROR_VMAX,
        "occupancy_boundary_color": "cyan",
        "residual_temporal_flicker": {
            name: float(
                (((decoded[name] - reference)[1:] - (decoded[name] - reference)[:-1]).abs().mean())
            )
            for name in CONDITIONS
        },
    }


@torch.inference_mode()
def decode_outputs(
    root: Path,
    descriptors: list[dict[str, object]],
    samples_cpu: dict[str, dict[str, torch.Tensor]],
    outputs: dict[str, dict[str, torch.Tensor]],
    rows: list[dict[str, object]],
    wan_root: Path,
    device: torch.device,
) -> dict[str, object]:
    row_index = {(row["sample_id"], row["condition"]): row for row in rows}
    vae = WanVAEWrapper(str(wan_root)).to(device=device, dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)
    a_sources: dict[str, torch.Tensor] = {}
    for scene in ("S0", "S1"):
        label = next(str(item["sample_id"]) for item in descriptors if item["scene"] == scene)
        prefix = samples_cpu[label]["latent_prefix_0_18"][:, :18]
        pixels = vae.decode_to_pixel(prefix.to(device=device, dtype=torch.bfloat16), use_cache=False)
        pixels = (pixels[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
        if pixels.shape[0] != 69:
            raise AssertionError(f"{scene}: A/source decode shape {pixels.shape}")
        a_sources[scene] = pixels[-12:]
        vae.model.clear_cache()

    audit = {}
    for descriptor in descriptors:
        label = str(descriptor["sample_id"])
        scene = str(descriptor["scene"])
        query = str(descriptor["query"])
        sample = samples_cpu[label]
        occupancy = sample["projected_occupancy1"].bool()
        composite = torch.where(
            occupancy.expand_as(outputs[label]["no_memory"]),
            sample["projected_memory_latent16"],
            outputs[label]["no_memory"],
        )
        latent_values = {"reference": composite, **outputs[label]}
        decoded_last: dict[str, torch.Tensor] = {}
        for name, query_latent in latent_values.items():
            full = torch.cat((sample["latent_prefix_0_18"], query_latent), dim=1)
            if tuple(full.shape) != (1, 60, 16, 60, 104):
                raise AssertionError(f"{label}/{name}: {full.shape}")
            pixels = vae.decode_to_pixel(full.to(device=device, dtype=torch.bfloat16), use_cache=False)
            pixels = (pixels[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
            if pixels.shape[0] != 237:
                raise AssertionError(f"{label}/{name}: decoded {pixels.shape}")
            if name in CONDITIONS:
                write_video(pixels, root / "videos" / scene / query / f"{name}.mp4")
            decoded_last[name] = pixels[-12:]
            vae.model.clear_cache()
            del pixels
            torch.cuda.empty_cache()

        occupancy_pixel = F.interpolate(
            occupancy[0].float(),
            size=decoded_last["reference"].shape[-2:],
            mode="nearest",
        )[:, 0].bool().repeat_interleave(4, dim=0)
        if tuple(occupancy_pixel.shape) != (12, 480, 832):
            raise AssertionError(f"{label}: pixel occupancy {occupancy_pixel.shape}")
        target_render = (sample["target_Aprime_render12"].float() * 0.5 + 0.5).clamp(0, 1)
        make_main_montage(
            root / "montages" / f"{label}_main.png",
            a_sources[scene],
            target_render,
            decoded_last,
        )
        error_audit = make_error_montage(
            root / "montages" / f"{label}_overlap_error.png",
            decoded_last,
            occupancy_pixel,
        )
        reference = decoded_last["reference"]
        for name in CONDITIONS:
            error = (decoded_last[name] - reference).abs().mean(dim=1)
            row_index[(label, name)]["overlap_decoded_pixel_l1"] = (
                float(error[occupancy_pixel].mean())
                if occupancy_pixel.any()
                else float("nan")
            )
        audit[label] = {
            "full_context_latent_frames": 60,
            "full_context_decoded_frames": 237,
            "fixed_montage_frames": [225, 237],
            "reference": "projected z_A inside overlap; no-memory outside; not GT",
            "video_conditions": list(CONDITIONS),
            **error_audit,
        }
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return audit


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate(
    descriptors: list[dict[str, object]],
    rows: list[dict[str, object]],
    audits: dict[str, object],
    training: dict[str, object],
    hashes: dict[str, str],
    montage_audit: dict[str, object],
) -> dict[str, object]:
    by_query = {
        str(item["sample_id"]): {
            row["condition"]: row
            for row in rows
            if row["sample_id"] == item["sample_id"]
        }
        for item in descriptors
    }
    eligible = [item for item in descriptors if bool(item["eligible"])]
    content_wins = []
    no_memory_wins = []
    per_query = {}
    for item in descriptors:
        label = str(item["sample_id"])
        values = by_query[label]
        correct = values["correct"]
        valid_gain = (
            float(values["no_memory"]["overlap_masked_latent_raw_l1"])
            - float(correct["overlap_masked_latent_raw_l1"])
        )
        net_gain = (
            float(values["no_memory"]["full_composite_raw_l1"])
            - float(correct["full_composite_raw_l1"])
        )
        content_win = (
            float(correct["overlap_masked_latent_raw_l1"])
            < float(values["mask_only"]["overlap_masked_latent_raw_l1"])
            and float(correct["overlap_masked_latent_raw_l1"])
            < float(values["wrong_same_mask"]["overlap_masked_latent_raw_l1"])
        )
        no_memory_win = (
            float(correct["overlap_masked_latent_raw_l1"])
            < float(values["no_memory"]["overlap_masked_latent_raw_l1"])
        )
        if bool(item["eligible"]):
            content_wins.append(content_win)
            no_memory_wins.append(no_memory_win)
        per_query[label] = {
            **item,
            "valid_gain": valid_gain,
            "invalid_spill": float(correct["invalid_spill_l1"]),
            "net_gain": net_gain,
            "correct_beats_mask_and_wrong": content_win,
            "correct_beats_no_memory": no_memory_win,
            "wrong_minus_correct_overlap_l1": (
                float(values["wrong_same_mask"]["overlap_masked_latent_raw_l1"])
                - float(correct["overlap_masked_latent_raw_l1"])
            ),
        }

    per_scene = {}
    for scene in ("S0", "S1"):
        selected_all = [
            per_query[str(item["sample_id"])]
            for item in descriptors
            if item["scene"] == scene
        ]
        selected = [item for item in selected_all if item["eligible"]]
        per_scene[scene] = {
            "aggregation_scope": "eligible queries only",
            "mean_valid_gain": float(np.mean([item["valid_gain"] for item in selected])),
            "mean_invalid_spill": float(np.mean([item["invalid_spill"] for item in selected])),
            "mean_net_gain": float(np.mean([item["net_gain"] for item in selected])),
            "all_query_diagnostic_means": {
                "valid_gain": float(np.mean([item["valid_gain"] for item in selected_all])),
                "invalid_spill": float(np.mean([item["invalid_spill"] for item in selected_all])),
                "net_gain": float(np.mean([item["net_gain"] for item in selected_all])),
            },
            "exact_medium_valid_gains": {
                item["query"]: item["valid_gain"]
                for item in selected
                if item["view_class"] in ("exact", "medium")
            },
        }
    content_rate = float(np.mean(content_wins)) if content_wins else 0.0
    no_memory_rate = float(np.mean(no_memory_wins)) if no_memory_wins else 0.0
    exact_medium_consistent = all(
        gain > 0
        for scene in per_scene.values()
        for gain in scene["exact_medium_valid_gains"].values()
    )
    engineering_pass = all(bool(audits[name]["passed"]) for name in ("shared_A", "projection", "hard_gate"))
    quantitative_gates = {
        "engineering_audits_all_pass": engineering_pass,
        "content_control_win_rate_at_least_0_80": content_rate >= 0.80,
        "no_memory_win_rate_at_least_0_70": no_memory_rate >= 0.70,
        "both_scenes_exact_medium_direction_consistent": exact_medium_consistent,
        "both_scene_net_gain_positive": all(item["mean_net_gain"] > 0 for item in per_scene.values()),
    }
    return {
        "scope": "capacity/overfit; S0/S1; no held-out generalization claim",
        "eligible_query_count": len(eligible),
        "total_query_count": len(descriptors),
        "win_rates": {
            "correct_beats_both_mask_only_and_wrong_same_mask": content_rate,
            "correct_beats_no_memory": no_memory_rate,
            "correct_vs_mask_only": float(np.mean([
                by_query[str(item["sample_id"])]["correct"]["overlap_masked_latent_raw_l1"]
                < by_query[str(item["sample_id"])]["mask_only"]["overlap_masked_latent_raw_l1"]
                for item in eligible
            ])),
            "correct_vs_wrong_same_mask": float(np.mean([
                by_query[str(item["sample_id"])]["correct"]["overlap_masked_latent_raw_l1"]
                < by_query[str(item["sample_id"])]["wrong_same_mask"]["overlap_masked_latent_raw_l1"]
                for item in eligible
            ])),
        },
        "per_scene": per_scene,
        "per_query": per_query,
        "quantitative_gates": quantitative_gates,
        "training": training,
        "audits": {
            "shared_A": audits["shared_A"]["passed"],
            "projection": audits["projection"]["passed"],
            "hard_gate": audits["hard_gate"]["passed"],
        },
        "montage_audit": montage_audit,
        **hashes,
    }


def plot_curves(path: Path, aggregate: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"S0": "tab:blue", "S1": "tab:orange"}
    for scene in ("S0", "S1"):
        values = [item for item in aggregate["per_query"].values() if item["scene"] == scene]
        values.sort(key=lambda item: (abs(float(item["actual_yaw_degrees"])), float(item["actual_yaw_degrees"])))
        angle = [abs(float(item["actual_yaw_degrees"])) for item in values]
        coverage = [float(item["overlap_coverage"]) for item in values]
        axes[0].plot(angle, [item["valid_gain"] for item in values], "o-", color=colors[scene], label=f"{scene} valid gain")
        axes[0].plot(angle, [item["invalid_spill"] for item in values], "x--", color=colors[scene], alpha=0.7, label=f"{scene} invalid spill")
        axes[1].scatter(coverage, [item["net_gain"] for item in values], color=colors[scene], label=scene)
        for x, y, item in zip(coverage, [item["net_gain"] for item in values], values):
            axes[1].annotate(item["query"], (x, y), fontsize=7)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("|actual yaw| (degrees)")
    axes[0].set_ylabel("raw L1 gain/spill")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("actual overlap coverage")
    axes[1].set_ylabel("net gain")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_preliminary_report(root: Path, aggregate: dict[str, object]) -> None:
    rates = aggregate["win_rates"]
    lines = [
        "# Phase 1 宽视角 shared-memory 最终验证",
        "",
        "本报告覆盖 S0/S1 各一个 shared A，以及 exact、±10°、按预注册 coverage 规则选出的 wide query。该实验仍是小样本 capacity/overfit 验证。",
        "",
        "## 核心量化",
        "",
        f"- eligible query: {aggregate['eligible_query_count']}/{aggregate['total_query_count']}",
        f"- correct 同时优于 mask-only 与 wrong-same-mask: {rates['correct_beats_both_mask_only_and_wrong_same_mask']:.1%}",
        f"- correct 优于 no-memory: {rates['correct_beats_no_memory']:.1%}",
    ]
    for scene, item in aggregate["per_scene"].items():
        lines.append(
            f"- {scene}: valid gain={item['mean_valid_gain']:.6f}, "
            f"invalid spill={item['mean_invalid_spill']:.6f}, net gain={item['mean_net_gain']:.6f}"
        )
    lines.extend([
        "",
        "## 审计",
        "",
        f"- shared-A: {aggregate['audits']['shared_A']}",
        f"- projection: {aggregate['audits']['projection']}",
        f"- hard gate / frozen parameters: {aggregate['audits']['hard_gate']}",
        "",
        "## 定性检查",
        "",
        "主 montage 固定展示最后 12 帧；reference 是 overlap 内 projected z_A、外部 no-memory 的 full-context decode，不是真实 GT。最终视觉观察和 Phase 决策由独立 finalize 步骤写入，避免在生成 montage 前预判。",
        "",
        "## 未证明边界",
        "",
        "不证明 held-out 泛化、更多场景、动态 memory 更新、完整 LSM 系统或四个 denoise steps 全部参与反传。本轮没有启动 Phase 2，也没有解冻 backbone。",
    ])
    (root / "PHASE1_WIDE_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument("--wan-root", default="/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B")
    args = parser.parse_args()
    if not 200 <= args.max_steps <= 1000:
        raise ValueError("exploratory training requires 200 <= max_steps <= 1000")
    if args.lr != 1e-3:
        raise ValueError("exploratory training keeps lr fixed at 1e-3")
    if args.preservation_weight not in (0.5, 1.0, 2.0):
        raise ValueError("preservation_weight must be one of 0.5, 1.0, 2.0")
    repo_root = Path(args.repo_root).resolve()
    expected_init = (repo_root / "artifacts/phase1_lsm/train/fixed8_projected/memory_adapter.safetensors").resolve()
    if Path(args.init_adapter).resolve() != expected_init:
        raise ValueError(f"initial adapter must be {expected_init}")

    root = Path(args.root)
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    descriptors = manifest["queries"]
    labels = [str(item["sample_id"]) for item in descriptors]
    if len(labels) != 10 or len(set(labels)) != 10:
        raise AssertionError(f"expected ten unique queries, got {labels}")
    train_labels = [
        str(item["sample_id"])
        for item in descriptors
        if bool(item["eligible"]) and float(item["overlap_coverage"]) > 0.0
    ]
    if not train_labels:
        raise AssertionError("no query has projected overlap")
    forbidden = (
        "memory_adapter.safetensors", "metrics_per_query.csv", "aggregate_metrics.json",
        "training_curve.csv", "training_curve.json", "PHASE_DECISION.md",
    )
    if any((root / name).exists() for name in forbidden):
        raise FileExistsError("refusing to overwrite train/eval outputs")
    for dirname in ("outputs", "videos", "montages"):
        (root / dirname).mkdir(parents=True, exist_ok=True)

    command_log = root / "COMMAND_LOG.md"
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Train/eval\n\n"
            f"- Start: {datetime.now().astimezone().isoformat()}\n"
            f"- In-process command: {shlex.join([sys.executable, *sys.argv])}\n"
            f"- Exploratory: AdamW, lr={args.lr:g}, steps={args.max_steps}, "
            f"preservation_weight={args.preservation_weight:g}, balanced cyclic, last denoise step backprop\n"
        )
    started = time.perf_counter()
    checkpoint_hash = sha256_file(args.checkpoint)
    init_hash = sha256_file(args.init_adapter)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    generator, config = load_generator(repo_root, args.checkpoint, device)
    load_adapter(generator.model.memory_adapter, args.init_adapter, device=device)
    adapter = generator.model.memory_adapter
    trainable = freeze_except_adapter(generator)
    if trainable != [adapter.proj.weight] or adapter.parameter_count != ADAPTER_PARAMETER_COUNT:
        raise AssertionError("trainable set or adapter parameter count changed")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    initial_weight = adapter.proj.weight.detach().cpu().clone()

    samples_cpu = {
        label: load_tensors(root / "samples" / label / "sample.safetensors")
        for label in labels
    }
    samples = {
        label: {name: samples_cpu[label][name].to(device) for name in MODEL_TENSORS}
        for label in labels
    }
    conditions = {label: build_conditions(samples[label]) for label in labels}
    hard_before = hard_gate_snapshot(adapter, labels, samples, conditions)

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
        fill_teacher_cache(
            generator, conditional, kv_cache, sample["block19_ref16"],
            sample["block18_previous"], render,
        )
        prediction, _ = denoise_block(
            generator, generator.scheduler, sample["denoise_step_inputs"][0],
            conditional, kv_cache, render_block=render, denoising_kv_size=1560 * 6,
            denoising_steps=denoising_steps, memory_condition=condition,
            memory_gate=None if condition is None else sample["projected_occupancy1"].float(),
            transition_noises=sample["transition_noises"],
        )
        return prediction

    no_memory = {}
    no_memory_runtime_ms = {}
    replay_equal = {}
    with torch.inference_mode():
        for label in labels:
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            no_memory[label] = run_sample(label, None)
            torch.cuda.synchronize(device)
            no_memory_runtime_ms[label] = (time.perf_counter() - tick) * 1000
            replay_equal[label] = bool(torch.equal(no_memory[label], samples[label]["z_Aprime_no_memory"]))
            if not replay_equal[label]:
                raise AssertionError(f"{label}: no-memory replay changed")

    curve = []
    gradient_names = None
    sample_counts = {label: 0 for label in train_labels}
    for step in range(1, args.max_steps + 1):
        label = train_labels[(step - 1) % len(train_labels)]
        sample_counts[label] += 1
        optimizer.zero_grad(set_to_none=True)
        prediction = run_sample(label, conditions[label]["correct"])
        memory_loss, _ = exact_memory_loss(
            prediction,
            samples[label]["projected_memory_latent16"],
            samples[label]["projected_occupancy1"],
        )
        preserve_loss = invalid_raw_l1(
            prediction, no_memory[label], samples[label]["projected_occupancy1"]
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
        record = {
            "step": step,
            "sample_id": label,
            "loss": float(loss.detach()),
            "valid_memory_loss": float(memory_loss.detach()),
            "invalid_raw_l1": float(preserve_loss.detach()),
        }
        curve.append(record)
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
    if max(sample_counts.values()) - min(sample_counts.values()) > 1:
        raise AssertionError(f"sampling is not balanced: {sample_counts}")
    if torch.equal(initial_weight, adapter.proj.weight.detach().cpu()):
        raise AssertionError("adapter did not update")

    generator.eval().requires_grad_(False)
    save_adapter(adapter, root)
    roundtrip = MemoryPatchAdapter().to(device=device, dtype=torch.bfloat16).eval()
    load_adapter(roundtrip, root / "memory_adapter.safetensors", device=device)
    roundtrip_equal = bool(torch.equal(roundtrip.proj.weight, adapter.proj.weight))
    test_label = labels[0]
    test_condition = conditions[test_label]["correct"].permute(0, 2, 1, 3, 4).contiguous()
    test_gate = samples[test_label]["projected_occupancy1"].permute(0, 2, 1, 3, 4).float()
    roundtrip_equal = roundtrip_equal and bool(torch.equal(
        gated_adapter_residual(adapter, test_condition, test_gate)[0],
        gated_adapter_residual(roundtrip, test_condition, test_gate)[0],
    ))
    if not roundtrip_equal:
        raise AssertionError("adapter save/load roundtrip changed output")
    del roundtrip
    hard_after = hard_gate_snapshot(adapter, labels, samples, conditions)
    hard_audit = {
        "passed": bool(hard_before["passed"] and hard_after["passed"]),
        "adapter_parameter_count": adapter.parameter_count,
        "injection_order": [
            "residual=memory_adapter(memory_condition)",
            "residual=residual*G_patch",
            "dit_embedding=dit_embedding+residual",
        ],
        "only_adapter_receives_gradient": gradient_names == ["model.memory_adapter.proj.weight"],
        "backbone_frozen": True,
        "no_memory_hard_bypass": True,
        "no_memory_replay_torch_equal": replay_equal,
        "adapter_save_load_torch_equal": roundtrip_equal,
        "before_training": hard_before,
        "after_training": hard_after,
        "evaluation_reuse": {
            "same_trajectory": True,
            "same_captured_causal_state": True,
            "same_denoise_step_inputs": True,
            "same_transition_noise": True,
            "same_occupancy": True,
            "same_projected_target": True,
            "parameters_updated_during_evaluation": False,
        },
    }
    (root / "hard_gate_audit.json").write_text(json.dumps(hard_audit, indent=2) + "\n", encoding="utf-8")

    outputs: dict[str, dict[str, torch.Tensor]] = {}
    rows = []
    descriptor_by_label = {str(item["sample_id"]): item for item in descriptors}
    with torch.inference_mode():
        for label in labels:
            descriptor = descriptor_by_label[label]
            sample = samples[label]
            outputs[label] = {}
            composite = torch.where(
                sample["projected_occupancy1"].bool().expand_as(no_memory[label]),
                sample["projected_memory_latent16"],
                no_memory[label],
            )
            for name in CONDITIONS:
                if name == "no_memory":
                    prediction = no_memory[label]
                    runtime_ms = no_memory_runtime_ms[label]
                else:
                    torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    prediction = run_sample(label, conditions[label][name])
                    torch.cuda.synchronize(device)
                    runtime_ms = (time.perf_counter() - tick) * 1000
                outputs[label][name] = prediction.detach().cpu().contiguous()
                rows.append({
                    "scene": descriptor["scene"],
                    "query": descriptor["query"],
                    "sample_id": label,
                    "condition": name,
                    "actual_yaw_degrees": descriptor["actual_yaw_degrees"],
                    "overlap_coverage": descriptor["overlap_coverage"],
                    "latent_displacement_mean_pixels": descriptor["latent_displacement_mean_pixels"],
                    "eligible": descriptor["eligible"],
                    "overlap_masked_latent_raw_l1": float(masked_raw_l1(
                        prediction,
                        sample["projected_memory_latent16"],
                        sample["projected_occupancy1"],
                    )),
                    "overlap_decoded_pixel_l1": float("nan"),
                    "invalid_spill_l1": float(invalid_raw_l1(
                        prediction, no_memory[label], sample["projected_occupancy1"]
                    )),
                    "full_composite_raw_l1": float(
                        (prediction.float() - composite.float()).abs().mean()
                    ),
                    "runtime_ms": runtime_ms,
                })
            save_file(outputs[label], root / "outputs" / f"{label}.safetensors")

    with (root / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    (root / "training_curve.json").write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")
    training = {
        "optimizer": "AdamW",
        "lr": args.lr,
        "steps": args.max_steps,
        "preservation_weight": args.preservation_weight,
        "preservation": "strict invalid raw L1; I_latent=~M_latent; empty exact invalid set contributes zero",
        "sampling": "balanced cyclic",
        "sample_counts": sample_counts,
        "backpropagated_denoise_steps": "last only (unchanged)",
        "requested_denoise_indices": requested_steps.tolist(),
        "actual_model_timesteps": [float(value) for value in denoising_steps],
        "gradient_parameter_names": gradient_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
    }

    del generator, kv_cache, samples, conditions, optimizer, no_memory
    gc.collect()
    torch.cuda.empty_cache()
    montage_audit = decode_outputs(
        root, descriptors, samples_cpu, outputs, rows, Path(args.wan_root), device
    )
    write_csv(root / "metrics_per_query.csv", rows)

    shared_audit = json.loads((root / "shared_A_audit.json").read_text())
    projection_audit = json.loads((root / "projection_audit.json").read_text())
    hashes = {
        "base_checkpoint_sha256_before": checkpoint_hash,
        "base_checkpoint_sha256_after": sha256_file(args.checkpoint),
        "initial_adapter_sha256_before": init_hash,
        "initial_adapter_sha256_after": sha256_file(args.init_adapter),
        "trained_adapter_sha256": sha256_file(root / "memory_adapter.safetensors"),
    }
    if hashes["base_checkpoint_sha256_after"] != checkpoint_hash:
        raise AssertionError("base checkpoint changed")
    if hashes["initial_adapter_sha256_after"] != init_hash:
        raise AssertionError("initial adapter changed")
    aggregate = build_aggregate(
        descriptors,
        rows,
        {
            "shared_A": shared_audit,
            "projection": projection_audit,
            "hard_gate": hard_audit,
        },
        training,
        hashes,
        montage_audit,
    )
    aggregate["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    aggregate["total_seconds"] = time.perf_counter() - started
    (root / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    plot_curves(root / "angle_coverage_curves.png", aggregate)
    write_preliminary_report(root, aggregate)
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- Finish: {datetime.now().astimezone().isoformat()}\n"
            f"- Total train/eval/decode seconds: {aggregate['total_seconds']:.3f}\n"
            f"- Peak VRAM GiB: {aggregate['peak_vram_gib']:.6f}\n"
            f"- Content-control win rate: {aggregate['win_rates']['correct_beats_both_mask_only_and_wrong_same_mask']:.6f}\n"
            f"- No-memory win rate: {aggregate['win_rates']['correct_beats_no_memory']:.6f}\n"
        )
    print(json.dumps({
        "status": "SUCCESS_PENDING_VISUAL_DECISION",
        "win_rates": aggregate["win_rates"],
        "per_scene": aggregate["per_scene"],
    }, indent=2), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
