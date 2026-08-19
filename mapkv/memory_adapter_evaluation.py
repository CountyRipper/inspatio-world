from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


METHODS = {
    "baseline": "methods/baseline",
    "episode_wre": "methods/episode_wre",
    "latent_anchor_all4": "methods/latent_anchor_all4",
    "adapter_patch_only": "methods/adapter_patch_only",
}
OPTIONAL_METHODS = {
    "adapter_patch_middle": "methods/adapter_patch_middle",
}


def _load_latent(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    return (value["pred_latents"] if isinstance(value, dict) else value).float()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.float()
    if expanded.ndim == 4:
        expanded = expanded.unsqueeze(2)
    denominator = float(expanded.sum().item()) * int(value.shape[2])
    if denominator <= 0:
        return float("nan")
    return float((value.float().abs() * expanded).sum().item() / denominator)


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float()
    if value.ndim == 5:
        value = value.squeeze(2)
    b, f, h, w = value.shape
    flat = value.reshape(b * f, 1, h, w)
    outer = F.max_pool2d(flat, 3, stride=1, padding=1)
    inner = 1.0 - F.max_pool2d(1.0 - flat, 3, stride=1, padding=1)
    return (outer - inner).clamp(0, 1).reshape(b, f, 1, h, w)


def _gradient_cosine(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    px = prediction[..., 1:] - prediction[..., :-1]
    tx = target[..., 1:] - target[..., :-1]
    py = prediction[..., 1:, :] - prediction[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    mx = torch.minimum(mask[..., 1:], mask[..., :-1]).expand_as(px)
    my = torch.minimum(mask[..., 1:, :], mask[..., :-1, :]).expand_as(py)
    left = torch.cat([px[mx > 0], py[my > 0]])
    right = torch.cat([tx[mx > 0], ty[my > 0]])
    if left.numel() == 0:
        return float("nan")
    return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item())


def _boundary_gradient_delta(
    value: torch.Tensor, baseline: torch.Tensor, band: torch.Tensor
) -> float:
    vx = value[..., 1:] - value[..., :-1]
    bx = baseline[..., 1:] - baseline[..., :-1]
    vy = value[..., 1:, :] - value[..., :-1, :]
    by = baseline[..., 1:, :] - baseline[..., :-1, :]
    mx = torch.maximum(band[..., 1:], band[..., :-1])
    my = torch.maximum(band[..., 1:, :], band[..., :-1, :])
    numerator = ((vx - bx).abs() * mx).sum() + ((vy - by).abs() * my).sum()
    denominator = (mx.sum() + my.sum()).clamp_min(1.0) * value.shape[2]
    return float((numerator / denominator).item())


def _temporal_peak(value: torch.Tensor, start: int, stop: int) -> float:
    segment = value[:, max(start - 1, 0):stop]
    if segment.shape[1] < 2:
        return 0.0
    deltas = (segment[:, 1:] - segment[:, :-1]).abs().mean(dim=(0, 2, 3, 4))
    return float(deltas.max().item())


def _case_metrics(root: Path, case_id: str) -> dict:
    case_root = root / case_id
    methods = {
        name: _load_latent(case_root / relative / "pred_latents.pt")
        for name, relative in METHODS.items()
        if (case_root / relative / "pred_latents.pt").exists()
    }
    required = set(METHODS)
    if set(methods) != required:
        raise FileNotFoundError(
            f"{case_id} missing methods: {sorted(required - set(methods))}"
        )
    overfit_path = case_root / "methods/adapter_overfit/pred_latents.pt"
    if overfit_path.exists():
        methods["adapter_overfit"] = _load_latent(overfit_path)
    for name, relative in OPTIONAL_METHODS.items():
        path = case_root / relative / "pred_latents.pt"
        if path.exists():
            methods[name] = _load_latent(path)
    gpu_by_method = {}
    method_paths = {**METHODS, **OPTIONAL_METHODS}
    method_paths["adapter_overfit"] = "methods/adapter_overfit"
    for name in methods:
        metadata = case_root / method_paths[name] / "run_metadata.json"
        if metadata.exists():
            gpu_by_method[name] = json.loads(metadata.read_text())["gpu"]
    if len(set(gpu_by_method.values())) != 1:
        raise RuntimeError(
            f"{case_id} mixes GPU models across counterfactuals: {gpu_by_method}"
        )
    baseline = methods["baseline"]
    mask_latent = torch.load(
        case_root / "methods/baseline/masks/mask_latent.pt",
        map_location="cpu", weights_only=True,
    ).float()
    phase_payload = json.loads(
        (case_root / "trajectory/phase_labels.json").read_text()
    )
    trajectory = json.loads(
        (case_root / "trajectory/trajectory_manifest.json").read_text()
    )
    phases = {value["name"]: value for value in phase_payload["phases"]}
    reentry_start = int(phases["Leave_to_B2"]["start_block"]) * 3
    reentry_stop = int(phases["B2_hold"]["stop_block_exclusive"]) * 3
    zero_root = case_root / "zero_init/memory_adapter"
    block_metrics: dict[str, dict] = {}
    aggregates = {
        name: {
            "appearance_numerator": 0.0,
            "appearance_denominator": 0.0,
            "source_numerator": 0.0,
            "source_denominator": 0.0,
            "boundary_numerator": 0.0,
            "boundary_denominator": 0.0,
            "feature": [],
        }
        for name in methods
    }
    coverages = []
    for block_root in sorted(zero_root.glob("block_*")):
        block_id = int(block_root.name.split("_")[-1])
        start, stop = block_id * 3, block_id * 3 + 3
        memory = torch.load(
            block_root / "L_mem.pt", map_location="cpu", weights_only=True
        ).float()
        need = torch.load(
            block_root / "M_need.pt", map_location="cpu", weights_only=True
        ).float()
        if need.ndim == 4:
            need = need.unsqueeze(2)
        source = (
            ((mask_latent[:, start:stop] + 1.0) * 0.5)
            .clamp(0, 1)
            .mean(dim=2, keepdim=True)
        )
        band = _boundary(need)
        coverages.append(float(need.mean().item()))
        block_metrics[str(block_id)] = {
            "need_coverage": coverages[-1], "methods": {},
        }
        for name, full in methods.items():
            current = full[:, start:stop]
            appearance = _masked_mean(current - memory, need)
            source_delta = _masked_mean(current - baseline[:, start:stop], source)
            boundary_error = _boundary_gradient_delta(
                current, baseline[:, start:stop], band
            )
            feature = _gradient_cosine(current, memory, need)
            block_metrics[str(block_id)]["methods"][name] = {
                "historical_appearance_l1": appearance,
                "generated_history_feature_similarity": feature,
                "source_region_delta_vs_baseline": source_delta,
                "boundary_band_error": boundary_error,
            }
            for key, value, weight in (
                ("appearance", appearance, float(need.sum().item())),
                ("source", source_delta, float(source.sum().item())),
                ("boundary", boundary_error, float(band.sum().item())),
            ):
                aggregates[name][f"{key}_numerator"] += value * weight
                aggregates[name][f"{key}_denominator"] += weight
            aggregates[name]["feature"].append(feature)
    summary = {}
    for name, values in aggregates.items():
        summary[name] = {
            "historical_appearance_l1": (
                values["appearance_numerator"]
                / max(values["appearance_denominator"], 1e-8)
            ),
            "generated_history_feature_similarity": float(
                np.nanmean(values["feature"])
            ),
            "source_region_delta_vs_baseline": (
                values["source_numerator"] / max(values["source_denominator"], 1e-8)
            ),
            "boundary_band_error": (
                values["boundary_numerator"]
                / max(values["boundary_denominator"], 1e-8)
            ),
            "reentry_temporal_peak": _temporal_peak(
                methods[name], reentry_start, reentry_stop
            ),
        }
    zero_metadata = json.loads(
        (case_root / "zero_init/run_metadata.json").read_text()
    )
    zero_replay = zero_metadata["replay"]
    zero_replay = zero_replay.get("against_saved_latents") or zero_replay
    return {
        "case_id": case_id,
        "canonical_history_chunk": int(trajectory["source_chunk"]),
        "target_chunk": int(trajectory["target_chunk"]),
        "trajectory": {
            "b1_yaw": float(trajectory["b1_theta_degrees"]),
            "leave_yaw": float(trajectory["leave_theta_degrees"]),
            "b2_yaw": float(trajectory["b2_theta_degrees"]),
        },
        "reentry_target_blocks": sorted(int(value) for value in block_metrics),
        "mean_need_coverage": float(np.mean(coverages)),
        "adapter_off_max_abs_diff": float(zero_replay["max_abs_diff"]),
        "gpu": next(iter(gpu_by_method.values())),
        "gpu_matched_across_methods": True,
        "methods": summary,
        "per_block": block_metrics,
    }


def evaluate_memory_adapter(root: str | Path) -> dict:
    root = Path(root).resolve()
    cases = {
        case_id: _case_metrics(root, case_id)
        for case_id in ("scene01", "scene02")
    }
    overfit = json.loads(
        (root / "training/overfit_scene01/training_summary.json").read_text()
    )
    joint = json.loads(
        (root / "training/joint_scene01_scene02/training_summary.json").read_text()
    )
    refine_path = root / "training/joint_patch_middle/training_summary.json"
    refine = json.loads(refine_path.read_text()) if refine_path.exists() else None
    review_path = root / "human_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    available_adapters = ["adapter_patch_only"]
    if all("adapter_patch_middle" in value["methods"] for value in cases.values()):
        available_adapters.append("adapter_patch_middle")
    requested_best = review.get("best_adapter")
    if requested_best in available_adapters:
        best_adapter = requested_best
    else:
        best_adapter = max(
            available_adapters,
            key=lambda name: float(np.mean([
                value["methods"][name]["generated_history_feature_similarity"]
                for value in cases.values()
            ])),
        )
    identity_metric_better = all(
        values["methods"][best_adapter]["historical_appearance_l1"]
        < values["methods"]["episode_wre"]["historical_appearance_l1"]
        and values["methods"][best_adapter]["generated_history_feature_similarity"]
        > values["methods"]["episode_wre"]["generated_history_feature_similarity"]
        for values in cases.values()
    )
    boundary_better_than_hard = all(
        values["methods"][best_adapter]["boundary_band_error"]
        < values["methods"]["latent_anchor_all4"]["boundary_band_error"]
        for values in cases.values()
    )
    review_go = bool(review) and all(
        review.get(case_id, {}).get(best_adapter, {}).get("identity") == "STRONG"
        and review.get(case_id, {}).get(best_adapter, {}).get("boundary")
        in {"CLEAN", "MINOR_SEAM"}
        and review.get(case_id, {}).get(best_adapter, {}).get("source_protection")
        in {"GOOD", "PARTIAL"}
        for case_id in cases
    )
    if review_go and identity_metric_better and boundary_better_than_hard:
        status = "LIGHTWEIGHT_MEMORY_ADAPTER_WORKS"
    elif identity_metric_better and any(
        review.get(case_id, {}).get(best_adapter, {}).get("identity") == "PARTIAL"
        or review.get(case_id, {}).get(best_adapter, {}).get("boundary")
        == "CLEAR_SEAM"
        for case_id in cases
    ):
        status = "ADAPTER_PROMISING_NEEDS_REFINEMENT"
    else:
        status = "CURRENT_ADAPTER_INTERFACE_INSUFFICIENT"
    payload = {
        "status": status,
        "focus_zh": "轻量 MemoryPatchAdapter：changed-view identity 与边界连续性",
        "cases": cases,
        "training": {
            "overfit_scene01": overfit,
            "joint_scene01_scene02": joint,
            "joint_patch_middle": refine,
        },
        "best_adapter": best_adapter,
        "human_review": review,
        "decisions": {
            "adapter_off_exact": all(
                value["adapter_off_max_abs_diff"] == 0.0 for value in cases.values()
            ),
            "joint_adapter_identity_metric_better_than_episode_wre": identity_metric_better,
            "joint_adapter_boundary_better_than_latent_anchor_all4": boundary_better_than_hard,
            "human_review_go": review_go,
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "status.json").write_text(
        json.dumps({"status": status}, indent=2), encoding="utf-8"
    )
    return payload


__all__ = ["METHODS", "OPTIONAL_METHODS", "evaluate_memory_adapter"]
