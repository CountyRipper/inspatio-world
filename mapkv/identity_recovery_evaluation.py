from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .continuous_cavr_evaluation import _video_transition
from .evaluation import _image, _masked_l1
from .locality_evaluation import (
    _block,
    _intrinsics,
    _rotation_warp,
    _save_mask,
    _save_rgb,
)


METHOD_ROOTS = {
    "baseline": "baseline",
    "current_masked": "generation/current_masked_wre",
    "strong_latent": "generation/strong_core_latent_wre",
    "rgb_warp_vae": "generation/rgb_warp_vae_wre",
    "canonical_kv": "generation/canonical_kv",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_root(root: Path, method: str) -> Path:
    return root / METHOD_ROOTS[method]


def _mask_to_image(mask: torch.Tensor, image_hw: tuple[int, int]) -> np.ndarray:
    value = mask.float()
    while value.ndim > 2:
        value = value.mean(dim=0)
    return cv2.resize(
        value.numpy(),
        (image_hw[1], image_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    ).clip(0, 1)


def _prefix_exact(metadata: dict, first_active: int) -> bool:
    per_chunk = metadata["replay"]["against_saved_latents"][
        "per_chunk_max_abs_diff"
    ]
    return all(float(per_chunk[str(chunk)]) == 0.0 for chunk in range(first_active))


def _coverage_signature(metadata: dict) -> list[tuple[int, float, float, str]]:
    return [
        (
            int(item["target_chunk"]),
            round(float(item["hard_coverage_fraction"]), 8),
            round(float(item["coverage_fraction"]), 8),
            str(item["status"]),
        )
        for item in metadata["mapkv"]["selections"]
    ]


def _identity_crop_assets(
    *,
    assets: Path,
    images: dict[str, np.ndarray],
) -> dict:
    height, width = next(iter(images.values())).shape[:2]
    # Stable normalized boxes around the three identity-bearing tabletop zones
    # called out in prior visual review.  The full-frame overview remains next
    # to the crops so these regions are never presented without context.
    definitions = {
        "pastry": (0.30, 0.38, 0.34, 0.44),
        "cup": (0.56, 0.28, 0.30, 0.54),
        "plate_local": (0.18, 0.48, 0.46, 0.45),
    }
    crop_root = assets / "identity_crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    boxes = {}
    for name, (x, y, w, h) in definitions.items():
        left = int(round(x * width))
        top = int(round(y * height))
        right = min(width, left + int(round(w * width)))
        bottom = min(height, top + int(round(h * height)))
        boxes[name] = [left, top, right, bottom]
        for method, image in images.items():
            crop = image[top:bottom, left:right]
            _save_rgb(crop_root / f"{name}_{method}.png", crop)

    overview = Image.fromarray(
        (images["warped_b1"].clip(0, 1) * 255).round().astype(np.uint8)
    )
    draw = ImageDraw.Draw(overview)
    colors = {"pastry": "#ff355e", "cup": "#20b26b", "plate_local": "#2775ff"}
    for name, box in boxes.items():
        draw.rectangle(box, outline=colors[name], width=4)
        draw.text((box[0] + 5, box[1] + 5), name, fill=colors[name])
    overview.save(crop_root / "crop_overview.png")
    return {"boxes_xyxy": boxes, "root": str(crop_root)}


def evaluate_identity_recovery(
    *,
    run_root: str | Path,
    case_dir: str | Path,
    source_chunk: int = 8,
    target_chunk: int = 22,
) -> dict:
    root = Path(run_root).resolve()
    case_dir = Path(case_dir).resolve()
    methods = {name: _method_root(root, name) for name in METHOD_ROOTS}
    source = _image(methods["baseline"] / "keyframes" / f"chunk_{source_chunk:04d}.png")
    baseline_target = _image(
        methods["baseline"] / "keyframes" / f"chunk_{target_chunk:04d}.png"
    )
    height, width = source.shape[:2]
    mapping = methods["baseline"] / "block_mapping.json"
    intrinsics = _intrinsics(case_dir / "intrinsics.txt", (height, width))
    warped_source, warp_valid, homography = _rotation_warp(
        source,
        np.asarray(_block(mapping, source_chunk)["c2w"]),
        np.asarray(_block(mapping, target_chunk)["c2w"]),
        intrinsics,
    )
    strong_state = torch.load(
        methods["strong_latent"]
        / "warp"
        / f"target_{target_chunk:04d}"
        / "warp_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    hard = _mask_to_image(strong_state["hard_coverage"], (height, width))
    memory = _mask_to_image(strong_state["memory_coverage"], (height, width))
    query = _mask_to_image(strong_state["query_gate_tokens"], (height, width))
    overlap = hard * warp_valid
    nonoverlap = 1.0 - memory
    if float(overlap.mean()) <= 0 or float(nonoverlap.mean()) <= 0:
        raise RuntimeError("Identity evaluation requires overlap and non-overlap")

    assets = root / "assets" / "identity_recovery"
    assets.mkdir(parents=True, exist_ok=True)
    _save_rgb(assets / "b1_source.png", source)
    _save_rgb(assets / "b1_warped_to_b2.png", warped_source)
    _save_rgb(assets / "b2_baseline.png", baseline_target)
    _save_mask(assets / "M_hard.png", hard)
    _save_mask(assets / "M_memory.png", memory)
    _save_mask(assets / "M_query.png", query)

    phase = _json(case_dir / "phase_labels.json")
    b2 = next(item for item in phase["phases"] if item["name"] == "B2_hold")
    ramp = next(item for item in phase["phases"] if item["name"] == "A_to_B2")
    metrics = {}
    metadata = {}
    crop_images = {"warped_b1": warped_source}
    for method, method_root in methods.items():
        target = _image(
            method_root / "keyframes" / f"chunk_{target_chunk:04d}.png"
        )
        crop_images[method] = target
        meta = _json(method_root / "run_metadata.json")
        metadata[method] = meta
        transition = _video_transition(
            method_root / "pred.mp4",
            int(b2["rgb_start"]),
            window_start=int(ramp["rgb_start"]),
            window_stop=int(b2["rgb_stop_exclusive"]),
        )
        metrics[method] = {
            "overlap_b1_to_b2_l1": _masked_l1(
                warped_source, target, overlap
            ),
            "nonoverlap_delta_vs_baseline_l1": _masked_l1(
                baseline_target, target, nonoverlap
            ),
            "whole_delta_vs_baseline_l1": float(
                np.abs(baseline_target - target).mean()
            ),
            "reentry_window_mean_l1": transition["reentry_window_mean_l1"],
            "reentry_window_peak_l1": transition["reentry_window_peak_l1"],
            "entrance_frame_l1": transition["entrance_frame_l1"],
            "generation_seconds": float(meta["timing_seconds"]["total"]),
        }
    crop_manifest = _identity_crop_assets(assets=assets, images=crop_images)

    canonical_audit = _json(methods["canonical_kv"] / "canonical_kv_audit.json")
    bank_root = Path(
        metadata["canonical_kv"]["mapkv"]["recent_bank_root"]
    ).resolve()
    bank_metadata = _json(bank_root / "metadata.json")
    bank_chunk = bank_metadata["chunks"][str(source_chunk)]
    bank_abs_mean_differences = {}
    for layer, writer_stats in canonical_audit["source_writer"][
        "layer_stats"
    ].items():
        payload = torch.load(
            bank_root / bank_chunk["layers"][layer]["path"],
            map_location="cpu",
            weights_only=True,
        )
        bank_abs_mean_differences[layer] = {
            "k_abs_mean_diff": abs(
                float(payload["k"].float().abs().mean())
                - float(writer_stats["k_abs_mean"])
            ),
            "v_abs_mean_diff": abs(
                float(payload["v"].float().abs().mean())
                - float(writer_stats["v_abs_mean"])
            ),
        }
    bank_stats_max_diff = max(
        max(item.values()) for item in bank_abs_mean_differences.values()
    )
    first_active = min(
        int(item["target_chunk"])
        for item in metadata["strong_latent"]["mapkv"]["selections"]
        if item["status"] == "scheduled_visible_support"
    )
    validity = {
        "fixed_source_chunk_8": all(
            all(int(item["source_chunk"]) == source_chunk for item in meta["mapkv"]["selections"])
            for name, meta in metadata.items()
            if name != "baseline"
        ),
        "strong_rgb_canonical_identical_geometry_masks": (
            _coverage_signature(metadata["strong_latent"])
            == _coverage_signature(metadata["rgb_warp_vae"])
            == _coverage_signature(metadata["canonical_kv"])
        ),
        "prefix_exact_before_first_visible_memory": all(
            _prefix_exact(metadata[name], first_active)
            for name in ("strong_latent", "rgb_warp_vae", "canonical_kv")
        ),
        "runtime_cache_unchanged": all(
            audit["unchanged"]
            for name in ("strong_latent", "rgb_warp_vae", "canonical_kv")
            for audit in metadata[name]["mapkv"]["cache_audits"].values()
        ),
        "canonical_source_native_reconstruction_exact": (
            float(canonical_audit["source_reconstruction_max_abs_diff"]) == 0.0
        ),
        "canonical_source_matches_original_b1_bank_stats": (
            bank_stats_max_diff <= 1e-6
        ),
        "future_leakage": any(
            frame["future_geometry_used"]
            for item in metadata["canonical_kv"]["mapkv"]["selections"]
            for frame in item["geometry_frames"]
        ),
        "first_active_chunk": first_active,
    }
    valid = all(
        value
        for key, value in validity.items()
        if key not in {"future_leakage", "first_active_chunk"}
    ) and not validity["future_leakage"]

    current = metrics["current_masked"]
    strong = metrics["strong_latent"]
    rgb = metrics["rgb_warp_vae"]
    canonical = metrics["canonical_kv"]
    p1_identity_gain = strong["overlap_b1_to_b2_l1"] < current["overlap_b1_to_b2_l1"]
    p1_locality_cost = strong["nonoverlap_delta_vs_baseline_l1"] > current[
        "nonoverlap_delta_vs_baseline_l1"
    ]
    p2_rgb_better = (
        rgb["overlap_b1_to_b2_l1"] < strong["overlap_b1_to_b2_l1"]
        and abs(
            rgb["nonoverlap_delta_vs_baseline_l1"]
            - strong["nonoverlap_delta_vs_baseline_l1"]
        ) < 0.01
    )
    canonical_gap = canonical["overlap_b1_to_b2_l1"] - rgb[
        "overlap_b1_to_b2_l1"
    ]
    p3_approximates_oracle = canonical_gap <= 0.005
    status = (
        "INVALID"
        if not valid
        else (
            "RGB_WARP_QUALITY_ORACLE_CANONICAL_K_GAP"
            if p2_rgb_better and not p3_approximates_oracle
            else "IDENTITY_RECOVERY_INCONCLUSIVE"
        )
    )
    result = {
        "status": status,
        "validity": validity,
        "trajectory": {
            "source_chunk": source_chunk,
            "target_chunk": target_chunk,
            "source_yaw_deg": 30.0,
            "target_yaw_deg": 20.0,
            "homography_source_to_target": homography.tolist(),
        },
        "masks": {
            "hard_fraction": float(hard.mean()),
            "memory_fraction": float(memory.mean()),
            "query_fraction": float(query.mean()),
            "overlap_metric_fraction": float(overlap.mean()),
            "nonoverlap_metric_fraction": float(nonoverlap.mean()),
        },
        "methods": metrics,
        "priority_1": {
            "mask_attenuation_detected": p1_identity_gain,
            "strong_core_has_locality_cost": p1_locality_cost,
            "overlap_gain": current["overlap_b1_to_b2_l1"]
            - strong["overlap_b1_to_b2_l1"],
        },
        "priority_2": {
            "rgb_warp_vae_better_than_latent_warp": p2_rgb_better,
            "overlap_gain": strong["overlap_b1_to_b2_l1"]
            - rgb["overlap_b1_to_b2_l1"],
            "quality_oracle": "rgb_warp_vae" if p2_rgb_better else "strong_latent",
        },
        "priority_3": {
            "canonical_approximates_quality_oracle": p3_approximates_oracle,
            "canonical_overlap_gap_to_rgb": canonical_gap,
            "source_reconstruction_max_abs_diff": canonical_audit[
                "source_reconstruction_max_abs_diff"
            ],
            "memory_bytes": canonical_audit["memory_bytes"],
            "original_b1_bank_abs_mean_max_diff": bank_stats_max_diff,
            "original_b1_bank_abs_mean_by_layer": bank_abs_mean_differences,
        },
        "identity_crops": crop_manifest,
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


__all__ = ["METHOD_ROOTS", "evaluate_identity_recovery"]
