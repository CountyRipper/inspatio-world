from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mapkv_proto.pose_utils import to_cut3r_c2w

from .continuous_cavr_evaluation import _video_transition
from .evaluation import _image, _masked_l1
from .locality_evaluation import (
    _block,
    _intrinsics,
    _rotation_warp,
    _save_mask,
    _save_rgb,
)
from .surfel_index import SurfelIndex
from .warp_reencode import _surfel_coverage_for_pose


METHOD_ROOTS = {
    "baseline": "baseline",
    "current_rgb_wre": "generation/current_rgb_wre",
    "source_protected": "generation/source_protected_rgb_wre",
    "middle10": "generation/source_protected_middle10",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _root(root: Path, method: str) -> Path:
    return root / METHOD_ROOTS[method]


def _state_mask(
    state: dict,
    key: str,
    image_hw: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    value = state[key].float()
    while value.ndim > 2:
        value = value.mean(dim=0)
    return cv2.resize(
        value.numpy(),
        (image_hw[1], image_hw[0]),
        interpolation=interpolation,
    ).clip(0, 1)


def _prefix_exact(metadata: dict, first_active: int) -> bool:
    values = metadata["replay"]["against_saved_latents"][
        "per_chunk_max_abs_diff"
    ]
    return all(
        float(values[str(chunk)]) == 0.0 for chunk in range(first_active)
    )


def _largest_regions(mask: np.ndarray, *, count: int = 2) -> list[list[int]]:
    binary = (mask > 0.25).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )
    labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    height, width = binary.shape
    candidates = []
    for label in range(1, labels):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < max(64, int(0.0025 * height * width)):
            continue
        pad = max(8, int(round(0.02 * max(height, width))))
        candidates.append(
            (
                area,
                [
                    max(0, x - pad),
                    max(0, y - pad),
                    min(width, x + w + pad),
                    min(height, y + h + pad),
                ],
            )
        )
    candidates.sort(reverse=True)
    if not candidates:
        ys, xs = np.nonzero(binary)
        if not len(xs):
            return []
        candidates = [
            (
                len(xs),
                [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
            )
        ]
    return [box for _, box in candidates[:count]]


def _save_region_crops(
    *,
    assets: Path,
    boxes: list[list[int]],
    images: dict[str, np.ndarray],
) -> dict:
    root = assets / "revisit_regions"
    root.mkdir(parents=True, exist_ok=True)
    for region, (left, top, right, bottom) in enumerate(boxes, start=1):
        for method, image in images.items():
            _save_rgb(
                root / f"region_{region}_{method}.png",
                image[top:bottom, left:right],
            )
    return {"boxes_xyxy": boxes, "root": str(root)}


def evaluate_source_protected_revisit(
    *,
    run_root: str | Path,
    case_dir: str | Path,
) -> dict:
    root = Path(run_root).resolve()
    case = Path(case_dir).resolve()
    manifest = _json(case / "trajectory_manifest.json")
    phases = _json(case / "phase_labels.json")
    source_chunk = int(manifest["source_chunk"])
    target_chunk = int(manifest["target_chunk"])
    source = _image(
        root / "baseline/keyframes" / f"chunk_{source_chunk:04d}.png"
    )
    baseline_target = _image(
        root / "baseline/keyframes" / f"chunk_{target_chunk:04d}.png"
    )
    image_hw = source.shape[:2]
    mapping = root / "baseline/block_mapping.json"
    source_pose = np.asarray(_block(mapping, source_chunk)["c2w"])
    target_pose = np.asarray(_block(mapping, target_chunk)["c2w"])
    intrinsics = _intrinsics(case / "intrinsics.txt", image_hw)
    warped_b1, warp_valid, homography = _rotation_warp(
        source, source_pose, target_pose, intrinsics
    )

    protected_root = _root(root, "source_protected")
    state = torch.load(
        protected_root
        / "warp"
        / f"target_{target_chunk:04d}"
        / "warp_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    m_history = _state_mask(state, "history_coverage", image_hw)
    m_ref_valid = _state_mask(state, "reference_valid_coverage", image_hw)
    m_ref_protected = _state_mask(
        state, "reference_protected_coverage", image_hw
    )
    m_need = _state_mask(state, "need_coverage", image_hw)
    m_memory = _state_mask(state, "memory_coverage", image_hw)
    m_query = _state_mask(
        state,
        "query_gate_tokens",
        image_hw,
        interpolation=cv2.INTER_LINEAR,
    )
    m_source = (m_ref_valid > 0.5).astype(np.float32)
    m_revisit = (m_need > 0.25).astype(np.float32) * warp_valid
    if float(m_source.mean()) <= 0:
        raise RuntimeError("The benchmark has no source-valid evaluation region")
    if float(m_revisit.mean()) <= 0:
        raise RuntimeError("The benchmark has no true generated-history revisit region")

    assets = root / "assets/source_protected"
    assets.mkdir(parents=True, exist_ok=True)
    _save_rgb(assets / "b1_first_visit.png", source)
    _save_rgb(assets / "b1_warped_to_b2.png", warped_b1)
    _save_rgb(assets / "b2_baseline.png", baseline_target)
    for name, value in {
        "B1_reference_blind": 1.0
        - (
            np.asarray(
                cv2.imread(
                    str(
                        root
                        / "baseline/masks"
                        / f"chunk_{source_chunk:04d}_reference_valid.png"
                    ),
                    cv2.IMREAD_GRAYSCALE,
                ),
                dtype=np.float32,
            )
            / 255.0
        ),
        "M_history": m_history,
        "M_ref_valid": m_ref_valid,
        "M_ref_protected": m_ref_protected,
        "M_need": m_need,
        "M_memory": m_memory,
        "M_query": m_query,
        "M_source_eval": m_source,
        "M_revisit_eval": m_revisit,
    }.items():
        _save_mask(assets / f"{name}.png", value)

    sequence = _json(root / "cut3r/sequence.json")
    source_frame = next(
        item for item in sequence["frames"]
        if int(item["chunk_id"]) == source_chunk
    )
    index = SurfelIndex.load(root / "surfel/surfel_index.npz")
    b1_generated, b1_geometry = _surfel_coverage_for_pose(
        surfel_index=index,
        source_chunk=source_chunk,
        target_chunk=source_chunk + 2,
        query_pose=to_cut3r_c2w(source_pose),
        intrinsics=np.asarray(source_frame["intrinsics"], dtype=np.float64),
        source_image_hw=tuple(int(v) for v in source_frame["shape"]),
        target_hw=(60, 104),
        generated_only=True,
    )
    _save_mask(
        assets / "B1_generated_only_surfels.png",
        cv2.resize(
            b1_generated.numpy(),
            (image_hw[1], image_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        ),
    )

    leave_to_b2 = next(
        item for item in phases["phases"] if item["name"] == "Leave_to_B2"
    )
    b2_hold = next(
        item for item in phases["phases"] if item["name"] == "B2_hold"
    )
    metrics = {}
    metadata = {}
    crop_images = {
        "b1_first": source,
        "b1_warped": warped_b1,
    }
    for method in METHOD_ROOTS:
        method_root = _root(root, method)
        target = _image(
            method_root / "keyframes" / f"chunk_{target_chunk:04d}.png"
        )
        crop_images[method] = target
        meta = _json(method_root / "run_metadata.json")
        metadata[method] = meta
        transition = _video_transition(
            method_root / "pred.mp4",
            int(b2_hold["rgb_start"]),
            window_start=int(leave_to_b2["rgb_start"]),
            window_stop=int(b2_hold["rgb_stop_exclusive"]),
        )
        metrics[method] = {
            "revisit_region_b1_to_b2_l1": _masked_l1(
                warped_b1, target, m_revisit
            ),
            "source_region_delta_vs_baseline_l1": _masked_l1(
                baseline_target, target, m_source
            ),
            "whole_delta_vs_baseline_l1": float(
                np.abs(baseline_target - target).mean()
            ),
            "reentry_mean_l1": transition["reentry_window_mean_l1"],
            "reentry_peak_l1": transition["reentry_window_peak_l1"],
            "generation_seconds": float(meta["timing_seconds"]["total"]),
        }
    boxes = _largest_regions(m_revisit)
    crops = _save_region_crops(
        assets=assets, boxes=boxes, images=crop_images
    )

    protected_meta = metadata["source_protected"]
    active = [
        item
        for item in protected_meta["mapkv"]["selections"]
        if item["status"] == "scheduled_visible_support"
    ]
    first_active = min(int(item["target_chunk"]) for item in active)
    gate_tokens = state["query_gate_tokens"].float()
    protected_latent = state["reference_protected_coverage"].float()
    batch, frames = protected_latent.shape[:2]
    protected_tokens = torch.nn.functional.adaptive_max_pool2d(
        protected_latent.reshape(batch * frames, 1, *protected_latent.shape[-2:]),
        gate_tokens.shape[-2:],
    ).reshape_as(gate_tokens)
    protected_gate_max = float(
        (gate_tokens * (protected_tokens > 0).float()).max().item()
    )
    surfel_stats = _json(root / "surfel/stats.json")
    validity = {
        "trajectory_exact_0_45_m20_35": (
            manifest["trajectory_type"] == "pure_yaw_source_protected_revisit"
            and float(manifest["b1_theta_degrees"]) == 45.0
            and float(manifest["leave_theta_degrees"]) == -20.0
            and float(manifest["b2_theta_degrees"]) == 35.0
        ),
        "fixed_historical_source": all(
            int(item["source_chunk"]) == source_chunk
            for item in protected_meta["mapkv"]["selections"]
        ),
        "generated_only_observations_tagged": bool(
            surfel_stats["reference_blind_at_write"]["enabled"]
        )
        and int(
            surfel_stats["reference_blind_at_write"]["tagged_observations"]
        )
        > 0,
        "source_protection_enabled": bool(
            protected_meta["mapkv"]["warp_reencode"][
                "source_protected_memory"
            ]
        ),
        "protected_query_gate_exact_zero": protected_gate_max == 0.0,
        "prefix_exact_before_memory": all(
            _prefix_exact(metadata[method], first_active)
            for method in ("current_rgb_wre", "source_protected", "middle10")
        ),
        "runtime_cache_unchanged": all(
            audit["unchanged"]
            for method in ("current_rgb_wre", "source_protected", "middle10")
            for audit in metadata[method]["mapkv"]["cache_audits"].values()
        ),
        "future_leakage": any(
            frame["future_geometry_used"]
            for item in protected_meta["mapkv"]["selections"]
            for frame in item["geometry_frames"]
        ),
    }
    if not all(
        value for key, value in validity.items() if key != "future_leakage"
    ) or validity["future_leakage"]:
        raise RuntimeError(f"Invalid source-protected run: {validity}")

    baseline = metrics["baseline"]
    current = metrics["current_rgb_wre"]
    protected = metrics["source_protected"]
    source_protection_gain = (
        current["source_region_delta_vs_baseline_l1"]
        - protected["source_region_delta_vs_baseline_l1"]
    )
    revisit_gain = (
        baseline["revisit_region_b1_to_b2_l1"]
        - protected["revisit_region_b1_to_b2_l1"]
    )
    source_stable = (
        protected["source_region_delta_vs_baseline_l1"]
        <= max(
            0.01,
            0.5 * current["source_region_delta_vs_baseline_l1"],
        )
    )
    revisit_improved = revisit_gain > (
        0.05 * baseline["revisit_region_b1_to_b2_l1"]
    )
    status = (
        "SOURCE_PROTECTED_REVISIT_WORKS"
        if source_stable and revisit_improved
        else "TRAINING_FREE_IDENTITY_LIMITED"
    )
    result = {
        "status": status,
        "validity": validity,
        "trajectory": {
            "source_chunk": source_chunk,
            "target_chunk": target_chunk,
            "source_yaw_degrees": 45.0,
            "leave_yaw_degrees": -20.0,
            "target_yaw_degrees": 35.0,
            "history_gap_chunks": target_chunk - source_chunk,
            "homography_source_to_target": homography.tolist(),
        },
        "masks": {
            "source_fraction": float(m_source.mean()),
            "history_fraction": float(m_history.mean()),
            "reference_protected_fraction": float(m_ref_protected.mean()),
            "revisit_fraction": float(m_revisit.mean()),
            "memory_fraction": float(m_memory.mean()),
            "query_fraction": float(m_query.mean()),
            "protected_query_gate_max": protected_gate_max,
        },
        "generated_only_b1": b1_geometry,
        "methods": metrics,
        "source_protection": {
            "source_stable": source_stable,
            "source_region_gain_vs_current": source_protection_gain,
        },
        "true_revisit": {
            "improved": revisit_improved,
            "revisit_region_gain_vs_baseline": revisit_gain,
        },
        "automatic_revisit_regions": crops,
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "status.json").write_text(
        json.dumps({"status": status}, indent=2), encoding="utf-8"
    )
    return result


__all__ = ["METHOD_ROOTS", "evaluate_source_protected_revisit"]
