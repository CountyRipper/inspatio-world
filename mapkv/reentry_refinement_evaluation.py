from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

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
    "current_source_protected": "generation/current_source_protected",
    "reentry_only": "generation/reentry_only",
    "view_adaptive": "generation/view_adaptive",
    "edge_safe": "generation/edge_safe",
    "final_step": "generation/final_step",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_map(case: Path) -> dict[str, dict]:
    payload = _json(case / "phase_labels.json")
    return {item["name"]: item for item in payload["phases"]}


def _method_root(root: Path, method: str) -> Path:
    return root / METHOD_ROOTS[method]


def _active_reentry(metadata: dict) -> dict:
    active = [
        item
        for item in metadata["mapkv"]["selections"]
        if item["status"] == "scheduled_reentry_read"
    ]
    if len(active) != 1:
        raise RuntimeError(
            f"Expected exactly one re-entry read, found {len(active)}"
        )
    return active[0]


def _prefix_exact(metadata: dict, first_active: int) -> bool:
    values = metadata["replay"]["against_saved_latents"][
        "per_chunk_max_abs_diff"
    ]
    return all(
        float(values[str(chunk)]) == 0.0 for chunk in range(first_active)
    )


def _timeline_asset(
    *,
    root: Path,
    selections: list[dict],
    ypr: np.ndarray,
    latent_length: int,
    rgb_length: int,
    frames_per_block: int,
) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for item in selections:
        chunk = int(item["target_chunk"])
        latent_center = chunk * frames_per_block + frames_per_block // 2
        rgb = int(
            round(
                latent_center
                * (rgb_length - 1)
                / max(latent_length - 1, 1)
            )
        )
        rows.append(
            {
                "chunk": chunk,
                "yaw_degrees": float(ypr[rgb, 0]),
                "historical_visibility_fraction": float(
                    item["historical_visibility_fraction"]
                ),
                "absence_count": int(item["absence_count"]),
                "state_before": str(item["state_before"]),
                "state_after": str(item["state_after"]),
                "selected_source_chunk": item.get("selected_source_chunk"),
                "read_coverage_fraction": float(
                    item.get("read_coverage_fraction", 0.0)
                ),
                "status": str(item["status"]),
            }
        )
    asset_root = root / "assets/reentry_refinement"
    asset_root.mkdir(parents=True, exist_ok=True)
    (root / "lifecycle_timeline.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    chunks = [item["chunk"] for item in rows]
    yaw = [item["yaw_degrees"] for item in rows]
    visibility = [item["historical_visibility_fraction"] for item in rows]
    read = [item["read_coverage_fraction"] for item in rows]
    fig, left = plt.subplots(figsize=(12, 4.5))
    left.plot(chunks, yaw, color="#325aa8", marker="o", label="yaw (deg)")
    left.set(xlabel="latent chunk", ylabel="yaw (degrees)")
    right = left.twinx()
    right.plot(
        chunks,
        visibility,
        color="#d97706",
        marker=".",
        label="historical visibility",
    )
    right.bar(
        chunks,
        read,
        color="#2f855a",
        alpha=0.55,
        label="long-term read coverage",
    )
    right.set(ylabel="coverage fraction", ylim=(0, 1))
    for item in rows:
        if item["state_after"] == "ABSENT":
            left.axvline(item["chunk"], color="#718096", alpha=0.18)
        if item["read_coverage_fraction"] > 0:
            left.axvline(
                item["chunk"], color="#6b46c1", linestyle="--", linewidth=2
            )
            left.text(
                item["chunk"],
                max(yaw) + 2,
                f"read src={item['selected_source_chunk']}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    handles = left.get_lines() + right.get_lines() + list(right.containers)
    labels = [handle.get_label() for handle in handles]
    left.legend(handles, labels, loc="lower right")
    left.set_title("Re-entry lifecycle: visibility → absence → one-shot read → handoff")
    fig.tight_layout()
    fig.savefig(asset_root / "lifecycle_timeline.png", dpi=160)
    plt.close(fig)
    return rows


def _save_right_edge_crops(
    *,
    assets: Path,
    images: dict[str, np.ndarray],
    fraction: float = 0.32,
) -> dict:
    height, width = next(iter(images.values())).shape[:2]
    left = int(round(width * (1.0 - fraction)))
    box = [left, 0, width, height]
    root = assets / "right_edge"
    root.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        _save_rgb(root / f"{name}.png", image[:, left:width])
    return {"box_xyxy": box, "fraction": float(fraction)}


def evaluate_reentry_refinement(
    *,
    run_root: str | Path,
    case_dir: str | Path,
) -> dict:
    root = Path(run_root).resolve()
    case = Path(case_dir).resolve()
    manifest = _json(case / "trajectory_manifest.json")
    phases = _phase_map(case)
    anchor_chunk = int(manifest["source_chunk"])
    target_chunk = int(manifest["target_chunk"])
    metadata = {
        method: _json(_method_root(root, method) / "run_metadata.json")
        for method in METHOD_ROOTS
    }
    edge_active = _active_reentry(metadata["edge_safe"])
    selected_chunk = int(edge_active["selected_source_chunk"])
    read_chunk = int(edge_active["target_chunk"])

    mapping = root / "baseline/block_mapping.json"
    selected_image = _image(
        root / "baseline/keyframes" / f"chunk_{selected_chunk:04d}.png"
    )
    anchor_image = _image(
        root / "baseline/keyframes" / f"chunk_{anchor_chunk:04d}.png"
    )
    baseline_target = _image(
        root / "baseline/keyframes" / f"chunk_{target_chunk:04d}.png"
    )
    image_hw = selected_image.shape[:2]
    selected_pose = np.asarray(_block(mapping, selected_chunk)["c2w"])
    anchor_pose = np.asarray(_block(mapping, anchor_chunk)["c2w"])
    target_pose = np.asarray(_block(mapping, target_chunk)["c2w"])
    intrinsics = _intrinsics(case / "intrinsics.txt", image_hw)
    warped_source, warp_valid, homography = _rotation_warp(
        selected_image, selected_pose, target_pose, intrinsics
    )
    warped_anchor, anchor_warp_valid, anchor_homography = _rotation_warp(
        anchor_image, anchor_pose, target_pose, intrinsics
    )

    sequence = _json(root / "cut3r/sequence.json")
    geometry_frame = next(
        item
        for item in sequence["frames"]
        if int(item["chunk_id"]) == selected_chunk
    )
    anchor_geometry_frame = next(
        item
        for item in sequence["frames"]
        if int(item["chunk_id"]) == anchor_chunk
    )
    index = SurfelIndex.load(root / "surfel/surfel_index.npz")
    selected_group = index.generated_only_cell_indices(
        selected_chunk, reference_blind_threshold=0.75
    )
    coverage_latent, coverage_audit = _surfel_coverage_for_pose(
        surfel_index=index,
        source_chunk=selected_chunk,
        target_chunk=target_chunk,
        query_pose=to_cut3r_c2w(target_pose),
        intrinsics=np.asarray(geometry_frame["intrinsics"], dtype=np.float64),
        source_image_hw=tuple(int(v) for v in geometry_frame["shape"]),
        target_hw=(60, 104),
        generated_only=False,
        eligible_indices=selected_group,
    )
    anchor_group = index.generated_only_cell_indices(
        anchor_chunk, reference_blind_threshold=0.5
    )
    anchor_coverage_latent, anchor_coverage_audit = (
        _surfel_coverage_for_pose(
            surfel_index=index,
            source_chunk=anchor_chunk,
            target_chunk=target_chunk,
            query_pose=to_cut3r_c2w(target_pose),
            intrinsics=np.asarray(
                anchor_geometry_frame["intrinsics"], dtype=np.float64
            ),
            source_image_hw=tuple(
                int(v) for v in anchor_geometry_frame["shape"]
            ),
            target_hw=(60, 104),
            generated_only=False,
            eligible_indices=anchor_group,
        )
    )
    history = cv2.resize(
        coverage_latent.numpy(),
        (image_hw[1], image_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    anchor_history = cv2.resize(
        anchor_coverage_latent.numpy(),
        (image_hw[1], image_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    ref_valid = (
        np.asarray(
            cv2.imread(
                str(
                    root
                    / "baseline/masks"
                    / f"chunk_{target_chunk:04d}_reference_valid.png"
                ),
                cv2.IMREAD_GRAYSCALE,
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    ref_protected = cv2.dilate(
        (ref_valid > 0).astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(np.float32)
    valid_safe = cv2.erode(
        (warp_valid > 0).astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(np.float32)
    revisit = (
        (history > 0).astype(np.float32)
        * (1.0 - ref_protected)
        * valid_safe
    )
    anchor_revisit = (
        (anchor_history > 0).astype(np.float32)
        * (1.0 - ref_protected)
        * (anchor_warp_valid > 0).astype(np.float32)
    )
    source_region = (ref_valid > 0.5).astype(np.float32)
    if float(revisit.mean()) <= 0 or float(anchor_revisit.mean()) <= 0:
        raise RuntimeError("No true generated-history revisit evaluation region")

    assets = root / "assets/reentry_refinement"
    assets.mkdir(parents=True, exist_ok=True)
    _save_rgb(assets / "anchor_b1_chunk.png", anchor_image)
    _save_rgb(assets / "selected_observation.png", selected_image)
    _save_rgb(assets / "selected_observation_warped_to_b2.png", warped_source)
    _save_rgb(assets / "anchor_b1_warped_to_b2.png", warped_anchor)
    _save_rgb(assets / "baseline_b2.png", baseline_target)
    _save_mask(assets / "M_history_b2.png", history)
    _save_mask(assets / "M_ref_valid_b2.png", ref_valid)
    _save_mask(assets / "M_revisit_eval.png", revisit)
    _save_mask(assets / "M_anchor_revisit_eval.png", anchor_revisit)
    _save_mask(assets / "M_warp_valid_eroded.png", valid_safe)

    images = {
        "anchor_b1": anchor_image,
        "selected_observation": selected_image,
        "selected_warped": warped_source,
    }
    metrics = {}
    leave = phases["B1_to_Leave"]
    reentry = phases["Leave_to_B2"]
    b2 = phases["B2_hold"]
    right_mask = np.zeros(image_hw, dtype=np.float32)
    right_mask[:, int(round(image_hw[1] * 0.68)) :] = 1.0
    right_revisit = right_mask * revisit
    anchor_right_revisit = right_mask * anchor_revisit
    for method in METHOD_ROOTS:
        method_root = _method_root(root, method)
        target = _image(
            method_root / "keyframes" / f"chunk_{target_chunk:04d}.png"
        )
        images[method] = target
        leave_transition = _video_transition(
            method_root / "pred.mp4",
            int(leave["rgb_start"]),
            window_start=int(phases["B1_hold"]["rgb_start"]),
            window_stop=int(leave["rgb_stop_exclusive"]),
        )
        reentry_transition = _video_transition(
            method_root / "pred.mp4",
            int(reentry["rgb_start"]),
            window_start=int(reentry["rgb_start"]),
            window_stop=int(b2["rgb_stop_exclusive"]),
        )
        anchor_l1 = _masked_l1(warped_anchor, target, anchor_revisit)
        adaptive_l1 = _masked_l1(warped_source, target, revisit)
        anchor_right_l1 = (
            _masked_l1(warped_anchor, target, anchor_right_revisit)
            if float(anchor_right_revisit.mean()) > 0
            else None
        )
        adaptive_right_l1 = (
            _masked_l1(warped_source, target, right_revisit)
            if float(right_revisit.mean()) > 0
            else None
        )
        uses_anchor = method in {
            "baseline",
            "current_source_protected",
            "reentry_only",
        }
        metrics[method] = {
            "revisit_reference_chunk": (
                anchor_chunk if uses_anchor else selected_chunk
            ),
            "revisit_region_b1_to_b2_l1": (
                anchor_l1 if uses_anchor else adaptive_l1
            ),
            "anchor_revisit_region_l1": anchor_l1,
            "adaptive_revisit_region_l1": adaptive_l1,
            "source_region_delta_vs_baseline_l1": _masked_l1(
                baseline_target, target, source_region
            ),
            "right_edge_revisit_l1": (
                anchor_right_l1 if uses_anchor else adaptive_right_l1
            ),
            "anchor_right_edge_revisit_l1": anchor_right_l1,
            "adaptive_right_edge_revisit_l1": adaptive_right_l1,
            "whole_delta_vs_baseline_l1": float(
                np.abs(baseline_target - target).mean()
            ),
            "leave_window_mean_l1": leave_transition[
                "reentry_window_mean_l1"
            ],
            "leave_window_peak_l1": leave_transition[
                "reentry_window_peak_l1"
            ],
            "leave_peak_rgb_frame": leave_transition[
                "reentry_window_peak_rgb_frame"
            ],
            "reentry_window_mean_l1": reentry_transition[
                "reentry_window_mean_l1"
            ],
            "reentry_window_peak_l1": reentry_transition[
                "reentry_window_peak_l1"
            ],
            "reentry_peak_rgb_frame": reentry_transition[
                "reentry_window_peak_rgb_frame"
            ],
            "generation_seconds": float(
                metadata[method]["timing_seconds"]["total"]
            ),
        }
    right_edge = _save_right_edge_crops(assets=assets, images=images)
    ypr = np.load(case / "yaw_pitch_roll.npy")
    lifecycle = _timeline_asset(
        root=root,
        selections=metadata["edge_safe"]["mapkv"]["selections"],
        ypr=ypr,
        latent_length=int(metadata["baseline"]["latent_length"]),
        rgb_length=int(metadata["baseline"]["decoded_rgb_length"]),
        frames_per_block=int(metadata["baseline"]["frames_per_block"]),
    )

    baseline = metrics["baseline"]
    current = metrics["current_source_protected"]
    reentry_only = metrics["reentry_only"]
    adaptive = metrics["view_adaptive"]
    edge = metrics["edge_safe"]
    final = metrics["final_step"]
    current_memory_gain = max(
        baseline["anchor_revisit_region_l1"]
        - current["anchor_revisit_region_l1"],
        0.0,
    )
    reentry_memory_gain = max(
        baseline["anchor_revisit_region_l1"]
        - reentry_only["anchor_revisit_region_l1"],
        0.0,
    )
    memory_gain_retention = (
        reentry_memory_gain / current_memory_gain
        if current_memory_gain > 1e-8
        else 0.0
    )
    identity_retained = memory_gain_retention >= 0.5
    first_departure_fixed = (
        reentry_only["leave_window_peak_l1"]
        <= 1.01 * baseline["leave_window_peak_l1"]
    )
    policy_works = (
        identity_retained
        and first_departure_fixed
        and reentry_only["reentry_window_peak_l1"]
        <= current["reentry_window_peak_l1"]
    )
    right_edge_improved = (
        edge["adaptive_right_edge_revisit_l1"] is not None
        and adaptive["adaptive_right_edge_revisit_l1"] is not None
        and edge["adaptive_right_edge_revisit_l1"]
        <= adaptive["adaptive_right_edge_revisit_l1"]
    )
    edge_fix_works = (
        right_edge_improved
        and edge["adaptive_revisit_region_l1"]
        <= max(
            baseline["adaptive_revisit_region_l1"] * 0.95,
            baseline["adaptive_revisit_region_l1"] - 0.005,
        )
        and edge["reentry_window_peak_l1"]
        <= current["reentry_window_peak_l1"]
    )
    final_step_useful = (
        final["source_region_delta_vs_baseline_l1"]
        < edge["source_region_delta_vs_baseline_l1"]
        and final["reentry_window_peak_l1"]
        <= edge["reentry_window_peak_l1"]
        and final["adaptive_revisit_region_l1"]
        < baseline["adaptive_revisit_region_l1"]
    )
    selected_sources = {
        method: int(_active_reentry(metadata[method])["selected_source_chunk"])
        for method in ("reentry_only", "view_adaptive", "edge_safe", "final_step")
    }
    active_chunks = {
        method: int(_active_reentry(metadata[method])["target_chunk"])
        for method in ("reentry_only", "view_adaptive", "edge_safe", "final_step")
    }
    validity = {
        "trajectory_exact": (
            float(manifest["b1_theta_degrees"]) == 45.0
            and float(manifest["leave_theta_degrees"]) == -20.0
            and float(manifest["b2_theta_degrees"]) == 35.0
        ),
        "one_reentry_read_per_method": len(set(active_chunks.values())) == 1,
        "fixed_source_control_is_anchor": (
            selected_sources["reentry_only"] == anchor_chunk
        ),
        "adaptive_source_locked": all(
            _active_reentry(metadata[method])["source_locked_for_episode"]
            for method in ("view_adaptive", "edge_safe", "final_step")
        ),
        "edge_safe_border_padding": (
            _active_reentry(metadata["edge_safe"])["rgb_padding_mode"]
            == "border"
        ),
        "edge_safe_threshold_075": (
            float(
                metadata["edge_safe"]["mapkv"]["warp_reencode"][
                    "generated_only_threshold"
                ]
            )
            == 0.75
        ),
        "prefix_exact_before_read": all(
            _prefix_exact(metadata[method], active_chunks[method])
            for method in (
                "reentry_only",
                "view_adaptive",
                "edge_safe",
                "final_step",
            )
        ),
        "runtime_cache_unchanged": all(
            audit["unchanged"]
            for method in (
                "reentry_only",
                "view_adaptive",
                "edge_safe",
                "final_step",
            )
            for audit in metadata[method]["mapkv"]["cache_audits"].values()
        ),
        "future_leakage": any(
            bool(frame.get("future_geometry_used", False))
            for method in ("reentry_only", "view_adaptive", "edge_safe")
            for audit in metadata[method]["mapkv"]["warp_reencode"][
                "audits"
            ].values()
            for frame in audit["geometry"]["per_frame"]
        ),
    }
    if not all(
        value for key, value in validity.items() if key != "future_leakage"
    ) or validity["future_leakage"]:
        raise RuntimeError(f"Invalid re-entry refinement run: {validity}")

    if not policy_works:
        primary_status = "REENTRY_HANDOFF_INSUFFICIENT"
    elif not edge_fix_works:
        primary_status = "VIEW_ADAPTIVE_EDGE_FIX_NOT_WORKING"
    else:
        primary_status = "VIEW_ADAPTIVE_EDGE_FIX_WORKS"
    statuses = [primary_status]
    if final_step_useful:
        statuses.append("FINAL_STEP_STABILIZATION_USEFUL")
    result = {
        "status": primary_status,
        "statuses": statuses,
        "validity": validity,
        "trajectory": {
            "anchor_chunk": anchor_chunk,
            "selected_source_chunk": selected_chunk,
            "read_chunk": read_chunk,
            "target_chunk": target_chunk,
            "history_gap_chunks": target_chunk - selected_chunk,
            "homography_source_to_target": homography.tolist(),
            "anchor_homography_source_to_target": (
                anchor_homography.tolist()
            ),
        },
        "observation_selection": {
            "fixed_source_chunk": anchor_chunk,
            "selected_sources": selected_sources,
            "active_chunks": active_chunks,
            "edge_safe_top3": edge_active.get("candidate_scores", [])[:3],
        },
        "regions": {
            "source_fraction": float(source_region.mean()),
            "true_revisit_fraction": float(revisit.mean()),
            "anchor_true_revisit_fraction": float(anchor_revisit.mean()),
            "right_edge_revisit_fraction": float(right_revisit.mean()),
            "coverage_audit": coverage_audit,
            "anchor_coverage_audit": anchor_coverage_audit,
            "right_edge": right_edge,
        },
        "methods": metrics,
        "decisions": {
            "reentry_policy_works": policy_works,
            "view_adaptive_edge_fix_works": edge_fix_works,
            "right_edge_metric_improved": right_edge_improved,
            "final_step_stabilization_useful": final_step_useful,
            "identity_retained_after_handoff": identity_retained,
            "first_departure_fixed": first_departure_fixed,
            "current_continuous_memory_gain": current_memory_gain,
            "reentry_only_memory_gain": reentry_memory_gain,
            "memory_gain_retention_ratio": memory_gain_retention,
        },
        "lifecycle_timeline": lifecycle,
        "baseline_reference": {
            "anchor_revisit_region_l1": baseline[
                "anchor_revisit_region_l1"
            ],
            "adaptive_revisit_region_l1": baseline[
                "adaptive_revisit_region_l1"
            ],
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "status.json").write_text(
        json.dumps(
            {"status": result["status"], "statuses": statuses},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result


__all__ = ["METHOD_ROOTS", "evaluate_reentry_refinement"]
