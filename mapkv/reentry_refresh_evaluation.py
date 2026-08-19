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
    "current_continuous": "generation/current_continuous",
    "one_shot": "generation/one_shot",
    "episode_continuous": "generation/episode_continuous",
    "per_surface_ttl": "generation/per_surface_ttl",
    "same_surface_adaptive": "generation/same_surface_adaptive",
    "edge_safe": "generation/edge_safe",
    "final_step": "generation/final_step",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_map(case: Path) -> dict[str, dict]:
    payload = _json(case / "phase_labels.json")
    return {item["name"]: item for item in payload["phases"]}


def _active_reads(metadata: dict) -> list[dict]:
    return [
        item
        for item in metadata["mapkv"]["selections"]
        if item["status"] == "scheduled_reentry_read"
    ]


def _prefix_exact(metadata: dict, first_active: int) -> bool:
    values = metadata["replay"]["against_saved_latents"][
        "per_chunk_max_abs_diff"
    ]
    return all(
        float(values[str(chunk)]) == 0.0 for chunk in range(first_active)
    )


def _timeline(
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
                "state_before": str(item["state_before"]),
                "state_after": str(item["state_after"]),
                "selected_source_chunk": item.get("selected_source_chunk"),
                "read_coverage_fraction": float(
                    item.get("read_coverage_fraction", 0.0)
                ),
                "visible_surface_count": int(
                    item.get("visible_surface_count", 0)
                ),
                "active_refresh_surface_count": int(
                    item.get("active_refresh_surface_count", 0)
                ),
                "newly_reentered_surface_count": int(
                    item.get("newly_reentered_surface_count", 0)
                ),
                "status": str(item["status"]),
            }
        )
    assets = root / "assets/reentry_refresh"
    assets.mkdir(parents=True, exist_ok=True)
    (root / "lifecycle_timeline.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    chunks = [item["chunk"] for item in rows]
    yaw = [item["yaw_degrees"] for item in rows]
    visibility = [item["historical_visibility_fraction"] for item in rows]
    read = [item["read_coverage_fraction"] for item in rows]
    fig, left = plt.subplots(figsize=(13, 4.8))
    left.plot(chunks, yaw, color="#325aa8", marker="o", label="yaw (deg)")
    left.set(xlabel="latent chunk", ylabel="yaw (degrees)")
    right = left.twinx()
    right.plot(
        chunks,
        visibility,
        color="#d97706",
        marker=".",
        label="anchor visibility",
    )
    right.bar(
        chunks,
        read,
        color="#2f855a",
        alpha=0.55,
        label="active refresh coverage",
    )
    right.set(ylabel="coverage fraction", ylim=(0, 1))
    handles = left.get_lines() + right.get_lines() + list(right.containers)
    labels = [handle.get_label() for handle in handles]
    left.legend(handles, labels, loc="lower right")
    left.set_title(
        "Re-entry refresh: write-only first visit, absence, active refresh"
    )
    fig.tight_layout()
    fig.savefig(assets / "lifecycle_timeline.png", dpi=160)
    plt.close(fig)
    return rows


def evaluate_reentry_refresh(
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
    available = {
        method: relative
        for method, relative in METHOD_ROOTS.items()
        if (root / relative / "run_metadata.json").exists()
    }
    required = {
        "baseline",
        "current_continuous",
        "one_shot",
        "episode_continuous",
    }
    if not required.issubset(available):
        raise FileNotFoundError(
            f"Priority-1 methods missing: {sorted(required - set(available))}"
        )
    metadata = {
        method: _json(root / relative / "run_metadata.json")
        for method, relative in available.items()
    }

    mapping = root / "baseline/block_mapping.json"
    anchor_image = _image(
        root / "baseline/keyframes" / f"chunk_{anchor_chunk:04d}.png"
    )
    baseline_target = _image(
        root / "baseline/keyframes" / f"chunk_{target_chunk:04d}.png"
    )
    image_hw = anchor_image.shape[:2]
    anchor_pose = np.asarray(_block(mapping, anchor_chunk)["c2w"])
    target_pose = np.asarray(_block(mapping, target_chunk)["c2w"])
    intrinsics = _intrinsics(case / "intrinsics.txt", image_hw)
    warped_anchor, anchor_warp_valid, homography = _rotation_warp(
        anchor_image, anchor_pose, target_pose, intrinsics
    )

    sequence = _json(root / "cut3r/sequence.json")
    geometry_frame = next(
        item
        for item in sequence["frames"]
        if int(item["chunk_id"]) == anchor_chunk
    )
    index = SurfelIndex.load(root / "surfel/surfel_index.npz")
    anchor_group = index.generated_only_cell_indices(
        anchor_chunk, reference_blind_threshold=0.5
    )
    coverage_latent, coverage_audit = _surfel_coverage_for_pose(
        surfel_index=index,
        source_chunk=anchor_chunk,
        target_chunk=target_chunk,
        query_pose=to_cut3r_c2w(target_pose),
        intrinsics=np.asarray(geometry_frame["intrinsics"], dtype=np.float64),
        source_image_hw=tuple(int(value) for value in geometry_frame["shape"]),
        target_hw=(60, 104),
        generated_only=False,
        eligible_indices=anchor_group,
    )
    history = cv2.resize(
        coverage_latent.numpy(),
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
    revisit = (
        (history > 0).astype(np.float32)
        * (1.0 - ref_protected)
        * (anchor_warp_valid > 0).astype(np.float32)
    )
    source_region = (ref_valid > 0.5).astype(np.float32)
    right = np.zeros(image_hw, dtype=np.float32)
    right[:, int(round(image_hw[1] * 0.68)) :] = 1.0
    right_revisit = right * revisit
    if float(revisit.mean()) <= 0:
        raise RuntimeError("No canonical chunk-11 revisit evaluation region")

    assets = root / "assets/reentry_refresh"
    assets.mkdir(parents=True, exist_ok=True)
    _save_rgb(assets / "canonical_b1_chunk11.png", anchor_image)
    _save_rgb(assets / "canonical_b1_warped_to_b2.png", warped_anchor)
    _save_rgb(assets / "baseline_b2.png", baseline_target)
    _save_mask(assets / "M_anchor_history_b2.png", history)
    _save_mask(assets / "M_canonical_revisit_eval.png", revisit)
    _save_mask(assets / "M_ref_valid_b2.png", ref_valid)

    leave = phases["B1_to_Leave"]
    reentry = phases["Leave_to_B2"]
    b2 = phases["B2_hold"]
    metrics = {}
    b2_images = {"canonical_b1_warped": warped_anchor}
    for method, relative in available.items():
        method_root = root / relative
        target = _image(
            method_root / "keyframes" / f"chunk_{target_chunk:04d}.png"
        )
        b2_images[method] = target
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
        reads = _active_reads(metadata[method]) if method != "baseline" else []
        metrics[method] = {
            "canonical_revisit_region_l1": _masked_l1(
                warped_anchor, target, revisit
            ),
            "source_region_delta_vs_baseline_l1": _masked_l1(
                baseline_target, target, source_region
            ),
            "right_edge_revisit_l1": (
                _masked_l1(warped_anchor, target, right_revisit)
                if float(right_revisit.mean()) > 0
                else None
            ),
            "whole_delta_vs_baseline_l1": float(
                np.abs(baseline_target - target).mean()
            ),
            "leave_window_mean_l1": leave_transition[
                "reentry_window_mean_l1"
            ],
            "leave_window_peak_l1": leave_transition[
                "reentry_window_peak_l1"
            ],
            "reentry_window_mean_l1": reentry_transition[
                "reentry_window_mean_l1"
            ],
            "reentry_window_peak_l1": reentry_transition[
                "reentry_window_peak_l1"
            ],
            "generation_seconds": float(
                metadata[method]["timing_seconds"]["total"]
            ),
            "active_read_chunks": [
                int(item["target_chunk"]) for item in reads
            ],
            "mean_active_read_coverage": (
                float(
                    np.mean(
                        [
                            float(item.get("read_coverage_fraction", 0.0))
                            for item in reads
                        ]
                    )
                )
                if reads
                else 0.0
            ),
            "selected_source_chunks": sorted(
                {
                    int(item["selected_source_chunk"])
                    for item in reads
                    if item.get("selected_source_chunk") is not None
                }
            ),
        }

    baseline = metrics["baseline"]
    current = metrics["current_continuous"]
    one_shot = metrics["one_shot"]
    episode = metrics["episode_continuous"]
    current_gain = max(
        baseline["canonical_revisit_region_l1"]
        - current["canonical_revisit_region_l1"],
        0.0,
    )
    episode_gain = max(
        baseline["canonical_revisit_region_l1"]
        - episode["canonical_revisit_region_l1"],
        0.0,
    )
    priority1_retention = (
        episode_gain / current_gain if current_gain > 1e-8 else 0.0
    )
    priority1_success = (
        priority1_retention >= 0.8
        and episode["leave_window_peak_l1"]
        <= 1.02
        * max(
            baseline["leave_window_peak_l1"],
            one_shot["leave_window_peak_l1"],
        )
    )

    priority2_success = None
    if "per_surface_ttl" in metrics:
        per_surface = metrics["per_surface_ttl"]
        per_surface_gain = max(
            baseline["canonical_revisit_region_l1"]
            - per_surface["canonical_revisit_region_l1"],
            0.0,
        )
        priority2_success = bool(
            per_surface_gain >= 0.9 * episode_gain
            and per_surface["leave_window_peak_l1"]
            <= 1.02 * episode["leave_window_peak_l1"]
            and per_surface["mean_active_read_coverage"]
            <= episode["mean_active_read_coverage"]
        )

    priority3_success = None
    if "same_surface_adaptive" in metrics:
        source = (
            metrics["per_surface_ttl"]
            if priority2_success
            else episode
        )
        adaptive = metrics["same_surface_adaptive"]
        priority3_success = bool(
            adaptive["canonical_revisit_region_l1"]
            <= 1.05 * source["canonical_revisit_region_l1"]
            and adaptive["right_edge_revisit_l1"]
            <= source["right_edge_revisit_l1"]
        )

    edge_safe_success = None
    if "edge_safe" in metrics:
        source_name = (
            "same_surface_adaptive"
            if priority3_success
            else (
                "per_surface_ttl"
                if priority2_success
                else "episode_continuous"
            )
        )
        source = metrics[source_name]
        edge = metrics["edge_safe"]
        edge_safe_success = bool(
            edge["right_edge_revisit_l1"]
            <= source["right_edge_revisit_l1"]
            and edge["canonical_revisit_region_l1"]
            <= 1.05 * source["canonical_revisit_region_l1"]
        )

    final_step_useful = None
    if "final_step" in metrics and "edge_safe" in metrics:
        final = metrics["final_step"]
        edge = metrics["edge_safe"]
        final_step_useful = bool(
            final["source_region_delta_vs_baseline_l1"]
            < edge["source_region_delta_vs_baseline_l1"]
            and final["reentry_window_peak_l1"]
            <= edge["reentry_window_peak_l1"]
            and final["canonical_revisit_region_l1"]
            <= 1.05 * edge["canonical_revisit_region_l1"]
        )

    if not priority1_success:
        status = "REENTRY_CONTINUOUS_REFRESH_NOT_WORKING"
    else:
        status = "REENTRY_EPISODE_CONTINUOUS_WORKS"

    timeline_source = (
        "final_step"
        if "final_step" in metadata
        else (
            "edge_safe"
            if "edge_safe" in metadata
            else (
                "same_surface_adaptive"
                if "same_surface_adaptive" in metadata
                else (
                    "per_surface_ttl"
                    if "per_surface_ttl" in metadata
                    else "episode_continuous"
                )
            )
        )
    )
    timeline = _timeline(
        root=root,
        selections=metadata[timeline_source]["mapkv"]["selections"],
        ypr=np.load(case / "yaw_pitch_roll.npy"),
        latent_length=int(metadata["baseline"]["latent_length"]),
        rgb_length=int(metadata["baseline"]["decoded_rgb_length"]),
        frames_per_block=int(metadata["baseline"]["frames_per_block"]),
    )

    validity = {
        "trajectory_exact": (
            float(manifest["b1_theta_degrees"]) == 45.0
            and float(manifest["leave_theta_degrees"]) == -20.0
            and float(manifest["b2_theta_degrees"]) == 35.0
        ),
        "canonical_identity_reference_chunk": anchor_chunk,
        "episode_fixed_source_is_anchor": (
            metrics["episode_continuous"]["selected_source_chunks"]
            == [anchor_chunk]
        ),
        "prefix_exact_before_first_read": all(
            _prefix_exact(
                metadata[method],
                min(
                    int(item["target_chunk"])
                    for item in _active_reads(metadata[method])
                ),
            )
            for method in available
            if method not in {"baseline", "current_continuous"}
            and _active_reads(metadata[method])
        ),
        "runtime_cache_unchanged": all(
            audit["unchanged"]
            for method in available
            if method != "baseline"
            for audit in metadata[method]["mapkv"]["cache_audits"].values()
        ),
        "future_leakage": any(
            bool(frame.get("future_geometry_used", False))
            for method in available
            if method not in {"baseline", "current_continuous"}
            for audit in metadata[method]["mapkv"]["warp_reencode"][
                "audits"
            ].values()
            for frame in audit["geometry"]["per_frame"]
        ),
    }
    if (
        not validity["trajectory_exact"]
        or validity["canonical_identity_reference_chunk"] != 11
        or not validity["episode_fixed_source_is_anchor"]
        or not validity["prefix_exact_before_first_read"]
        or not validity["runtime_cache_unchanged"]
        or validity["future_leakage"]
    ):
        raise RuntimeError(f"Invalid re-entry refresh run: {validity}")

    right_root = assets / "right_edge"
    right_root.mkdir(parents=True, exist_ok=True)
    right_left = int(round(image_hw[1] * 0.68))
    for method, image in b2_images.items():
        _save_rgb(right_root / f"{method}.png", image[:, right_left:])

    result = {
        "status": status,
        "validity": validity,
        "trajectory": {
            "anchor_chunk": anchor_chunk,
            "target_chunk": target_chunk,
            "homography_source_to_target": homography.tolist(),
        },
        "regions": {
            "canonical_revisit_fraction": float(revisit.mean()),
            "source_fraction": float(source_region.mean()),
            "right_edge_revisit_fraction": float(right_revisit.mean()),
            "coverage_audit": coverage_audit,
        },
        "methods": metrics,
        "decisions": {
            "priority1_episode_continuous_works": priority1_success,
            "priority1_memory_gain_retention_ratio": priority1_retention,
            "priority2_per_surface_ttl_works": priority2_success,
            "priority3_same_surface_adaptive_works": priority3_success,
            "edge_safe_support_works": edge_safe_success,
            "final_step_stabilization_useful": final_step_useful,
        },
        "lifecycle_timeline_method": timeline_source,
        "lifecycle_timeline": timeline,
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "status.json").write_text(
        json.dumps({"status": status}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


__all__ = ["METHOD_ROOTS", "evaluate_reentry_refresh"]
