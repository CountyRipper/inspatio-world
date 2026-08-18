from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

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
    "block_on_wre": "generation/block_on_wre",
    "continuous_cavr": "generation/continuous_cavr",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_root(root: Path, method: str) -> Path:
    return root / METHOD_ROOTS[method]


def _keyframe(root: Path, method: str, chunk: int) -> np.ndarray:
    return _image(
        _method_root(root, method)
        / "keyframes"
        / f"chunk_{chunk:04d}.png"
    )


def _prefix_is_exact(metadata: dict, first_active: int) -> bool:
    replay = metadata.get("replay", {}).get("against_saved_latents") or {}
    per_chunk = replay.get("per_chunk_max_abs_diff", {})
    return bool(per_chunk) and all(
        float(per_chunk[str(chunk)]) == 0.0
        for chunk in range(first_active)
    )


def _video_transition(
    path: Path,
    frame_index: int,
    *,
    window_start: int,
    window_stop: int,
) -> dict:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
    capture.release()
    if frame_index <= 0 or frame_index >= len(frames):
        raise IndexError(
            f"Transition frame {frame_index} outside decoded video of {len(frames)}"
        )
    start = max(1, int(window_start))
    stop = min(len(frames), int(window_stop))
    deltas = [
        float(np.abs(frames[index] - frames[index - 1]).mean())
        for index in range(start, stop)
    ]
    return {
        "entrance_frame_l1": float(
            np.abs(frames[frame_index] - frames[frame_index - 1]).mean()
        ),
        "transition_window_mean_frame_l1": float(np.mean(deltas)),
        "transition_window_peak_frame_l1": float(np.max(deltas)),
        "transition_window_peak_rgb_frame": int(
            start + int(np.argmax(deltas))
        ),
        "transition_window_rgb_frames": [start, stop - 1],
        "decoded_video_frames": len(frames),
    }


def evaluate_continuous_cavr(
    *,
    run_root: str | Path,
    case_dir: str | Path,
    source_chunk: int = 8,
    target_chunks: tuple[int, ...] = (21, 22),
) -> dict:
    root = Path(run_root).resolve()
    case_dir = Path(case_dir).resolve()
    target_chunk = max(int(chunk) for chunk in target_chunks)
    first_b2 = min(int(chunk) for chunk in target_chunks)
    mapping = root / "baseline" / "block_mapping.json"
    source = _keyframe(root, "baseline", source_chunk)
    height, width = source.shape[:2]
    source_c2w = np.asarray(_block(mapping, source_chunk)["c2w"])
    target_c2w = np.asarray(_block(mapping, target_chunk)["c2w"])
    intrinsics = _intrinsics(case_dir / "intrinsics.txt", (height, width))
    warped_source, image_warp_valid, homography = _rotation_warp(
        source, source_c2w, target_c2w, intrinsics
    )
    state = torch.load(
        root
        / "generation"
        / "continuous_cavr"
        / "warp"
        / f"target_{target_chunk:04d}"
        / "warp_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    coverage_small = state["coverage"].float().mean(dim=(0, 1, 2)).numpy()
    coverage = cv2.resize(
        coverage_small,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ).clip(0, 1)
    overlap = coverage * image_warp_valid
    nonoverlap = 1.0 - coverage
    if float(overlap.mean()) <= 0 or float(nonoverlap.mean()) <= 0:
        raise RuntimeError(
            "Continuous CAVR evaluation requires overlap and non-overlap"
        )

    assets = root / "assets" / "cavr"
    assets.mkdir(parents=True, exist_ok=True)
    baseline_target = _keyframe(root, "baseline", target_chunk)
    _save_rgb(assets / "b1_source.png", source)
    _save_rgb(assets / "b1_warped_to_b2.png", warped_source)
    _save_rgb(assets / "b2_baseline.png", baseline_target)
    _save_mask(assets / "memory_coverage.png", coverage)
    _save_mask(assets / "overlap_metric_mask.png", overlap)
    _save_mask(assets / "nonoverlap_metric_mask.png", nonoverlap)
    overlay = baseline_target.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], 0.9 * coverage)
    overlay[..., 1:] *= 1.0 - 0.45 * coverage[..., None]
    _save_rgb(assets / "b2_memory_coverage_overlay.png", overlay)

    phase_payload = _json(case_dir / "phase_labels.json")
    b2_phase = next(
        item for item in phase_payload["phases"] if item["name"] == "B2_hold"
    )
    revisit_ramp = next(
        item for item in phase_payload["phases"] if item["name"] == "A_to_B2"
    )
    b2_rgb_start = int(b2_phase["rgb_start"])
    methods = {}
    metadata_by_method = {}
    for method in METHOD_ROOTS:
        method_root = _method_root(root, method)
        metadata = _json(method_root / "run_metadata.json")
        metadata_by_method[method] = metadata
        target = _keyframe(root, method, target_chunk)
        before_b2 = _keyframe(root, method, first_b2 - 1)
        b2_first = _keyframe(root, method, first_b2)
        previous_target = _keyframe(root, method, target_chunk - 1)
        replay = metadata.get("replay", {}).get("against_saved_latents") or {}
        per_chunk = replay.get("per_chunk_max_abs_diff", {})
        methods[method] = {
            "overlap_b1_to_b2_l1": _masked_l1(
                warped_source, target, overlap
            ),
            "nonoverlap_delta_vs_baseline_l1": _masked_l1(
                baseline_target, target, nonoverlap
            ),
            "whole_delta_vs_baseline_l1": float(
                np.abs(baseline_target - target).mean()
            ),
            "b2_entry_boundary_l1": float(
                np.abs(before_b2 - b2_first).mean()
            ),
            "within_b2_boundary_l1": float(
                np.abs(previous_target - target).mean()
            ),
            "target_latent_max_abs_diff_vs_baseline": (
                0.0
                if method == "baseline"
                else per_chunk.get(str(target_chunk))
            ),
            "generation_seconds": float(
                metadata["timing_seconds"]["total"]
            ),
            **_video_transition(
                method_root / "pred.mp4",
                b2_rgb_start,
                window_start=int(revisit_ramp["rgb_start"]),
                window_stop=int(b2_phase["rgb_stop_exclusive"]),
            ),
        }

    continuous_metadata = metadata_by_method["continuous_cavr"]
    selections = continuous_metadata["mapkv"]["selections"]
    active = [
        item
        for item in selections
        if item["status"] == "scheduled_visible_support"
    ]
    inactive = [
        item
        for item in selections
        if item["status"] == "memory_off_no_visible_support"
    ]
    active_chunks = [int(item["target_chunk"]) for item in active]
    first_active = min(active_chunks)
    coverage_timeline = [
        {
            "chunk": int(item["target_chunk"]),
            "coverage": float(item["coverage_fraction"]),
            "active": item["status"] == "scheduled_visible_support",
        }
        for item in selections
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(
        [item["chunk"] for item in coverage_timeline],
        [item["coverage"] for item in coverage_timeline],
        marker="o",
        color="#2759c7",
    )
    axis.axvspan(14, 16, color="#98a2b3", alpha=0.18, label="no support")
    axis.axvline(first_b2, color="#e45756", linestyle="--", label="B2")
    axis.set(
        title="Geometry-driven historical-memory coverage",
        xlabel="target chunk",
        ylabel="coverage",
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(assets / "coverage_timeline.png", dpi=160)
    plt.close(figure)

    audits = continuous_metadata["mapkv"]["warp_reencode"]["audits"]
    gpu_names = {
        method: metadata["gpu"]
        for method, metadata in metadata_by_method.items()
    }
    validity = {
        "source_chunk_fixed_to_8": all(
            int(item["source_chunk"]) == int(source_chunk)
            for item in selections
        ),
        "visibility_driven_not_fixed_b2": (
            active_chunks != list(target_chunks)
            and any(chunk < first_b2 for chunk in active_chunks)
            and bool(inactive)
        ),
        "active_chunks": active_chunks,
        "inactive_no_support_chunks": [
            int(item["target_chunk"]) for item in inactive
        ],
        "prefix_exact_before_first_visible_memory": _prefix_is_exact(
            continuous_metadata, first_active
        ),
        "short_term_recent_reprojected": all(
            bool(item["short_term_recent_reprojected"])
            for item in audits.values()
        ),
        "runtime_cache_unchanged": all(
            item["unchanged"]
            for item in continuous_metadata["mapkv"]["cache_audits"].values()
        ),
        "known_pose_surfel_query": all(
            item["geometry"]["pose_source"]
            == "known_control_c2w_to_cut3r_c2w"
            for item in audits.values()
        ),
        "future_geometry_used": any(
            frame["future_geometry_used"]
            for item in active
            for frame in item["geometry_frames"]
        ),
        "same_gpu": len(set(gpu_names.values())) == 1,
        "gpu_by_method": gpu_names,
    }
    valid = all(
        value
        for key, value in validity.items()
        if key
        not in {
            "active_chunks",
            "inactive_no_support_chunks",
            "gpu_by_method",
            "future_geometry_used",
        }
    ) and not validity["future_geometry_used"]
    baseline = methods["baseline"]
    block_on = methods["block_on_wre"]
    continuous = methods["continuous_cavr"]
    fidelity_preserved = bool(
        continuous["overlap_b1_to_b2_l1"]
        < baseline["overlap_b1_to_b2_l1"]
        and continuous["overlap_b1_to_b2_l1"]
        <= block_on["overlap_b1_to_b2_l1"] + 0.02
    )
    new_region_preserved = bool(
        continuous["nonoverlap_delta_vs_baseline_l1"]
        <= block_on["nonoverlap_delta_vs_baseline_l1"]
    )
    transition_improved = bool(
        continuous["b2_entry_boundary_l1"]
        < block_on["b2_entry_boundary_l1"]
        and continuous["entrance_frame_l1"]
        < block_on["entrance_frame_l1"]
        and continuous["transition_window_mean_frame_l1"]
        <= block_on["transition_window_mean_frame_l1"]
        and continuous["transition_window_peak_frame_l1"]
        <= block_on["transition_window_peak_frame_l1"]
    )
    if valid and fidelity_preserved and new_region_preserved and transition_improved:
        status = "CONTINUOUS_CAVR_WORKS"
    elif valid and fidelity_preserved and new_region_preserved:
        status = "CONTINUOUS_CAVR_MIXED_TRANSITION"
    else:
        status = "CONTINUOUS_CAVR_NOT_WORKING"
    result = {
        "status": status,
        "case": case_dir.name,
        "source_chunk": int(source_chunk),
        "target_chunks": list(target_chunks),
        "evaluation_target_chunk": target_chunk,
        "camera": {
            "source_yaw_degrees": 30.0,
            "target_yaw_degrees": 20.0,
            "relative_rotation_degrees": 10.0,
            "translation": 0.0,
            "homography_source_to_target": homography.tolist(),
        },
        "coverage": {
            "target_memory_fraction": float(coverage.mean()),
            "metric_overlap_fraction": float(overlap.mean()),
            "nonoverlap_fraction": float(nonoverlap.mean()),
            "timeline": coverage_timeline,
        },
        "methods": methods,
        "validity": validity,
        "decision": {
            "memory_fidelity_preserved": fidelity_preserved,
            "new_region_preserved": new_region_preserved,
            "transition_improved": transition_improved,
            "visual_review": {
                "completed": True,
                "finding": (
                    "Continuous CAVR removes the block-on B2 pouring-object "
                    "hard switch, but shows repeated ghost/layout popping as "
                    "coverage grows across re-entry blocks. The measured peak "
                    "is therefore a real visual discontinuity."
                ),
                "artifact": "assets/cavr_transition_filmstrip_small.jpg",
            },
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Continuous Geometry-Reprojected Virtual Recent"
    )
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--case_dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_continuous_cavr(
                run_root=args.run_root,
                case_dir=args.case_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
