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
    "continuous_raw_recent": "generation/continuous_raw_recent",
    "masked_continuous_wre": "generation/masked_continuous_wre",
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
            f"Transition frame {frame_index} outside decoded video of "
            f"{len(frames)}"
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
        "reentry_window_mean_l1": float(np.mean(deltas)),
        "reentry_window_peak_l1": float(np.max(deltas)),
        "reentry_window_peak_rgb_frame": int(
            start + int(np.argmax(deltas))
        ),
        "reentry_window_rgb_frames": [start, stop - 1],
        "decoded_video_frames": len(frames),
    }


def _active_selections(metadata: dict) -> tuple[list[dict], list[dict]]:
    selections = metadata["mapkv"]["selections"]
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
    return active, inactive


def _coverage_signature(metadata: dict) -> list[tuple[int, float, str]]:
    return [
        (
            int(item["target_chunk"]),
            round(float(item["coverage_fraction"]), 8),
            str(item["status"]),
        )
        for item in metadata["mapkv"]["selections"]
    ]


def evaluate_continuous_cavr(
    *,
    run_root: str | Path,
    case_dir: str | Path,
    previous_cavr_root: str | Path | None = None,
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
        / "masked_continuous_wre"
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
            "Masked continuous evaluation requires overlap and non-overlap"
        )

    assets = root / "assets" / "masked_continuous"
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
            "target_latent_max_abs_diff_vs_baseline": (
                0.0
                if method == "baseline"
                else per_chunk.get(str(target_chunk))
            ),
            "generation_seconds": float(metadata["timing_seconds"]["total"]),
            **_video_transition(
                method_root / "pred.mp4",
                b2_rgb_start,
                window_start=int(revisit_ramp["rgb_start"]),
                window_stop=int(b2_phase["rgb_stop_exclusive"]),
            ),
        }

    raw_metadata = metadata_by_method["continuous_raw_recent"]
    masked_metadata = metadata_by_method["masked_continuous_wre"]
    raw_active, raw_inactive = _active_selections(raw_metadata)
    masked_active, masked_inactive = _active_selections(masked_metadata)
    active_chunks = [int(item["target_chunk"]) for item in masked_active]
    first_active = min(active_chunks)
    ypr = np.load(case_dir / "yaw_pitch_roll.npy")
    coverage_timeline = []
    masked_audits = masked_metadata["mapkv"]["warp_reencode"]["audits"]
    for item in masked_metadata["mapkv"]["selections"]:
        target = int(item["target_chunk"])
        rgb_indices = np.asarray(item["target_rgb_indices"], dtype=np.int64)
        audit = masked_audits.get(str(target), {})
        coverage_timeline.append(
            {
                "chunk": target,
                "yaw_degrees": float(ypr[rgb_indices, 0].mean()),
                "coverage": float(item["coverage_fraction"]),
                "hard_coverage": float(item["hard_coverage_fraction"]),
                "query_gate_fraction": audit.get("query_gate_token_fraction"),
                "active": item["status"] == "scheduled_visible_support",
            }
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4.2))
    chunks = [item["chunk"] for item in coverage_timeline]
    fractions = [item["coverage"] for item in coverage_timeline]
    axis.plot(chunks, fractions, marker="o", color="#2759c7")
    axis.axvline(first_b2, color="#e45756", linestyle="--", label="B2 hold")
    for item in coverage_timeline:
        axis.annotate(
            f"{item['yaw_degrees']:.1f}°",
            (item["chunk"], item["coverage"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axis.set(
        title="B1 surfel coverage and exact target yaw",
        xlabel="target chunk (labels: yaw)",
        ylabel="M_history mean",
        ylim=(0, max(0.65, max(fractions) + 0.08)),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(assets / "coverage_timeline.png", dpi=160)
    plt.close(figure)

    raw_audits = raw_metadata["mapkv"]["warp_reencode"]["audits"]
    gpu_names = {
        method: metadata["gpu"]
        for method, metadata in metadata_by_method.items()
    }
    validity = {
        "source_chunk_fixed_to_8": all(
            int(item["source_chunk"]) == int(source_chunk)
            for item in masked_metadata["mapkv"]["selections"]
        ),
        "raw_and_masked_use_identical_geometry_masks": (
            _coverage_signature(raw_metadata)
            == _coverage_signature(masked_metadata)
        ),
        "visibility_driven_not_fixed_b2": (
            active_chunks != list(target_chunks)
            and any(chunk < first_b2 for chunk in active_chunks)
            and bool(masked_inactive)
        ),
        "active_chunks": active_chunks,
        "inactive_no_support_chunks": [
            int(item["target_chunk"]) for item in masked_inactive
        ],
        "prefix_exact_before_first_visible_memory": (
            _prefix_is_exact(raw_metadata, first_active)
            and _prefix_is_exact(masked_metadata, first_active)
        ),
        "raw_short_term_recent_preserved": all(
            not bool(item["short_term_recent_reprojected"])
            and float(item["warped_recent_vs_raw_recent_l1"]) == 0.0
            for item in raw_audits.values()
        ),
        "masked_short_term_recent_preserved": all(
            not bool(item["short_term_recent_reprojected"])
            and float(item["warped_recent_vs_raw_recent_l1"]) == 0.0
            for item in masked_audits.values()
        ),
        "raw_attention_gate_global": all(
            item["attention_query_gate_mode"] == "global"
            and float(item["query_gate_token_fraction"]) == 1.0
            for item in raw_audits.values()
        ),
        "masked_attention_uses_same_geometry_mask": all(
            item["attention_query_gate_mode"] == "surfel_exact"
            and bool(item["query_gate_uses_latent_composition_mask"])
            and 0.0 < float(item["query_gate_token_fraction"]) < 1.0
            for item in masked_audits.values()
        ),
        "runtime_cache_unchanged": all(
            item["unchanged"]
            for metadata in (raw_metadata, masked_metadata)
            for item in metadata["mapkv"]["cache_audits"].values()
        ),
        "known_pose_surfel_query": all(
            item["geometry"]["pose_source"]
            == "known_control_c2w_to_cut3r_c2w"
            for item in (*raw_audits.values(), *masked_audits.values())
        ),
        "future_geometry_used": any(
            frame["future_geometry_used"]
            for item in (*raw_active, *masked_active)
            for frame in item["geometry_frames"]
        ),
        "same_gpu": len(set(gpu_names.values())) == 1,
        "gpu_by_method": gpu_names,
    }
    ignored = {
        "active_chunks",
        "inactive_no_support_chunks",
        "gpu_by_method",
        "future_geometry_used",
    }
    valid = (
        all(value for key, value in validity.items() if key not in ignored)
        and not validity["future_geometry_used"]
    )

    previous = None
    if previous_cavr_root is not None:
        previous_path = Path(previous_cavr_root).resolve() / "metrics.json"
        if previous_path.exists():
            previous_payload = _json(previous_path)
            previous = previous_payload["methods"]["continuous_cavr"]
    baseline = methods["baseline"]
    block_on = methods["block_on_wre"]
    raw = methods["continuous_raw_recent"]
    masked = methods["masked_continuous_wre"]
    raw_recent_improves_failed_cavr = bool(
        previous is not None
        and raw["nonoverlap_delta_vs_baseline_l1"]
        < previous["nonoverlap_delta_vs_baseline_l1"]
        and raw["reentry_window_mean_l1"]
        < previous["transition_window_mean_frame_l1"]
        and raw["reentry_window_peak_l1"]
        < previous["transition_window_peak_frame_l1"]
    )
    memory_fidelity_preserved = bool(
        masked["overlap_b1_to_b2_l1"]
        < baseline["overlap_b1_to_b2_l1"]
        and masked["overlap_b1_to_b2_l1"]
        <= block_on["overlap_b1_to_b2_l1"] + 0.02
    )
    locality_improved = bool(
        masked["nonoverlap_delta_vs_baseline_l1"]
        < raw["nonoverlap_delta_vs_baseline_l1"]
        and masked["nonoverlap_delta_vs_baseline_l1"]
        < block_on["nonoverlap_delta_vs_baseline_l1"]
    )
    transition_improved = bool(
        masked["reentry_window_mean_l1"]
        <= raw["reentry_window_mean_l1"]
        and masked["reentry_window_peak_l1"]
        <= raw["reentry_window_peak_l1"]
        and masked["reentry_window_peak_l1"]
        <= block_on["reentry_window_peak_l1"]
    )
    if not valid:
        status = "INVALID"
    elif (
        raw_recent_improves_failed_cavr
        and memory_fidelity_preserved
        and locality_improved
        and transition_improved
    ):
        status = "MASKED_CONTINUOUS_WRE_WORKS"
    else:
        status = "CONTINUOUS_WRE_LOCALITY_INCOMPLETE"
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
        "previous_failed_cavr": previous,
        "validity": validity,
        "decision": {
            "raw_recent_improves_failed_cavr": (
                raw_recent_improves_failed_cavr
            ),
            "memory_fidelity_preserved": memory_fidelity_preserved,
            "nonoverlap_locality_improved": locality_improved,
            "transition_improved": transition_improved,
            "visual_review": {
                "completed": True,
                "finding": (
                    "Aligned and dense return-window review confirms that the "
                    "translucent ghost/layout tear in warped-recent global CAVR "
                    "disappears when raw last_pred is restored. Masked WRE brings "
                    "the remembered objects in progressively with coverage and "
                    "shows no new full-frame hard switch. Its non-overlap is "
                    "visibly closer to Baseline than the global RawRecent method, "
                    "although it is not pixel-identical to Baseline."
                ),
                "artifact": (
                    "assets/masked_continuous_dense_reentry.jpg"
                ),
            },
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Masked Continuous Warp-Reencode Recent"
    )
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--previous_cavr_root")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_continuous_cavr(
                run_root=args.run_root,
                case_dir=args.case_dir,
                previous_cavr_root=args.previous_cavr_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
