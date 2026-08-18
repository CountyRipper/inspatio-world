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
    "hard_recentkv": "generation/hard_recentkv",
    "warp_reencode": "generation/warp_reencode",
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


def _prefix_is_exact(metadata: dict, first_target: int) -> bool:
    replay = metadata.get("replay", {}).get("against_saved_latents") or {}
    per_chunk = replay.get("per_chunk_max_abs_diff", {})
    return bool(per_chunk) and all(
        float(per_chunk[str(chunk)]) == 0.0
        for chunk in range(first_target)
    )


def evaluate_warp_reencode(
    *,
    run_root: str | Path,
    case_dir: str | Path,
    source_chunk: int = 8,
    target_chunks: tuple[int, ...] = (21, 22),
) -> dict:
    root = Path(run_root).resolve()
    case_dir = Path(case_dir).resolve()
    target_chunk = max(int(chunk) for chunk in target_chunks)
    first_target = min(int(chunk) for chunk in target_chunks)
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
        / "warp_reencode"
        / "warp"
        / f"target_{target_chunk:04d}"
        / "warp_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    coverage_small = (
        state["coverage"].float().mean(dim=(0, 1, 2)).numpy()
    )
    coverage = cv2.resize(
        coverage_small,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ).clip(0, 1)
    overlap = coverage * image_warp_valid
    nonoverlap = 1.0 - coverage
    if float(overlap.mean()) <= 0 or float(nonoverlap.mean()) <= 0:
        raise RuntimeError(
            "Warp-reencode evaluation requires both overlap and non-overlap"
        )
    baseline_target = _keyframe(root, "baseline", target_chunk)
    assets = root / "assets" / "warp_reencode"
    _save_rgb(assets / "b1_source.png", source)
    _save_rgb(assets / "b1_warped_to_b2.png", warped_source)
    _save_rgb(assets / "b2_baseline.png", baseline_target)
    _save_mask(assets / "warp_coverage.png", coverage)
    _save_mask(assets / "overlap_metric_mask.png", overlap)
    _save_mask(assets / "nonoverlap_metric_mask.png", nonoverlap)
    overlay = baseline_target.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], 0.9 * coverage)
    overlay[..., 1:] *= 1.0 - 0.45 * coverage[..., None]
    _save_rgb(assets / "b2_coverage_overlay.png", overlay)

    method_metrics = {}
    metadata_by_method = {}
    for method in METHOD_ROOTS:
        method_root = _method_root(root, method)
        metadata = _json(method_root / "run_metadata.json")
        metadata_by_method[method] = metadata
        target = _keyframe(root, method, target_chunk)
        before_b2 = _keyframe(root, method, first_target - 1)
        first_b2 = _keyframe(root, method, first_target)
        previous_target = _keyframe(root, method, target_chunk - 1)
        replay = metadata.get("replay", {}).get("against_saved_latents") or {}
        per_chunk = replay.get("per_chunk_max_abs_diff", {})
        method_metrics[method] = {
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
                np.abs(before_b2 - first_b2).mean()
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
            "target_block_seconds": metadata["timing_seconds"][
                "per_block"
            ].get(str(target_chunk)),
        }

    hard = method_metrics["hard_recentkv"]
    warp = method_metrics["warp_reencode"]
    baseline = method_metrics["baseline"]
    warp_overlap_gain = (
        baseline["overlap_b1_to_b2_l1"]
        - warp["overlap_b1_to_b2_l1"]
    )
    works = bool(
        warp_overlap_gain > 0.0
        and warp["overlap_b1_to_b2_l1"]
        < hard["overlap_b1_to_b2_l1"]
        and warp["nonoverlap_delta_vs_baseline_l1"]
        < hard["nonoverlap_delta_vs_baseline_l1"]
    )
    warp_metadata = metadata_by_method["warp_reencode"]
    hard_metadata = metadata_by_method["hard_recentkv"]
    hard_sources = {
        int(item["source_chunk"])
        for item in hard_metadata["mapkv"]["selections"]
    }
    warp_sources = {
        int(item["source_chunk"])
        for item in warp_metadata["mapkv"]["selections"]
    }
    gpu_names = {
        method: metadata["gpu"]
        for method, metadata in metadata_by_method.items()
    }
    validity = {
        "source_chunk_fixed_to_8": (
            hard_sources == {int(source_chunk)}
            and warp_sources == {int(source_chunk)}
        ),
        "target_chunks": list(target_chunks),
        "same_gpu": len(set(gpu_names.values())) == 1,
        "gpu_by_method": gpu_names,
        "warp_prefix_exact_through_chunk_20": _prefix_is_exact(
            warp_metadata, first_target
        ),
        "runtime_cache_unchanged": all(
            item["unchanged"]
            for item in warp_metadata["mapkv"]["cache_audits"].values()
        ),
        "writer_isolated_from_runtime_cache": bool(
            warp_metadata["mapkv"]["warp_reencode"][
                "writer_isolated_from_runtime_cache"
            ]
        ),
        "no_future_memory": all(
            int(item["source_chunk"]) < int(item["target_chunk"]) - 1
            for item in warp_metadata["mapkv"]["selections"]
        ),
    }
    valid = all(
        value
        for key, value in validity.items()
        if key not in {"target_chunks", "gpu_by_method"}
    )
    status = (
        "WARP_REENCODE_WORKS"
        if valid and works
        else "WARP_REENCODE_NOT_WORKING"
    )
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
            "latent_warp_fraction": float(coverage.mean()),
            "metric_overlap_fraction": float(overlap.mean()),
            "nonoverlap_fraction": float(nonoverlap.mean()),
        },
        "validity": validity,
        "methods": method_metrics,
        "decision": {
            "automatic_status_uses_metrics": True,
            "warp_overlap_gain_vs_baseline": float(warp_overlap_gain),
            "warp_overlap_improvement_vs_hard": float(
                hard["overlap_b1_to_b2_l1"]
                - warp["overlap_b1_to_b2_l1"]
            ),
            "warp_nonoverlap_reduction_vs_hard": float(
                hard["nonoverlap_delta_vs_baseline_l1"]
                - warp["nonoverlap_delta_vs_baseline_l1"]
            ),
            "visual_review_required": True,
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-source camera-aligned recent re-encoding"
    )
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--source_chunk", type=int, default=8)
    parser.add_argument("--target_chunks", type=int, nargs="+", default=[21, 22])
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_warp_reencode(
                run_root=args.run_root,
                case_dir=args.case_dir,
                source_chunk=args.source_chunk,
                target_chunks=tuple(args.target_chunks),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
