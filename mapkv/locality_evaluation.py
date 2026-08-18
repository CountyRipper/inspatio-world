from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .evaluation import _image, _mask, _masked_l1, evaluate


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_root(root: Path, method: str) -> Path:
    return root / "baseline" if method == "baseline" else root / "generation" / method


def _method_image(root: Path, method: str, chunk: int) -> np.ndarray:
    return _image(
        _method_root(root, method)
        / "keyframes"
        / f"chunk_{chunk:04d}.png"
    )


def _blocks(path: Path) -> list[dict]:
    payload = _json(path)
    return payload["blocks"] if isinstance(payload, dict) else payload


def _block(path: Path, chunk: int) -> dict:
    return next(
        item for item in _blocks(path) if int(item["chunk_id"]) == int(chunk)
    )


def _intrinsics(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    raw = path.read_text(encoding="utf-8").strip()
    try:
        matrix = np.asarray(ast.literal_eval(raw), dtype=np.float64)
    except (SyntaxError, ValueError):
        matrix = np.loadtxt(path, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(3, 3)
    matrix = matrix[:3, :3]
    source_width = max(2.0 * float(matrix[0, 2]), 1.0)
    source_height = max(2.0 * float(matrix[1, 2]), 1.0)
    height, width = image_hw
    scale = np.diag(
        [width / source_width, height / source_height, 1.0]
    )
    return scale @ matrix


def _rotation_warp(
    source: np.ndarray,
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = source.shape[:2]
    relative = (
        np.asarray(target_c2w, dtype=np.float64)[:3, :3].T
        @ np.asarray(source_c2w, dtype=np.float64)[:3, :3]
    )
    homography = intrinsics @ relative @ np.linalg.inv(intrinsics)
    warped = cv2.warpPerspective(
        source,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = cv2.warpPerspective(
        np.ones((height, width), dtype=np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped, valid.clip(0, 1), homography


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        (np.asarray(array).clip(0, 1) * 255).round().astype(np.uint8)
    ).save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        (np.asarray(mask).clip(0, 1) * 255).round().astype(np.uint8)
    ).save(path)


def evaluate_replication(
    *,
    run_root: str | Path,
    case_dir: str | Path,
    source_chunk: int,
    target_chunk: int,
) -> dict:
    root = Path(run_root).resolve()
    result = evaluate(
        run_root=root,
        case_dir=case_dir,
        source_chunk=source_chunk,
        target_chunk=target_chunk,
        methods=["baseline", "manualcorrect", "wrongkv", "surfelkv"],
    )
    generation = result["generation"]
    baseline = generation["baseline"]["b1_b2_generated_region_l1"]
    correct = generation["manualcorrect"]["b1_b2_generated_region_l1"]
    wrong = generation["wrongkv"]["b1_b2_generated_region_l1"]
    surfel = generation["surfelkv"]["b1_b2_generated_region_l1"]
    mask = _mask(
        root
        / "baseline"
        / "masks"
        / f"chunk_{target_chunk:04d}_generated_region.png",
        _method_image(root, "baseline", target_chunk).shape[:2],
    )
    manual_surfel_l1 = _masked_l1(
        _method_image(root, "manualcorrect", target_chunk),
        _method_image(root, "surfelkv", target_chunk),
        mask,
    )
    passed = bool(
        correct < baseline
        and correct < wrong
        and surfel < baseline
        and manual_surfel_l1 < 0.01
        and result["retrieval"].get("positive_cluster_hit", False)
    )
    result["replication"] = {
        "status": "PASS" if passed else "FAIL",
        "manual_improvement": float(baseline - correct),
        "wrong_vs_correct_margin": float(wrong - correct),
        "surfel_vs_manual_generated_region_l1": manual_surfel_l1,
        "selected_chunk": result["retrieval"]["selected_chunks"][0],
        "positive_cluster": result["retrieval"]["positive_cluster"],
    }
    result["status"] = "PASS" if passed else "FAIL"
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def evaluate_locality(
    *,
    run_root: str | Path,
    case_dir: str | Path,
    source_chunk: int,
    target_chunk: int,
    methods: tuple[str, ...] = (
        "baseline",
        "global_surfelkv",
        "gated_surfelkv",
    ),
) -> dict:
    root = Path(run_root).resolve()
    case_dir = Path(case_dir).resolve()
    retrieval_payload = _json(root / "retrieval" / "retrieval.json")
    retrieval = next(
        entry
        for entry in retrieval_payload["targets"]
        if int(entry["target_chunk"]) == int(target_chunk)
    )
    locality_control_path = root / "retrieval/locality_control.json"
    addressing_context = {
        "locality_control_used": locality_control_path.exists(),
        "locality_plan_scope": retrieval.get(
            "retrieval_scope", "all_causal_history"
        ),
        "locality_selected_chunks": retrieval["selected_chunks"],
    }
    if locality_control_path.exists():
        audit = _json(locality_control_path)
        unconstrained_payload = _json(
            root / "retrieval/unconstrained_retrieval.json"
        )
        unconstrained = next(
            entry
            for entry in unconstrained_payload["targets"]
            if int(entry["target_chunk"]) == int(target_chunk)
        )
        addressing_context.update(
            {
                "control": audit,
                "unconstrained_selected_chunks": unconstrained[
                    "selected_chunks"
                ],
                "unconstrained_top3_chunks": unconstrained["top3_chunks"],
                "unconstrained_positive_cluster_rank": unconstrained[
                    "positive_cluster_best_rank"
                ],
            }
        )
    coverage_path = (
        root / "retrieval" / retrieval["coverage_mask_path"]
    )
    with np.load(coverage_path) as payload:
        coverage_small = payload["coverage"].astype(np.float32)

    source = _method_image(root, "baseline", source_chunk)
    baseline_target = _method_image(root, "baseline", target_chunk)
    height, width = source.shape[:2]
    coverage = cv2.resize(
        coverage_small,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).clip(0, 1)
    mapping = root / "baseline" / "block_mapping.json"
    source_c2w = np.asarray(_block(mapping, source_chunk)["c2w"])
    target_c2w = np.asarray(_block(mapping, target_chunk)["c2w"])
    k = _intrinsics(case_dir / "intrinsics.txt", (height, width))
    warped_source, warp_valid, homography = _rotation_warp(
        source, source_c2w, target_c2w, k
    )
    overlap = coverage * warp_valid
    nonoverlap = 1.0 - coverage
    if float(overlap.mean()) <= 0:
        raise RuntimeError("Projected surfel overlap is empty")
    if float(nonoverlap.mean()) <= 0:
        raise RuntimeError("Partial-overlap case has no non-overlap region")

    assets = root / "assets" / "locality"
    _save_rgb(assets / "b1_source.png", source)
    _save_rgb(assets / "b1_warped_to_b2.png", warped_source)
    _save_rgb(assets / "b2_baseline.png", baseline_target)
    _save_mask(assets / "projected_support.png", coverage)
    _save_mask(assets / "overlap_metric_mask.png", overlap)
    _save_mask(assets / "nonoverlap_metric_mask.png", nonoverlap)
    overlay = baseline_target.copy()
    overlay[..., 0] = np.maximum(
        overlay[..., 0], 0.85 * coverage
    )
    overlay[..., 1:] *= (1.0 - 0.45 * coverage[..., None])
    _save_rgb(assets / "b2_support_overlay.png", overlay)

    method_metrics = {}
    for method in methods:
        target = _method_image(root, method, target_chunk)
        previous = _method_image(root, method, target_chunk - 1)
        metadata = _json(_method_root(root, method) / "run_metadata.json")
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
            "boundary_l1": float(np.abs(previous - target).mean()),
            "generation_seconds": float(
                metadata["timing_seconds"]["total"]
            ),
            "target_block_seconds": metadata["timing_seconds"][
                "per_block"
            ].get(str(target_chunk)),
        }
    baseline_error = method_metrics["baseline"]["overlap_b1_to_b2_l1"]
    global_values = method_metrics["global_surfelkv"]
    gated_values = method_metrics["gated_surfelkv"]
    global_gain = baseline_error - global_values["overlap_b1_to_b2_l1"]
    gated_gain = baseline_error - gated_values["overlap_b1_to_b2_l1"]
    recovery_retained = (
        gated_gain / global_gain if global_gain > 1e-8 else None
    )
    global_disturbance = global_values[
        "nonoverlap_delta_vs_baseline_l1"
    ]
    gated_disturbance = gated_values[
        "nonoverlap_delta_vs_baseline_l1"
    ]
    disturbance_ratio = (
        gated_disturbance / global_disturbance
        if global_disturbance > 1e-8
        else 0.0
    )
    query_sufficient = bool(
        global_gain > 0.01
        and gated_gain > 0.0
        and recovery_retained is not None
        and recovery_retained >= 0.60
        and (
            gated_disturbance <= 0.01
            or disturbance_ratio <= 0.60
        )
    )
    token_values = method_metrics.get("token_selected_surfelkv")
    token_decision = None
    token_selection_needed = False
    if token_values is not None:
        token_gain = (
            baseline_error - token_values["overlap_b1_to_b2_l1"]
        )
        token_overlap_improvement_vs_gated = (
            gated_values["overlap_b1_to_b2_l1"]
            - token_values["overlap_b1_to_b2_l1"]
        )
        token_nonoverlap = token_values[
            "nonoverlap_delta_vs_baseline_l1"
        ]
        token_selection_needed = bool(
            token_gain > 0.0
            and token_overlap_improvement_vs_gated > 0.01
            and token_nonoverlap <= gated_disturbance
        )
        token_decision = {
            "overlap_gain_vs_baseline": float(token_gain),
            "overlap_improvement_vs_gated": float(
                token_overlap_improvement_vs_gated
            ),
            "nonoverlap_disturbance": float(token_nonoverlap),
            "materially_improves_locality": token_selection_needed,
        }
    status = (
        "QUERY_GATING_SUFFICIENT"
        if query_sufficient
        else (
            "TOKEN_SELECTION_NEEDED"
            if token_selection_needed
            else "PARTIAL_OVERLAP_NOT_WORKING"
        )
    )
    result = {
        "status": status,
        "source_chunk": int(source_chunk),
        "target_chunk": int(target_chunk),
        "query_pose_mode": retrieval["query_pose_mode"],
        "addressing_context": addressing_context,
        "retrieval": retrieval,
        "coverage": {
            "raw_fraction": float(coverage.mean()),
            "metric_overlap_fraction": float(overlap.mean()),
            "nonoverlap_fraction": float(nonoverlap.mean()),
            "homography_b1_to_b2": homography.tolist(),
        },
        "methods": method_metrics,
        "decision": {
            "global_overlap_gain": float(global_gain),
            "gated_overlap_gain": float(gated_gain),
            "gated_recovery_retained": (
                None if recovery_retained is None else float(recovery_retained)
            ),
            "global_nonoverlap_disturbance": float(global_disturbance),
            "gated_nonoverlap_disturbance": float(gated_disturbance),
            "gated_to_global_disturbance_ratio": float(disturbance_ratio),
            "token_selection": token_decision,
            "thresholds_are_triage": True,
        },
    }
    (root / "locality_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def evaluate_layer_budget(
    *,
    output_dir: str | Path,
    case_dir: str | Path,
    baseline_root: str | Path,
    bank_root: str | Path,
    source_chunk: int,
    target_chunk: int,
    method_roots: dict[str, str | Path],
    layer_sets: dict[str, list[int]],
) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_root = Path(baseline_root).resolve()
    source = _image(
        baseline_root / "keyframes" / f"chunk_{source_chunk:04d}.png"
    )
    mask = _mask(
        baseline_root
        / "masks"
        / f"chunk_{target_chunk:04d}_generated_region.png",
        source.shape[:2],
    )
    bank = _json(Path(bank_root).resolve() / "metadata.json")
    layer_meta = bank["chunks"][str(source_chunk)]["layers"]
    rows = {}
    for name, method_path in method_roots.items():
        method_path = Path(method_path).resolve()
        target = _image(
            method_path / "keyframes" / f"chunk_{target_chunk:04d}.png"
        )
        previous = _image(
            method_path
            / "keyframes"
            / f"chunk_{target_chunk - 1:04d}.png"
        )
        metadata = _json(method_path / "run_metadata.json")
        layers = layer_sets[name]
        memory_bytes = sum(
            int(layer_meta[str(layer)].get("memory_bytes", 0))
            or (Path(bank_root) / layer_meta[str(layer)]["path"]).stat().st_size
            for layer in layers
        )
        rows[name] = {
            "layers": layers,
            "num_layers": len(layers),
            "b1_to_b2_generated_region_l1": _masked_l1(
                source, target, mask
            ),
            "boundary_l1": float(np.abs(previous - target).mean()),
            "generation_seconds": float(
                metadata["timing_seconds"]["total"]
            ),
            "target_block_seconds": metadata["timing_seconds"][
                "per_block"
            ].get(str(target_chunk)),
            "memory_bytes": int(memory_bytes),
        }
    all_values = rows["all"]
    for values in rows.values():
        values["runtime_relative_to_all"] = (
            values["generation_seconds"]
            / max(all_values["generation_seconds"], 1e-8)
        )
        values["memory_relative_to_all"] = (
            values["memory_bytes"]
            / max(all_values["memory_bytes"], 1)
        )
    result = {
        "case_dir": str(Path(case_dir).resolve()),
        "source_chunk": source_chunk,
        "target_chunk": target_chunk,
        "methods": rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MapKV next-stage evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    replication = subparsers.add_parser("replication")
    replication.add_argument("--run_root", required=True)
    replication.add_argument("--case_dir", required=True)
    replication.add_argument("--source_chunk", type=int, required=True)
    replication.add_argument("--target_chunk", type=int, required=True)
    locality = subparsers.add_parser("locality")
    locality.add_argument("--run_root", required=True)
    locality.add_argument("--case_dir", required=True)
    locality.add_argument("--source_chunk", type=int, required=True)
    locality.add_argument("--target_chunk", type=int, required=True)
    args = parser.parse_args()
    if args.command == "replication":
        evaluate_replication(
            run_root=args.run_root,
            case_dir=args.case_dir,
            source_chunk=args.source_chunk,
            target_chunk=args.target_chunk,
        )
    else:
        evaluate_locality(
            run_root=args.run_root,
            case_dir=args.case_dir,
            source_chunk=args.source_chunk,
            target_chunk=args.target_chunk,
        )


if __name__ == "__main__":
    main()
