from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from mapkv_proto.pose_utils import to_cut3r_c2w

from .surfel_index import SurfelIndex
from .surfel_rgb_options import (
    _prepare_cut3r_rgb,
    render_target_rgb,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_grid(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    yy = np.rint(np.linspace(0, array.shape[0] - 1, shape[0])).astype(int)
    xx = np.rint(np.linspace(0, array.shape[1] - 1, shape[1])).astype(int)
    return array[np.ix_(yy, xx)]


def _cloud(sequence_root: Path, frame: dict) -> np.ndarray:
    payload = np.load(sequence_root / frame["data_path"])
    points = _sample_grid(payload["pts3d"], (30, 52))
    confidence = _sample_grid(payload["confidence"], (30, 52))
    finite = np.isfinite(points).all(-1) & np.isfinite(confidence)
    values = confidence[finite]
    threshold = (
        float(np.quantile(values, 0.35)) if values.size else float("inf")
    )
    return points[finite & (confidence >= threshold)]


def _rotation_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left)[:3, :3].T @ np.asarray(right)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pair_consistency(
    sequence_path: Path,
    surfel_stats: dict,
) -> dict:
    sequence = _json(sequence_path)
    frames = sequence["frames"]
    thresholds = {
        int(item["chunk_id"]): float(item["position_threshold"])
        for item in surfel_stats["insertions"]
    }
    clouds = {
        int(frame["chunk_id"]): _cloud(sequence_path.parent, frame)
        for frame in frames
    }
    pairs = []
    for left, right in zip(frames[:-1], frames[1:]):
        angle = _rotation_degrees(
            np.asarray(left["camera_pose"]),
            np.asarray(right["camera_pose"]),
        )
        if not 5.0 <= angle <= 10.0:
            continue
        left_chunk = int(left["chunk_id"])
        right_chunk = int(right["chunk_id"])
        a, b = clouds[left_chunk], clouds[right_chunk]
        distances = np.concatenate(
            [
                cKDTree(b).query(a, k=1)[0],
                cKDTree(a).query(b, k=1)[0],
            ]
        )
        merge_threshold = max(
            thresholds[left_chunk], thresholds[right_chunk]
        )
        left_camera = np.asarray(left["camera_pose"])[:3, 3]
        right_camera = np.asarray(right["camera_pose"])[:3, 3]
        a_rays = a - left_camera
        b_rays = b - right_camera
        a_rays /= np.maximum(
            np.linalg.norm(a_rays, axis=-1, keepdims=True), 1e-8
        )
        b_rays /= np.maximum(
            np.linalg.norm(b_rays, axis=-1, keepdims=True), 1e-8
        )
        angular_chords = np.concatenate(
            [
                cKDTree(b_rays).query(a_rays, k=1)[0],
                cKDTree(a_rays).query(b_rays, k=1)[0],
            ]
        )
        angular_degrees = np.degrees(
            2.0
            * np.arcsin(np.clip(0.5 * angular_chords, 0.0, 1.0))
        )
        pairs.append(
            {
                "chunks": [left_chunk, right_chunk],
                "rotation_degrees": angle,
                "nn_median": float(np.median(distances)),
                "nn_p90": float(np.quantile(distances, 0.9)),
                "within_merge_fraction": float(
                    np.mean(distances <= merge_threshold)
                ),
                "merge_threshold": merge_threshold,
                "angular_nn_median_degrees": float(
                    np.median(angular_degrees)
                ),
                "angular_nn_p90_degrees": float(
                    np.quantile(angular_degrees, 0.9)
                ),
            }
        )
    return {
        "pairs": pairs,
        "mean_within_merge_fraction": (
            float(np.mean([item["within_merge_fraction"] for item in pairs]))
            if pairs
            else 0.0
        ),
        "median_nn": (
            float(np.median([item["nn_median"] for item in pairs]))
            if pairs
            else float("inf")
        ),
        "median_angular_nn_degrees": (
            float(
                np.median(
                    [item["angular_nn_median_degrees"] for item in pairs]
                )
            )
            if pairs
            else float("inf")
        ),
    }


def _anchor_overlap(
    index: SurfelIndex,
    anchor_chunk: int,
    revisit_chunk: int,
) -> dict:
    anchor = {
        cell_index
        for cell_index, cell in enumerate(index.cells)
        if anchor_chunk in cell.observing_chunks
    }
    revisit = {
        cell_index
        for cell_index, cell in enumerate(index.cells)
        if revisit_chunk in cell.observing_chunks
    }
    stable_anchor = {
        cell_index for cell_index in anchor if index.cells[cell_index].stable
    }
    stable_revisit = {
        cell_index for cell_index in revisit if index.cells[cell_index].stable
    }
    shared = anchor & revisit
    stable_shared = stable_anchor & stable_revisit
    return {
        "anchor_cells": len(anchor),
        "revisit_cells": len(revisit),
        "shared_cells": len(shared),
        "anchor_recall": len(shared) / max(len(anchor), 1),
        "stable_anchor_cells": len(stable_anchor),
        "stable_revisit_cells": len(stable_revisit),
        "stable_shared_cells": len(stable_shared),
        "stable_anchor_recall": len(stable_shared)
        / max(len(stable_anchor), 1),
    }


def _edge_correlation(
    reference: np.ndarray,
    rendered: np.ndarray,
    mask: np.ndarray,
) -> float:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    reference_edge = np.hypot(
        cv2.Sobel(reference_gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(reference_gray, cv2.CV_32F, 0, 1),
    )
    rendered_edge = np.hypot(
        cv2.Sobel(rendered_gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(rendered_gray, cv2.CV_32F, 0, 1),
    )
    if int(mask.sum()) < 2:
        return 0.0
    return float(np.corrcoef(reference_edge[mask], rendered_edge[mask])[0, 1])


def _candidate_observation_colors(
    index: SurfelIndex,
    sequence: dict,
    candidate_chunks: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    frames = {
        int(item["chunk_id"]): item for item in sequence["frames"]
    }
    images = {
        chunk: _prepare_cut3r_rgb(
            frames[chunk]["image_path"],
            tuple(int(value) for value in frames[chunk]["shape"]),
        )
        for chunk in candidate_chunks
        if chunk in frames
    }
    colors = np.zeros((len(index.cells), 3), dtype=np.uint8)
    valid = np.zeros(len(index.cells), dtype=bool)
    for cell_index, cell in enumerate(index.cells):
        for chunk in candidate_chunks:
            pixel = cell.source_pixels.get(chunk)
            image = images.get(chunk)
            if pixel is None or image is None:
                continue
            y, x = np.rint(pixel).astype(int)
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                colors[cell_index] = image[y, x]
                valid[cell_index] = True
                break
    return colors, valid


def _render_gate_view(
    *,
    run_root: Path,
    case_dir: Path,
    target_chunk: int,
    prefix_last_chunk: int,
    output_name: str,
    candidate_chunks: tuple[int, ...],
) -> dict:
    sequence_path = run_root / "cut3r/sequence.json"
    sequence = _json(sequence_path)
    index = SurfelIndex.load(run_root / "surfel/surfel_index.npz")
    colors, valid = _candidate_observation_colors(
        index, sequence, candidate_chunks
    )
    mapping = _json(run_root / "baseline/block_mapping.json")["blocks"]
    target = next(
        item for item in mapping if int(item["chunk_id"]) == target_chunk
    )
    query_pose = to_cut3r_c2w(
        np.asarray(target["c2w"], dtype=np.float64)
    )
    image_hw = tuple(int(value) for value in sequence["frames"][0]["shape"])
    rendered, mask, stats = render_target_rgb(
        index,
        colors,
        valid,
        query_pose,
        np.asarray(sequence["query_intrinsics"], dtype=np.float64),
        image_hw,
        eligible_chunks=set(candidate_chunks),
        eligible_max_chunk=prefix_last_chunk,
        stable_only=True,
    )
    target_rgb = _prepare_cut3r_rgb(
        run_root / "baseline" / target["png_path"], image_hw
    )
    assets = run_root / "assets/geometry_gate"
    assets.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered).save(assets / f"{output_name}_zbuffer.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(
        assets / f"{output_name}_coverage.png"
    )
    overlay = target_rgb.copy()
    overlay[mask] = np.clip(
        0.72 * rendered[mask].astype(np.float32)
        + 0.28 * target_rgb[mask].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    Image.fromarray(overlay).save(assets / f"{output_name}_overlay.png")
    return {
        **stats,
        "rgb_l1": (
            float(
                np.abs(
                    rendered.astype(np.float32)
                    - target_rgb.astype(np.float32)
                )[mask].mean()
                / 255.0
            )
            if mask.any()
            else 1.0
        ),
        "edge_correlation": _edge_correlation(target_rgb, rendered, mask),
    }


def evaluate_geometry_gate(
    *,
    run_root: str | Path,
    legacy_root: str | Path,
    case_dir: str | Path,
) -> dict:
    root = Path(run_root).resolve()
    legacy = Path(legacy_root).resolve()
    case = Path(case_dir).resolve()
    manifest = _json(case / "trajectory_manifest.json")
    anchor_chunk = int(manifest["source_chunk"])
    target_chunk = int(manifest["target_chunk"])
    revisit_history_chunk = target_chunk - 2
    new_sequence = _json(root / "cut3r/sequence.json")
    new_stats = _json(root / "surfel/stats.json")
    legacy_stats = _json(legacy / "surfel/stats.json")
    new_index = SurfelIndex.load(root / "surfel/surfel_index.npz")
    legacy_index = SurfelIndex.load(legacy / "surfel/surfel_index.npz")
    new_pair = _pair_consistency(
        root / "cut3r/sequence.json", new_stats
    )
    legacy_pair = _pair_consistency(
        legacy / "cut3r/sequence.json", legacy_stats
    )
    new_overlap = _anchor_overlap(
        new_index, anchor_chunk, revisit_history_chunk
    )
    legacy_overlap = _anchor_overlap(
        legacy_index, anchor_chunk, revisit_history_chunk
    )
    new_render = _render_gate_view(
        run_root=root,
        case_dir=case,
        target_chunk=target_chunk,
        prefix_last_chunk=int(new_sequence["prefix_last_chunk"]),
        output_name="fixed_global",
        candidate_chunks=tuple(range(anchor_chunk - 3, anchor_chunk + 1)),
    )
    legacy_shadow = root / "legacy_shadow"
    legacy_shadow.mkdir(exist_ok=True)
    for name in ("baseline", "cut3r", "surfel"):
        destination = legacy_shadow / name
        if not destination.exists():
            destination.symlink_to(legacy / name, target_is_directory=True)
    legacy_render = _render_gate_view(
        run_root=legacy_shadow,
        case_dir=case,
        target_chunk=target_chunk,
        prefix_last_chunk=int(
            _json(legacy / "cut3r/sequence.json")["prefix_last_chunk"]
        ),
        output_name="legacy",
        candidate_chunks=tuple(range(anchor_chunk - 3, anchor_chunk + 1)),
    )
    retrieval = _json(root / "retrieval/retrieval.json")["targets"][-1]
    checks = {
        "fixed_global_alignment": bool(
            new_sequence.get("fixed_global_alignment", False)
        ),
        "previous_depths_frozen": max(
            (
                float(item["previous_depth_max_abs_change"])
                for item in new_sequence.get("alignment_audits", [])
            ),
            default=float("inf"),
        )
        <= 1e-6,
        "stable_core_nonempty": int(new_stats["stable_cells"]) > 0,
        "adjacent_consistency_improved": (
            new_pair["median_angular_nn_degrees"]
            <= 0.75
        ),
        "anchor_overlap_improved": (
            new_overlap["stable_anchor_recall"]
            >= max(
                0.1,
                2.0 * legacy_overlap["stable_anchor_recall"],
            )
        ),
        "target_edge_improved": (
            new_render["rgb_l1"] <= 0.85 * legacy_render["rgb_l1"]
            and new_render["edge_correlation"]
            > legacy_render["edge_correlation"]
        ),
        "retrieval_positive_hit": bool(
            retrieval.get("positive_cluster_hit", False)
        ),
        "retrieval_confident": (
            float(retrieval.get("top1_margin", 0.0)) >= 0.1
            and float(retrieval.get("normalized_entropy", 1.0)) <= 0.9
        ),
    }
    status = (
        "GEOMETRY_GATE_PASS"
        if all(checks.values())
        else "GEOMETRY_GATE_FAIL"
    )
    result = {
        "status": status,
        "checks": checks,
        "trajectory": {
            "anchor_chunk": anchor_chunk,
            "revisit_history_chunk": revisit_history_chunk,
            "target_chunk": target_chunk,
        },
        "fixed_global": {
            "surfel": new_stats,
            "pair_consistency": new_pair,
            "anchor_overlap": new_overlap,
            "target_render": new_render,
            "retrieval": retrieval,
        },
        "legacy": {
            "surfel": legacy_stats,
            "pair_consistency": legacy_pair,
            "anchor_overlap": legacy_overlap,
            "target_render": legacy_render,
        },
    }
    (root / "geometry_gate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (root / "status.json").write_text(
        json.dumps({"status": status}, indent=2), encoding="utf-8"
    )
    return result


__all__ = ["evaluate_geometry_gate"]
