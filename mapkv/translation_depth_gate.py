from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry_gate import _anchor_overlap, _render_gate_view
from .surfel_index import SurfelIndex


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_indices(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    yy = np.rint(np.linspace(0, shape[0] - 1, 30)).astype(int)
    xx = np.rint(np.linspace(0, shape[1] - 1, 52)).astype(int)
    return yy, xx


def _directed_depth_cycle(
    source: dict,
    target: dict,
    root: Path,
) -> tuple[np.ndarray, int]:
    source_payload = np.load(root / source["data_path"])
    target_payload = np.load(root / target["data_path"])
    yy, xx = _sample_indices(source_payload["depth"].shape)
    points = source_payload["pts3d"][np.ix_(yy, xx)]
    confidence = source_payload["confidence"][np.ix_(yy, xx)]
    finite = np.isfinite(points).all(-1) & np.isfinite(confidence)
    threshold = (
        float(np.quantile(confidence[finite], 0.35))
        if finite.any()
        else float("inf")
    )
    points = points[finite & (confidence >= threshold)]
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    camera = (
        homogeneous
        @ np.linalg.inv(np.asarray(target["camera_pose"], dtype=np.float32)).T
    )[:, :3]
    intrinsic = np.asarray(target["intrinsics"], dtype=np.float64)
    positive = camera[:, 2] > 1e-6
    u = (
        intrinsic[0, 0] * camera[:, 0] / np.maximum(camera[:, 2], 1e-8)
        + intrinsic[0, 2]
    )
    v = (
        intrinsic[1, 1] * camera[:, 1] / np.maximum(camera[:, 2], 1e-8)
        + intrinsic[1, 2]
    )
    x = np.rint(u).astype(int)
    y = np.rint(v).astype(int)
    depth = target_payload["depth"]
    inside = (
        positive
        & (x >= 0)
        & (x < depth.shape[1])
        & (y >= 0)
        & (y < depth.shape[0])
    )
    predicted = camera[inside, 2]
    observed = depth[y[inside], x[inside]]
    valid = (
        np.isfinite(predicted)
        & np.isfinite(observed)
        & (observed > 1e-6)
    )
    relative = np.abs(predicted[valid] - observed[valid]) / np.maximum(
        observed[valid], 1e-6
    )
    return relative, int(valid.sum())


def _translation_cycle(sequence_path: Path) -> dict:
    sequence = _json(sequence_path)
    frames = sequence["frames"]
    pairs = []
    all_errors = []
    for left, right in zip(frames[:-1], frames[1:]):
        translation = float(
            np.linalg.norm(
                np.asarray(left["camera_pose"])[:3, 3]
                - np.asarray(right["camera_pose"])[:3, 3]
            )
        )
        if translation <= 1e-5:
            continue
        forward, forward_count = _directed_depth_cycle(
            left, right, sequence_path.parent
        )
        backward, backward_count = _directed_depth_cycle(
            right, left, sequence_path.parent
        )
        errors = np.concatenate([forward, backward])
        if not len(errors):
            continue
        all_errors.append(errors)
        pairs.append(
            {
                "chunks": [
                    int(left["chunk_id"]),
                    int(right["chunk_id"]),
                ],
                "translation": translation,
                "valid_correspondences": forward_count + backward_count,
                "relative_depth_error_median": float(np.median(errors)),
                "relative_depth_error_p90": float(
                    np.quantile(errors, 0.9)
                ),
            }
        )
    combined = np.concatenate(all_errors) if all_errors else np.empty(0)
    return {
        "pairs": pairs,
        "relative_depth_error_median": (
            float(np.median(combined)) if len(combined) else float("inf")
        ),
        "relative_depth_error_p90": (
            float(np.quantile(combined, 0.9))
            if len(combined)
            else float("inf")
        ),
        "valid_correspondences": int(len(combined)),
    }


def _surface_neighbor_persistence(
    index: SurfelIndex,
    anchor_chunk: int,
    revisit_chunk: int,
) -> dict:
    anchor = [
        cell
        for cell in index.cells
        if cell.stable and anchor_chunk in cell.observing_chunks
    ]
    revisit = [
        cell
        for cell in index.cells
        if cell.stable and revisit_chunk in cell.observing_chunks
    ]
    if not anchor or not revisit:
        return {"matched": 0, "anchor": len(anchor), "recall": 0.0}
    revisit_positions = np.asarray([cell.xyz for cell in revisit])
    tree = cKDTree(revisit_positions)
    matched = 0
    for cell in anchor:
        radius = max(2.5 * float(cell.radius), 2.0 * index.voxel_size)
        candidates = tree.query_ball_point(cell.xyz, radius)
        if any(
            revisit[item].normal is not None
            and cell.normal is not None
            and float(np.dot(cell.normal, revisit[item].normal)) >= 0.8
            for item in candidates
        ):
            matched += 1
    return {
        "matched": matched,
        "anchor": len(anchor),
        "recall": matched / max(len(anchor), 1),
    }


def evaluate_translation_depth_gate(
    *,
    run_root: str | Path,
    case_dir: str | Path,
) -> dict:
    root = Path(run_root).resolve()
    case = Path(case_dir).resolve()
    manifest = _json(case / "trajectory_manifest.json")
    anchor_chunk = int(manifest["source_chunk"])
    target_chunk = int(manifest["target_chunk"])
    sequence = _json(root / "cut3r/sequence.json")
    stats = _json(root / "surfel/stats.json")
    index = SurfelIndex.load(root / "surfel/surfel_index.npz")
    overlap = _anchor_overlap(index, anchor_chunk, target_chunk - 2)
    neighbor_persistence = _surface_neighbor_persistence(
        index, anchor_chunk, target_chunk - 2
    )
    cycle = _translation_cycle(root / "cut3r/sequence.json")
    rendered = _render_gate_view(
        run_root=root,
        case_dir=case,
        target_chunk=target_chunk,
        prefix_last_chunk=int(sequence["prefix_last_chunk"]),
        output_name="translation",
        candidate_chunks=tuple(
            range(anchor_chunk - 3, anchor_chunk + 1)
        ),
    )
    retrieval = _json(root / "retrieval/retrieval.json")["targets"][-1]
    checks = {
        "translation_nonzero": float(manifest["translation_distance"]) > 0,
        "same_pose_revisit": bool(
            _json(case / "pose_validation.json")["valid"]
        ),
        "fixed_global_alignment": bool(
            sequence.get("fixed_global_alignment", False)
        ),
        "stable_core": (
            float(stats["stable_cell_fraction"]) >= 0.1
            and int(stats["stable_cells"]) >= 500
        ),
        "depth_cycle": (
            cycle["valid_correspondences"] >= 1000
            and cycle["relative_depth_error_median"] <= 0.1
            and cycle["relative_depth_error_p90"] <= 0.3
        ),
        "anchor_surface_persistence": (
            neighbor_persistence["recall"] >= 0.5
        ),
        "target_render": (
            rendered["rgb_l1"] <= 0.2
            and rendered["edge_correlation"] >= 0.05
        ),
        "retrieval": (
            bool(retrieval.get("positive_cluster_hit", False))
            and float(
                retrieval.get("positive_vs_negative_margin", 0.0)
            )
            >= 0.1
        ),
    }
    status = (
        "TRANSLATION_DEPTH_GATE_PASS"
        if all(checks.values())
        else "TRANSLATION_DEPTH_GATE_FAIL"
    )
    result = {
        "status": status,
        "checks": checks,
        "cycle": cycle,
        "anchor_overlap": overlap,
        "surface_neighbor_persistence": neighbor_persistence,
        "target_render": rendered,
        "retrieval": retrieval,
        "surfel": stats,
    }
    (root / "translation_depth_gate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


__all__ = ["evaluate_translation_depth_gate"]
