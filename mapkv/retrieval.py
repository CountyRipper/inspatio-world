from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .surfel_index import SurfelCell, SurfelIndex


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left)[:3, :3].T @ np.asarray(right)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def contiguous_pose_clusters(
    sequence: dict,
    *,
    rotation_tolerance_degrees: float = 0.25,
    translation_tolerance: float = 1e-4,
) -> list[list[int]]:
    """Group only consecutive chunks on the same controlled plateau."""
    frames = sorted(
        sequence["frames"], key=lambda item: int(item["chunk_id"])
    )
    clusters: list[list[int]] = []
    cluster_poses: list[np.ndarray] = []
    for item in frames:
        chunk = int(item["chunk_id"])
        pose = np.asarray(item["camera_pose"], dtype=np.float64)
        if not clusters:
            clusters.append([chunk])
            cluster_poses.append(pose)
            continue
        reference = cluster_poses[-1]
        translation = float(
            np.linalg.norm(pose[:3, 3] - reference[:3, 3])
        )
        rotation = float(
            np.degrees(_rotation_distance(reference, pose))
        )
        if (
            chunk == clusters[-1][-1] + 1
            and translation <= translation_tolerance
            and rotation <= rotation_tolerance_degrees
        ):
            clusters[-1].append(chunk)
        else:
            clusters.append([chunk])
            cluster_poses.append(pose)
    return clusters


class GeometryChunkRetriever:
    def __init__(
        self,
        *,
        min_history_gap_chunks: int = 2,
        use_view_alignment: bool = True,
        use_occlusion: bool = True,
        stable_only: bool = True,
        view_orientation_sigma_degrees: float = 15.0,
        view_translation_sigma: float = 0.03,
        surface_neighbor_radius_scale: float = 2.5,
        surface_neighbor_normal_cosine: float = 0.8,
    ):
        self.min_history_gap_chunks = int(min_history_gap_chunks)
        self.use_view_alignment = bool(use_view_alignment)
        self.use_occlusion = bool(use_occlusion)
        self.stable_only = bool(stable_only)
        self.view_orientation_sigma_degrees = float(
            view_orientation_sigma_degrees
        )
        self.view_translation_sigma = float(view_translation_sigma)
        self.surface_neighbor_radius_scale = float(
            surface_neighbor_radius_scale
        )
        self.surface_neighbor_normal_cosine = float(
            surface_neighbor_normal_cosine
        )

    def retrieve(
        self,
        surfel_index: SurfelIndex,
        query_pose: np.ndarray,
        intrinsics: np.ndarray,
        *,
        source_image_size: tuple[int, int],
        image_size: tuple[int, int],
        current_chunk: int,
        top_k: int,
        positive_chunks: list[int] | tuple[int, ...] = (),
        chunk_clusters: list[list[int]] | None = None,
        query_pose_mode: str = "controlled_same_pose_known",
        candidate_chunks: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[dict, dict]:
        """Vote from target-visible, causally eligible surfels.

        Eligibility is applied inside visible_cells before projection and
        z-buffering. Therefore an immediate-recent-only surface cannot occlude
        an older long-term candidate and then disappear during voting.
        """
        started = time.perf_counter()
        eligible_max = int(current_chunk) - self.min_history_gap_chunks
        candidate_set = (
            None
            if candidate_chunks is None
            else {int(chunk) for chunk in candidate_chunks}
        )
        visible = surfel_index.visible_cells(
            query_pose,
            intrinsics,
            image_size,
            source_image_size=source_image_size,
            eligible_max_chunk=eligible_max,
            eligible_chunks=candidate_set,
            stable_only=self.stable_only,
            use_occlusion=self.use_occlusion,
        )
        scores: dict[int, float] = {}
        support: dict[int, int] = {}
        pixel_support: dict[int, int] = {}
        confidence_sum: dict[int, float] = {}
        cell_support: dict[int, set[int]] = {}
        cell_contributions: dict[int, set[int]] = {}
        query_camera = np.asarray(query_pose)[:3, 3]
        query_forward = np.asarray(query_pose)[:3, 2].astype(np.float64)
        query_forward /= max(float(np.linalg.norm(query_forward)), 1e-8)
        visible_cell_ids = np.unique(visible["indices"])
        stable_ids = np.asarray(
            [
                index
                for index, candidate in enumerate(surfel_index.cells)
                if (not self.stable_only or candidate.stable)
            ],
            dtype=np.int32,
        )
        stable_positions = np.asarray(
            [surfel_index.cells[int(index)].xyz for index in stable_ids],
            dtype=np.float32,
        ).reshape(-1, 3)
        try:
            from scipy.spatial import cKDTree

            stable_tree = (
                cKDTree(stable_positions) if len(stable_positions) else None
            )
        except ImportError:
            stable_tree = None
        neighbor_counts: list[int] = []
        total_visible_weight = 0.0
        for cell_index in visible_cell_ids:
            cell_index = int(cell_index)
            cell = surfel_index.cells[cell_index]
            offsets = np.flatnonzero(visible["indices"] == cell_index)
            visible_offset = int(offsets[0])
            neighbor_radius = max(
                self.surface_neighbor_radius_scale
                * max(float(cell.radius), surfel_index.voxel_size),
                2.0 * surfel_index.voxel_size,
            )
            if stable_tree is None:
                neighbor_ids = np.asarray([cell_index], dtype=np.int32)
            else:
                local = stable_tree.query_ball_point(
                    cell.xyz, neighbor_radius
                )
                neighbor_ids = stable_ids[np.asarray(local, dtype=np.int32)]
            if cell.normal is not None and len(neighbor_ids):
                neighbor_ids = np.asarray(
                    [
                        index
                        for index in neighbor_ids
                        if surfel_index.cells[int(index)].normal is not None
                        and float(
                            np.dot(
                                cell.normal,
                                surfel_index.cells[int(index)].normal,
                            )
                        )
                        >= self.surface_neighbor_normal_cosine
                    ],
                    dtype=np.int32,
                )
            if cell_index not in set(int(value) for value in neighbor_ids):
                neighbor_ids = np.append(neighbor_ids, cell_index)
            neighbor_counts.append(int(len(neighbor_ids)))
            observation_cells: dict[int, list[SurfelCell]] = {}
            for neighbor_id in neighbor_ids:
                neighbor = surfel_index.cells[int(neighbor_id)]
                for chunk in neighbor.observing_chunks:
                    chunk = int(chunk)
                    if not 0 <= chunk <= eligible_max:
                        continue
                    if candidate_set is not None and chunk not in candidate_set:
                        continue
                    observation_cells.setdefault(chunk, []).append(neighbor)
            eligible_observations = sorted(observation_cells)
            if not eligible_observations:
                raise AssertionError(
                    "Ineligible surfel survived pre-z-buffer causal filtering"
                )
            confidence = float(
                np.clip(cell.calibrated_confidence, 0.05, 1.0)
            )
            normal_factor = max(
                float(visible["normal_cosine"][visible_offset]), 0.05
            )
            depth_factor = 1.0 + float(visible["depth"][visible_offset])
            base = confidence * normal_factor / depth_factor
            total_visible_weight += base
            target_direction = query_camera - cell.xyz
            target_direction /= max(
                float(np.linalg.norm(target_direction)), 1e-8
            )
            contributed: set[int] = set()
            # Every eligible observation receives the full surfel weight.
            # There is intentionally no per-cell chunk-weight partitioning.
            for chunk in eligible_observations:
                alignment = 1.0
                alignments = []
                for observation_cell in observation_cells[chunk]:
                    surface_alignment = 1.0
                    if (
                        self.use_view_alignment
                        and chunk in observation_cell.view_dirs
                    ):
                        surface_alignment = max(
                            0.0,
                            float(
                                np.dot(
                                    target_direction,
                                    observation_cell.view_dirs[chunk],
                                )
                            ),
                        )
                    source_forward = observation_cell.camera_forwards.get(
                        chunk
                    )
                    if self.use_view_alignment and source_forward is not None:
                        source_forward = np.asarray(
                            source_forward, dtype=np.float64
                        )
                        source_forward /= max(
                            float(np.linalg.norm(source_forward)), 1e-8
                        )
                        angle = float(
                            np.arccos(
                                np.clip(
                                    np.dot(query_forward, source_forward),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                        sigma = np.deg2rad(
                            self.view_orientation_sigma_degrees
                        )
                        surface_alignment *= float(
                            np.exp(
                                -0.5
                                * (angle / max(sigma, 1e-8)) ** 2
                            )
                        )
                    source_center = observation_cell.camera_centers.get(
                        chunk
                    )
                    if self.use_view_alignment and source_center is not None:
                        center_distance = float(
                            np.linalg.norm(
                                query_camera
                                - np.asarray(source_center, dtype=np.float64)
                            )
                        )
                        surface_alignment *= float(
                            np.exp(
                                -0.5
                                * (
                                    center_distance
                                    / max(self.view_translation_sigma, 1e-8)
                                )
                                ** 2
                            )
                        )
                    alignments.append(surface_alignment)
                alignment = max(alignments, default=0.0)
                contribution = base * alignment
                if contribution <= 0:
                    continue
                contributed.add(chunk)
                scores[chunk] = scores.get(chunk, 0.0) + contribution
                support[chunk] = support.get(chunk, 0) + 1
                confidence_sum[chunk] = (
                    confidence_sum.get(chunk, 0.0) + cell.confidence
                )
                cell_support.setdefault(chunk, set()).add(cell_index)
            cell_contributions[cell_index] = contributed
        for cell_index, count in zip(
            *np.unique(visible["indices"], return_counts=True)
        ):
            for chunk in cell_contributions.get(int(cell_index), set()):
                pixel_support[chunk] = pixel_support.get(chunk, 0) + int(count)

        clusters: list[list[int]] = []
        assigned: set[int] = set()
        for cluster in chunk_clusters or []:
            eligible = [
                int(chunk)
                for chunk in cluster
                if int(chunk) in scores
            ]
            if eligible:
                clusters.append(eligible)
                assigned.update(eligible)
        clusters.extend(
            [chunk] for chunk in scores if chunk not in assigned
        )
        cluster_scores = [
            max(scores[chunk] for chunk in cluster)
            for cluster in clusters
        ]
        ranked_cluster_indices = sorted(
            range(len(clusters)),
            key=lambda index: (
                -cluster_scores[index],
                min(clusters[index]),
            ),
        )
        selected_cluster_indices = ranked_cluster_indices[
            : max(1, top_k)
        ]
        explained_cluster_indices = ranked_cluster_indices[
            : max(3, top_k)
        ]
        cluster_score_sum = float(sum(cluster_scores))
        cluster_probabilities = (
            np.asarray(cluster_scores, dtype=np.float64)
            / max(cluster_score_sum, 1e-12)
        )
        normalized_entropy = (
            float(
                -np.sum(
                    cluster_probabilities
                    * np.log(np.maximum(cluster_probabilities, 1e-12))
                )
                / np.log(len(cluster_probabilities))
            )
            if len(cluster_probabilities) > 1
            else 0.0
        )
        top1_margin = (
            float(
                (
                    cluster_scores[ranked_cluster_indices[0]]
                    - cluster_scores[ranked_cluster_indices[1]]
                )
                / max(
                    cluster_scores[ranked_cluster_indices[0]],
                    1e-12,
                )
            )
            if len(ranked_cluster_indices) > 1
            else (1.0 if ranked_cluster_indices else 0.0)
        )

        def representative(cluster_index: int) -> int:
            return min(
                clusters[cluster_index],
                key=lambda chunk: (-scores[chunk], chunk),
            )

        selected = [
            representative(index) for index in selected_cluster_indices
        ]
        explained = [
            representative(index)
            for index in explained_cluster_indices
        ]
        score_sum = sum(scores.values())
        retrieved = [
            {
                "chunk_id": int(chunk),
                "cluster_chunks": [
                    int(member)
                    for member in clusters[cluster_index]
                ],
                "score": float(
                    cluster_scores[cluster_index]
                    / max(score_sum, 1e-12)
                ),
                "individual_score": float(
                    scores[chunk] / max(score_sum, 1e-12)
                ),
                "coverage_score": float(
                    scores[chunk] / max(total_visible_weight, 1e-12)
                ),
                "raw_score": float(scores[chunk]),
                "cluster_raw_score": float(
                    cluster_scores[cluster_index]
                ),
                "visible_support": int(support[chunk]),
                "visible_pixel_support": int(
                    pixel_support.get(chunk, 0)
                ),
                "visible_cell_support": int(len(cell_support[chunk])),
                "mean_confidence": float(
                    confidence_sum[chunk] / max(support[chunk], 1)
                ),
                "temporal_gap_chunks": int(current_chunk - chunk),
            }
            for chunk, cluster_index in zip(
                explained, explained_cluster_indices
            )
        ]
        coverage = np.zeros(image_size, dtype=np.float32)
        selected_chunk = selected[0] if selected else None
        if selected_chunk is not None:
            for pixel, contributions in zip(
                visible["pixels"],
                (
                    cell_contributions.get(int(cell_index), set())
                    for cell_index in visible["indices"]
                ),
            ):
                if selected_chunk in contributions:
                    coverage[int(pixel[0]), int(pixel[1])] = 1.0
        positive = sorted({int(chunk) for chunk in positive_chunks})
        positive_cluster_indices = [
            index
            for index, cluster in enumerate(clusters)
            if any(chunk in positive for chunk in cluster)
        ]
        negative_cluster_indices = [
            index
            for index in range(len(clusters))
            if index not in positive_cluster_indices
        ]
        best_positive_score = max(
            (cluster_scores[index] for index in positive_cluster_indices),
            default=0.0,
        )
        best_negative_score = max(
            (cluster_scores[index] for index in negative_cluster_indices),
            default=0.0,
        )
        positive_vs_negative_margin = float(
            (best_positive_score - best_negative_score)
            / max(best_positive_score, 1e-12)
        )
        positive_score_mass = float(
            sum(cluster_scores[index] for index in positive_cluster_indices)
            / max(sum(cluster_scores), 1e-12)
        )
        result = {
            "target_chunk": int(current_chunk),
            "query_pose_mode": query_pose_mode,
            "retrieval_scope": (
                "all_causal_history"
                if candidate_set is None
                else "explicit_candidate_control"
            ),
            "candidate_chunks": (
                None
                if candidate_set is None
                else sorted(candidate_set)
            ),
            "voting_mode": "simple_observing_chunk_vote",
            "voting_unit": "unique_stable_surfel",
            "cluster_aggregation": "max",
            "stable_only": self.stable_only,
            "surface_neighbor_radius_scale": (
                self.surface_neighbor_radius_scale
            ),
            "surface_neighbor_normal_cosine": (
                self.surface_neighbor_normal_cosine
            ),
            "mean_surface_neighbors": (
                float(np.mean(neighbor_counts)) if neighbor_counts else 0.0
            ),
            "top1_margin": top1_margin,
            "normalized_entropy": normalized_entropy,
            "eligibility_before_zbuffer": True,
            "eligible_max_chunk": eligible_max,
            "num_eligible_surfels": int(visible["num_eligible_cells"]),
            "num_visible_surfels": int(visible["num_visible_cells"]),
            "num_visible_pixels": int(len(visible["indices"])),
            "eligible_chunks": sorted(int(chunk) for chunk in scores),
            "retrieved": retrieved,
            "top3_chunks": [int(chunk) for chunk in explained[:3]],
            "selected_chunks": [int(chunk) for chunk in selected],
            "weights": (
                [
                    float(
                        cluster_scores[cluster_index]
                        / max(
                            sum(
                                cluster_scores[index]
                                for index in selected_cluster_indices
                            ),
                            1e-12,
                        )
                    )
                    for cluster_index in selected_cluster_indices
                ]
                if selected
                else []
            ),
            "scores": {
                str(chunk): float(value / max(score_sum, 1e-12))
                for chunk, value in sorted(scores.items())
            },
            "pose_equivalent_clusters": [
                [int(chunk) for chunk in cluster]
                for cluster in clusters
            ],
            "cluster_scores": {
                ",".join(
                    str(chunk) for chunk in clusters[index]
                ): float(
                    cluster_scores[index] / max(score_sum, 1e-12)
                )
                for index in ranked_cluster_indices
            },
            "all_chunk_support": {
                str(chunk): {
                    "visible_pixels": int(support[chunk]),
                    "visible_cells": int(len(cell_support[chunk])),
                    "mean_confidence": float(
                        confidence_sum[chunk] / max(support[chunk], 1)
                    ),
                    "coverage_score": float(
                        scores[chunk]
                        / max(total_visible_weight, 1e-12)
                    ),
                }
                for chunk in sorted(scores)
            },
            "positive_cluster": positive,
            "positive_vs_negative_margin": positive_vs_negative_margin,
            "positive_score_mass": positive_score_mass,
            "positive_cluster_hit": bool(
                selected and selected[0] in positive
            ),
            "positive_cluster_best_rank": next(
                (
                    rank
                    for rank, cluster_index in enumerate(
                        ranked_cluster_indices, start=1
                    )
                    if any(
                        chunk in positive
                        for chunk in clusters[cluster_index]
                    )
                ),
                None,
            ),
            "coverage_mask_path": None,
            "coverage_fraction": float(coverage.mean()),
            "total_visible_weight": float(total_visible_weight),
            "confidence_calibration": "per_frame_quantile_at_insert",
            "retrieval_ms": (
                time.perf_counter() - started
            )
            * 1000.0,
            "min_history_gap_chunks": self.min_history_gap_chunks,
            "use_view_alignment": self.use_view_alignment,
            "view_orientation_sigma_degrees": (
                self.view_orientation_sigma_degrees
            ),
            "view_translation_sigma": self.view_translation_sigma,
            "use_occlusion": self.use_occlusion,
        }
        return result, {"visible": visible, "coverage": coverage}


def pose_retrieve(
    sequence: dict,
    *,
    current_chunk: int,
    query_pose: np.ndarray,
    top_k: int,
    min_history_gap_chunks: int,
    positive_chunks: list[int] | tuple[int, ...] = (),
    query_pose_mode: str = "controlled_same_pose_known",
) -> dict:
    candidates = [
        item
        for item in sequence["frames"]
        if 0
        <= int(item["chunk_id"])
        <= current_chunk - min_history_gap_chunks
    ]
    distances = {
        int(item["chunk_id"]): _rotation_distance(
            np.asarray(item["camera_pose"]), query_pose
        )
        for item in candidates
    }
    ranked = sorted(
        distances, key=lambda chunk: (distances[chunk], chunk)
    )[: max(1, top_k)]
    inverse = {
        chunk: 1.0 / (1e-6 + distances[chunk])
        for chunk in distances
    }
    normalizer = sum(inverse[chunk] for chunk in ranked)
    positive = sorted({int(chunk) for chunk in positive_chunks})
    return {
        "target_chunk": int(current_chunk),
        "query_pose_mode": query_pose_mode,
        "eligible_chunks": sorted(distances),
        "rotation_distance_radians": {
            str(chunk): float(distance)
            for chunk, distance in sorted(distances.items())
        },
        "selected_chunks": [int(chunk) for chunk in ranked],
        "weights": [
            float(inverse[chunk] / normalizer) for chunk in ranked
        ],
        "scores": {
            str(chunk): float(1.0 / (1.0 + distance))
            for chunk, distance in sorted(distances.items())
        },
        "positive_cluster": positive,
        "positive_cluster_hit": bool(
            ranked and ranked[0] in positive
        ),
        "coverage_mask_path": None,
    }


def _save_support(
    path: Path,
    diagnostics: dict,
    image_size: tuple[int, int],
) -> None:
    support = np.zeros((*image_size, 3), dtype=np.uint8)
    visible = diagnostics["visible"]
    support[
        visible["pixels"][:, 0], visible["pixels"][:, 1]
    ] = [70, 120, 210]
    support[diagnostics["coverage"] > 0] = [235, 85, 70]
    from PIL import Image

    Image.fromarray(support).resize((832, 480)).save(path)


def build_retrieval(
    *,
    sequence_path: str | Path,
    surfel_index_path: str | Path,
    output_dir: str | Path,
    target_chunks: list[int],
    top_k: int = 1,
    min_history_gap_chunks: int = 2,
    use_view_alignment: bool = True,
    use_occlusion: bool = True,
    stable_only: bool = True,
    positive_chunks: list[int] | tuple[int, ...] = (),
    candidate_chunks: list[int] | tuple[int, ...] | None = None,
    pose_cluster_rotation_tolerance_degrees: float = 0.25,
    pose_cluster_translation_tolerance: float = 1e-4,
) -> tuple[list[dict], list[dict]]:
    sequence_path = Path(sequence_path).resolve()
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if sequence.get("cut3r_predicted_pose_used_for_map", True):
        raise ValueError(
            "Core-repair retrieval requires known-pose CUT3R geometry"
        )
    target_chunks = sorted({int(chunk) for chunk in target_chunks})
    if int(sequence["prefix_last_chunk"]) >= min(target_chunks):
        raise ValueError(
            "CUT3R prefix reaches the first memory target; future leakage"
        )
    index = SurfelIndex.load(surfel_index_path)
    query_pose = np.asarray(sequence["query_pose"], dtype=np.float32)
    query_intrinsics = np.asarray(
        sequence["query_intrinsics"], dtype=np.float32
    )
    query_pose_mode = str(
        sequence.get("query_pose_mode", "controlled_same_pose_known")
    )
    query_frame = next(
        item
        for item in sequence["frames"]
        if int(item["chunk_id"])
        == int(sequence["query_source_chunk"])
    )
    source_hw = tuple(int(x) for x in query_frame["shape"])
    render_hw = (60, 104)
    retriever = GeometryChunkRetriever(
        min_history_gap_chunks=min_history_gap_chunks,
        use_view_alignment=use_view_alignment,
        use_occlusion=use_occlusion,
        stable_only=stable_only,
    )
    chunk_clusters = contiguous_pose_clusters(
        sequence,
        rotation_tolerance_degrees=(
            pose_cluster_rotation_tolerance_degrees
        ),
        translation_tolerance=pose_cluster_translation_tolerance,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    surfel_entries = []
    pose_entries = []
    diagnostics_by_target = {}
    for target_chunk in target_chunks:
        surfel, diagnostics = retriever.retrieve(
            index,
            query_pose,
            query_intrinsics,
            source_image_size=source_hw,
            image_size=render_hw,
            current_chunk=target_chunk,
            top_k=top_k,
            positive_chunks=positive_chunks,
            chunk_clusters=chunk_clusters,
            query_pose_mode=query_pose_mode,
            candidate_chunks=candidate_chunks,
        )
        pose = pose_retrieve(
            sequence,
            current_chunk=target_chunk,
            query_pose=query_pose,
            top_k=top_k,
            min_history_gap_chunks=min_history_gap_chunks,
            positive_chunks=positive_chunks,
            query_pose_mode=query_pose_mode,
        )
        coverage_name = (
            f"selected_coverage_target_{target_chunk:04d}.npz"
        )
        np.savez_compressed(
            output_dir / coverage_name,
            coverage=diagnostics["coverage"],
        )
        surfel["coverage_mask_path"] = coverage_name
        surfel_entries.append(surfel)
        pose_entries.append(pose)
        diagnostics_by_target[target_chunk] = diagnostics
        _save_support(
            output_dir
            / f"visible_support_target_{target_chunk:04d}.png",
            diagnostics,
            render_hw,
        )

    payload = {
        "version": 2,
        "targets": surfel_entries,
        "future_leakage": False,
        "prefix_last_chunk": int(sequence["prefix_last_chunk"]),
        "target_chunks": target_chunks,
        "positive_cluster": sorted(
            {int(chunk) for chunk in positive_chunks}
        ),
        "geometry_pose_source": "known_control_c2w",
        "query_pose_mode": query_pose_mode,
        "retrieval_scope": (
            "all_causal_history"
            if candidate_chunks is None
            else "explicit_candidate_control"
        ),
        "candidate_chunks": (
            None
            if candidate_chunks is None
            else sorted({int(chunk) for chunk in candidate_chunks})
        ),
        "pose_cluster_rule": {
            "contiguous_only": True,
            "rotation_tolerance_degrees": (
                pose_cluster_rotation_tolerance_degrees
            ),
            "translation_tolerance": (
                pose_cluster_translation_tolerance
            ),
        },
    }
    (output_dir / "retrieval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "pose_plan.json").write_text(
        json.dumps(
            {"version": 2, "targets": pose_entries}, indent=2
        ),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(surfel_entries),
        1,
        figsize=(10, max(4, 3 * len(surfel_entries))),
        squeeze=False,
    )
    for axis, entry in zip(axes[:, 0], surfel_entries):
        chunks = [int(chunk) for chunk in entry["scores"]]
        values = [entry["scores"][str(chunk)] for chunk in chunks]
        colors = [
            "#54a24b"
            if chunk in entry["positive_cluster"]
            else "#4f7cac"
            for chunk in chunks
        ]
        axis.bar(chunks, values, color=colors)
        for item in entry["retrieved"]:
            axis.axvline(
                item["chunk_id"], color="#e45756", alpha=0.5
            )
        axis.set(
            title=(
                "Eligible-first visible-surfel vote for target "
                f"{entry['target_chunk']}"
            ),
            xlabel="historical chunk",
            ylabel="normalized score",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_timeline.png", dpi=160)
    plt.close(fig)

    primary_target = target_chunks[-1]
    _save_support(
        output_dir / "visible_support.png",
        diagnostics_by_target[primary_target],
        render_hw,
    )
    return surfel_entries, pose_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve historical KV chunks from eligible target-visible "
            "surfels"
        )
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--surfel_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_chunk", type=int)
    parser.add_argument("--target_chunks", nargs="+", type=int)
    parser.add_argument("--positive_chunks", nargs="*", type=int, default=[])
    parser.add_argument("--candidate_chunks", nargs="*", type=int)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--min_history_gap_chunks", type=int, default=2
    )
    parser.add_argument("--no_view_alignment", action="store_true")
    parser.add_argument("--no_occlusion", action="store_true")
    parser.add_argument("--include_tentative", action="store_true")
    parser.add_argument(
        "--pose_cluster_rotation_tolerance_degrees",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--pose_cluster_translation_tolerance",
        type=float,
        default=1e-4,
    )
    args = parser.parse_args()
    targets = args.target_chunks
    if targets is None:
        if args.target_chunk is None:
            raise ValueError(
                "Use --target_chunk or --target_chunks"
            )
        targets = [args.target_chunk]
    build_retrieval(
        sequence_path=args.sequence,
        surfel_index_path=args.surfel_index,
        output_dir=args.output_dir,
        target_chunks=targets,
        top_k=args.top_k,
        min_history_gap_chunks=args.min_history_gap_chunks,
        use_view_alignment=not args.no_view_alignment,
        use_occlusion=not args.no_occlusion,
        stable_only=not args.include_tentative,
        positive_chunks=args.positive_chunks,
        candidate_chunks=args.candidate_chunks,
        pose_cluster_rotation_tolerance_degrees=(
            args.pose_cluster_rotation_tolerance_degrees
        ),
        pose_cluster_translation_tolerance=(
            args.pose_cluster_translation_tolerance
        ),
    )


if __name__ == "__main__":
    main()
