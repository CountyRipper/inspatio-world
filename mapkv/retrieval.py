from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .surfel_index import SurfelIndex


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
    ):
        self.min_history_gap_chunks = int(min_history_gap_chunks)
        self.use_view_alignment = bool(use_view_alignment)
        self.use_occlusion = bool(use_occlusion)

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
    ) -> tuple[dict, dict]:
        """Vote from target-visible, causally eligible surfels.

        Eligibility is applied inside visible_cells before projection and
        z-buffering. Therefore an immediate-recent-only surface cannot occlude
        an older long-term candidate and then disappear during voting.
        """
        started = time.perf_counter()
        eligible_max = int(current_chunk) - self.min_history_gap_chunks
        visible = surfel_index.visible_cells(
            query_pose,
            intrinsics,
            image_size,
            source_image_size=source_image_size,
            eligible_max_chunk=eligible_max,
            use_occlusion=self.use_occlusion,
        )
        scores: dict[int, float] = {}
        support: dict[int, int] = {}
        confidence_sum: dict[int, float] = {}
        cell_support: dict[int, set[int]] = {}
        cell_contributions: list[set[int]] = []
        query_camera = np.asarray(query_pose)[:3, 3]
        visible_cell_ids = np.unique(visible["indices"])
        confidence_values = np.asarray(
            [
                surfel_index.cells[int(index)].confidence
                for index in visible_cell_ids
            ],
            dtype=np.float32,
        )
        confidence_cap = (
            float(np.quantile(confidence_values, 0.95))
            if confidence_values.size
            else 1.0
        )
        total_visible_weight = 0.0
        for visible_offset, cell_index in enumerate(visible["indices"]):
            cell_index = int(cell_index)
            cell = surfel_index.cells[cell_index]
            eligible_observations = sorted(
                {
                    int(chunk)
                    for chunk in cell.observing_chunks
                    if 0 <= int(chunk) <= eligible_max
                }
            )
            if not eligible_observations:
                raise AssertionError(
                    "Ineligible surfel survived pre-z-buffer causal filtering"
                )
            confidence = min(max(float(cell.confidence), 0.0), confidence_cap)
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
                if self.use_view_alignment and chunk in cell.view_dirs:
                    alignment = max(
                        0.0,
                        float(
                            np.dot(
                                target_direction,
                                cell.view_dirs[chunk],
                            )
                        ),
                    )
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
            cell_contributions.append(contributed)

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
            sum(scores[chunk] for chunk in cluster)
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
                visible["pixels"], cell_contributions
            ):
                if selected_chunk in contributions:
                    coverage[int(pixel[0]), int(pixel[1])] = 1.0
        positive = sorted({int(chunk) for chunk in positive_chunks})
        result = {
            "target_chunk": int(current_chunk),
            "query_pose_mode": "controlled_same_pose_known",
            "voting_mode": "simple_observing_chunk_vote",
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
            "confidence_cap_p95": confidence_cap,
            "retrieval_ms": (
                time.perf_counter() - started
            )
            * 1000.0,
            "min_history_gap_chunks": self.min_history_gap_chunks,
            "use_view_alignment": self.use_view_alignment,
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
        "query_pose_mode": "controlled_same_pose_known",
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
    positive_chunks: list[int] | tuple[int, ...] = (),
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
    )
    chunk_clusters = contiguous_pose_clusters(sequence)
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
        )
        pose = pose_retrieve(
            sequence,
            current_chunk=target_chunk,
            query_pose=query_pose,
            top_k=top_k,
            min_history_gap_chunks=min_history_gap_chunks,
            positive_chunks=positive_chunks,
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
        "pose_cluster_rule": {
            "contiguous_only": True,
            "rotation_tolerance_degrees": 0.25,
            "translation_tolerance": 0.0001,
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
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--min_history_gap_chunks", type=int, default=2
    )
    parser.add_argument("--no_view_alignment", action="store_true")
    parser.add_argument("--no_occlusion", action="store_true")
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
        positive_chunks=args.positive_chunks,
    )


if __name__ == "__main__":
    main()
