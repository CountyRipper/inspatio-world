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
    ) -> tuple[dict, dict]:
        started = time.perf_counter()
        visible = surfel_index.visible_cells(
            query_pose,
            intrinsics,
            image_size,
            source_image_size=source_image_size,
            use_occlusion=self.use_occlusion,
        )
        eligible_max = int(current_chunk) - self.min_history_gap_chunks
        scores: dict[int, float] = {}
        support: dict[int, int] = {}
        confidence_sum: dict[int, float] = {}
        cell_contributions: list[dict[int, float]] = []
        query_camera = np.asarray(query_pose)[:3, 3]
        total_visible_weight = 0.0
        for visible_offset, cell_index in enumerate(visible["indices"]):
            cell = surfel_index.cells[int(cell_index)]
            base = float(cell.confidence) / (1.0 + float(visible["depth"][visible_offset]))
            total_chunk_weight = sum(
                weight for chunk, weight in cell.chunk_weights.items()
                if 0 <= int(chunk) <= eligible_max
            )
            contributions = {}
            if total_chunk_weight <= 0:
                cell_contributions.append(contributions)
                continue
            total_visible_weight += base
            target_direction = query_camera - cell.xyz
            target_direction /= max(float(np.linalg.norm(target_direction)), 1e-8)
            for chunk, chunk_weight in cell.chunk_weights.items():
                chunk = int(chunk)
                if chunk < 0 or chunk > eligible_max:
                    continue
                fraction = float(chunk_weight) / total_chunk_weight
                alignment = 1.0
                if self.use_view_alignment and chunk in cell.view_dirs:
                    alignment = max(
                        0.0, float(np.dot(target_direction, cell.view_dirs[chunk]))
                    )
                contribution = base * fraction * alignment
                contributions[chunk] = contribution
                scores[chunk] = scores.get(chunk, 0.0) + contribution
                support[chunk] = support.get(chunk, 0) + 1
                confidence_sum[chunk] = confidence_sum.get(chunk, 0.0) + cell.confidence
            cell_contributions.append(contributions)
        ranked = sorted(scores, key=lambda chunk: (-scores[chunk], chunk))[: max(1, top_k)]
        score_sum = sum(scores.values())
        retrieved = [
            {
                "chunk_id": int(chunk),
                "score": float(scores[chunk] / max(score_sum, 1e-12)),
                "raw_score": float(scores[chunk]),
                "visible_support": int(support[chunk]),
                "mean_confidence": float(confidence_sum[chunk] / max(support[chunk], 1)),
                "temporal_gap_chunks": int(current_chunk - chunk),
            }
            for chunk in ranked
        ]
        selected = ranked[0] if ranked else None
        coverage = np.zeros(image_size, dtype=np.float32)
        if selected is not None:
            for pixel, contributions in zip(visible["pixels"], cell_contributions):
                if contributions.get(selected, 0.0) > 0:
                    coverage[int(pixel[0]), int(pixel[1])] = 1.0
        result = {
            "target_chunk": int(current_chunk),
            "query_pose_mode": "controlled_same_pose",
            "num_visible_surfels": int(len(visible["indices"])),
            "eligible_chunks": sorted(int(chunk) for chunk in scores),
            "retrieved": retrieved,
            "selected_chunks": [int(chunk) for chunk in ranked],
            "weights": (
                [float(scores[chunk] / max(sum(scores[c] for c in ranked), 1e-12)) for chunk in ranked]
                if ranked else []
            ),
            "scores": {
                str(chunk): float(value / max(score_sum, 1e-12))
                for chunk, value in sorted(scores.items())
            },
            "coverage_mask_path": "selected_coverage.npz",
            "coverage_fraction": float(coverage.mean()),
            "total_visible_weight": float(total_visible_weight),
            "retrieval_ms": (time.perf_counter() - started) * 1000.0,
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
) -> dict:
    candidates = [
        item for item in sequence["frames"]
        if 0 <= int(item["chunk_id"]) <= current_chunk - min_history_gap_chunks
    ]
    distances = {
        int(item["chunk_id"]): _rotation_distance(
            np.asarray(item["camera_pose"]), query_pose
        )
        for item in candidates
    }
    ranked = sorted(distances, key=lambda chunk: (distances[chunk], chunk))[: max(1, top_k)]
    inverse = {chunk: 1.0 / (1e-6 + distances[chunk]) for chunk in distances}
    normalizer = sum(inverse[chunk] for chunk in ranked)
    return {
        "target_chunk": int(current_chunk),
        "query_pose_mode": "controlled_same_pose",
        "eligible_chunks": sorted(distances),
        "rotation_distance_radians": {
            str(chunk): float(distance) for chunk, distance in sorted(distances.items())
        },
        "selected_chunks": [int(chunk) for chunk in ranked],
        "weights": [float(inverse[chunk] / normalizer) for chunk in ranked],
        "scores": {
            str(chunk): float(1.0 / (1.0 + distance))
            for chunk, distance in sorted(distances.items())
        },
        "coverage_mask_path": None,
    }


def build_retrieval(
    *,
    sequence_path: str | Path,
    surfel_index_path: str | Path,
    output_dir: str | Path,
    target_chunk: int,
    top_k: int = 1,
    min_history_gap_chunks: int = 2,
    use_view_alignment: bool = True,
    use_occlusion: bool = True,
) -> tuple[dict, dict]:
    sequence_path = Path(sequence_path).resolve()
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    index = SurfelIndex.load(surfel_index_path)
    query_pose = np.asarray(sequence["query_pose"], dtype=np.float32)
    query_intrinsics = np.asarray(sequence["query_intrinsics"], dtype=np.float32)
    query_frame = next(
        item for item in sequence["frames"]
        if int(item["chunk_id"]) == int(sequence["query_source_chunk"])
    )
    source_hw = tuple(int(x) for x in query_frame["shape"])
    render_hw = (60, 104)
    retriever = GeometryChunkRetriever(
        min_history_gap_chunks=min_history_gap_chunks,
        use_view_alignment=use_view_alignment,
        use_occlusion=use_occlusion,
    )
    surfel, diagnostics = retriever.retrieve(
        index,
        query_pose,
        query_intrinsics,
        source_image_size=source_hw,
        image_size=render_hw,
        current_chunk=target_chunk,
        top_k=top_k,
    )
    pose = pose_retrieve(
        sequence,
        current_chunk=target_chunk,
        query_pose=query_pose,
        top_k=top_k,
        min_history_gap_chunks=min_history_gap_chunks,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "selected_coverage.npz",
        coverage=diagnostics["coverage"],
    )
    payload = {
        "version": 1,
        "targets": [surfel],
        "future_leakage": False,
        "prefix_last_chunk": int(sequence["prefix_last_chunk"]),
        "target_chunk": int(target_chunk),
    }
    (output_dir / "retrieval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "pose_plan.json").write_text(
        json.dumps({"version": 1, "targets": [pose]}, indent=2), encoding="utf-8"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunks = [int(chunk) for chunk in surfel["scores"]]
    values = [surfel["scores"][str(chunk)] for chunk in chunks]
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(chunks, values, color="#4f7cac")
    for item in surfel["retrieved"]:
        axis.axvline(item["chunk_id"], color="#e45756", alpha=0.6)
    axis.set(
        title=f"Visible-surfel chunk vote for target {target_chunk}",
        xlabel="historical chunk",
        ylabel="normalized score",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_timeline.png", dpi=160)
    plt.close(fig)

    support = np.zeros((*render_hw, 3), dtype=np.uint8)
    visible = diagnostics["visible"]
    support[visible["pixels"][:, 0], visible["pixels"][:, 1]] = [70, 120, 210]
    support[diagnostics["coverage"] > 0] = [235, 85, 70]
    from PIL import Image
    Image.fromarray(support).resize((832, 480)).save(output_dir / "visible_support.png")
    return surfel, pose


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve historical KV chunks from visible surfels")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--surfel_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_chunk", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--min_history_gap_chunks", type=int, default=2)
    parser.add_argument("--no_view_alignment", action="store_true")
    parser.add_argument("--no_occlusion", action="store_true")
    args = parser.parse_args()
    build_retrieval(
        sequence_path=args.sequence,
        surfel_index_path=args.surfel_index,
        output_dir=args.output_dir,
        target_chunk=args.target_chunk,
        top_k=args.top_k,
        min_history_gap_chunks=args.min_history_gap_chunks,
        use_view_alignment=not args.no_view_alignment,
        use_occlusion=not args.no_occlusion,
    )


if __name__ == "__main__":
    main()
