from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .pose_utils import pose_distance
from .revisit_pair import load_blocks


def causal_candidates(target_chunk: int, available_chunks: Iterable[int]) -> list[int]:
    """Exclude future/current/immediate-previous chunks from the address space."""
    return sorted(
        int(chunk) for chunk in available_chunks
        if int(chunk) >= 0 and int(chunk) < target_chunk - 1
    )


def build_pose_retrieval_plan(
    blocks: list[dict],
    *,
    target_chunks: Iterable[int],
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    oracle_sources: dict[int, int] | None = None,
) -> list[dict]:
    by_id = {int(block["chunk_id"]): block for block in blocks}
    plan = []
    for target_chunk in target_chunks:
        target_chunk = int(target_chunk)
        candidates = causal_candidates(target_chunk, by_id)
        if not candidates:
            plan.append(
                {
                    "target_chunk": target_chunk,
                    "candidate_chunks": [],
                    "scores": {},
                    "selected_chunks": [],
                    "weights": [],
                    "oracle_hit": False,
                    "coverage_mask_path": None,
                }
            )
            continue
        target_pose = np.asarray(by_id[target_chunk]["c2w"])
        distances = {}
        details = {}
        for chunk in candidates:
            distance, translation, rotation = pose_distance(
                np.asarray(by_id[chunk]["c2w"]),
                target_pose,
                translation_weight=translation_weight,
                rotation_weight=rotation_weight,
            )
            distances[chunk] = distance
            details[str(chunk)] = {
                "translation_distance": translation,
                "rotation_distance_radians": rotation,
            }
        selected = min(distances, key=distances.get)
        scores = {
            str(chunk): float(1.0 / (1.0 + distance))
            for chunk, distance in distances.items()
        }
        oracle = (oracle_sources or {}).get(target_chunk)
        plan.append(
            {
                "target_chunk": target_chunk,
                "candidate_chunks": candidates,
                "scores": scores,
                "pose_distance_components": details,
                "selected_chunks": [selected],
                "weights": [1.0],
                "oracle_hit": oracle is not None and selected == oracle,
                "coverage_mask_path": None,
            }
        )
    return plan


class RetrievalPlan:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        entries = payload.get("targets", payload) if isinstance(payload, dict) else payload
        self.entries = {int(entry["target_chunk"]): entry for entry in entries}

    def entry(self, target_chunk: int) -> dict:
        if target_chunk not in self.entries:
            raise KeyError(f"Target chunk {target_chunk} is absent from {self.path}")
        return self.entries[target_chunk]

    def selected_chunk(self, target_chunk: int) -> int | None:
        selected = self.entry(target_chunk).get("selected_chunks", [])
        if not selected:
            return None
        if len(selected) != 1:
            raise ValueError("MapKV prototype supports top_k=1 only")
        return int(selected[0])

    def load_coverage(self, target_chunk: int) -> torch.Tensor | None:
        relative = self.entry(target_chunk).get("coverage_mask_path")
        if not relative:
            return None
        path = Path(relative)
        if not path.is_absolute():
            path = self.path.parent / path
        with np.load(path) as payload:
            if "coverage" not in payload:
                raise KeyError(f"Coverage file has no 'coverage' array: {path}")
            return torch.from_numpy(payload["coverage"].astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pose-only top-1 KV retrieval plan")
    parser.add_argument("--block_mapping", required=True)
    parser.add_argument("--target_chunks", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--translation_weight", type=float, default=1.0)
    parser.add_argument("--rotation_weight", type=float, default=1.0)
    args = parser.parse_args()
    blocks, _ = load_blocks(args.block_mapping)
    plan = build_pose_retrieval_plan(
        blocks,
        target_chunks=args.target_chunks,
        translation_weight=args.translation_weight,
        rotation_weight=args.rotation_weight,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
