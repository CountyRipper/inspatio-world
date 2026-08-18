from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .surfel_index import SurfelIndex


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _project(
    positions: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    homogeneous = np.concatenate(
        [positions, np.ones((len(positions), 1), dtype=np.float64)],
        axis=1,
    )
    camera = (homogeneous @ np.linalg.inv(c2w).T)[:, :3]
    depth = camera[:, 2]
    u = intrinsics[0, 0] * camera[:, 0] / np.maximum(depth, 1e-8)
    u += intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / np.maximum(depth, 1e-8)
    v += intrinsics[1, 2]
    return u, v, depth


def _token_indices(
    *,
    index: SurfelIndex,
    visible_cell_ids: np.ndarray,
    source_pose: np.ndarray,
    source_intrinsics: np.ndarray,
    source_hw: tuple[int, int],
    token_hw: tuple[int, int],
    frames_per_block: int,
    neighborhood: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    positions = np.asarray(
        [index.cells[int(cell)].xyz for cell in visible_cell_ids],
        dtype=np.float64,
    )
    u, v, depth = _project(
        positions, source_pose, source_intrinsics
    )
    source_h, source_w = source_hw
    valid = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(u)
        & np.isfinite(v)
        & (depth > 1e-5)
        & (u >= 0)
        & (u < source_w)
        & (v >= 0)
        & (v < source_h)
    )
    token_h, token_w = token_hw
    token_x = np.floor(u[valid] * token_w / source_w).astype(np.int64)
    token_y = np.floor(v[valid] * token_h / source_h).astype(np.int64)
    spatial: set[int] = set()
    for y, x in zip(token_y, token_x):
        for dy in range(-neighborhood, neighborhood + 1):
            for dx in range(-neighborhood, neighborhood + 1):
                yy, xx = int(y + dy), int(x + dx)
                if 0 <= yy < token_h and 0 <= xx < token_w:
                    spatial.add(yy * token_w + xx)
    spatial_indices = np.asarray(sorted(spatial), dtype=np.int64)
    tokens_per_frame = token_h * token_w
    token_indices = np.concatenate(
        [
            spatial_indices + frame * tokens_per_frame
            for frame in range(frames_per_block)
        ]
    )
    spatial_mask = np.zeros(token_hw, dtype=np.float32)
    if len(spatial_indices):
        spatial_mask[
            spatial_indices // token_w,
            spatial_indices % token_w,
        ] = 1.0
    return token_indices, spatial_mask, {
        "visible_cells": int(len(visible_cell_ids)),
        "source_projectable_cells": int(np.count_nonzero(valid)),
        "selected_spatial_tokens": int(len(spatial_indices)),
        "selected_tokens": int(len(token_indices)),
        "selected_token_fraction": float(
            len(token_indices)
            / max(frames_per_block * tokens_per_frame, 1)
        ),
    }


def _save_overlay(
    source_path: Path,
    spatial_mask: np.ndarray,
    destination: Path,
) -> None:
    image = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32)
    mask = np.asarray(
        Image.fromarray((spatial_mask * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
        ),
        dtype=np.float32,
    ) / 255.0
    overlay = image.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], 235.0 * mask)
    overlay[..., 1:] *= 1.0 - 0.45 * mask[..., None]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(
        destination
    )


def build_token_selection(
    *,
    sequence_path: str | Path,
    surfel_index_path: str | Path,
    retrieval_plan_path: str | Path,
    output_plan_path: str | Path,
    token_hw: tuple[int, int] = (30, 52),
    frames_per_block: int = 3,
    neighborhood: int = 1,
) -> dict:
    sequence_path = Path(sequence_path).resolve()
    plan_path = Path(retrieval_plan_path).resolve()
    output_path = Path(output_plan_path).resolve()
    sequence = _json(sequence_path)
    plan = _json(plan_path)
    index = SurfelIndex.load(surfel_index_path)
    query_pose = np.asarray(sequence["query_pose"], dtype=np.float64)
    query_intrinsics = np.asarray(
        sequence["query_intrinsics"], dtype=np.float64
    )
    query_source = next(
        item
        for item in sequence["frames"]
        if int(item["chunk_id"]) == int(sequence["query_source_chunk"])
    )
    query_source_hw = tuple(int(value) for value in query_source["shape"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    selection_cache: dict[
        int, tuple[np.ndarray, np.ndarray, dict]
    ] = {}
    for entry in plan["targets"]:
        target_chunk = int(entry["target_chunk"])
        selected = [int(chunk) for chunk in entry["selected_chunks"]]
        if len(selected) != 1:
            raise ValueError("Token selection supports top-K=1 only")
        source_chunk = selected[0]
        source_frame = next(
            item
            for item in sequence["frames"]
            if int(item["chunk_id"]) == source_chunk
        )
        if source_chunk not in selection_cache:
            visible = index.visible_cells(
                query_pose,
                query_intrinsics,
                (60, 104),
                source_image_size=query_source_hw,
                eligible_max_chunk=target_chunk - 2,
                eligible_chunks={source_chunk},
                use_occlusion=True,
            )
            visible_ids = np.unique(visible["indices"])
            source_hw = tuple(
                int(value) for value in source_frame["shape"]
            )
            selection_cache[source_chunk] = _token_indices(
                index=index,
                visible_cell_ids=visible_ids,
                source_pose=np.asarray(
                    source_frame["camera_pose"], dtype=np.float64
                ),
                source_intrinsics=np.asarray(
                    source_frame["intrinsics"], dtype=np.float64
                ),
                source_hw=source_hw,
                token_hw=token_hw,
                frames_per_block=frames_per_block,
                neighborhood=neighborhood,
            )
        token_indices, spatial_mask, stats = selection_cache[source_chunk]
        if not len(token_indices):
            raise RuntimeError(
                f"No historical KV tokens selected for target {target_chunk}"
            )
        relative = Path(
            f"selected_tokens_target_{target_chunk:04d}.npz"
        )
        np.savez_compressed(
            output_path.parent / relative,
            token_indices=token_indices,
            spatial_mask=spatial_mask,
            source_chunk=np.asarray([source_chunk], dtype=np.int32),
        )
        entry["selected_token_indices_path"] = str(relative)
        entry["selected_token_count"] = int(len(token_indices))
        entry["selected_token_fraction"] = stats[
            "selected_token_fraction"
        ]
        entry["token_selection"] = {
            "mode": "target_visible_surfel_to_source_token",
            "token_hw": list(token_hw),
            "frames_per_block": frames_per_block,
            "neighborhood": neighborhood,
            **stats,
        }
        overlay = output_path.parent / (
            f"selected_tokens_source_target_{target_chunk:04d}.png"
        )
        _save_overlay(
            Path(source_frame["image_path"]), spatial_mask, overlay
        )
        diagnostics.append(
            {
                "target_chunk": target_chunk,
                "source_chunk": source_chunk,
                "overlay_path": overlay.name,
                **stats,
            }
        )
    plan["payload_scope"] = "surfel_selected_historical_tokens"
    plan["token_selection"] = {
        "coordinate_source": "selected chunk known c2w + CUT3R intrinsics",
        "token_hw": list(token_hw),
        "frames_per_block": frames_per_block,
        "neighborhood": neighborhood,
        "spatial_rope_readdressed": False,
    }
    output_path.write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    result = {
        "output_plan": str(output_path),
        "targets": diagnostics,
    }
    (output_path.parent / "token_selection_stats.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select historical KV tokens through visible surfels"
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--surfel_index", required=True)
    parser.add_argument("--retrieval_plan", required=True)
    parser.add_argument("--output_plan", required=True)
    parser.add_argument("--token_height", type=int, default=30)
    parser.add_argument("--token_width", type=int, default=52)
    parser.add_argument("--frames_per_block", type=int, default=3)
    parser.add_argument("--neighborhood", type=int, default=1)
    args = parser.parse_args()
    result = build_token_selection(
        sequence_path=args.sequence,
        surfel_index_path=args.surfel_index,
        retrieval_plan_path=args.retrieval_plan,
        output_plan_path=args.output_plan,
        token_hw=(args.token_height, args.token_width),
        frames_per_block=args.frames_per_block,
        neighborhood=args.neighborhood,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
