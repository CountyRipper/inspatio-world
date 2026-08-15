from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mapkv_proto.cut3r.surfel_index import SurfelIndex
from mapkv_proto.pose_utils import scale_intrinsics


def chunk_color(chunk_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(chunk_id + 104729)
    return tuple(int(value) for value in rng.integers(48, 256, size=3))


def save_visualizations(
    *,
    view: dict,
    views_root: Path,
    rendered: dict[str, np.ndarray],
    selected_chunk: int | None,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    image = Image.open(views_root / view["image_path"]).convert("RGB")
    coverage = Image.fromarray(
        (rendered["coverage"].clip(0, 1) * 255).astype(np.uint8)
    ).resize(image.size, Image.Resampling.NEAREST)
    color = chunk_color(selected_chunk) if selected_chunk is not None else (255, 0, 0)
    overlay = Image.new("RGBA", image.size, color + (0,))
    overlay.putalpha(coverage.point(lambda value: int(0.55 * value)))
    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(composed)
    draw.rectangle((0, 0, composed.width, 24), fill="black")
    draw.text((6, 5), f"selected chunk: {selected_chunk}", fill="white")
    composed.save(output_root / f"target_{int(view['chunk_id']):04d}_overlay.png")

    ids = rendered["surfel_id"]
    color_map = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for surfel_id in np.unique(ids[ids >= 0]):
        color_map[ids == surfel_id] = chunk_color(int(surfel_id))
    Image.fromarray(color_map).resize(image.size, Image.Resampling.NEAREST).save(
        output_root / f"target_{int(view['chunk_id']):04d}_surfel_ids.png"
    )
    depth = rendered["depth"]
    valid = depth > 0
    depth_image = np.zeros_like(depth, dtype=np.uint8)
    if valid.any():
        low, high = np.quantile(depth[valid], [0.02, 0.98])
        depth_image[valid] = (
            255 * np.clip((depth[valid] - low) / max(high - low, 1e-8), 0, 1)
        ).astype(np.uint8)
    Image.fromarray(depth_image).resize(image.size, Image.Resampling.NEAREST).save(
        output_root / f"target_{int(view['chunk_id']):04d}_depth.png"
    )


def build_plan(
    *,
    surfel_index_path: str | Path,
    views_json: str | Path,
    output_path: str | Path,
    target_chunks: list[int],
    grid_hw: tuple[int, int] = (30, 52),
    oracle_sources: dict[int, int] | None = None,
    address_plan_path: str | Path | None = None,
    fixed_selected_chunk: int | None = None,
) -> list[dict]:
    index = SurfelIndex.load(surfel_index_path)
    views_path = Path(views_json)
    payload = json.loads(views_path.read_text(encoding="utf-8"))
    by_chunk = {
        int(view["chunk_id"]): view
        for view in payload["views"]
        if not view["is_anchor"]
    }
    output_path = Path(output_path)
    output_root = output_path.parent
    coverage_root = output_root / "coverage"
    visualization_root = output_root / "retrieval_visualization"
    coverage_root.mkdir(parents=True, exist_ok=True)
    address_entries = None
    if address_plan_path is not None and fixed_selected_chunk is not None:
        raise ValueError("Use either address_plan_path or fixed_selected_chunk, not both")
    if address_plan_path is not None:
        address_payload = json.loads(
            Path(address_plan_path).read_text(encoding="utf-8")
        )
        raw_entries = (
            address_payload.get("targets", address_payload)
            if isinstance(address_payload, dict)
            else address_payload
        )
        address_entries = {
            int(entry["target_chunk"]): entry for entry in raw_entries
        }
    elif fixed_selected_chunk is not None:
        address_entries = {
            int(target): {
                "target_chunk": int(target),
                "candidate_chunks": [int(fixed_selected_chunk)],
                "scores": {str(int(fixed_selected_chunk)): 1.0},
                "selected_chunks": [int(fixed_selected_chunk)],
                "weights": [1.0],
                "address_mode": "fixed_oracle",
            }
            for target in target_chunks
        }
    plan = []
    for target_chunk in target_chunks:
        if target_chunk not in by_chunk:
            raise KeyError(f"No exported view for target chunk {target_chunk}")
        view = by_chunk[target_chunk]
        intrinsic = scale_intrinsics(
            np.asarray(view["intrinsics"]),
            source_hw=tuple(view["image_hw"]),
            target_hw=grid_hw,
        )
        geometry_entry, rendered = index.retrieve(
            target_chunk=target_chunk,
            c2w=np.asarray(view["c2w_cut3r"]),
            intrinsic=intrinsic,
            image_hw=grid_hw,
            oracle_chunk=(oracle_sources or {}).get(target_chunk),
        )
        if address_entries is None:
            entry = geometry_entry
        else:
            if target_chunk not in address_entries:
                raise KeyError(
                    f"Address plan has no entry for target chunk {target_chunk}"
                )
            entry = dict(address_entries[target_chunk])
            selected_chunks = entry.get("selected_chunks", [])
            if len(selected_chunks) > 1:
                raise ValueError("The first MapKV prototype supports top_k=1 only")
            selected_for_gate = selected_chunks[0] if selected_chunks else None
            rendered["coverage"] = index.coverage_for_chunk(
                rendered,
                chunk_id=selected_for_gate,
                target_chunk=target_chunk,
            )
            oracle = (oracle_sources or {}).get(target_chunk)
            entry["oracle_hit"] = (
                oracle is not None and selected_for_gate == oracle
            )
            entry["geometry_vote_diagnostic"] = geometry_entry
        coverage_path = coverage_root / f"target_{target_chunk:04d}.npz"
        np.savez_compressed(
            coverage_path,
            coverage=rendered["coverage"],
            depth=rendered["depth"],
            surfel_id=rendered["surfel_id"],
            cosine=rendered["cosine"],
        )
        entry["coverage_mask_path"] = str(coverage_path.relative_to(output_root))
        selected = entry["selected_chunks"][0] if entry["selected_chunks"] else None
        save_visualizations(
            view=view,
            views_root=views_path.parent,
            rendered=rendered,
            selected_chunk=selected,
            output_root=visualization_root,
        )
        plan.append(entry)
    output_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Render/vote a causal surfel retrieval plan")
    parser.add_argument("--surfel_index", required=True)
    parser.add_argument("--views_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_chunks", nargs="+", type=int, required=True)
    parser.add_argument("--grid_height", type=int, default=30)
    parser.add_argument("--grid_width", type=int, default=52)
    parser.add_argument("--oracle_source", type=int)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--address_plan",
        help="Optional pose/oracle address plan; geometry is then used only for its gate.",
    )
    parser.add_argument(
        "--fixed_selected_chunk",
        type=int,
        help="Use one manually fixed address and build its matched surfel gate.",
    )
    args = parser.parse_args()
    if args.top_k != 1:
        raise ValueError("The first MapKV prototype supports top_k=1 only")
    oracle_sources = None
    if args.oracle_source is not None:
        oracle_sources = {target: args.oracle_source for target in args.target_chunks}
    build_plan(
        surfel_index_path=args.surfel_index,
        views_json=args.views_json,
        output_path=args.output,
        target_chunks=args.target_chunks,
        grid_hw=(args.grid_height, args.grid_width),
        oracle_sources=oracle_sources,
        address_plan_path=args.address_plan,
        fixed_selected_chunk=args.fixed_selected_chunk,
    )


if __name__ == "__main__":
    main()
