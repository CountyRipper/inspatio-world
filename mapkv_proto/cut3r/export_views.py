from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from mapkv_proto.pose_utils import scale_intrinsics, tcw_to_c2w, to_cut3r_c2w
from mapkv_proto.revisit_pair import load_blocks


def load_intrinsic(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path)
    if values.shape[0] < 3 or values.shape[1] < 3:
        raise ValueError(f"Invalid intrinsic matrix file: {path}")
    return values[:3, :3].astype(np.float64)


def export_views(
    *,
    block_mapping: str | Path,
    anchor_image: str | Path,
    intrinsic_path: str | Path,
    output_root: str | Path,
    intrinsic_source_hw: tuple[int, int] | None = None,
    anchor_tcw: np.ndarray | None = None,
) -> dict:
    blocks, mapping_root = load_blocks(block_mapping)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    first_image = Image.open(mapping_root / blocks[0]["png_path"])
    target_hw = (first_image.height, first_image.width)
    intrinsic = load_intrinsic(intrinsic_path)
    if intrinsic_source_hw is None:
        inferred_h = max(int(round(2 * intrinsic[1, 2])), 1)
        inferred_w = max(int(round(2 * intrinsic[0, 2])), 1)
        intrinsic_source_hw = (inferred_h, inferred_w)
    intrinsic = scale_intrinsics(
        intrinsic, source_hw=intrinsic_source_hw, target_hw=target_hw
    )

    anchor_destination = output_root / "anchor.png"
    shutil.copy2(anchor_image, anchor_destination)
    if anchor_tcw is None:
        anchor_tcw = np.asarray(blocks[0]["Tcw"], dtype=np.float64)
    anchor_c2w = tcw_to_c2w(anchor_tcw)
    views = [
        {
            "view_id": 0,
            "chunk_id": -1,
            "image_path": anchor_destination.name,
            "rgb_frame_index": 0,
            "Tcw": np.asarray(anchor_tcw).tolist(),
            "c2w": anchor_c2w.tolist(),
            "c2w_cut3r": to_cut3r_c2w(anchor_c2w).tolist(),
            "intrinsics": intrinsic.tolist(),
            "image_hw": list(target_hw),
            "is_anchor": True,
        }
    ]
    for block in blocks:
        destination = output_root / f"chunk_{int(block['chunk_id']):04d}.png"
        shutil.copy2(mapping_root / block["png_path"], destination)
        c2w = np.asarray(block["c2w"], dtype=np.float64)
        views.append(
            {
                "view_id": len(views),
                "chunk_id": int(block["chunk_id"]),
                "image_path": destination.name,
                "rgb_frame_index": int(block["rgb_center_index"]),
                "Tcw": block["Tcw"],
                "c2w": block["c2w"],
                "c2w_cut3r": to_cut3r_c2w(c2w).tolist(),
                "intrinsics": intrinsic.tolist(),
                "image_hw": list(target_hw),
                "is_anchor": False,
            }
        )
    payload = {
        "coordinate_convention": "c2w_cut3r = c2w @ diag(1,-1,-1,1)",
        "intrinsic_source_hw": list(intrinsic_source_hw),
        "generated_image_hw": list(target_hw),
        "views": views,
    }
    (output_root / "views.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic baseline views to CUT3R")
    parser.add_argument("--block_mapping", required=True)
    parser.add_argument("--anchor_image", required=True)
    parser.add_argument("--intrinsic_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--intrinsic_source_height", type=int)
    parser.add_argument("--intrinsic_source_width", type=int)
    args = parser.parse_args()
    source_hw = None
    if args.intrinsic_source_height and args.intrinsic_source_width:
        source_hw = (args.intrinsic_source_height, args.intrinsic_source_width)
    export_views(
        block_mapping=args.block_mapping,
        anchor_image=args.anchor_image,
        intrinsic_path=args.intrinsic_path,
        output_root=args.output_root,
        intrinsic_source_hw=source_hw,
    )


if __name__ == "__main__":
    main()
