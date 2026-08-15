from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .pose_utils import pose_distance, tcw_to_c2w


def monotonic_index(source_index: int, source_length: int, target_length: int) -> int:
    if source_length <= 1 or target_length <= 1:
        return 0
    return int(round(source_index * (target_length - 1) / (source_length - 1)))


def _save_rgb_tensor(frame: torch.Tensor, path: Path) -> None:
    frame = frame.detach().float().clamp(0, 1).cpu()
    if frame.ndim != 3:
        raise ValueError(f"Expected [C,H,W] RGB frame, got {tuple(frame.shape)}")
    array = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def build_block_mapping(
    *,
    pred_video: torch.Tensor,
    latent_length: int,
    mask_latent: torch.Tensor,
    target_tcw: torch.Tensor | np.ndarray,
    frames_per_block: int,
    output_root: str | Path,
) -> list[dict]:
    """Save lossless center keyframes and actual latent/RGB/pose monotonic mapping."""
    output_root = Path(output_root)
    keyframe_root = output_root / "keyframes"
    keyframe_root.mkdir(parents=True, exist_ok=True)
    if pred_video.ndim != 5:
        raise ValueError("pred_video must be [B,T,C,H,W]")
    if latent_length % frames_per_block:
        raise ValueError("latent_length must be divisible by frames_per_block")
    rgb_length = pred_video.shape[1]
    poses_tensor = torch.as_tensor(target_tcw).cpu()
    while poses_tensor.ndim > 3 and poses_tensor.shape[0] == 1:
        poses_tensor = poses_tensor[0]
    if poses_tensor.ndim == 2:
        poses_tensor = poses_tensor.unsqueeze(0)
    poses = np.asarray(poses_tensor, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"target_tcw must resolve to [T,4,4], got {poses.shape}")
    mapping = []
    for chunk_id in range(latent_length // frames_per_block):
        latent_indices = list(
            range(chunk_id * frames_per_block, (chunk_id + 1) * frames_per_block)
        )
        center_latent = latent_indices[len(latent_indices) // 2]
        rgb_index = monotonic_index(center_latent, latent_length, rgb_length)
        pose_index = monotonic_index(rgb_index, rgb_length, len(poses))
        png_path = keyframe_root / f"chunk_{chunk_id:04d}.png"
        _save_rgb_tensor(pred_video[0, rgb_index], png_path)
        mask01 = ((mask_latent[:, latent_indices].float() + 1.0) * 0.5).clamp(0, 1)
        tcw = poses[pose_index]
        mapping.append(
            {
                "chunk_id": chunk_id,
                "latent_indices": latent_indices,
                "rgb_center_index": rgb_index,
                "pose_index": pose_index,
                "png_path": str(png_path.relative_to(output_root)),
                "Tcw": tcw.tolist(),
                "c2w": tcw_to_c2w(tcw).tolist(),
                "reference_valid_fraction": float(mask01.mean().item()),
            }
        )
    payload = {
        "latent_length": latent_length,
        "rgb_length": rgb_length,
        "pose_length": len(poses),
        "frames_per_block": frames_per_block,
        "mapping_rule": "round(index * (target_length - 1) / max(source_length - 1, 1))",
        "blocks": mapping,
    }
    (output_root / "block_mapping.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return mapping


def load_blocks(path: str | Path) -> tuple[list[dict], Path]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks = payload["blocks"] if isinstance(payload, dict) else payload
    return blocks, path.parent


def rank_revisit_pairs(
    blocks: list[dict],
    *,
    minimum_gap: int = 3,
    minimum_invalid_fraction: float = 0.10,
    exclude_source_chunk_zero: bool = True,
) -> list[dict]:
    candidates = []
    for source in blocks:
        if exclude_source_chunk_zero and source["chunk_id"] == 0:
            continue
        source_invalid = 1.0 - float(source["reference_valid_fraction"])
        if source_invalid < minimum_invalid_fraction:
            continue
        for target in blocks:
            gap = int(target["chunk_id"]) - int(source["chunk_id"])
            if gap < minimum_gap:
                continue
            target_invalid = 1.0 - float(target["reference_valid_fraction"])
            if target_invalid < minimum_invalid_fraction:
                continue
            distance, translation, rotation = pose_distance(
                np.asarray(source["c2w"]), np.asarray(target["c2w"])
            )
            candidates.append(
                {
                    "source_chunk": int(source["chunk_id"]),
                    "target_chunk": int(target["chunk_id"]),
                    "temporal_gap": gap,
                    "pose_distance": distance,
                    "translation_distance": translation,
                    "rotation_distance_radians": rotation,
                    "source_invalid_fraction": source_invalid,
                    "target_invalid_fraction": target_invalid,
                }
            )
    return sorted(candidates, key=lambda item: item["pose_distance"])


def choose_wrong_chunk(blocks: list[dict], *, source_chunk: int, target_chunk: int) -> int:
    by_id = {int(block["chunk_id"]): block for block in blocks}
    source = by_id[source_chunk]
    target = by_id[target_chunk]
    candidates = [
        block for block in blocks
        if int(block["chunk_id"]) < target_chunk - 1 and int(block["chunk_id"]) != source_chunk
    ]
    if not candidates:
        raise ValueError("No causally valid wrong-chunk candidate")
    def score(block):
        return (
            pose_distance(np.asarray(block["c2w"]), np.asarray(source["c2w"]))[0]
            + pose_distance(np.asarray(block["c2w"]), np.asarray(target["c2w"]))[0]
        )
    return int(max(candidates, key=score)["chunk_id"])


def save_candidate_contact_sheet(
    candidates: list[dict],
    blocks: list[dict],
    *,
    mapping_root: Path,
    output_path: str | Path,
    limit: int = 5,
    thumbnail_size: tuple[int, int] = (416, 240),
) -> None:
    by_id = {int(block["chunk_id"]): block for block in blocks}
    rows = []
    for candidate in candidates[:limit]:
        source = by_id[candidate["source_chunk"]]
        target = by_id[candidate["target_chunk"]]
        images = []
        for label, block in (("source", source), ("revisit", target)):
            image = Image.open(mapping_root / block["png_path"]).convert("RGB")
            image.thumbnail(thumbnail_size)
            canvas = Image.new("RGB", thumbnail_size, "black")
            canvas.paste(image, ((thumbnail_size[0] - image.width) // 2, 0))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, thumbnail_size[1] - 24, thumbnail_size[0], thumbnail_size[1]), fill="black")
            draw.text(
                (6, thumbnail_size[1] - 20),
                f"{label} chunk {block['chunk_id']}  valid={block['reference_valid_fraction']:.2f}",
                fill="white",
            )
            images.append(canvas)
        row = Image.new("RGB", (2 * thumbnail_size[0], thumbnail_size[1] + 22), "white")
        row.paste(images[0], (0, 0))
        row.paste(images[1], (thumbnail_size[0], 0))
        ImageDraw.Draw(row).text(
            (6, thumbnail_size[1] + 3),
            f"gap={candidate['temporal_gap']} pose={candidate['pose_distance']:.4f}",
            fill="black",
        )
        rows.append(row)
    if not rows:
        raise ValueError("No revisit candidates passed the filters")
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank generated-region revisit chunk pairs")
    parser.add_argument("--block_mapping", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--contact_sheet", required=True)
    parser.add_argument("--minimum_gap", type=int, default=3)
    parser.add_argument("--minimum_invalid_fraction", type=float, default=0.10)
    args = parser.parse_args()
    blocks, root = load_blocks(args.block_mapping)
    candidates = rank_revisit_pairs(
        blocks,
        minimum_gap=args.minimum_gap,
        minimum_invalid_fraction=args.minimum_invalid_fraction,
    )
    top_candidates = candidates[:5]
    for candidate in top_candidates:
        candidate["suggested_wrong_chunk"] = choose_wrong_chunk(
            blocks,
            source_chunk=int(candidate["source_chunk"]),
            target_chunk=int(candidate["target_chunk"]),
        )
    Path(args.output_json).write_text(
        json.dumps(top_candidates, indent=2), encoding="utf-8"
    )
    save_candidate_contact_sheet(candidates, blocks, mapping_root=root, output_path=args.contact_sheet)


if __name__ == "__main__":
    main()
