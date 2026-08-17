from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .pose_utils import pose_distance, tcw_to_c2w
from .trajectory_builder import sha256_file


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


def _masked_l1(
    source_image: Path, target_image: Path, generated_mask: Path
) -> tuple[float, float]:
    source = np.asarray(Image.open(source_image).convert("RGB"), dtype=np.float32) / 255.0
    target = np.asarray(Image.open(target_image).convert("RGB"), dtype=np.float32) / 255.0
    mask_image = Image.open(generated_mask).convert("L").resize(
        (source.shape[1], source.shape[0]), Image.Resampling.NEAREST
    )
    mask = np.asarray(mask_image, dtype=np.float32) / 255.0
    weighted = np.abs(source - target).mean(axis=2) * mask
    return float(weighted.sum() / max(mask.sum(), 1.0)), float(mask.mean())


def _phase_contains(phase_payload: dict, name: str, chunk_id: int) -> bool:
    phase = next(item for item in phase_payload["phases"] if item["name"] == name)
    return int(phase["start_block"]) <= chunk_id < int(phase["stop_block_exclusive"])


def _save_control_contact_sheet(
    *, blocks: dict[int, dict], root: Path, case_dir: Path, output_path: Path
) -> None:
    phase = json.loads((case_dir / "phase_labels.json").read_text(encoding="utf-8"))
    ids = [
        ("B1 source", int(phase["source_chunk"])),
        ("B2 baseline", int(phase["target_chunk"])),
        ("Wrong source", int(phase["wrong_chunk"])),
    ]
    panels = []
    for label, chunk_id in ids:
        block = blocks[chunk_id]
        image = Image.open(root / block["png_path"]).convert("RGB").resize((416, 240))
        mask_path = root / "masks" / f"chunk_{chunk_id:04d}_generated_region.png"
        mask = Image.open(mask_path).convert("L").resize(image.size)
        overlay = Image.new("RGB", image.size, (255, 0, 0))
        image = Image.composite(overlay, image, mask.point(lambda value: value // 3))
        canvas = Image.new("RGB", (416, 276), "black")
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 246),
            (
                f"{label}: chunk {chunk_id}  "
                f"blind={1.0 - float(block['reference_valid_fraction']):.3f}"
            ),
            fill="white",
        )
        panels.append(canvas)
    sheet = Image.new("RGB", (sum(panel.width for panel in panels), 276), "black")
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def validate_control_pair(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Validate a manifest-declared control pair")
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--alpha_zero_root", required=True)
    parser.add_argument("--b1_quality_pass", action="store_true")
    parser.add_argument("--headroom_pass", action="store_true")
    parser.add_argument("--output_json")
    parser.add_argument("--contact_sheet")
    args = parser.parse_args(argv)

    case_dir = Path(args.case_dir).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    alpha_zero_root = Path(args.alpha_zero_root).resolve()
    trajectory = json.loads(
        (case_dir / "trajectory_manifest.json").read_text(encoding="utf-8")
    )
    phases = json.loads((case_dir / "phase_labels.json").read_text(encoding="utf-8"))
    pose_validation = json.loads(
        (case_dir / "pose_validation.json").read_text(encoding="utf-8")
    )
    render_validation = json.loads(
        (case_dir / "render_revisit_diff.json").read_text(encoding="utf-8")
    )
    mapping_payload = json.loads(
        (baseline_root / "block_mapping.json").read_text(encoding="utf-8")
    )
    blocks = {int(item["chunk_id"]): item for item in mapping_payload["blocks"]}
    source_chunk = int(phases["source_chunk"])
    target_chunk = int(phases["target_chunk"])
    wrong_chunk = int(phases["wrong_chunk"])
    source = blocks[source_chunk]
    target = blocks[target_chunk]
    _, translation_distance, rotation_radians = pose_distance(
        np.asarray(source["c2w"]), np.asarray(target["c2w"])
    )
    rotation_degrees = float(np.degrees(rotation_radians))

    generated_mask = (
        baseline_root / "masks" / f"chunk_{target_chunk:04d}_generated_region.png"
    )
    masked_l1, target_blind_from_png = _masked_l1(
        baseline_root / source["png_path"],
        baseline_root / target["png_path"],
        generated_mask,
    )
    source_blind = 1.0 - float(source["reference_valid_fraction"])
    target_blind = 1.0 - float(target["reference_valid_fraction"])

    baseline_metadata = json.loads(
        (baseline_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    alpha_zero_metadata = json.loads(
        (alpha_zero_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    baseline_benchmark = baseline_metadata.get("benchmark") or {}
    alpha_zero_benchmark = alpha_zero_metadata.get("benchmark") or {}
    alpha_zero_replay = (
        alpha_zero_metadata.get("replay", {}).get("against_saved_latents") or {}
    )
    bank_metadata = json.loads(
        (baseline_root / "kv_bank" / "metadata.json").read_text(encoding="utf-8")
    )

    bank_checks = []
    layer_shapes = {}
    layer_checksums = {}
    for chunk_id in (source_chunk, wrong_chunk):
        chunk = bank_metadata["chunks"].get(str(chunk_id))
        if chunk is None:
            bank_checks.append(False)
            continue
        bank_checks.append(chunk.get("rope_layout") == "recent_slot_t3_t5")
        for layer, info in chunk["layers"].items():
            layer_path = baseline_root / "kv_bank" / info["path"]
            actual_checksum = sha256_file(layer_path)
            bank_checks.append(actual_checksum == info.get("sha256"))
            layer_shapes.setdefault(layer, []).append(info["shape"])
            layer_checksums[f"{chunk_id}:{layer}"] = actual_checksum
    same_shapes = all(
        len(shapes) == 2 and shapes[0] == shapes[1] for shapes in layer_shapes.values()
    )
    bank_checks.append(same_shapes)

    exact_inputs_equal = (
        baseline_benchmark.get("input_checksums")
        == alpha_zero_benchmark.get("input_checksums")
        and baseline_benchmark.get("target_pose_sha256")
        == trajectory.get("target_pose_sha256")
        == alpha_zero_benchmark.get("target_pose_sha256")
    )
    checks = {
        "V1_rotation_distance_le_0_25_degrees": rotation_degrees <= 0.25,
        "V2_translation_distance_le_1e_4": translation_distance <= 1e-4,
        "V3_both_inside_endpoint_plateaus": (
            _phase_contains(phases, "B1_hold", source_chunk)
            and _phase_contains(phases, "B2_hold", target_chunk)
        ),
        "V4_gap_and_active_cache_exclusion": (
            target_chunk - source_chunk >= 4 and source_chunk < target_chunk - 1
        ),
        "V5_source_not_initial_reference_only": source_chunk > 0,
        "V6_reference_blind_evaluable": (
            0.10 <= source_blind <= 0.70 and 0.10 <= target_blind <= 0.70
        ),
        "V7_B1_quality_and_camera_control": bool(args.b1_quality_pass),
        "V8_baseline_headroom": bool(args.headroom_pass and masked_l1 > 0.005),
        "V9_counterfactual_inputs_identical": bool(
            exact_inputs_equal
            and pose_validation.get("valid")
            and render_validation.get("same_view_pass")
            and alpha_zero_replay.get("max_abs_diff") == 0.0
        ),
        "V10_KV_ids_checksums_shapes_layers_rope": all(bank_checks),
    }
    benchmark_valid = all(checks.values())
    payload = {
        "benchmark_valid": benchmark_valid,
        "decision": "VALID" if benchmark_valid else "INVALID_CASE",
        "checks": checks,
        "source_chunk": source_chunk,
        "target_chunk": target_chunk,
        "wrong_chunk": wrong_chunk,
        "temporal_gap_chunks": target_chunk - source_chunk,
        "rotation_distance_degrees": rotation_degrees,
        "translation_distance": translation_distance,
        "source_reference_blind_fraction": source_blind,
        "target_reference_blind_fraction": target_blind,
        "target_mask_png_fraction": target_blind_from_png,
        "baseline_masked_source_target_l1": masked_l1,
        "alpha_zero_max_abs_diff": alpha_zero_replay.get("max_abs_diff"),
        "target_pose_sha256": trajectory.get("target_pose_sha256"),
        "input_checksums": baseline_benchmark.get("input_checksums"),
        "kv_layer_shapes": layer_shapes,
        "kv_layer_checksums": layer_checksums,
        "rope_layout": bank_metadata.get("rope_layout"),
    }
    output_json = Path(args.output_json or case_dir / "pair_validation.json")
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    contact_sheet = Path(args.contact_sheet or case_dir / "pair_contact_sheet.png")
    _save_control_contact_sheet(
        blocks=blocks,
        root=baseline_root,
        case_dir=case_dir,
        output_path=contact_sheet,
    )
    print(json.dumps(payload, indent=2))
    if not benchmark_valid:
        raise SystemExit(2)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_control_pair(sys.argv[2:])
        return
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
