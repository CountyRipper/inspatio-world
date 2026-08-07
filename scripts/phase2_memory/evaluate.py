#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from phase1_lsm.data_prep import sha256_file
from phase2_memory.data import load_scene_geometry, prepare_trajectory
from phase2_memory.manifest import load_manifest
from phase2_memory.rollout import ROLLOUT_VARIANTS, run_online_rollout
from phase2_memory.trajectory import block_keyframes
from scripts.phase2_memory.common import (
    DEFAULT_CHECKPOINT,
    cpu,
    finish_distributed,
    init_distributed,
    json_dump,
    load_pipeline,
)
from scripts.render_point_cloud import open_ffmpeg_writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/phase2_memory_manifest.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", choices=("train", "heldout"))
    parser.add_argument("--group-id", action="append")
    parser.add_argument("--phase2a-only", action="store_true")
    parser.add_argument(
        "--condition",
        action="append",
        choices=ROLLOUT_VARIANTS,
    )
    parser.add_argument("--anchoring-strength", type=float, default=0.0)
    return parser.parse_args()


def write_video(frames: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[-2:]
    writer = open_ffmpeg_writer(str(path), width, height, 24)
    try:
        array = (
            frames.permute(0, 2, 3, 1).numpy().clip(0, 1) * 255
        ).round().astype(np.uint8)
        for frame in array:
            writer.stdin.write(frame.tobytes())
    finally:
        writer.stdin.close()
        writer.wait()
    if writer.returncode:
        raise RuntimeError(f"ffmpeg failed for {path}")


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    occupancy: torch.Tensor,
) -> float:
    selected = occupancy.bool().expand_as(prediction)
    return float((prediction.float() - target.float()).abs()[selected].mean())


def masked_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    occupancy: torch.Tensor,
) -> float:
    values = F.cosine_similarity(
        prediction.float(), target.float(), dim=2, eps=1e-8
    )
    return float(values[occupancy[:, :, 0].bool()].mean())


def make_montage(
    path: Path,
    selected_frames: dict[str, torch.Tensor],
    conditions: tuple[str, ...],
    labels: list[str],
) -> None:
    cell_w, cell_h = 208, 120
    title_h, header_h = 38, 34
    canvas = Image.new(
        "RGB",
        (cell_w * len(conditions), title_h + header_h + cell_h * len(labels)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 7),
        "Full-context decode; rows are three keyframes at every return",
        fill="black",
    )
    for column, condition in enumerate(conditions):
        draw.text((column * cell_w + 5, title_h + 6), condition, fill="black")
        for row, label in enumerate(labels):
            frame = selected_frames[condition][row]
            array = (
                frame.permute(1, 2, 0).numpy().clip(0, 1) * 255
            ).round().astype(np.uint8)
            image = Image.fromarray(array).resize(
                (cell_w, cell_h), Image.Resampling.LANCZOS
            )
            canvas.paste(
                image,
                (column * cell_w, title_h + header_h + row * cell_h),
            )
            if column == 0:
                draw.text(
                    (5, title_h + header_h + row * cell_h + 5),
                    label,
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )
    canvas.save(path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 <= args.anchoring_strength <= 1.0:
        raise ValueError("anchoring strength must be in [0,1]")
    conditions = tuple(
        args.condition
        if args.condition is not None
        else ("no_memory", "correct", "wrong")
    )
    if "correct" not in conditions or "no_memory" not in conditions or "wrong" not in conditions:
        raise ValueError("evaluation requires no_memory, correct, and wrong")
    root = Path(args.output_root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    manifest = load_manifest(args.manifest)
    groups = manifest.select(
        split=args.split,
        group_ids=None if args.group_id is None else set(args.group_id),
        phase2a_only=args.phase2a_only,
    )
    if not groups:
        raise ValueError("no evaluation groups selected")

    device = init_distributed()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = Path(args.checkpoint).resolve()
    adapter = Path(args.adapter).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    adapter_hash = sha256_file(adapter)
    pipeline = load_pipeline(
        Path(args.repo_root).resolve(), checkpoint, adapter, device
    )
    scene_cache = {}
    latent_cache = {}
    aggregate = []
    started = time.perf_counter()

    for group in groups:
        tick = time.perf_counter()
        group_root = root / "groups" / group.group_id
        group_root.mkdir(parents=True)
        if group.scene_id not in scene_cache:
            scene_cache[group.scene_id] = load_scene_geometry(
                manifest.scenes[group.scene_id]
            )
        geometry = scene_cache[group.scene_id]
        cache = latent_cache.get(group.scene_id)
        prepared = prepare_trajectory(
            pipeline,
            geometry,
            group,
            group_root / "trajectory.txt",
            device,
            cached_ref_latent=None if cache is None else cache["ref_latent"],
            cached_conditional=None if cache is None else cache["conditional"],
        )
        latent_cache[group.scene_id] = {
            "ref_latent": prepared.ref_latent,
            "conditional": prepared.conditional,
        }
        results = {
            condition: run_online_rollout(
                pipeline,
                prepared,
                variant=condition,
                anchoring_strength=(
                    args.anchoring_strength
                    if condition in ("correct", "wrong")
                    else 0.0
                ),
                capture_records=False,
            )
            for condition in conditions
        }
        reference = {
            item.block: item for item in results["correct"].observations
        }
        no_memory_output = results["no_memory"].output
        rows = []
        for block, target_observation in reference.items():
            start = block * 3
            target = target_observation.correct_target
            occupancy = target_observation.correct_occupancy
            invalid = (~occupancy.bool()).expand_as(target)
            for condition in conditions:
                prediction = results[condition].output[:, start:start + 3]
                spill = (
                    float(
                        (
                            prediction.float()
                            - no_memory_output[:, start:start + 3].float()
                        ).abs()[invalid].mean()
                    )
                    if invalid.any()
                    else 0.0
                )
                rows.append({
                    "block": block,
                    "memory_id": target_observation.memory_id,
                    "memory_version": target_observation.retrieved_versions[0],
                    "condition": condition,
                    "overlap_l1": masked_l1(prediction, target, occupancy),
                    "overlap_feature_cosine": masked_cosine(
                        prediction, target, occupancy
                    ),
                    "invalid_spill_l1": spill,
                    "occupancy_fraction": target_observation.occupancy_fraction,
                })
        by_key = {
            (row["block"], row["condition"]): row for row in rows
        }
        advantages = [{
            "block": block,
            "memory_id": reference[block].memory_id,
            "correct_vs_no_memory_l1_gain": (
                by_key[(block, "no_memory")]["overlap_l1"]
                - by_key[(block, "correct")]["overlap_l1"]
            ),
            "wrong_minus_correct_l1": (
                by_key[(block, "wrong")]["overlap_l1"]
                - by_key[(block, "correct")]["overlap_l1"]
            ),
        } for block in sorted(reference)]

        save_file(
            {
                condition: cpu(results[condition].output, torch.bfloat16)
                for condition in conditions
            },
            group_root / "outputs.safetensors",
        )
        frame_indices = [
            int(frame)
            for station in group.return_stations
            for frame in block_keyframes(station.block)
        ]
        frame_labels = [
            f"{station.memory_id}v{1 if station.action == 'return_write' else 2}"
            f"/b{station.block}/k{slot}"
            for station in group.return_stations
            for slot in range(3)
        ]
        selected_frames = {}
        for condition in conditions:
            pixels = pipeline.vae.decode_to_pixel(
                results[condition].output, use_cache=False
            )
            pixels = (pixels[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
            if pixels.shape[0] != 237:
                raise AssertionError(f"decoded frame count: {pixels.shape}")
            write_video(pixels, group_root / "videos" / f"{condition}.mp4")
            selected_frames[condition] = pixels[frame_indices]
            pipeline.vae.model.clear_cache()
            del pixels
            torch.cuda.empty_cache()
        make_montage(
            group_root / "montage.png",
            selected_frames,
            conditions,
            frame_labels,
        )
        metrics = {
            "group_id": group.group_id,
            "scene_id": group.scene_id,
            "split": group.split,
            "family": group.family,
            "conditions": conditions,
            "anchoring_strength": args.anchoring_strength,
            "rows": rows,
            "advantages": advantages,
            "retrieval_correct_all_returns": all(
                item.retrieved_ids[0] == item.memory_id
                for result in results.values()
                for item in result.observations
            ),
            "all_projections_non_identity": all(
                item.projection_non_identity
                for result in results.values()
                for item in result.observations
            ),
            "memory_writeback_reread": all(
                item.retrieved_versions[0] == 2
                for item in results["correct"].observations[-2:]
            ),
            "seconds": time.perf_counter() - tick,
        }
        json_dump(group_root / "metrics.json", metrics)
        aggregate.append(metrics)
        print(json.dumps({
            "group": group.group_id,
            "advantages": advantages,
            "seconds": metrics["seconds"],
        }), flush=True)
        del prepared, results, selected_frames
        gc.collect()
        torch.cuda.empty_cache()

    gains = [
        row["correct_vs_no_memory_l1_gain"]
        for group in aggregate for row in group["advantages"]
    ]
    wrong_gaps = [
        row["wrong_minus_correct_l1"]
        for group in aggregate for row in group["advantages"]
    ]
    correct_rows = [
        row for group in aggregate for row in group["rows"]
        if row["condition"] == "correct"
    ]
    no_memory_rows = [
        row for group in aggregate for row in group["rows"]
        if row["condition"] == "no_memory"
    ]
    wrong_rows = [
        row for group in aggregate for row in group["rows"]
        if row["condition"] == "wrong"
    ]

    def breakdown(selected: list[dict[str, object]]) -> dict[str, float | int]:
        selected_gains = [
            row["correct_vs_no_memory_l1_gain"]
            for group in selected for row in group["advantages"]
        ]
        selected_wrong_gaps = [
            row["wrong_minus_correct_l1"]
            for group in selected for row in group["advantages"]
        ]
        return {
            "group_count": len(selected),
            "return_count": len(selected_gains),
            "mean_correct_vs_no_memory_l1_gain": float(np.mean(selected_gains)),
            "correct_beats_no_memory_rate": float(
                np.mean([value > 0 for value in selected_gains])
            ),
            "mean_wrong_minus_correct_l1": float(np.mean(selected_wrong_gaps)),
            "correct_beats_wrong_rate": float(
                np.mean([value > 0 for value in selected_wrong_gaps])
            ),
        }

    summary = {
        "manifest": str(manifest.path),
        "heldout_unit": manifest.heldout_unit,
        "group_count": len(groups),
        "return_count": len(gains),
        "conditions": conditions,
        "anchoring_strength": args.anchoring_strength,
        "mean_no_memory_overlap_l1": float(np.mean([
            row["overlap_l1"] for row in no_memory_rows
        ])),
        "mean_correct_overlap_l1": float(np.mean([
            row["overlap_l1"] for row in correct_rows
        ])),
        "mean_wrong_overlap_l1": float(np.mean([
            row["overlap_l1"] for row in wrong_rows
        ])),
        "mean_correct_overlap_feature_cosine": float(np.mean([
            row["overlap_feature_cosine"] for row in correct_rows
        ])),
        "mean_correct_invalid_spill_l1": float(np.mean([
            row["invalid_spill_l1"] for row in correct_rows
        ])),
        "mean_occupancy_fraction": float(np.mean([
            row["occupancy_fraction"] for row in correct_rows
        ])),
        "mean_correct_vs_no_memory_l1_gain": float(np.mean(gains)),
        "correct_beats_no_memory_rate": float(np.mean([value > 0 for value in gains])),
        "mean_wrong_minus_correct_l1": float(np.mean(wrong_gaps)),
        "correct_beats_wrong_rate": float(np.mean([value > 0 for value in wrong_gaps])),
        "by_family": {
            family: breakdown([
                group for group in aggregate if group["family"] == family
            ])
            for family in sorted({group["family"] for group in aggregate})
        },
        "by_scene": {
            scene_id: breakdown([
                group for group in aggregate if group["scene_id"] == scene_id
            ])
            for scene_id in sorted({group["scene_id"] for group in aggregate})
        },
        "retrieval_correct_all_groups": all(
            group["retrieval_correct_all_returns"] for group in aggregate
        ),
        "projection_non_identity_all_groups": all(
            group["all_projections_non_identity"] for group in aggregate
        ),
        "writeback_reread_all_groups": all(
            group["memory_writeback_reread"] for group in aggregate
        ),
        "checkpoint_sha256": checkpoint_hash,
        "adapter_sha256": adapter_hash,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "seconds": time.perf_counter() - started,
    }
    json_dump(root / "aggregate_metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if sha256_file(checkpoint) != checkpoint_hash or sha256_file(adapter) != adapter_hash:
        raise AssertionError("immutable checkpoint or adapter changed")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    finish_distributed()


if __name__ == "__main__":
    main()
