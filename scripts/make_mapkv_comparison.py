#!/usr/bin/env python3
"""Create the fixed MapKV evidence bundle after all four replays finish."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.io import read_video, write_video

from mapkv_proto.metrics import compute_revisit_metrics, save_metrics
from mapkv_proto.visualization import make_contact_sheet


FAILURES = {
    "1": "historical KV payload is not usable",
    "2": "injection is unstable",
    "3": "CUT3R geometry/alignment is wrong",
    "4": "surfel retrieval is wrong",
    "5": "retrieval is right but global/chunk-level KV is too coarse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_run", required=True)
    parser.add_argument("--pose_run", required=True)
    parser.add_argument("--geometry_run", required=True)
    parser.add_argument("--oracle_run", required=True)
    parser.add_argument("--pose_plan", required=True)
    parser.add_argument("--geometry_plan", required=True)
    parser.add_argument("--source_chunk", type=int, required=True)
    parser.add_argument("--target_chunk", type=int, required=True)
    parser.add_argument("--wrong_chunk", type=int, required=True)
    parser.add_argument("--output_root", default="artifacts/final")
    parser.add_argument("--phase1_conclusion", choices=("GO", "NO-GO"), required=True)
    parser.add_argument("--phase2_conclusion", choices=("GO", "NO-GO"), required=True)
    parser.add_argument("--primary_failure", choices=tuple(FAILURES), required=True)
    parser.add_argument("--next_action", required=True)
    parser.add_argument("--why_revisit", required=True)
    parser.add_argument("--oracle_effect", required=True)
    parser.add_argument("--wrong_effect", required=True)
    parser.add_argument("--activation_discontinuity", required=True)
    return parser.parse_args()


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan_entry(path: str | Path, target_chunk: int) -> dict:
    payload = load_json(path)
    entries = payload.get("targets", payload) if isinstance(payload, dict) else payload
    for entry in entries:
        if int(entry["target_chunk"]) == target_chunk:
            return entry
    raise KeyError(f"Target chunk {target_chunk} absent from {path}")


def selected(entry: dict) -> int | None:
    values = entry.get("selected_chunks", [])
    return int(values[0]) if values else None


def load_run(path: str | Path) -> dict:
    root = Path(path).resolve()
    mapping_payload = load_json(root / "block_mapping.json")
    return {
        "root": root,
        "video": root / "pred.mp4",
        "metadata": load_json(root / "run_metadata.json"),
        "mapping": mapping_payload["blocks"],
        "latent_length": int(mapping_payload["latent_length"]),
        "rgb_length": int(mapping_payload["rgb_length"]),
    }


def chunk_for_rgb(run: dict, frame_index: int) -> int:
    return int(
        min(
            run["mapping"],
            key=lambda block: abs(int(block["rgb_center_index"]) - frame_index),
        )["chunk_id"]
    )


def phase_label(chunk: int, source: int, target: int) -> str:
    if chunk == source:
        return f"FIRST VISIT — source {source}"
    if chunk == target:
        return f"REVISIT — target {target}"
    if source < chunk < target:
        return "LEFT SOURCE / recent cache eviction"
    if chunk < source:
        return "APPROACHING SOURCE"
    return "AFTER REVISIT"


def labeled_cell(
    image: Image.Image,
    label: str,
    phase: str,
    *,
    cell_size: tuple[int, int] = (416, 240),
) -> np.ndarray:
    image = image.convert("RGB").resize(cell_size, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, cell_size[0], 38), fill="black")
    draw.text((6, 4), label, fill="white")
    draw.text((6, 21), phase, fill=(255, 224, 96))
    return np.asarray(image)


def make_video(
    *,
    runs: dict[str, dict],
    baseline: dict,
    output: Path,
    source_chunk: int,
    target_chunk: int,
    retrieval_ids: dict[str, int | None],
) -> None:
    decoded = {}
    fps_values = []
    for name, run in runs.items():
        frames, _, info = read_video(str(run["video"]), pts_unit="sec")
        decoded[name] = frames
        fps_values.append(float(info.get("video_fps", 24.0)))
    length = min(frames.shape[0] for frames in decoded.values())
    rows = []
    for frame_index in range(length):
        chunk = chunk_for_rgb(baseline, frame_index)
        phase = phase_label(chunk, source_chunk, target_chunk)
        method_cells = []
        for name in ("Baseline", "PoseKV", "GeometryKV", "OracleKV"):
            suffix = ""
            if name != "Baseline":
                suffix = f" | retrieved={retrieval_ids.get(name)}"
            frame = Image.fromarray(decoded[name][frame_index].numpy())
            method_cells.append(labeled_cell(frame, name + suffix, phase))

        valid_path = (
            baseline["root"] / "masks" / f"chunk_{chunk:04d}_reference_valid.png"
        )
        generated_path = (
            baseline["root"] / "masks" / f"chunk_{chunk:04d}_generated_region.png"
        )
        mask_cells = [
            labeled_cell(Image.open(valid_path).convert("RGB"), "Reference valid mask", phase),
            labeled_cell(Image.open(generated_path).convert("RGB"), "Generated-region mask", phase),
        ]
        top = np.concatenate(method_cells[:3], axis=1)
        bottom = np.concatenate([method_cells[3], *mask_cells], axis=1)
        rows.append(np.concatenate([top, bottom], axis=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(output),
        torch.from_numpy(np.stack(rows)),
        fps=min(fps_values),
        video_codec="h264",
        options={"crf": "18"},
    )


def keyframe(run: dict, chunk: int) -> Path:
    block = next(item for item in run["mapping"] if int(item["chunk_id"]) == chunk)
    return run["root"] / block["png_path"]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = load_run(args.baseline_run)
    pose = load_run(args.pose_run)
    geometry = load_run(args.geometry_run)
    oracle = load_run(args.oracle_run)
    runs = {
        "Baseline": baseline,
        "PoseKV": pose,
        "GeometryKV": geometry,
        "OracleKV": oracle,
    }
    pose_entry = plan_entry(args.pose_plan, args.target_chunk)
    geometry_entry = plan_entry(args.geometry_plan, args.target_chunk)
    oracle_source = int(
        oracle["metadata"]["mapkv"]["selections"][0]["source_chunk"]
    )
    retrieval_ids = {
        "PoseKV": selected(pose_entry),
        "GeometryKV": selected(geometry_entry),
        "OracleKV": oracle_source,
    }
    comparison_path = output_root / "baseline_vs_pose_vs_geometry_vs_oracle.mp4"
    make_video(
        runs=runs,
        baseline=baseline,
        output=comparison_path,
        source_chunk=args.source_chunk,
        target_chunk=args.target_chunk,
        retrieval_ids=retrieval_ids,
    )

    contact_images = {
        f"First visit source {args.source_chunk}": keyframe(baseline, args.source_chunk),
        "Baseline revisit": keyframe(baseline, args.target_chunk),
        f"PoseKV revisit ({retrieval_ids['PoseKV']})": keyframe(pose, args.target_chunk),
        f"GeometryKV revisit ({retrieval_ids['GeometryKV']})": keyframe(
            geometry, args.target_chunk
        ),
        f"OracleKV revisit ({oracle_source})": keyframe(oracle, args.target_chunk),
    }
    make_contact_sheet(contact_images, output_root / "contact_sheet.png")

    boundary_pairs = {}
    latencies = {}
    for name, run in runs.items():
        if args.target_chunk > 0:
            boundary_pairs[name] = (
                keyframe(run, args.target_chunk - 1),
                keyframe(run, args.target_chunk),
            )
        latencies[name] = run["metadata"]["timing_seconds"]["per_block"]
    metrics = compute_revisit_metrics(
        source_keyframe=keyframe(baseline, args.source_chunk),
        revisit_keyframes={
            name: keyframe(run, args.target_chunk) for name, run in runs.items()
        },
        boundary_pairs=boundary_pairs,
        block_latencies=latencies,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    metrics.update(
        {
            "source_chunk": args.source_chunk,
            "target_chunk": args.target_chunk,
            "wrong_chunk": args.wrong_chunk,
            "pose_plan_entry": pose_entry,
            "geometry_plan_entry": geometry_entry,
            "oracle_source_chunk": oracle_source,
            "artifact_paths": {
                "comparison": str(comparison_path),
                "contact_sheet": str(output_root / "contact_sheet.png"),
            },
        }
    )
    save_metrics(metrics, output_root / "metrics.json")

    baseline_meta = baseline["metadata"]
    target_block = next(
        block for block in baseline["mapping"]
        if int(block["chunk_id"]) == args.target_chunk
    )
    source_block = next(
        block for block in baseline["mapping"]
        if int(block["chunk_id"]) == args.source_chunk
    )
    replay_info = baseline_meta["replay"]["in_process_memory_off"]
    geometry_scores = geometry_entry.get("scores", {})
    report = f"""# CUT3R-Surfel KV Prototype Report

## Environment
- InSpatio commit: {baseline_meta['git_commit']}
- VMem commit: 39291e4f272f6b4f270691d930926ab5930f942e
- CUT3R checkpoint: cut3r_512_dpt_4_64.pth (see geometry/surfel_metadata.json)
- GPU: {baseline_meta['gpu']}
- Config: configs/mapkv_proto.yaml

## Revisit case
- Source chunk: {args.source_chunk}
- Target chunk: {args.target_chunk}
- Temporal gap: {args.target_chunk - args.source_chunk}
- Reference-blind fraction: source={1.0 - source_block['reference_valid_fraction']:.4f}, target={1.0 - target_block['reference_valid_fraction']:.4f}
- Why this is a generated-region revisit: {args.why_revisit}

## Phase I — Oracle KV
- Baseline vs AlphaZero equality: baseline repeat max_abs_diff={replay_info['max_abs_diff'] if replay_info else 'not run'}; inspect AlphaZero run_metadata.json for its saved-baseline comparison.
- Correct Oracle visual effect: {args.oracle_effect}
- WrongKV visual effect: {args.wrong_effect}
- Best alpha/layer/step: alpha={oracle['metadata']['mapkv']['alpha']}, layers={oracle['metadata']['mapkv']['selected_layers_resolved']}, steps={oracle['metadata']['mapkv']['selected_steps_resolved']}
- Activation discontinuity: {args.activation_discontinuity}
- Conclusion: {args.phase1_conclusion}

## Phase II — Geometry Retrieval
- Oracle source chunk: {oracle_source}
- PoseKV selected chunk: {retrieval_ids['PoseKV']}
- GeometryKV selected chunk: {retrieval_ids['GeometryKV']}
- Geometry top-K scores: {json.dumps(geometry_scores, sort_keys=True)}
- Retrieval visualization: ../geometry/retrieval_visualization/target_{args.target_chunk:04d}_overlay.png
- Video comparison: baseline_vs_pose_vs_geometry_vs_oracle.mp4
- Conclusion: {args.phase2_conclusion}

## Failure localization
{args.primary_failure}. {FAILURES[args.primary_failure]}.

## Next action
{args.next_action}
"""
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")

    oracle_comparison = output_root.parent / "oracle" / "comparison.mp4"
    oracle_comparison.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(comparison_path, oracle_comparison)


if __name__ == "__main__":
    main()
