#!/usr/bin/env python3
"""Finalize a truthful evidence bundle when Phase I is NO-GO."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.io import read_video, write_video

from mapkv_proto.metrics import compute_revisit_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", default="artifacts")
    parser.add_argument("--source_chunk", type=int, default=12)
    parser.add_argument("--target_chunk", type=int, default=19)
    parser.add_argument("--wrong_chunk", type=int, default=16)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def mapping_block(mapping: list[dict], chunk: int) -> dict:
    return next(block for block in mapping if int(block["chunk_id"]) == chunk)


def keyframe(run_root: Path, mapping: list[dict], chunk: int) -> Path:
    return run_root / mapping_block(mapping, chunk)["png_path"]


def chunk_for_frame(mapping: list[dict], frame_index: int) -> int:
    return int(
        min(
            mapping,
            key=lambda block: abs(int(block["rgb_center_index"]) - frame_index),
        )["chunk_id"]
    )


def phase(chunk: int, source: int, target: int) -> str:
    if chunk == source:
        return f"FIRST VISIT source={source}"
    if chunk == target:
        return f"REVISIT target={target}"
    if source < chunk < target:
        return "LEFT SOURCE / evicted from recent cache"
    if chunk < source:
        return "APPROACHING SOURCE"
    return "AFTER REVISIT"


def labeled_cell(
    image: Image.Image,
    title: str,
    phase_text: str,
    size: tuple[int, int] = (416, 240),
) -> np.ndarray:
    image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0], 39), fill="black")
    draw.text((6, 4), title, fill="white")
    draw.text((6, 22), phase_text, fill=(255, 222, 80))
    return np.asarray(image)


def make_phase1_video(
    *,
    methods: dict[str, tuple[Path, str]],
    baseline_root: Path,
    mapping: list[dict],
    output: Path,
    source_chunk: int,
    target_chunk: int,
) -> None:
    videos = {}
    fps_values = []
    for title, (run_root, _) in methods.items():
        frames, _, info = read_video(str(run_root / "pred.mp4"), pts_unit="sec")
        videos[title] = frames
        fps_values.append(float(info.get("video_fps", 24.0)))
    length = min(len(frames) for frames in videos.values())
    masks = {}
    output_frames = []
    for frame_index in range(length):
        chunk = chunk_for_frame(mapping, frame_index)
        phase_text = phase(chunk, source_chunk, target_chunk)
        cells = []
        for title, (_, retrieved) in methods.items():
            label = title if not retrieved else f"{title} | retrieved={retrieved}"
            cells.append(
                labeled_cell(
                    Image.fromarray(videos[title][frame_index].numpy()),
                    label,
                    phase_text,
                )
            )
        if chunk not in masks:
            masks[chunk] = Image.open(
                baseline_root / "masks" / f"chunk_{chunk:04d}_generated_region.png"
            ).convert("RGB")
        cells.append(labeled_cell(masks[chunk], "Generated-region gate", phase_text))
        top = np.concatenate(cells[:3], axis=1)
        bottom = np.concatenate(cells[3:6], axis=1)
        output_frames.append(np.concatenate([top, bottom], axis=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(output),
        torch.from_numpy(np.stack(output_frames)),
        fps=int(round(min(fps_values))),
        video_codec="h264",
        options={"crf": "18"},
    )


def save_contact_sheet(items: list[tuple[str, Path]], output: Path) -> None:
    cells = []
    for title, path in items:
        image = Image.open(path).convert("RGB").resize((416, 240), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (416, 264), "white")
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text((6, 244), title, fill="black")
        cells.append(canvas)
    sheet = Image.new("RGB", (3 * 416, 2 * 264), "white")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % 3) * 416, (index // 3) * 264))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def run_metrics(
    *,
    baseline_root: Path,
    run_roots: dict[str, Path],
    mapping: list[dict],
    source_chunk: int,
    target_chunk: int,
) -> dict:
    source = load_rgb(keyframe(baseline_root, mapping, source_chunk))
    mask = np.asarray(
        Image.open(
            baseline_root / "masks" / f"chunk_{target_chunk:04d}_generated_region.png"
        ).convert("L").resize((source.shape[1], source.shape[0]), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    baseline_latents = torch.load(
        baseline_root / "pred_latents.pt", map_location="cpu", weights_only=True
    ).float()
    metrics = {"methods": {}}
    for name, root in run_roots.items():
        target = load_rgb(keyframe(root, mapping, target_chunk))
        error = np.abs(target - source).mean(axis=2)
        previous = load_rgb(keyframe(root, mapping, target_chunk - 1))
        metadata = load_json(root / "run_metadata.json")
        method = {
            "source_revisit_pixel_l1_full": float(error.mean()),
            "source_revisit_pixel_l1_reference_blind_weighted": float(
                (error * mask).sum() / max(mask.sum(), 1e-8)
            ),
            "boundary_pixel_l1": float(np.abs(previous - target).mean()),
            "target_latency_seconds": float(
                metadata["timing_seconds"]["per_block"][str(target_chunk)]
            ),
            "activation_audit": metadata["mapkv"]["activation_audit"],
        }
        if name == "baseline":
            method["latent_max_abs_diff_from_baseline"] = 0.0
            method["nonzero_latent_blocks"] = []
        else:
            latents = torch.load(
                root / "pred_latents.pt", map_location="cpu", weights_only=True
            ).float()
            block_diffs = [
                float(
                    (
                        latents[:, start:start + 3]
                        - baseline_latents[:, start:start + 3]
                    ).abs().max()
                )
                for start in range(0, baseline_latents.shape[1], 3)
            ]
            method["latent_max_abs_diff_from_baseline"] = max(block_diffs)
            method["nonzero_latent_blocks"] = [
                index for index, value in enumerate(block_diffs) if value != 0.0
            ]
        metrics["methods"][name] = method
    metrics.update(
        compute_revisit_metrics(
            source_keyframe=keyframe(baseline_root, mapping, source_chunk),
            revisit_keyframes={
                name: keyframe(root, mapping, target_chunk)
                for name, root in run_roots.items()
            },
            boundary_pairs={
                name: (
                    keyframe(root, mapping, target_chunk - 1),
                    keyframe(root, mapping, target_chunk),
                )
                for name, root in run_roots.items()
            },
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    return metrics


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    baseline_root = artifact_root / "baseline"
    oracle_root = artifact_root / "oracle"
    final_root = artifact_root / "final"
    mapping_payload = load_json(baseline_root / "block_mapping.json")
    mapping = mapping_payload["blocks"]
    run_roots = {
        "baseline": baseline_root,
        "alpha_zero": oracle_root / "runs" / "alpha_zero",
        "wrong_a010": oracle_root / "runs" / "wrong_kv",
        "oracle_a005": oracle_root / "runs" / "oracle_a005",
        "oracle_a010": oracle_root / "runs" / "oracle_a010",
        "oracle_a020": oracle_root / "runs" / "oracle_a020",
    }
    methods = {
        "Baseline": (run_roots["baseline"], ""),
        "WrongKV a=.10": (run_roots["wrong_a010"], str(args.wrong_chunk)),
        "OracleKV a=.05": (run_roots["oracle_a005"], str(args.source_chunk)),
        "OracleKV a=.10": (run_roots["oracle_a010"], str(args.source_chunk)),
        "OracleKV a=.20": (run_roots["oracle_a020"], str(args.source_chunk)),
    }
    comparison = oracle_root / "comparison.mp4"
    make_phase1_video(
        methods=methods,
        baseline_root=baseline_root,
        mapping=mapping,
        output=comparison,
        source_chunk=args.source_chunk,
        target_chunk=args.target_chunk,
    )
    final_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(comparison, final_root / "phase1_baseline_vs_wrong_vs_oracle.mp4")
    save_contact_sheet(
        [
            (f"First visit source {args.source_chunk}", keyframe(baseline_root, mapping, args.source_chunk)),
            (f"Baseline revisit {args.target_chunk}", keyframe(baseline_root, mapping, args.target_chunk)),
            (f"WrongKV {args.wrong_chunk} a=.10", keyframe(run_roots["wrong_a010"], mapping, args.target_chunk)),
            ("OracleKV a=.05", keyframe(run_roots["oracle_a005"], mapping, args.target_chunk)),
            ("OracleKV a=.10", keyframe(run_roots["oracle_a010"], mapping, args.target_chunk)),
            ("OracleKV a=.20", keyframe(run_roots["oracle_a020"], mapping, args.target_chunk)),
        ],
        final_root / "contact_sheet.png",
    )
    metrics = run_metrics(
        baseline_root=baseline_root,
        run_roots=run_roots,
        mapping=mapping,
        source_chunk=args.source_chunk,
        target_chunk=args.target_chunk,
    )
    baseline_meta = load_json(baseline_root / "run_metadata.json")
    alpha_zero_meta = load_json(run_roots["alpha_zero"] / "run_metadata.json")
    legacy_meta = load_json(
        baseline_root / "upstream_attention_check" / "run_metadata.json"
    )
    metrics.update(
        {
            "phase1_conclusion": "NO-GO",
            "phase2_executed": False,
            "phase2_stop_reason": "OracleKV did not clearly work; Situation A requires stopping GeometryKV generation.",
            "source_chunk": args.source_chunk,
            "target_chunk": args.target_chunk,
            "wrong_chunk": args.wrong_chunk,
            "baseline_repeat_max_abs_diff": baseline_meta["replay"]["in_process_memory_off"]["max_abs_diff"],
            "legacy_attention_max_abs_diff": legacy_meta["replay"]["against_saved_latents"]["max_abs_diff"],
            "alpha_zero_max_abs_diff": alpha_zero_meta["replay"]["against_saved_latents"]["max_abs_diff"],
        }
    )
    metrics_text = json.dumps(metrics, indent=2)
    (oracle_root / "metrics.json").write_text(metrics_text, encoding="utf-8")
    (final_root / "metrics.json").write_text(metrics_text, encoding="utf-8")
    geometry_root = artifact_root / "geometry"
    geometry_root.mkdir(parents=True, exist_ok=True)
    (geometry_root / "SKIPPED.md").write_text(
        "# Phase II not executed\n\n"
        "Phase I was a NO-GO: stable OracleKV injection did not clearly restore "
        "the first-visit appearance. Per Situation A in the experiment protocol, "
        "CUT3R/PoseKV/GeometryKV generation was stopped instead of obscuring the "
        "payload result with a geometry experiment. This is not a negative result "
        "for CUT3R retrieval.\n",
        encoding="utf-8",
    )

    source_block = mapping_block(mapping, args.source_chunk)
    target_block = mapping_block(mapping, args.target_chunk)
    stable = metrics["methods"]
    if metrics["lpips_available"]:
        revisit_lpips = metrics["revisit_lpips"]
        boundary_lpips = metrics["boundary_lpips"]
        lpips_summary = (
            f" LPIPS was baseline={revisit_lpips['baseline']:.9f}, "
            f"WrongKV={revisit_lpips['wrong_a010']:.9f}, "
            f"Oracle .05/.10/.20={revisit_lpips['oracle_a005']:.9f}/"
            f"{revisit_lpips['oracle_a010']:.9f}/{revisit_lpips['oracle_a020']:.9f}."
        )
        boundary_lpips_summary = (
            f" Boundary LPIPS was baseline={boundary_lpips['baseline']:.9f} and "
            f"Oracle .05/.10/.20={boundary_lpips['oracle_a005']:.9f}/"
            f"{boundary_lpips['oracle_a010']:.9f}/{boundary_lpips['oracle_a020']:.9f}."
        )
    else:
        lpips_summary = " LPIPS was unavailable in the runtime environment."
        boundary_lpips_summary = " Boundary LPIPS was unavailable in the runtime environment."
    report = f"""# CUT3R-Surfel KV Prototype Report

## Environment
- InSpatio base commit: 2d15b7c742fbc90bfd7e67052a260ff87d97abc3
- Prototype run commit: {baseline_meta['git_commit']}
- VMem commit: 39291e4f272f6b4f270691d930926ab5930f942e (pinned reference; Phase II not executed)
- CUT3R checkpoint: not used because Phase I was NO-GO
- GPU: {baseline_meta['gpu']}
- Config: configs/mapkv_proto.yaml

## Revisit case
- Source chunk: {args.source_chunk}
- Target chunk: {args.target_chunk}
- Temporal gap: {args.target_chunk - args.source_chunk}
- Reference-blind fraction: source={1.0 - source_block['reference_valid_fraction']:.6f}, target={1.0 - target_block['reference_valid_fraction']:.6f}
- Why this is a generated-region revisit: the circular trajectory returns to the same cup/saucer/tray region after seven chunks; both lossless keyframes contain reference-invalid cup/saucer, tray-edge, and image-boundary pixels.

## Phase I — Oracle KV
- Baseline vs AlphaZero equality: legacy-attention vs memory-off max_abs_diff={metrics['legacy_attention_max_abs_diff']}; memory-off repeat={metrics['baseline_repeat_max_abs_diff']}; AlphaZero vs baseline={metrics['alpha_zero_max_abs_diff']}.
- Correct Oracle visual effect: alpha 0.05–0.20 produced small, target-local changes but no clearly visible restoration of source-region identity. Blind-weighted source L1 was baseline={stable['baseline']['source_revisit_pixel_l1_reference_blind_weighted']:.9f}, Oracle .05={stable['oracle_a005']['source_revisit_pixel_l1_reference_blind_weighted']:.9f}, .10={stable['oracle_a010']['source_revisit_pixel_l1_reference_blind_weighted']:.9f}, .20={stable['oracle_a020']['source_revisit_pixel_l1_reference_blind_weighted']:.9f}; the tiny/non-monotonic deltas are not evidence of recovery.{lpips_summary}
- WrongKV visual effect: WrongKV also produced only a small change and did not improve identity; blind-weighted source L1={stable['wrong_a010']['source_revisit_pixel_l1_reference_blind_weighted']:.9f}.
- Best alpha/layer/step: no successful alpha; tested 0.05/0.10/0.20 on layers 26–29 at step index 3. Alpha 1.0 was diagnostic only and moved target latent max_abs_diff to 0.98046875 without moving the image toward the source.
- Activation discontinuity: no obvious scene cut or camera failure. Boundary L1 stayed near baseline {stable['baseline']['boundary_pixel_l1']:.9f} (Oracle .05/.10/.20: {stable['oracle_a005']['boundary_pixel_l1']:.9f}/{stable['oracle_a010']['boundary_pixel_l1']:.9f}/{stable['oracle_a020']['boundary_pixel_l1']:.9f}).{boundary_lpips_summary}
- Conclusion: NO-GO

## Phase II — Geometry Retrieval
- Oracle source chunk: not applicable because Oracle payload test was NO-GO
- PoseKV selected chunk: not run
- GeometryKV selected chunk: not run
- Geometry top-K scores: not run
- Retrieval visualization: not generated
- Video comparison: Phase-I-only comparison at ../oracle/comparison.mp4
- Conclusion: NO-GO (protocol stop; this is not a negative measurement of CUT3R retrieval)

## Failure localization
1. historical KV payload is not usable.

## Next action
Test a spatially aligned latent-tile Oracle KV payload on this fixed revisit before spending another generation run on geometry addressing.
"""
    (final_root / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
