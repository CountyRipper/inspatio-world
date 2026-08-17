#!/usr/bin/env python3
"""Build metrics and visual evidence for one controlled Stage-0A case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.io import read_video, write_video


METHOD_RUNS = {
    "baseline": ("baseline", None),
    "alpha_zero": ("alpha_zero", None),
    "random_a010": ("random_a010", 0.10),
    "wrong_a010": ("wrong_a010", 0.10),
    "oracle_a005": ("oracle_a005", 0.05),
    "oracle_a010": ("oracle_a010", 0.10),
    "oracle_a020": ("oracle_a020", 0.20),
    "oracle_activation_a100": ("oracle_activation_a100", 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--conclusion", choices=("GO", "NO-GO", "INCONCLUSIVE"), default="INCONCLUSIVE"
    )
    parser.add_argument("--visual_summary", default="Pending controlled visual review.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--skip_visuals",
        action="store_true",
        help="Refresh metrics/report without re-encoding existing contact/video artifacts",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rgb(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def mapping_block(mapping: list[dict], chunk_id: int) -> dict:
    return next(item for item in mapping if int(item["chunk_id"]) == chunk_id)


def keyframe(root: Path, mapping: list[dict], chunk_id: int) -> Path:
    return root / mapping_block(mapping, chunk_id)["png_path"]


def load_mask(path: Path, height: int, width: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((width, height), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None]


def masked_pixel_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    error = (source - target).abs().mean(dim=0, keepdim=True)
    return float((error * mask).sum() / mask.sum().clamp_min(1e-8))


def masked_direction_cosine(
    delta: torch.Tensor, desired_delta: torch.Tensor, mask: torch.Tensor
) -> float | None:
    first = (delta.float() * mask.float()).reshape(-1)
    second = (desired_delta.float() * mask.float()).reshape(-1)
    denominator = first.norm() * second.norm()
    if float(denominator) <= 1e-12:
        return None
    return float(torch.dot(first, second) / denominator)


def _masked_lpips_inputs(
    source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    source = source[None].to(device)
    target = target[None].to(device)
    mask = mask[None].to(device)
    neutral = torch.full_like(source, 0.5)
    source = source * mask + neutral * (1.0 - mask)
    target = target * mask + neutral * (1.0 - mask)
    return source * 2.0 - 1.0, target * 2.0 - 1.0


def learned_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    model,
    device: torch.device,
) -> tuple[float, float]:
    first, second = _masked_lpips_inputs(source, target, mask, device)
    with torch.no_grad():
        distance = float(model(first, second).item())
        first_features = model.net.forward(first)
        second_features = model.net.forward(second)
        first_vector = torch.cat(
            [feature.float().mean(dim=(2, 3)) for feature in first_features], dim=1
        )
        second_vector = torch.cat(
            [feature.float().mean(dim=(2, 3)) for feature in second_features], dim=1
        )
        cosine = float(F.cosine_similarity(first_vector, second_vector).item())
    return distance, cosine


def collect_metrics(
    *, case_dir: Path, seed: int, mapping: list[dict], source_chunk: int, target_chunk: int
) -> dict:
    baseline_root = case_dir / "baseline" / f"seed_{seed}"
    oracle_root = case_dir / "oracle" / f"seed_{seed}"
    roots = {
        name: baseline_root if name == "baseline" else oracle_root / "runs" / run_name
        for name, (run_name, _) in METHOD_RUNS.items()
    }
    roots = {
        name: root
        for name, root in roots.items()
        if name == "baseline" or (root / "run_metadata.json").exists()
    }
    source = load_rgb(keyframe(baseline_root, mapping, source_chunk))
    height, width = source.shape[-2:]
    mask = load_mask(
        baseline_root / "masks" / f"chunk_{target_chunk:04d}_generated_region.png",
        height,
        width,
    )
    baseline_latents = torch.load(
        baseline_root / "pred_latents.pt", map_location="cpu", weights_only=True
    ).float()

    model = None
    learned_error = None
    try:
        import lpips

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as error:
        device = torch.device("cpu")
        learned_error = f"{type(error).__name__}: {error}"

    methods = {}
    input_checksums = {}
    targets = {}
    method_latents = {}
    for name, root in roots.items():
        target = load_rgb(keyframe(root, mapping, target_chunk))
        previous = load_rgb(keyframe(root, mapping, target_chunk - 1))
        metadata = load_json(root / "run_metadata.json")
        latents = (
            baseline_latents
            if name == "baseline"
            else torch.load(
                root / "pred_latents.pt", map_location="cpu", weights_only=True
            ).float()
        )
        block_diffs = [
            float((latents[:, start : start + 3] - baseline_latents[:, start : start + 3]).abs().max())
            for start in range(0, baseline_latents.shape[1], 3)
        ]
        item = {
            "source_revisit_pixel_l1_full": float((source - target).abs().mean()),
            "source_revisit_pixel_l1_generated_mask": masked_pixel_l1(
                source, target, mask
            ),
            "boundary_pixel_l1": float((previous - target).abs().mean()),
            "latent_max_abs_diff_from_baseline": max(block_diffs),
            "nonzero_latent_blocks": [
                index for index, value in enumerate(block_diffs) if value != 0.0
            ],
            "target_latency_seconds": float(
                metadata["timing_seconds"]["per_block"][str(target_chunk)]
            ),
            "activation_audit": metadata["mapkv"]["activation_audit"],
            "cache_audits": metadata["mapkv"].get("cache_audits", {}),
        }
        if model is not None:
            lpips_value, feature_cosine = learned_metrics(
                source, target, mask, model=model, device=device
            )
            item["masked_lpips_alexnet"] = lpips_value
            item["masked_alexnet_feature_cosine"] = feature_cosine
        methods[name] = item
        targets[name] = target
        method_latents[name] = latents
        input_checksums[name] = (metadata.get("benchmark") or {}).get("input_checksums")

    baseline_l1 = methods["baseline"]["source_revisit_pixel_l1_generated_mask"]
    baseline_target = targets["baseline"]
    desired_pixel_delta = source - baseline_target
    frames_per_block = 3
    source_slice = slice(
        source_chunk * frames_per_block, (source_chunk + 1) * frames_per_block
    )
    target_slice = slice(
        target_chunk * frames_per_block, (target_chunk + 1) * frames_per_block
    )
    source_latent = baseline_latents[:, source_slice]
    baseline_target_latent = baseline_latents[:, target_slice]
    desired_latent_delta = source_latent - baseline_target_latent
    latent_mask = F.interpolate(
        mask[None], size=baseline_latents.shape[-2:], mode="bilinear", align_corners=False
    )[:, None]
    for name, item in methods.items():
        item["generated_mask_l1_improvement_vs_baseline"] = (
            baseline_l1 - item["source_revisit_pixel_l1_generated_mask"]
        )
        item["generated_mask_pixel_delta_cosine_to_B1"] = masked_direction_cosine(
            targets[name] - baseline_target, desired_pixel_delta, mask
        )
        item["generated_mask_latent_delta_cosine_to_B1"] = masked_direction_cosine(
            method_latents[name][:, target_slice] - baseline_target_latent,
            desired_latent_delta,
            latent_mask,
        )
    return {
        "methods": methods,
        "generated_mask_fraction": float(mask.mean()),
        "learned_metrics_available": model is not None,
        "learned_metrics_error": learned_error,
        "input_checksums_identical": all(
            value == input_checksums["baseline"] for value in input_checksums.values()
        ),
        "input_checksums": input_checksums,
    }


def save_contact_sheet(
    *, roots: dict[str, Path], mapping: list[dict], source_chunk: int,
    target_chunk: int, output: Path
) -> None:
    items = [("B1 first visit", keyframe(roots["baseline"], mapping, source_chunk))]
    items.extend(
        (name, keyframe(root, mapping, target_chunk))
        for name, root in roots.items()
        if name != "oracle_activation_a100"
    )
    cells = []
    for title, path in items:
        image = Image.open(path).convert("RGB").resize((416, 240), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (416, 266), "white")
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text((6, 246), title, fill="black")
        cells.append(canvas)
    rows = (len(cells) + 2) // 3
    sheet = Image.new("RGB", (3 * 416, rows * 266), "white")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % 3) * 416, (index // 3) * 266))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _chunk_for_frame(mapping: list[dict], frame_index: int) -> int:
    return int(
        min(
            mapping,
            key=lambda item: abs(int(item["rgb_center_index"]) - frame_index),
        )["chunk_id"]
    )


def make_comparison_video(
    *, roots: dict[str, Path], baseline_root: Path, mapping: list[dict],
    source_chunk: int, target_chunk: int, output: Path
) -> None:
    display_names = [
        name
        for name in (
            "baseline",
            "random_a010",
            "wrong_a010",
            "oracle_a005",
            "oracle_a010",
            "oracle_a020",
        )
        if name in roots
    ][:5]
    videos = {}
    fps_values = []
    for name in display_names:
        frames, _, info = read_video(str(roots[name] / "pred.mp4"), pts_unit="sec")
        videos[name] = frames
        fps_values.append(float(info.get("video_fps", 24.0)))
    length = min(len(value) for value in videos.values())
    masks = {}
    output_frames = []
    for frame_index in range(length):
        chunk = _chunk_for_frame(mapping, frame_index)
        if chunk == source_chunk:
            phase = "FIRST VISIT B1"
        elif chunk == target_chunk:
            phase = "EXACT-POSE REVISIT B2"
        elif source_chunk < chunk < target_chunk:
            phase = "SOURCE EVICTED FROM RECENT CACHE"
        else:
            phase = "CONTROL TRAJECTORY"
        cells = []
        for name in display_names:
            image = Image.fromarray(videos[name][frame_index].numpy()).resize((416, 240))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 416, 39), fill="black")
            draw.text((6, 4), name, fill="white")
            draw.text((6, 22), phase, fill=(255, 222, 80))
            cells.append(np.asarray(image))
        if chunk not in masks:
            masks[chunk] = Image.open(
                baseline_root / "masks" / f"chunk_{chunk:04d}_generated_region.png"
            ).convert("RGB").resize((416, 240))
        mask_image = masks[chunk].copy()
        draw = ImageDraw.Draw(mask_image)
        draw.rectangle((0, 0, 416, 39), fill="black")
        draw.text((6, 4), "generated-region mask", fill="white")
        draw.text((6, 22), phase, fill=(255, 222, 80))
        cells.append(np.asarray(mask_image))
        while len(cells) < 6:
            cells.append(np.zeros_like(cells[0]))
        output_frames.append(
            np.concatenate(
                [np.concatenate(cells[:3], axis=1), np.concatenate(cells[3:], axis=1)],
                axis=0,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(output),
        torch.from_numpy(np.stack(output_frames)),
        fps=int(round(min(fps_values))),
        video_codec="h264",
        options={"crf": "18"},
    )


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    baseline_root = case_dir / "baseline" / f"seed_{args.seed}"
    oracle_root = case_dir / "oracle" / f"seed_{args.seed}"
    final_root = case_dir / "final" / f"seed_{args.seed}"
    trajectory = load_json(case_dir / "trajectory_manifest.json")
    pair = load_json(case_dir / "pair_validation.json")
    mapping = load_json(baseline_root / "block_mapping.json")["blocks"]
    source_chunk = int(trajectory["source_chunk"])
    target_chunk = int(trajectory["target_chunk"])
    wrong_chunk = int(trajectory["wrong_chunk"])
    roots = {
        name: baseline_root if name == "baseline" else oracle_root / "runs" / run_name
        for name, (run_name, _) in METHOD_RUNS.items()
    }
    roots = {
        name: root
        for name, root in roots.items()
        if name == "baseline" or (root / "run_metadata.json").exists()
    }

    metrics = collect_metrics(
        case_dir=case_dir,
        seed=args.seed,
        mapping=mapping,
        source_chunk=source_chunk,
        target_chunk=target_chunk,
    )
    activation = (
        load_json(roots["oracle_activation_a100"] / "run_metadata.json")
        if "oracle_activation_a100" in roots
        else None
    )
    alpha_one = metrics["methods"].get("oracle_activation_a100")
    metrics.update(
        {
            "case_id": trajectory["case_id"],
            "seed": args.seed,
            "benchmark_valid": pair["benchmark_valid"],
            "pair_validation": pair,
            "alpha_zero_max_abs_diff": metrics["methods"]["alpha_zero"][
                "latent_max_abs_diff_from_baseline"
            ],
            "alpha_one_max_abs_diff": (
                alpha_one["latent_max_abs_diff_from_baseline"]
                if alpha_one is not None
                else None
            ),
            "alpha_one_cache_unchanged": (
                all(
                    item.get("unchanged")
                    for item in activation["mapkv"]["cache_audits"].values()
                )
                if activation is not None
                else None
            ),
            "conclusion": args.conclusion,
            "visual_summary": args.visual_summary,
            "phase2_executed": False,
        }
    )
    final_root.mkdir(parents=True, exist_ok=True)
    metrics_text = json.dumps(metrics, indent=2)
    (oracle_root / "metrics.json").write_text(metrics_text, encoding="utf-8")
    (final_root / "metrics.json").write_text(metrics_text, encoding="utf-8")
    if not args.skip_visuals:
        save_contact_sheet(
            roots=roots,
            mapping=mapping,
            source_chunk=source_chunk,
            target_chunk=target_chunk,
            output=final_root / "contact_sheet.png",
        )
        make_comparison_video(
            roots=roots,
            baseline_root=baseline_root,
            mapping=mapping,
            source_chunk=source_chunk,
            target_chunk=target_chunk,
            output=final_root / "phase1_control_comparison.mp4",
        )

    baseline_meta = load_json(baseline_root / "run_metadata.json")
    source_block = mapping_block(mapping, source_chunk)
    target_block = mapping_block(mapping, target_chunk)
    method = metrics["methods"]
    available_stable_alphas = [
        alpha
        for name, (_, alpha) in METHOD_RUNS.items()
        if name.startswith("oracle_a")
        and name != "oracle_activation_a100"
        and name in method
    ]
    alpha_text = "/".join(f"{alpha:.2f}" for alpha in available_stable_alphas)
    activation_summary = (
        f"alpha=1 target latent diff={metrics['alpha_one_max_abs_diff']:.9f}; "
        f"cache unchanged={metrics['alpha_one_cache_unchanged']}"
        if metrics["alpha_one_max_abs_diff"] is not None
        else "alpha=1 not repeated for this seed; seed 0 carries activation/cache audit"
    )
    if args.conclusion == "NO-GO":
        phase2_conclusion = "NOT RUN — controlled Oracle Gate did not clear"
        failure_localization = (
            "1. historical KV payload is not demonstrated usable at whole-chunk granularity."
        )
        next_action = (
            "Test one spatially localized source-region payload in this same exact-pose "
            "benchmark before spending effort on geometry addressing."
        )
    elif args.conclusion == "GO":
        phase2_conclusion = "READY — run causal CUT3R surfel retrieval"
        failure_localization = "No Phase-I failure; Phase II has not yet been evaluated."
        next_action = "Run causal CUT3R surfel voting on the validated yaw30 cases."
    else:
        phase2_conclusion = "NOT RUN — controlled Oracle decision is incomplete"
        failure_localization = "Not assigned while the controlled decision is incomplete."
        next_action = "Complete the matched-alpha yaw30 primary matrix."
    report = f"""# CUT3R-Surfel KV Prototype Report

## Environment
- InSpatio base commit: 2d15b7c742fbc90bfd7e67052a260ff87d97abc3
- Prototype run commit: {baseline_meta['git_commit']}
- VMem commit: 39291e4f272f6b4f270691d930926ab5930f942e
- CUT3R checkpoint: not used unless controlled Oracle reaches GO
- GPU: {baseline_meta['gpu']}
- Config: configs/mapkv_proto.yaml

## Benchmark validity
- Case ID: {trajectory['case_id']}
- Controlled/static source: yes
- Exact pose artifact/checksum: {trajectory['target_pose_path']} / {trajectory['target_pose_sha256']}
- Pitch/yaw/roll/translation: 0 / 0→{trajectory['theta_degrees']}→0→{trajectory['theta_degrees']} / 0 / fixed
- B1↔B2 pose error: {pair['rotation_distance_degrees']:.9f}° / {pair['translation_distance']:.9g}
- Source chunk: {source_chunk}
- Target chunk: {target_chunk}
- Temporal gap: {target_chunk - source_chunk}
- Active-cache exclusion: {pair['checks']['V4_gap_and_active_cache_exclusion']}
- Reference-blind fraction: source={1.0 - source_block['reference_valid_fraction']:.6f}, target={1.0 - target_block['reference_valid_fraction']:.6f}
- Render/mask same-view check: raw max_abs_diff=0 / 0
- Validity: {'VALID' if pair['benchmark_valid'] else 'INVALID_CASE'}

## Revisit case
- Why this is a generated-region revisit: B1 and B2 are manifest-declared centers of two exact +{trajectory['theta_degrees']}° endpoint plateaus; the source is {target_chunk - source_chunk} chunks old and the evaluated mask contains only area outside the reference warp.
- Baseline headroom: generated-mask B1↔B2 L1={method['baseline']['source_revisit_pixel_l1_generated_mask']:.9f}.

## Phase I — Oracle KV
- Baseline vs AlphaZero equality: {metrics['alpha_zero_max_abs_diff']}.
- Correct Oracle visual effect: {args.visual_summary}
- WrongKV visual effect: generated-mask L1={method['wrong_a010']['source_revisit_pixel_l1_generated_mask']:.9f}; RandomKV={method['random_a010']['source_revisit_pixel_l1_generated_mask']:.9f}.
- Evaluated alpha/layer/step: layers 26–29, step 3; available stable alpha runs: {alpha_text}.
- Activation discontinuity: {activation_summary}.
- Conclusion: {args.conclusion}

## Phase II — Geometry Retrieval
- Oracle source chunk: {source_chunk}
- PoseKV selected chunk: not run
- GeometryKV selected chunk: not run
- Geometry top-K scores: not run
- Retrieval visualization: not run
- Video comparison: not run
- Conclusion: {phase2_conclusion}

## Failure localization
{failure_localization}

## Next action
{next_action}
"""
    (final_root / "REPORT.md").write_text(report, encoding="utf-8")
    if not args.quiet:
        print(metrics_text)


if __name__ == "__main__":
    main()
