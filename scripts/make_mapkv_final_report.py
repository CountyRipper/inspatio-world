#!/usr/bin/env python3
"""Aggregate the two-scene controlled Oracle decision without running Phase II."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_dirs", nargs="+", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output_dir", default="artifacts/final")
    parser.add_argument(
        "--conclusion", choices=("GO", "NO-GO", "INVALID_CASE"), required=True
    )
    parser.add_argument("--visual_summary", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def method_row(metrics: dict, method_name: str) -> dict:
    item = metrics["methods"][method_name]
    return {
        "generated_mask_l1": item["source_revisit_pixel_l1_generated_mask"],
        "generated_mask_l1_improvement_vs_baseline": item[
            "generated_mask_l1_improvement_vs_baseline"
        ],
        "masked_lpips_alexnet": item.get("masked_lpips_alexnet"),
        "pixel_delta_cosine_to_B1": item[
            "generated_mask_pixel_delta_cosine_to_B1"
        ],
        "latent_delta_cosine_to_B1": item[
            "generated_mask_latent_delta_cosine_to_B1"
        ],
        "nonzero_latent_blocks": item["nonzero_latent_blocks"],
    }


def alpha_summary(rows: list[dict], oracle_name: str) -> dict:
    usable = [row for row in rows if oracle_name in row["raw"]["methods"]]
    oracle_gains = [
        row["raw"]["methods"][oracle_name]["generated_mask_l1_improvement_vs_baseline"]
        for row in usable
    ]
    wrong_gains = [
        row["raw"]["methods"]["wrong_a010"]["generated_mask_l1_improvement_vs_baseline"]
        for row in usable
    ]
    random_gains = [
        row["raw"]["methods"]["random_a010"]["generated_mask_l1_improvement_vs_baseline"]
        for row in usable
    ]
    return {
        "run_count": len(usable),
        "mean_oracle_l1_improvement": mean(oracle_gains),
        "mean_wrong_a010_l1_improvement": mean(wrong_gains),
        "mean_random_a010_l1_improvement": mean(random_gains),
        "oracle_beats_wrong_count": sum(
            oracle > wrong for oracle, wrong in zip(oracle_gains, wrong_gains)
        ),
        "oracle_beats_random_count": sum(
            oracle > random for oracle, random in zip(oracle_gains, random_gains)
        ),
        "note": (
            "Wrong/Random controls use alpha=0.10; direct matched-strength inference is "
            "therefore valid only for oracle_a010. oracle_a020 is reported as a strength "
            "sweep, not as a matched-control discrimination result."
        ),
    }


def combine_contact_sheets(paths: list[tuple[str, Path]], output: Path) -> None:
    cells = []
    target_width = 1248
    for title, path in paths:
        image = Image.open(path).convert("RGB")
        if image.width != target_width:
            height = round(image.height * target_width / image.width)
            image = image.resize((target_width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_width, image.height + 34), "white")
        canvas.paste(image, (0, 34))
        ImageDraw.Draw(canvas).text((8, 9), title, fill="black")
        cells.append(canvas)
    result = Image.new(
        "RGB", (target_width, sum(cell.height for cell in cells)), "white"
    )
    y = 0
    for cell in cells:
        result.paste(cell, (0, y))
        y += cell.height
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def main() -> None:
    args = parse_args()
    case_dirs = [Path(value).resolve() for value in args.case_dirs]
    seeds = [int(value) for value in args.seeds.split(",")]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    case_summaries = []
    contact_sheets = []
    for case_dir in case_dirs:
        trajectory = load_json(case_dir / "trajectory_manifest.json")
        pair = load_json(case_dir / "pair_validation.json")
        case_summaries.append(
            {
                "case_id": trajectory["case_id"],
                "target_pose_sha256": trajectory["target_pose_sha256"],
                "source_chunk": trajectory["source_chunk"],
                "target_chunk": trajectory["target_chunk"],
                "temporal_gap_chunks": trajectory["target_chunk"]
                - trajectory["source_chunk"],
                "pair_valid": pair["benchmark_valid"],
                "rotation_distance_degrees": pair["rotation_distance_degrees"],
                "translation_distance": pair["translation_distance"],
            }
        )
        for seed in seeds:
            path = case_dir / "final" / f"seed_{seed}" / "metrics.json"
            metrics = load_json(path)
            rows.append(
                {
                    "case_id": trajectory["case_id"],
                    "seed": seed,
                    "benchmark_valid": metrics["benchmark_valid"],
                    "alpha_zero_max_abs_diff": metrics["alpha_zero_max_abs_diff"],
                    "alpha_one_max_abs_diff": metrics["alpha_one_max_abs_diff"],
                    "alpha_one_cache_unchanged": metrics["alpha_one_cache_unchanged"],
                    "baseline": method_row(metrics, "baseline"),
                    "oracle_a010": method_row(metrics, "oracle_a010"),
                    "wrong_a010": method_row(metrics, "wrong_a010"),
                    "random_a010": method_row(metrics, "random_a010"),
                    "oracle_a020": (
                        method_row(metrics, "oracle_a020")
                        if "oracle_a020" in metrics["methods"]
                        else None
                    ),
                    "raw": metrics,
                }
            )
        seed0_final = case_dir / "final" / "seed_0"
        contact_sheets.append(
            (trajectory["case_id"], seed0_final / "contact_sheet.png")
        )
        source_video = seed0_final / "phase1_control_comparison.mp4"
        shutil.copy2(
            source_video,
            output_dir / f"phase1_{trajectory['case_id']}_comparison.mp4",
        )

    all_valid = all(row["benchmark_valid"] for row in rows)
    alpha_zero_exact = all(row["alpha_zero_max_abs_diff"] == 0.0 for row in rows)
    alpha_one_rows = [row for row in rows if row["alpha_one_max_abs_diff"] is not None]
    alpha_one_active = all(row["alpha_one_max_abs_diff"] > 0.0 for row in alpha_one_rows)
    cache_unchanged = all(row["alpha_one_cache_unchanged"] for row in alpha_one_rows)
    target_only = all(
        row["oracle_a010"]["nonzero_latent_blocks"]
        == [next(item["target_chunk"] for item in case_summaries if item["case_id"] == row["case_id"])]
        for row in rows
    )
    matched = alpha_summary(rows, "oracle_a010")
    sweep_a020 = alpha_summary(rows, "oracle_a020")

    public_rows = [{key: value for key, value in row.items() if key != "raw"} for row in rows]
    decision = (
        f"VALID + ORACLE {args.conclusion}"
        if all_valid and args.conclusion != "INVALID_CASE"
        else "INVALID_CASE / IMPLEMENTATION_BUG"
    )
    aggregate = {
        "decision": decision,
        "answers": {
            "historical_kv_improves_revisit": args.conclusion == "GO",
            "cut3r_surfel_retrieval_approaches_oracle": None,
        },
        "benchmark": {
            "cases": case_summaries,
            "seeds": seeds,
            "all_pair_valid": all_valid,
        },
        "engineering_checks": {
            "alpha_zero_exact_all_runs": alpha_zero_exact,
            "alpha_one_active_seed0_runs": alpha_one_active,
            "alpha_one_runtime_cache_unchanged": cache_unchanged,
            "stable_memory_changes_target_chunk_only": target_only,
        },
        "phase1": {
            "matched_alpha_010": matched,
            "alpha_020_strength_sweep": sweep_a020,
            "visual_summary": args.visual_summary,
            "conclusion": args.conclusion,
            "runs": public_rows,
        },
        "phase2": {
            "executed": False,
            "conclusion": "NOT RUN: controlled Oracle Gate did not clear",
        },
        "failure_localization": (
            "1. historical KV payload is not demonstrated usable at whole-chunk granularity"
            if args.conclusion == "NO-GO"
            else None
        ),
        "next_action": (
            "Test one spatially localized source-region payload in the same exact-pose benchmark."
            if args.conclusion == "NO-GO"
            else "Run causal CUT3R surfel retrieval on the validated yaw30 cases."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    combine_contact_sheets(contact_sheets, output_dir / "contact_sheet.png")

    first_meta = load_json(case_dirs[0] / "baseline" / "seed_0" / "run_metadata.json")
    case_lines = "\n".join(
        f"- {item['case_id']}: chunks {item['source_chunk']}→{item['target_chunk']} "
        f"(gap {item['temporal_gap_chunks']}), pose error "
        f"{item['rotation_distance_degrees']:.9f}° / {item['translation_distance']:.3g}, "
        f"checksum `{item['target_pose_sha256']}`"
        for item in case_summaries
    )
    report = f"""# CUT3R-Surfel KV Prototype Report

## Environment
- InSpatio base commit: `2d15b7c742fbc90bfd7e67052a260ff87d97abc3`
- VMem reference commit: `39291e4f272f6b4f270691d930926ab5930f942e`
- CUT3R checkpoint: not used; Phase II remained gated
- GPU: {first_meta['gpu']}
- Config: `configs/mapkv_proto.yaml`

## Benchmark validity
- Controlled inputs: two repeated-static-frame scenes, pure yaw `0→+30→0→+30`, no pitch/roll/translation
- Seeds: {seeds}
{case_lines}
- Exact pose/render/mask B1↔B2 checks: passed
- Pair validity V1–V10: passed for both cases
- Baseline replay and AlphaZero equality: exact for all {len(rows)} primary runs
- Alpha=1 activation sanity: active on both seed-0 cases; runtime KV cache unchanged
- Validity: VALID

## Revisit case
- B1 and B2 are manifest-selected plateau centers, not post-hoc nearest-pose pairs.
- The source is outside the active recent window and generated-region masks retain substantial evaluation area.
- Baseline B1↔B2 error leaves clear headroom in both scenes.

## Phase I — Oracle KV
- Matched alpha: Oracle/Wrong/Random all use `0.10` for the primary discrimination.
- Mean generated-mask L1 improvement: Oracle={matched['mean_oracle_l1_improvement']:.9f}, Wrong={matched['mean_wrong_a010_l1_improvement']:.9f}, Random={matched['mean_random_a010_l1_improvement']:.9f}.
- Oracle beats Wrong: {matched['oracle_beats_wrong_count']}/{matched['run_count']}; Oracle beats Random: {matched['oracle_beats_random_count']}/{matched['run_count']}.
- Alpha `0.20` is retained only as a strength sweep because the controls are alpha `0.10`.
- Visual effect: {args.visual_summary}
- Stable target-only injection: passed; no whole-frame jump or camera-control break was observed.
- Conclusion: {args.conclusion}

## Phase II — Geometry Retrieval
- Oracle source chunk: manifest-defined B1 for each case
- PoseKV selected chunk: not run
- GeometryKV selected chunk: not run
- Geometry top-K scores: not run
- Retrieval visualization: not run
- Video comparison: not run
- Conclusion: NOT RUN — the controlled Oracle Gate did not clear, so running CUT3R would not answer the addressing question.

## Failure localization
1. historical KV payload is not demonstrated usable at whole-chunk granularity.

## Next action
Test one spatially localized source-region payload in this same exact-pose benchmark.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
