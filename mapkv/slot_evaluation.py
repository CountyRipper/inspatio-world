from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import _image, _mask, _masked_l1, evaluate


SLOT_GROUPS = {
    "recent": {
        "zero": "recentzero",
        "wrong": "recentwrong",
        "correct": "recentb1",
        "injection_mode": "replace_recent_delta",
    },
    "reference": {
        "zero": "refzero",
        "wrong": "refwrong",
        "correct": "refb1",
        "injection_mode": "replace_ref_delta",
    },
    "both": {
        "zero": None,
        "wrong": "bothwrong",
        "correct": "bothb1",
        "injection_mode": "replace_both_delta",
    },
}


def _method_image(root: Path, method: str, target_chunk: int) -> np.ndarray:
    method_root = root / "baseline" if method == "baseline" else root / "generation" / method
    return _image(method_root / "keyframes" / f"chunk_{target_chunk:04d}.png")


def _select_best_slot(groups: dict, ranking: list[tuple[float, float, str]]) -> str:
    """Prefer the simplest source-specific slot unless both is materially better.

    ``both`` is a diagnostic for distributed control, not an automatic winner when
    its Wrong intervention happens to be slightly more destructive. It should win
    only when it improves the correct B1 reconstruction over the best viable single
    slot by a visible normalized-L1 margin and also has a stronger aggregate score.
    """
    score_by_slot = {slot: score for score, _effect, slot in ranking}

    def source_specific(summary: dict) -> bool:
        return (
            float(summary["source_specificity_error_margin"]) > 0.002
            and float(summary["correct_improvement_vs_baseline"]) > 0.0
            and float(summary["correct_intervention_l1"]) > 0.002
        )

    viable_single = [
        slot
        for slot in ("recent", "reference")
        if slot in groups and source_specific(groups[slot])
    ]
    if not viable_single:
        if "both" in groups and source_specific(groups["both"]):
            return "both"
        return ranking[0][2]

    best_single = min(
        viable_single,
        key=lambda slot: (
            float(groups[slot]["correct_b1_b2_generated_region_l1"]),
            -score_by_slot[slot],
        ),
    )
    if "both" not in groups or not source_specific(groups["both"]):
        return best_single

    single_error = float(groups[best_single]["correct_b1_b2_generated_region_l1"])
    both_error = float(groups["both"]["correct_b1_b2_generated_region_l1"])
    score_gain = score_by_slot["both"] - score_by_slot[best_single]
    if both_error < single_error - 0.01 and score_gain > 0.01:
        return "both"
    return best_single


def evaluate_slots(
    *,
    run_root: str | Path,
    case_dir: str | Path,
    source_chunk: int,
    target_chunk: int,
    methods: list[str],
) -> dict:
    root = Path(run_root).resolve()
    result = evaluate(
        run_root=root,
        case_dir=case_dir,
        source_chunk=source_chunk,
        target_chunk=target_chunk,
        methods=methods,
    )
    generated = result["generation"]
    mask = _mask(
        root / "baseline" / "masks" / f"chunk_{target_chunk:04d}_generated_region.png",
        _method_image(root, "baseline", target_chunk).shape[:2],
    )
    groups = {}
    ranking = []
    for slot, names in SLOT_GROUPS.items():
        correct_name = names["correct"]
        wrong_name = names["wrong"]
        zero_name = names["zero"]
        if correct_name not in generated or wrong_name not in generated:
            continue
        correct = generated[correct_name]
        wrong = generated[wrong_name]
        correct_image = _method_image(root, correct_name, target_chunk)
        wrong_image = _method_image(root, wrong_name, target_chunk)
        zero = generated.get(zero_name) if zero_name else None
        summary = {
            "injection_mode": names["injection_mode"],
            "correct_method": correct_name,
            "wrong_method": wrong_name,
            "zero_method": zero_name,
            "correct_b1_b2_generated_region_l1": correct[
                "b1_b2_generated_region_l1"
            ],
            "wrong_b1_b2_generated_region_l1": wrong[
                "b1_b2_generated_region_l1"
            ],
            "zero_b1_b2_generated_region_l1": (
                None if zero is None else zero["b1_b2_generated_region_l1"]
            ),
            "correct_improvement_vs_baseline": correct[
                "b1_b2_generated_region_l1_improvement_vs_baseline"
            ],
            "correct_intervention_l1": correct[
                "b2_vs_baseline_generated_region_l1"
            ],
            "wrong_intervention_l1": wrong[
                "b2_vs_baseline_generated_region_l1"
            ],
            "zero_intervention_l1": (
                None if zero is None else zero["b2_vs_baseline_generated_region_l1"]
            ),
            "source_specificity_error_margin": (
                wrong["b1_b2_generated_region_l1"]
                - correct["b1_b2_generated_region_l1"]
            ),
            "correct_vs_wrong_generated_region_l1": _masked_l1(
                correct_image, wrong_image, mask
            ),
            "correct_boundary_l1": correct["target_boundary_l1"],
        }
        groups[slot] = summary
        ranking.append(
            (
                summary["source_specificity_error_margin"]
                + max(summary["correct_improvement_vs_baseline"], 0.0),
                summary["correct_intervention_l1"],
                slot,
            )
        )
    if not ranking:
        raise RuntimeError("No complete Zero/Wrong/Correct slot group was evaluated")
    ranking.sort(reverse=True)
    best_slot = _select_best_slot(groups, ranking)
    best = SLOT_GROUPS[best_slot]
    best_method = best["correct"]
    latent_hard = generated.get("latenthard")
    latent_soft = generated.get("latentsoft")
    best_values = generated[best_method]
    surfel_values = generated.get("surfelkv")
    best_image = _method_image(root, best_method, target_chunk)
    surfel_match = None
    if surfel_values is not None:
        surfel_image = _method_image(root, "surfelkv", target_chunk)
        surfel_match = {
            "best_manual_method": best_method,
            "generated_region_l1": _masked_l1(best_image, surfel_image, mask),
            "whole_frame_l1": float(np.abs(best_image - surfel_image).mean()),
            "target_latent_max_abs_diff_vs_baseline": surfel_values[
                "target_latent_max_abs_diff_vs_baseline"
            ],
        }
    result["slot_ablation"] = {
        "groups": groups,
        "ranking": [
            {"score": score, "intervention_l1": effect, "slot": slot}
            for score, effect, slot in ranking
        ],
        "most_influential_context_channel": best_slot,
        "best_slot_mode": best["injection_mode"],
        "best_manual_method": best_method,
        "best_kv_generated_region_l1": best_values["b1_b2_generated_region_l1"],
        "latent_hard_generated_region_l1": (
            None if latent_hard is None else latent_hard["b1_b2_generated_region_l1"]
        ),
        "latent_soft_generated_region_l1": (
            None if latent_soft is None else latent_soft["b1_b2_generated_region_l1"]
        ),
        "surfel_vs_best_manual": surfel_match,
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MapKV context-slot topology")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--source_chunk", type=int, required=True)
    parser.add_argument("--target_chunk", type=int, required=True)
    parser.add_argument("--methods", required=True)
    args = parser.parse_args()
    evaluate_slots(
        run_root=args.run_root,
        case_dir=args.case_dir,
        source_chunk=args.source_chunk,
        target_chunk=args.target_chunk,
        methods=[item for item in args.methods.split(",") if item],
    )
