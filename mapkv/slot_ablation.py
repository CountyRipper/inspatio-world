from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image

from .fast_pipeline import (
    _base_inference_command,
    _json,
    _link,
    _run,
    _sha256,
    _transcode_videos,
)
from .slot_evaluation import evaluate_slots
from .surfel_visualization import visualize_existing


STAGES = {"capture", "generation", "report", "full"}
ALL_METHODS = [
    "baseline",
    "recentzero",
    "recentwrong",
    "recentb1",
    "refzero",
    "refwrong",
    "refb1",
    "bothwrong",
    "bothb1",
    "latenthard",
    "latentsoft",
    "surfelkv",
]
SLOT_METHODS = {
    "recentzero": ("replace_recent_delta", "zero"),
    "recentwrong": ("replace_recent_delta", "wrong"),
    "recentb1": ("replace_recent_delta", "oracle"),
    "refzero": ("replace_ref_delta", "zero"),
    "refwrong": ("replace_ref_delta", "wrong"),
    "refb1": ("replace_ref_delta", "oracle"),
    "bothwrong": ("replace_both_delta", "wrong"),
    "bothb1": ("replace_both_delta", "oracle"),
}


def _copy_frozen_artifacts(source: Path, destination: Path, case_dir: Path) -> None:
    for name in ("baseline", "cut3r", "retrieval"):
        _link(source / name, destination / name)
    surfel_destination = destination / "surfel"
    if not surfel_destination.exists():
        shutil.copytree(source / "surfel", surfel_destination)
    trajectory = destination / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    for name in (
        "target_poses.npy",
        "yaw_pitch_roll.npy",
        "phase_labels.json",
        "trajectory_manifest.json",
    ):
        shutil.copy2(case_dir / name, trajectory / name)
    plots = destination / "assets" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case_dir / "pair_contact_sheet.png", plots / "pair_contact_sheet.png")
    posters = destination / "assets" / "posters"
    posters.mkdir(parents=True, exist_ok=True)
    Image.open(case_dir / "source_frame.png").convert("RGB").save(
        posters / "source.jpg", quality=88
    )


def _reference_bank_stats(path: Path) -> dict:
    payload = _json(path / "metadata.json")
    return {
        "num_chunks": len(payload["chunks"]),
        "available_chunks": sorted(int(chunk) for chunk in payload["chunks"]),
        "selected_layers": payload["selected_layers"],
        "num_layers": payload["num_layers"],
        "capture_type": payload["capture_type"],
        "rope_state": payload["rope_state"],
        "rope_layout": payload["rope_layout"],
        "memory_bytes": payload["memory_bytes"],
        "slot_len": payload["recent_slot_len"],
    }


def _status_from_metrics(metrics: dict) -> tuple[str, str, str]:
    slot = metrics["slot_ablation"]
    groups = slot["groups"]
    best_name = slot["most_influential_context_channel"]
    best = groups[best_name]
    specificity = float(best["source_specificity_error_margin"])
    improvement = float(best["correct_improvement_vs_baseline"])
    effect = float(best["correct_intervention_l1"])
    latent = metrics["generation"].get("latenthard", {})
    latent_improvement = float(
        latent.get("b1_b2_generated_region_l1_improvement_vs_baseline") or 0.0
    )
    # Triage only; synchronized B2 videos remain the primary judgement.
    source_separated = specificity > 0.002 and effect > 0.002 and improvement > 0
    if source_separated:
        status = {
            "recent": "KV_RECENT_VIABLE",
            "reference": "KV_REFERENCE_VIABLE",
            "both": "KV_MULTI_SLOT_VIABLE",
        }[best_name]
        if best_name == "recent":
            conclusion = (
                "Recent B1 KV strongly and source-specifically restores B1 at B2; "
                "reference-only is weak, and Both adds no correct-memory benefit."
            )
        elif best_name == "reference":
            conclusion = (
                "Reference-layout B1 KV is the strongest source-specific native-KV "
                "interface in the frozen checkpoint."
            )
        else:
            conclusion = (
                "Only the joint Recent+Reference intervention provides a strong "
                "source-specific native-KV signal in the frozen checkpoint."
            )
        next_action = (
            f"Repeat the frozen {best_name.title()}-slot SurfelKV closed loop on one "
            "second static yaw30 scene before partial-overlap experiments."
        )
    elif latent_improvement > max(0.01, improvement * 2.0) and effect < 0.01:
        status = "KV_TRAINING_FREE_LIMITED"
        conclusion = (
            "Recent/reference/both native-KV replacement is weak relative to the direct "
            "clean-latent control upper bound."
        )
        next_action = (
            "Choose geometry-indexed latent payload or one lightweight memory-aware adapter; "
            "do not continue tuning native-KV slots."
        )
    else:
        status = "INCONCLUSIVE"
        conclusion = (
            "The slot interventions are active, but source specificity and the latent-path "
            "gap require visual adjudication."
        )
        next_action = "Review synchronized B2 windows and make one status override if clear."
    return status, conclusion, next_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen InSpatio context-slot ablation")
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--inspatio-python", default=sys.executable)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    base_root = Path(args.base_run_root).resolve()
    case_dir = Path(args.case_dir).resolve()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _copy_frozen_artifacts(base_root, root, case_dir)

    manifest = _json(case_dir / "trajectory_manifest.json")
    phase_payload = _json(case_dir / "phase_labels.json")
    phases = {phase["name"]: phase for phase in phase_payload["phases"]}
    positive_chunks = list(
        range(
            phases["B1_hold"]["start_block"],
            phases["B1_hold"]["stop_block_exclusive"],
        )
    )
    target_chunks = list(
        range(
            phases["B2_hold"]["start_block"],
            phases["B2_hold"]["stop_block_exclusive"],
        )
    )
    source_chunk = int(manifest["source_chunk"])
    wrong_chunk = int(manifest["wrong_chunk"])
    target_chunk = int(manifest["target_chunk"])
    selected_steps = list(
        range(
            len(
                yaml.safe_load(
                    (repo / "configs/inference_1.3b.yaml").read_text()
                )["denoising_step_list"]
            )
        )
    )
    if source_chunk not in positive_chunks or target_chunk not in target_chunks:
        raise ValueError("Manifest B1/B2 chunks do not match controlled plateaus")

    common = Path(
        subprocess.check_output(
            [
                "git", "-C", str(repo), "rev-parse",
                "--path-format=absolute", "--git-common-dir",
            ],
            text=True,
        ).strip()
    )
    asset_root = common.parent
    noise_bundle = root / "kv" / "noise_bundle.pt"
    noise_bundle.parent.mkdir(parents=True, exist_ok=True)
    if not noise_bundle.exists():
        _link(base_root / "kv/noise_bundle.pt", noise_bundle)
    recent_bank = root / "kv/recent_bank_all"
    reference_bank = root / "kv/reference_bank_all"
    capture_root = root / "capture_all"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(repo)
            + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )

    config = {
        "case": "yaw30_scene01",
        "seed": args.seed,
        "base_run_root": str(base_root),
        "source_chunk": source_chunk,
        "wrong_chunk": wrong_chunk,
        "target_chunk": target_chunk,
        "positive_chunks": positive_chunks,
        "memory_target_chunks": target_chunks,
        "memory_layers": "all",
        "selected_step_indices": selected_steps,
        "alpha": 1.0,
        "gate": "global",
        "recent_bank": str(recent_bank),
        "reference_bank": str(reference_bank),
        "geometry_frozen": True,
        "retrieval_plan_sha256": _sha256(base_root / "retrieval/retrieval.json"),
        "trajectory_sha256": _sha256(case_dir / "target_poses.npy"),
        "source_sha256": _sha256(case_dir / "static_source.mp4"),
        "latent_controls": {
            "latenthard": {
                "mode": "direct_clean_x0_block_override",
                "strengths": [1.0, 1.0],
            },
            "latentsoft": {
                "mode": "direct_clean_x0_block_lerp",
                "strengths": [0.5, 0.25],
            },
            "note": (
                "Explicit upper bound; not native KV and not the separately trained "
                "Conv3D sidecar."
            ),
        },
    }
    (root / "config_resolved.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (root / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    def base_command(output: Path) -> list[str]:
        return _base_inference_command(
            python=args.inspatio_python,
            repo=repo,
            asset_root=asset_root,
            case_dir=case_dir,
            output_dir=output,
            noise_bundle=noise_bundle,
            bank_root=recent_bank,
            seed=args.seed,
            memory_layers="all",
        )

    if args.stage in {"capture", "full"}:
        if not (args.reuse and (reference_bank / "metadata.json").exists()):
            command = base_command(capture_root) + [
                "--run_name", "slot_capture_all",
                "--mode", "baseline",
                "--capture_kv",
                "--capture_chunks", str(source_chunk), str(wrong_chunk),
                "--capture_ref_kv",
                "--ref_bank_root", str(reference_bank),
                "--ref_chunks", str(source_chunk), str(wrong_chunk),
                "--compare_latents_to", str(root / "baseline/pred_latents.pt"),
                "--require_replay_tolerance",
            ]
            _run("slot_capture_all", command, root, env)
        _run(
            "recent_bank_stats",
            [
                args.inspatio_python, "-m", "mapkv.kv_bank",
                "--bank", str(recent_bank),
                "--output", str(root / "kv/bank_stats.json"),
                "--capture_manifest", str(root / "kv/capture_manifest.json"),
            ],
            root,
            env,
        )
        (root / "kv/reference_bank_stats.json").write_text(
            json.dumps(_reference_bank_stats(reference_bank), indent=2),
            encoding="utf-8",
        )
        capture_metadata = _json(capture_root / "run_metadata.json")
        replay = capture_metadata["replay"]["against_saved_latents"]
        if not replay["within_tolerance"]:
            raise RuntimeError(f"All-layer capture changed the baseline: {replay}")
        frozen_sanity = _json(base_root / "kv/sanity_metrics.json")
        sanity = {
            "alpha0_vs_baseline": frozen_sanity["alpha0_vs_baseline"],
            "all_layer_capture_vs_frozen_baseline": replay["max_abs_diff"],
            "capture_type": "clean_context",
            "reference_capture_type": "clean_reference_reencode",
            "recent_rope_layout": "recent_slot_t3_t5",
            "reference_rope_layout": "reference_slot_t0_t2",
            "selected_step_indices": selected_steps,
            "memory_target_chunks": target_chunks,
            "runtime_cache_unchanged": None,
        }
        (root / "kv/sanity_metrics.json").write_text(
            json.dumps(sanity, indent=2), encoding="utf-8"
        )

    def run_slot_method(name: str, injection_mode: str, source_mode: str) -> None:
        destination = root / "generation" / name
        if args.reuse and (destination / "run_metadata.json").exists():
            print("[MapKV] reuse", name)
            return
        command = base_command(destination) + [
            "--run_name", name,
            "--target_chunks", *[str(chunk) for chunk in target_chunks],
            "--selected_steps", *[str(step) for step in selected_steps],
            "--alpha", "1",
            "--injection_mode", injection_mode,
            "--gate_mode", "global",
            "--compare_latents_to", str(root / "baseline/pred_latents.pt"),
        ]
        if injection_mode in {"replace_ref_delta", "replace_both_delta"}:
            command += ["--ref_bank_root", str(reference_bank)]
        if source_mode == "zero":
            command += ["--mode", "zero", "--source_chunk", str(source_chunk)]
        elif source_mode == "wrong":
            command += ["--mode", "wrong", "--wrong_chunk", str(wrong_chunk)]
        elif source_mode == "oracle":
            command += ["--mode", "oracle", "--source_chunk", str(source_chunk)]
        elif source_mode == "geometry":
            command += [
                "--mode", "geometry",
                "--retrieval_plan", str(root / "retrieval/retrieval.json"),
            ]
        else:
            raise ValueError(source_mode)
        _run(name, command, root, env)

    def run_latent_method(name: str, strengths: list[float]) -> None:
        destination = root / "generation" / name
        if args.reuse and (destination / "run_metadata.json").exists():
            print("[MapKV] reuse", name)
            return
        if len(positive_chunks) != len(target_chunks) or len(strengths) != len(target_chunks):
            raise ValueError("Latent B1/B2 mapping must be one-to-one")
        command = base_command(destination) + [
            "--run_name", name,
            "--mode", "baseline",
            "--latent_memory_path", str(root / "baseline/pred_latents.pt"),
            "--latent_source_chunks", *[str(chunk) for chunk in positive_chunks],
            "--latent_target_chunks", *[str(chunk) for chunk in target_chunks],
            "--latent_strengths", *[str(value) for value in strengths],
            "--compare_latents_to", str(root / "baseline/pred_latents.pt"),
        ]
        _run(name, command, root, env)

    if args.stage in {"generation", "full"}:
        for name in (
            "recentzero", "recentwrong", "recentb1",
            "refzero", "refwrong", "refb1",
            "bothwrong", "bothb1",
        ):
            run_slot_method(name, *SLOT_METHODS[name])
        run_latent_method("latenthard", [1.0, 1.0])
        run_latent_method("latentsoft", [0.5, 0.25])

        methods_before_surfel = ALL_METHODS[:-1]
        metrics = evaluate_slots(
            run_root=root,
            case_dir=case_dir,
            source_chunk=source_chunk,
            target_chunk=target_chunk,
            methods=methods_before_surfel,
        )
        best_mode = metrics["slot_ablation"]["best_slot_mode"]
        run_slot_method("surfelkv", best_mode, "geometry")
        metrics = evaluate_slots(
            run_root=root,
            case_dir=case_dir,
            source_chunk=source_chunk,
            target_chunk=target_chunk,
            methods=ALL_METHODS,
        )
        cache_unchanged = True
        for name in SLOT_METHODS:
            metadata = _json(root / "generation" / name / "run_metadata.json")
            for target in target_chunks:
                cache_unchanged &= bool(
                    metadata["mapkv"]["cache_audits"][str(target)]["unchanged"]
                )
        sanity = _json(root / "kv/sanity_metrics.json")
        sanity["runtime_cache_unchanged"] = cache_unchanged
        sanity["memory_branch_effect"] = metrics["kv_sanity"]["memory_branch_effect"]
        (root / "kv/sanity_metrics.json").write_text(
            json.dumps(sanity, indent=2), encoding="utf-8"
        )
        evaluate_slots(
            run_root=root,
            case_dir=case_dir,
            source_chunk=source_chunk,
            target_chunk=target_chunk,
            methods=ALL_METHODS,
        )

    if args.stage in {"report", "full"}:
        visualize_existing(root / "surfel/surfel_index.npz", root / "surfel")
        metrics = _json(root / "metrics.json")
        best_mode = metrics["slot_ablation"]["best_slot_mode"]
        recent_stats = _json(root / "kv/bank_stats.json")
        reference_stats = _json(root / "kv/reference_bank_stats.json")
        architecture = {
            "backbone": "InSpatio-World-1.3B frozen student",
            "num_frame_per_block": 3,
            "geometry": {
                "frozen_from": str(base_root),
                "retrieval_plan_sha256": config["retrieval_plan_sha256"],
                "retrieved_chunk": source_chunk,
            },
            "memory": {
                "payload": "native_post_rope_kv",
                "granularity": "whole_chunk",
                "layers": recent_stats["selected_layers"],
                "recent_layout": "t3_t5",
                "reference_layout": "clean_generated_reencode_t0_t2",
                "recent_bytes": recent_stats["memory_bytes"],
                "reference_bytes": reference_stats["memory_bytes"],
            },
            "intervention": {
                "modes": [
                    "replace_recent_delta", "replace_ref_delta", "replace_both_delta"
                ],
                "alpha": 1.0,
                "steps": selected_steps,
                "target_chunks": target_chunks,
                "gate": "global",
                "best_slot_mode": best_mode,
            },
            "latent_upper_bound": config["latent_controls"],
        }
        (root / "architecture_state.json").write_text(
            json.dumps(architecture, indent=2), encoding="utf-8"
        )
        _transcode_videos(
            root, ALL_METHODS, target_chunk, int(manifest["target_rgb_index"])
        )
        status, conclusion, next_action = _status_from_metrics(metrics)
        _run(
            "slot_report",
            [
                args.inspatio_python, "-m", "mapkv.slot_report",
                "--run_root", str(root),
                "--status", status,
                "--conclusion", conclusion,
                "--next_action", next_action,
            ],
            root,
            env,
        )
    print(json.dumps({"stage": args.stage, "output": str(root)}, indent=2))


if __name__ == "__main__":
    main()
