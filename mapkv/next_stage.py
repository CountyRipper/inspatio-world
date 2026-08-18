from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .fast_pipeline import (
    _base_inference_command,
    _json,
    _link,
    _run,
    _transcode_videos,
)
from .locality_evaluation import (
    evaluate_layer_budget,
    evaluate_locality,
    evaluate_replication,
)


STAGES = {"replication", "partial", "layers", "report", "full"}


def _phases(case_dir: Path) -> tuple[dict, list[int], list[int]]:
    manifest = _json(case_dir / "trajectory_manifest.json")
    payload = _json(case_dir / "phase_labels.json")
    phases = {item["name"]: item for item in payload["phases"]}
    positive = list(
        range(
            int(phases["B1_hold"]["start_block"]),
            int(phases["B1_hold"]["stop_block_exclusive"]),
        )
    )
    targets = list(
        range(
            int(phases["B2_hold"]["start_block"]),
            int(phases["B2_hold"]["stop_block_exclusive"]),
        )
    )
    return manifest, positive, targets


def _environment(repo: Path, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": str(repo)
            + (
                ":" + env["PYTHONPATH"]
                if env.get("PYTHONPATH")
                else ""
            ),
        }
    )
    return env


def _asset_root(repo: Path) -> Path:
    common = Path(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            text=True,
        ).strip()
    )
    return common.parent


def _all_steps(repo: Path) -> list[int]:
    config = yaml.safe_load(
        (repo / "configs/inference_1.3b.yaml").read_text(
            encoding="utf-8"
        )
    )
    return list(range(len(config["denoising_step_list"])))


def _geometry(
    *,
    repo: Path,
    root: Path,
    case_dir: Path,
    positive: list[int],
    targets: list[int],
    source_chunk: int,
    query_pose_mode: str,
    inspatio_python: str,
    cut3r_python: str,
    env: dict[str, str],
) -> None:
    cut3r_root = repo / "third_party/CUT3R"
    checkpoint = cut3r_root / "src/cut3r_512_dpt_4_64.pth"
    cut3r_output = root / "cut3r"
    if not (cut3r_output / "sequence.json").exists():
        command = [
            cut3r_python,
            "-m",
            "mapkv.cut3r_adapter",
            "--baseline_root",
            str(root / "baseline"),
            "--block_mapping",
            str(root / "baseline/block_mapping.json"),
            "--cut3r_root",
            str(cut3r_root),
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(cut3r_output),
            "--target_chunk",
            str(min(targets)),
            "--query_source_chunk",
            str(source_chunk),
            "--query_pose_mode",
            query_pose_mode,
            "--query_target_chunk",
            str(min(targets)),
            "--confidence_threshold",
            "1.5",
            "--device",
            "cuda",
        ]
        _run("cut3r", command, root, env)
    if not (root / "surfel/surfel_index.npz").exists():
        _run(
            "surfel",
            [
                inspatio_python,
                "-m",
                "mapkv.surfel_index",
                "--sequence",
                str(root / "cut3r/sequence.json"),
                "--output_dir",
                str(root / "surfel"),
                "--confidence_threshold",
                "1.5",
                "--voxel_size_mode",
                "relative_scene",
                "--relative_scene_fraction",
                "0.005",
                "--grid_height",
                "30",
                "--grid_width",
                "52",
                "--radius_scale",
                "0.5",
                "--merge_normal_cosine",
                "0.6",
            ],
            root,
            env,
        )
    if not (root / "retrieval/retrieval.json").exists():
        _run(
            "retrieval",
            [
                inspatio_python,
                "-m",
                "mapkv.retrieval",
                "--sequence",
                str(root / "cut3r/sequence.json"),
                "--surfel_index",
                str(root / "surfel/surfel_index.npz"),
                "--output_dir",
                str(root / "retrieval"),
                "--target_chunks",
                *[str(chunk) for chunk in targets],
                "--positive_chunks",
                *[str(chunk) for chunk in positive],
                "--top_k",
                "1",
                "--min_history_gap_chunks",
                "2",
            ],
            root,
            env,
        )


def _bank_stats(
    python: str, root: Path, bank: Path, env: dict[str, str]
) -> None:
    _run(
        "bank_stats",
        [
            python,
            "-m",
            "mapkv.kv_bank",
            "--bank",
            str(bank),
            "--output",
            str(root / "kv/bank_stats.json"),
            "--capture_manifest",
            str(root / "kv/capture_manifest.json"),
        ],
        root,
        env,
    )


def _run_recent(
    *,
    name: str,
    root: Path,
    case_dir: Path,
    bank: Path,
    noise: Path,
    baseline_latents: Path,
    source_chunk: int,
    wrong_chunk: int,
    targets: list[int],
    steps: list[int],
    gate: str,
    mode: str,
    python: str,
    repo: Path,
    asset_root: Path,
    env: dict[str, str],
    layers: list[int] | None = None,
    injection_mode: str = "replace_recent_delta",
    retrieval_plan: Path | None = None,
) -> None:
    destination = root / "generation" / name
    if (destination / "run_metadata.json").exists():
        print("[MapKV] reuse", name)
        return
    layer_mode = "all" if layers is None else "explicit"
    command = _base_inference_command(
        python=python,
        repo=repo,
        asset_root=asset_root,
        case_dir=case_dir,
        output_dir=destination,
        noise_bundle=noise,
        bank_root=bank,
        seed=0,
        memory_layers=layer_mode,
    )
    if layers is not None:
        command += ["--selected_layers", *[str(layer) for layer in layers]]
    command += [
        "--run_name",
        name,
        "--target_chunks",
        *[str(chunk) for chunk in targets],
        "--selected_steps",
        *[str(step) for step in steps],
        "--alpha",
        "1",
        "--injection_mode",
        injection_mode,
        "--gate_mode",
        gate,
        "--compare_latents_to",
        str(baseline_latents),
    ]
    if mode == "manual":
        command += ["--mode", "oracle", "--source_chunk", str(source_chunk)]
    elif mode == "wrong":
        command += ["--mode", "wrong", "--wrong_chunk", str(wrong_chunk)]
    elif mode == "geometry":
        command += [
            "--mode",
            "geometry",
            "--retrieval_plan",
            str(
                retrieval_plan
                or root / "retrieval/retrieval.json"
            ),
        ]
    else:
        raise ValueError(mode)
    _run(name, command, root, env)


def run_replication(args: argparse.Namespace, repo: Path) -> dict:
    root = Path(args.replication_root).resolve()
    case_dir = Path(args.scene02_case).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest, positive, targets = _phases(case_dir)
    baseline_source = (
        case_dir / "baseline" / f"seed_{args.seed}"
    ).resolve()
    if not (root / "baseline").exists():
        _link(baseline_source, root / "baseline")
    env = _environment(repo, args.gpu)
    steps = _all_steps(repo)
    _geometry(
        repo=repo,
        root=root,
        case_dir=case_dir,
        positive=positive,
        targets=targets,
        source_chunk=int(manifest["source_chunk"]),
        query_pose_mode="controlled_same_pose_known",
        inspatio_python=args.inspatio_python,
        cut3r_python=args.cut3r_python,
        env=env,
    )
    plan = _json(root / "retrieval/retrieval.json")
    selected = {
        tuple(entry["selected_chunks"])
        for entry in plan["targets"]
    }
    if len(selected) != 1:
        raise RuntimeError(
            f"Scene02 B2 chunks select inconsistent histories: {selected}"
        )
    source_chunk = int(next(iter(selected))[0])
    if source_chunk not in positive:
        raise RuntimeError(
            f"Scene02 retrieval missed B1 cluster {positive}: {source_chunk}"
        )
    wrong_chunk = int(manifest["wrong_chunk"])
    bank = root / "kv/recent_bank_all"
    noise = baseline_source / "noise_bundle.pt"
    capture = root / "capture_all"
    asset_root = _asset_root(repo)
    if not (bank / "metadata.json").exists():
        command = _base_inference_command(
            python=args.inspatio_python,
            repo=repo,
            asset_root=asset_root,
            case_dir=case_dir,
            output_dir=capture,
            noise_bundle=noise,
            bank_root=bank,
            seed=args.seed,
            memory_layers="all",
        ) + [
            "--run_name",
            "scene02_capture_all",
            "--mode",
            "baseline",
            "--capture_kv",
            "--capture_chunks",
            str(source_chunk),
            str(wrong_chunk),
            "--compare_latents_to",
            str(root / "baseline/pred_latents.pt"),
            "--require_replay_tolerance",
        ]
        _run("capture_all", command, root, env)
    _bank_stats(args.inspatio_python, root, bank, env)
    replay = _json(capture / "run_metadata.json")["replay"][
        "against_saved_latents"
    ]
    (root / "kv/sanity_metrics.json").write_text(
        json.dumps(
            {
                "all_layer_capture_vs_baseline": replay["max_abs_diff"],
                "capture_type": "clean_context",
                "rope_state": "post_rope",
                "selected_steps": steps,
                "target_chunks": targets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, mode in (
        ("manualcorrect", "manual"),
        ("wrongkv", "wrong"),
        ("surfelkv", "geometry"),
    ):
        _run_recent(
            name=name,
            root=root,
            case_dir=case_dir,
            bank=bank,
            noise=noise,
            baseline_latents=root / "baseline/pred_latents.pt",
            source_chunk=source_chunk,
            wrong_chunk=wrong_chunk,
            targets=targets,
            steps=steps,
            gate="global",
            mode=mode,
            python=args.inspatio_python,
            repo=repo,
            asset_root=asset_root,
            env=env,
        )
    metrics = evaluate_replication(
        run_root=root,
        case_dir=case_dir,
        source_chunk=source_chunk,
        target_chunk=int(manifest["target_chunk"]),
    )
    _transcode_videos(
        root,
        ["baseline", "manualcorrect", "wrongkv", "surfelkv"],
        int(manifest["target_chunk"]),
        int(manifest["target_rgb_index"]),
    )
    return metrics


def run_partial(args: argparse.Namespace, repo: Path) -> dict:
    root = Path(args.partial_root).resolve()
    case_dir = Path(args.partial_case).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest, positive, targets = _phases(case_dir)
    env = _environment(repo, args.gpu)
    steps = _all_steps(repo)
    asset_root = _asset_root(repo)
    baseline = root / "baseline"
    bank = root / "kv/recent_bank_all"
    noise = root / "kv/noise_bundle.pt"
    if not (baseline / "run_metadata.json").exists():
        command = _base_inference_command(
            python=args.inspatio_python,
            repo=repo,
            asset_root=asset_root,
            case_dir=case_dir,
            output_dir=baseline,
            noise_bundle=noise,
            bank_root=bank,
            seed=args.seed,
            memory_layers="all",
        ) + [
            "--run_name",
            "partial_baseline_capture",
            "--mode",
            "baseline",
            "--create_noise_bundle",
            "--capture_kv",
            "--capture_chunks",
            *[str(chunk) for chunk in positive],
            str(manifest["wrong_chunk"]),
            "--verify_memory_off_replay",
            "--require_replay_tolerance",
        ]
        _run("partial_baseline", command, root, env)
    _bank_stats(args.inspatio_python, root, bank, env)
    _geometry(
        repo=repo,
        root=root,
        case_dir=case_dir,
        positive=positive,
        targets=targets,
        source_chunk=int(manifest["source_chunk"]),
        query_pose_mode="known_target_pose",
        inspatio_python=args.inspatio_python,
        cut3r_python=args.cut3r_python,
        env=env,
    )
    plan = _json(root / "retrieval/retrieval.json")
    locality_audit = root / "retrieval/locality_control.json"
    if locality_audit.exists():
        plan = _json(root / "retrieval/retrieval.json")
    else:
        unconstrained_selected = {
            tuple(entry["selected_chunks"])
            for entry in plan["targets"]
        }
        if not all(
            selected_chunk in positive
            for selection in unconstrained_selected
            for selected_chunk in selection[:1]
        ):
            retrieval_root = root / "retrieval"
            shutil.copy2(
                retrieval_root / "retrieval.json",
                retrieval_root / "unconstrained_retrieval.json",
            )
            shutil.copy2(
                retrieval_root / "pose_plan.json",
                retrieval_root / "unconstrained_pose_plan.json",
            )
            _run(
                "retrieval_b1_locality_control",
                [
                    args.inspatio_python,
                    "-m",
                    "mapkv.retrieval",
                    "--sequence",
                    str(root / "cut3r/sequence.json"),
                    "--surfel_index",
                    str(root / "surfel/surfel_index.npz"),
                    "--output_dir",
                    str(retrieval_root),
                    "--target_chunks",
                    *[str(chunk) for chunk in targets],
                    "--positive_chunks",
                    *[str(chunk) for chunk in positive],
                    "--candidate_chunks",
                    *[str(chunk) for chunk in positive],
                    "--top_k",
                    "1",
                    "--min_history_gap_chunks",
                    "2",
                ],
                root,
                env,
            )
            locality_audit.write_text(
                json.dumps(
                    {
                        "mode": "B1_positive_cluster_candidate_control",
                        "reason": (
                            "Stage C isolates query-side locality after the "
                            "unconstrained partial-overlap address selected another "
                            "valid historical observer."
                        ),
                        "unconstrained_selected": [
                            list(selection)
                            for selection in sorted(unconstrained_selected)
                        ],
                        "candidate_chunks": positive,
                        "geometry_parameters_changed": False,
                        "score_parameters_changed": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            plan = _json(retrieval_root / "retrieval.json")
    selected = {
        tuple(entry["selected_chunks"])
        for entry in plan["targets"]
    }
    if len(selected) != 1:
        raise RuntimeError(
            f"Partial B2 chunks select inconsistent histories: {selected}"
        )
    source_chunk = int(next(iter(selected))[0])
    if source_chunk not in positive:
        raise RuntimeError(
            "Partial-overlap generation requires a B1-cluster hit before "
            f"injection; selected {source_chunk}, expected {positive}"
        )
    wrong_chunk = int(manifest["wrong_chunk"])
    sanity = {
        "baseline_in_process_replay": _json(
            baseline / "run_metadata.json"
        )["replay"]["in_process_memory_off"]["max_abs_diff"],
        "capture_type": "clean_context",
        "rope_state": "post_rope",
        "selected_steps": steps,
        "target_chunks": targets,
    }
    (root / "kv/sanity_metrics.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )
    for name, gate in (
        ("global_surfelkv", "global"),
        ("gated_surfelkv", "surfel"),
    ):
        _run_recent(
            name=name,
            root=root,
            case_dir=case_dir,
            bank=bank,
            noise=noise,
            baseline_latents=baseline / "pred_latents.pt",
            source_chunk=source_chunk,
            wrong_chunk=wrong_chunk,
            targets=targets,
            steps=steps,
            gate=gate,
            mode="geometry",
            python=args.inspatio_python,
            repo=repo,
            asset_root=asset_root,
            env=env,
        )
    metrics = evaluate_locality(
        run_root=root,
        case_dir=case_dir,
        source_chunk=source_chunk,
        target_chunk=int(manifest["target_chunk"]),
    )
    methods = ["baseline", "global_surfelkv", "gated_surfelkv"]
    if metrics["status"] != "QUERY_GATING_SUFFICIENT":
        token_plan = root / "retrieval/token_selected_plan.json"
        if not token_plan.exists():
            _run(
                "token_selection",
                [
                    args.inspatio_python,
                    "-m",
                    "mapkv.token_selection",
                    "--sequence",
                    str(root / "cut3r/sequence.json"),
                    "--surfel_index",
                    str(root / "surfel/surfel_index.npz"),
                    "--retrieval_plan",
                    str(root / "retrieval/retrieval.json"),
                    "--output_plan",
                    str(token_plan),
                    "--neighborhood",
                    "1",
                ],
                root,
                env,
            )
        _run_recent(
            name="token_selected_surfelkv",
            root=root,
            case_dir=case_dir,
            bank=bank,
            noise=noise,
            baseline_latents=baseline / "pred_latents.pt",
            source_chunk=source_chunk,
            wrong_chunk=wrong_chunk,
            targets=targets,
            steps=steps,
            gate="surfel",
            mode="geometry",
            python=args.inspatio_python,
            repo=repo,
            asset_root=asset_root,
            env=env,
            injection_mode="selected_recent_delta",
            retrieval_plan=token_plan,
        )
        methods.append("token_selected_surfelkv")
        metrics = evaluate_locality(
            run_root=root,
            case_dir=case_dir,
            source_chunk=source_chunk,
            target_chunk=int(manifest["target_chunk"]),
            methods=tuple(methods),
        )
    _transcode_videos(
        root,
        methods,
        int(manifest["target_chunk"]),
        int(manifest["target_rgb_index"]),
    )
    return metrics


def run_layers(args: argparse.Namespace, repo: Path) -> dict:
    root = Path(args.layer_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(args.scene01_slot_root).resolve()
    case_dir = Path(args.scene01_case).resolve()
    manifest, _positive, targets = _phases(case_dir)
    bank = source / "kv/recent_bank_all"
    noise = source / "kv/noise_bundle.pt"
    baseline = source / "baseline"
    env = _environment(repo, args.gpu)
    asset_root = _asset_root(repo)
    steps = _all_steps(repo)
    layer_sets = {
        "all": list(range(30)),
        "early10": list(range(0, 10)),
        "middle10": list(range(10, 20)),
        "late10": list(range(20, 30)),
    }
    method_roots: dict[str, Path] = {
        "all": source / "generation/recentb1"
    }
    for name in ("early10", "middle10", "late10"):
        _run_recent(
            name=name,
            root=root,
            case_dir=case_dir,
            bank=bank,
            noise=noise,
            baseline_latents=baseline / "pred_latents.pt",
            source_chunk=int(manifest["source_chunk"]),
            wrong_chunk=int(manifest["wrong_chunk"]),
            targets=targets,
            steps=steps,
            gate="global",
            mode="manual",
            python=args.inspatio_python,
            repo=repo,
            asset_root=asset_root,
            env=env,
            layers=layer_sets[name],
        )
        method_roots[name] = root / "generation" / name
    metrics = evaluate_layer_budget(
        output_dir=root,
        case_dir=case_dir,
        baseline_root=baseline,
        bank_root=bank,
        source_chunk=int(manifest["source_chunk"]),
        target_chunk=int(manifest["target_chunk"]),
        method_roots=method_roots,
        layer_sets=layer_sets,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replication, partial-overlap locality, and layer budget"
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--inspatio-python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    parser.add_argument(
        "--cut3r-python",
        default=str(
            Path(__file__).resolve().parents[1]
            / "third_party/mapkv_cut3r_env/bin/python"
        ),
    )
    parser.add_argument(
        "--scene02-case", default="artifacts/control/yaw30_scene02"
    )
    parser.add_argument(
        "--partial-case", default="artifacts/control/yaw30to20_scene01"
    )
    parser.add_argument(
        "--scene01-case", default="artifacts/control/yaw30_scene01"
    )
    parser.add_argument(
        "--scene01-slot-root",
        default=(
            "results/mapkv_fast/"
            "yaw30_scene01_seed0_slot_ablation"
        ),
    )
    parser.add_argument(
        "--replication-root",
        default=(
            "results/mapkv_fast/"
            "yaw30_scene02_seed0_replication"
        ),
    )
    parser.add_argument(
        "--partial-root",
        default=(
            "results/mapkv_fast/"
            "yaw30to20_scene01_seed0_locality"
        ),
    )
    parser.add_argument(
        "--layer-root",
        default=(
            "results/mapkv_fast/"
            "yaw30_scene01_seed0_layer_budget"
        ),
    )
    parser.add_argument(
        "--report-root",
        default=(
            "results/mapkv_fast/"
            "mapkv_next_stage_seed0"
        ),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result: dict[str, object] = {"stage": args.stage}
    if args.stage in {"replication", "full"}:
        replication = run_replication(args, repo)
        result["replication"] = replication["replication"]
        if replication["replication"]["status"] != "PASS":
            print(json.dumps(result, indent=2))
            raise SystemExit("REPLICATION_FAILED")
    if args.stage in {"partial", "full"}:
        result["partial"] = run_partial(args, repo)["status"]
    if args.stage in {"layers", "full"}:
        result["layers"] = run_layers(args, repo)["methods"]
    if args.stage in {"report", "full"}:
        from .next_stage_report import build_report

        result["report"] = build_report(
            output_root=args.report_root,
            replication_root=args.replication_root,
            partial_root=args.partial_root,
            layer_root=args.layer_root,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
