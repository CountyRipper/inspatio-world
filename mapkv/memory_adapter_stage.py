from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from .fast_pipeline import _base_inference_command
from .memory_adapter_report import build_memory_adapter_report
from .memory_adapter_evaluation import evaluate_memory_adapter
from .memory_interface_stage import _interface_command
from .reentry_refinement_stage import _link_path, _memory_command
from .source_protected_stage import (
    _asset_root,
    _baseline_command,
    _build_geometry,
    _environment,
)


STAGES = {
    "prepare", "scene02", "zero_init", "baselines", "overfit", "joint", "refine",
    "evaluate", "report", "full",
}


def _run(name: str, command: list[str], root: Path, env: dict[str, str]) -> None:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    print(f"\n[MapKV adapter] {name}\n{' '.join(command)}")
    with (logs / f"{name}.log").open("w", encoding="utf-8") as handle:
        subprocess.run(
            command, cwd=str(Path(__file__).resolve().parents[1]), env=env,
            check=True, stdout=handle, stderr=subprocess.STDOUT,
        )


def _manifest(case: Path) -> tuple[dict, dict[str, dict]]:
    trajectory = json.loads((case / "trajectory_manifest.json").read_text())
    labels = json.loads((case / "phase_labels.json").read_text())
    return trajectory, {value["name"]: value for value in labels["phases"]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_contract(
    *, repo: Path, root: Path, cases: dict[str, dict[str, Path]], seed: int
) -> None:
    git_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    )
    payload = {
        "focus_zh": "Frozen InSpatio + Camera-Aligned Lightweight MemoryPatchAdapter",
        "git_commit": git_commit,
        "working_tree_clean": not bool(git_status.strip()),
        "seed": int(seed),
        "checkpoint": str(
            _asset_root(repo)
            / "checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"
        ),
        "adapter": {
            "input": ["L_mem", "raw_last_pred", "M_need"],
            "hidden_channels": 32,
            "injection": "parallel_masked_patch_token_residual",
            "zero_init": True,
        },
        "frozen": [
            "InSpatio backbone", "text encoder", "Wan VAE", "scheduler",
            "source/render pipeline", "CUT3R", "surfel map", "camera warp",
            "re-entry lifecycle",
        ],
        "cases": {},
    }
    for case_id, paths in cases.items():
        trajectory, _ = _manifest(paths["case"])
        payload["cases"][case_id] = {
            "case_id": trajectory["case_id"],
            "source_chunk": int(trajectory["source_chunk"]),
            "target_chunk": int(trajectory["target_chunk"]),
            "trajectory": [0.0, float(trajectory["b1_theta_degrees"]),
                           float(trajectory["leave_theta_degrees"]),
                           float(trajectory["b2_theta_degrees"])],
            "target_pose_sha256": _sha256(paths["case"] / "target_poses.npy"),
            "source_sha256": _sha256(paths["case"] / "static_source.mp4"),
        }
    (root / "run_contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if git_status:
        (root / "git_diff_stat.txt").write_text(
            subprocess.check_output(
                ["git", "-C", str(repo), "diff", "--stat"], text=True
            ), encoding="utf-8",
        )


def _case_paths(repo: Path, root: Path) -> dict[str, dict[str, Path]]:
    return {
        "scene01": {
            "case": repo / "artifacts/control/yaw45m20to35_scene01",
            "source": repo / "results/mapkv_fast/yaw45m20to35_scene01_seed0_reentry_refresh",
            "interfaces": repo / "results/mapkv_fast/yaw45m20to35_scene01_seed0_memory_interface",
            "output": root / "scene01",
        },
        "scene02": {
            "case": repo / "artifacts/control/yaw45m20to35_scene02",
            "source": repo / "results/mapkv_fast/yaw45m20to35_scene02_seed0_source_protected",
            "interfaces": repo / "results/mapkv_fast/yaw45m20to35_scene02_seed0_source_protected",
            "output": root / "scene02",
        },
    }


def _ensure_scene02_case(
    *, repo: Path, case: Path, python: str, root: Path,
    env: dict[str, str], gpu: str,
) -> None:
    if (case / "trajectory_manifest.json").exists():
        return
    _run(
        "build_scene02_control",
        [
            python, str(repo / "scripts/build_mapkv_control_case.py"),
            "--case_id", "yaw45m20to35_scene02",
            "--source_json", str(repo / "test/example2/new.json"),
            "--data_path_root", str(_asset_root(repo)),
            "--source_frame_index", "150",
            "--theta", "45", "--leave_theta", "-45",
            "--revisit_theta", "35",
            "--vae_calibration_metadata",
            str(repo / "artifacts/control/yaw30_scene01/baseline/seed_0/run_metadata.json"),
            "--vae_time_map", str(repo / "artifacts/control/vae_time_map.json"),
            "--render", "--render_device", str(gpu),
        ],
        root, env,
    )


def _prepare(root: Path, cases: dict[str, dict[str, Path]]) -> None:
    for case_id, paths in cases.items():
        output = paths["output"]
        methods = output / "methods"
        methods.mkdir(parents=True, exist_ok=True)
        _link_path(paths["source"] / "baseline", methods / "baseline")
        _link_path(paths["source"] / "cut3r", output / "cut3r")
        _link_path(paths["source"] / "surfel", output / "surfel")
        _link_path(paths["source"] / "kv", output / "kv")
        for name in (
            "target_poses.npy", "yaw_pitch_roll.npy", "phase_labels.json",
            "trajectory_manifest.json", "pose_validation.json",
        ):
            _link_path(paths["case"] / name, output / "trajectory" / name)
        if case_id == "scene01":
            _link_path(
                paths["source"] / "generation/episode_continuous",
                methods / "episode_wre",
            )
            _link_path(
                paths["interfaces"] / "generation/latent_anchor_all4",
                methods / "latent_anchor_all4",
            )


def _ensure_scene02_source(
    *, paths: dict[str, Path], python: str, repo: Path, root: Path,
    env: dict[str, str], seed: int,
) -> None:
    source = paths["source"]
    baseline = source / "baseline"
    noise = source / "kv/noise_bundle.pt"
    if not (baseline / "run_metadata.json").exists():
        command = _baseline_command(
            python=python, repo=repo, case_dir=paths["case"], output=baseline,
            noise=noise, seed=seed,
        )
        _run("scene02_baseline", command, root, env)
    trajectory, phases = _manifest(paths["case"])
    _build_geometry(
        repo=repo, root=source, baseline=baseline,
        source_chunk=int(trajectory["source_chunk"]),
        first_target=int(phases["B2_hold"]["start_block"]),
        inspatio_python=python,
        cut3r_python=str(repo / "third_party/mapkv_cut3r_env/bin/python"),
        env=env,
    )
def _adapter_inference_command(
    *,
    python: str,
    repo: Path,
    paths: dict[str, Path],
    output: Path,
    checkpoint: Path | None,
    zero_init: bool,
    middle: bool,
    seed: int,
) -> list[str]:
    trajectory, phases = _manifest(paths["case"])
    command = _base_inference_command(
        python=python,
        repo=repo,
        asset_root=_asset_root(repo),
        case_dir=paths["case"],
        output_dir=output,
        noise_bundle=paths["source"] / "kv/noise_bundle.pt",
        bank_root=paths["output"] / "unused",
        seed=seed,
        memory_layers="all",
    ) + [
        "--run_name",
        (
            "adapter_zero_init_middle"
            if zero_init and middle
            else "adapter_zero_init"
            if zero_init
            else "adapter_patch_middle"
            if middle
            else "adapter_patch_only"
        ),
        "--mode", "oracle",
        "--source_chunk", str(trajectory["source_chunk"]),
        "--selected_steps", "0", "1", "2", "3",
        "--alpha", "1",
        "--injection_mode", "replace_recent_delta",
        "--gate_mode", "global",
        "--warp_history_representation", "rgb_warp_vae",
        "--warp_source_latents", str(paths["source"] / "baseline/pred_latents.pt"),
        "--warp_intrinsics_path", str(paths["case"] / "intrinsics.txt"),
        "--warp_surfel_index", str(paths["source"] / "surfel/surfel_index.npz"),
        "--warp_surfel_sequence", str(paths["source"] / "cut3r/sequence.json"),
        "--warp_memory_dilation_kernel", "3",
        "--warp_query_feather_kernel", "3",
        "--source_protected_memory",
        "--warp_reference_protection_kernel", "3",
        "--warp_generated_only_threshold", "0.5",
        "--reentry_memory",
        "--reentry_observation_start_chunk", str(phases["B1_hold"]["start_block"]),
        "--reentry_absent_blocks", "2",
        "--reentry_refresh_policy", "episode_continuous",
        "--compare_latents_to", str(paths["source"] / "baseline/pred_latents.pt"),
    ]
    if zero_init:
        command += [
            "--memory_adapter_zero_init", "--require_replay_tolerance",
            "--replay_tolerance", "0",
        ]
    else:
        command += ["--memory_adapter_checkpoint", str(checkpoint)]
    if middle:
        command.append("--memory_adapter_middle")
    return command


def _ensure_zero_init(
    *, case_id: str, paths: dict[str, Path], python: str, repo: Path,
    root: Path, env: dict[str, str], seed: int,
) -> None:
    output = paths["output"] / "zero_init"
    metadata = output / "run_metadata.json"
    if not metadata.exists():
        _run(
            f"{case_id}_zero_init",
            _adapter_inference_command(
                python=python, repo=repo, paths=paths, output=output,
                checkpoint=None, zero_init=True, middle=False, seed=seed,
            ),
            root, env,
        )
    payload = json.loads(metadata.read_text())
    replay = payload.get("replay") or {}
    replay = replay.get("against_saved_latents") or replay
    if float(replay.get("max_abs_diff", float("inf"))) != 0.0:
        raise RuntimeError(f"{case_id} zero-init adapter is not exact baseline: {replay}")


def _ensure_scene02_baselines(
    *, paths: dict[str, Path], python: str, repo: Path, root: Path,
    env: dict[str, str], seed: int,
) -> None:
    trajectory, phases = _manifest(paths["case"])
    source_root = paths["source"]
    episode = source_root / "generation/episode_continuous"
    if not (episode / "run_metadata.json").exists():
        command = _memory_command(
            name="episode_continuous", python=python, repo=repo,
            case=paths["case"], root=source_root,
            anchor_chunk=int(trajectory["source_chunk"]),
            observation_start_chunk=int(phases["B1_hold"]["start_block"]),
            seed=seed, view_adaptive=False, edge_safe=False,
            final_step_stabilized=False, refresh_policy="episode_continuous",
        )
        _run("scene02_episode_wre", command, root, env)
    latent = source_root / "generation/latent_anchor_all4"
    if not (latent / "run_metadata.json").exists():
        command = _interface_command(
            name="latent_anchor_all4", interface="latent_anchor",
            interface_steps=(0, 1, 2, 3), python=python, repo=repo,
            case=paths["case"], root=source_root,
            source_chunk=int(trajectory["source_chunk"]),
            observation_start_chunk=int(phases["B1_hold"]["start_block"]),
            seed=seed,
        )
        _run("scene02_latent_anchor_all4", command, root, env)
    _link_path(episode, paths["output"] / "methods/episode_wre")
    _link_path(latent, paths["output"] / "methods/latent_anchor_all4")


def _train_command(
    *, python: str, repo: Path, cases: list[dict[str, Path]],
    output: Path, steps: int, seed: int, middle: bool,
) -> list[str]:
    command = [
        python, "-m", "mapkv.memory_adapter_training",
        "--repo", str(repo), "--asset_root", str(_asset_root(repo)),
        "--output_dir", str(output), "--device", "cuda:0",
        "--hidden_channels", "32", "--steps", str(steps),
        "--learning_rate", "0.001", "--rgb_every", "8",
        "--minimum_coverage", "0.05", "--seed", str(seed),
    ]
    for paths in cases:
        value = "::".join(
            [
                paths["case"].name,
                str(paths["case"]),
                str(paths["source"] / "baseline"),
                str(paths["output"] / "zero_init"),
                str(paths["source"] / "kv/noise_bundle.pt"),
            ]
        )
        command += ["--case", value]
    if middle:
        command.append("--inject_middle")
    return command


def _train(
    *, name: str, case_paths: list[dict[str, Path]], output: Path,
    steps: int, python: str, repo: Path, root: Path, env: dict[str, str], seed: int,
    middle: bool = False,
) -> dict:
    summary_path = output / "training_summary.json"
    if not summary_path.exists():
        _run(
            name,
            _train_command(
                python=python, repo=repo, cases=case_paths,
                output=output, steps=steps, seed=seed, middle=middle,
            ),
            root, env,
        )
    return json.loads(summary_path.read_text())


def _evaluate_checkpoint(
    *, label: str, checkpoint: Path, cases: dict[str, dict[str, Path]],
    python: str, repo: Path, root: Path, env: dict[str, str], seed: int,
    middle: bool = False,
    env_by_case: dict[str, dict[str, str]] | None = None,
) -> None:
    for case_id, paths in cases.items():
        output = paths["output"] / "methods" / label
        if (output / "run_metadata.json").exists():
            continue
        _run(
            f"{case_id}_{label}",
            _adapter_inference_command(
                python=python, repo=repo, paths=paths, output=output,
                checkpoint=checkpoint, zero_init=False, middle=middle, seed=seed,
            ),
            root, (env_by_case or {}).get(case_id, env),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-scene MapKV memory adapter stage")
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--scene02_gpu",
        help="Optional scene02 GPU; must match the GPU used for its baseline.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_root", default="results/mapkv_fast/memory_adapter_two_scene"
    )
    parser.add_argument(
        "--inspatio_python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = _case_paths(repo, root)
    env = _environment(repo, args.gpu)
    scene02_gpu = args.scene02_gpu or args.gpu
    scene02_env = _environment(repo, scene02_gpu)
    env_by_case = {"scene01": env, "scene02": scene02_env}
    _ensure_scene02_case(
        repo=repo, case=cases["scene02"]["case"],
        python=args.inspatio_python, root=root, env=scene02_env, gpu=scene02_gpu,
    )
    _prepare(root, cases)
    _write_run_contract(repo=repo, root=root, cases=cases, seed=args.seed)
    config = yaml.safe_load((repo / "configs/mapkv_memory_adapter.yaml").read_text())
    (root / "adapter_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if args.stage == "prepare":
        return
    if args.stage in {"scene02", "zero_init", "baselines", "full"}:
        _ensure_scene02_source(
            paths=cases["scene02"], python=args.inspatio_python, repo=repo,
            root=root, env=scene02_env, seed=args.seed,
        )
        if args.stage == "scene02":
            return
    if args.stage in {"zero_init", "full"}:
        for case_id, paths in cases.items():
            _ensure_zero_init(
                case_id=case_id, paths=paths, python=args.inspatio_python,
                repo=repo, root=root, env=env_by_case[case_id], seed=args.seed,
            )
        if args.stage == "zero_init":
            return
    if args.stage in {"baselines", "full"}:
        _ensure_scene02_baselines(
            paths=cases["scene02"], python=args.inspatio_python, repo=repo,
            root=root, env=scene02_env, seed=args.seed,
        )
        if args.stage == "baselines":
            return
    overfit_root = root / "training/overfit_scene01"
    if args.stage in {"overfit", "full"}:
        summary = _train(
            name="train_overfit_scene01", case_paths=[cases["scene01"]],
            output=overfit_root, steps=int(config["training"]["example_a_overfit_steps"]),
            python=args.inspatio_python, repo=repo, root=root, env=env, seed=args.seed,
        )
        capacity_reduction = summary.get("matched_core_loss_reduction_fraction")
        if capacity_reduction is None:
            capacity_reduction = summary.get("loss_reduction_fraction")
        if float(capacity_reduction or 0.0) <= 0.0:
            (root / "status.json").write_text(
                json.dumps({"status": "CURRENT_ADAPTER_INTERFACE_INSUFFICIENT"}, indent=2)
            )
            raise RuntimeError("Example-A overfit did not reduce the supervised objective")
        _evaluate_checkpoint(
            label="adapter_overfit", checkpoint=overfit_root / "checkpoint/adapter.pt",
            cases={"scene01": cases["scene01"]}, python=args.inspatio_python,
            repo=repo, root=root, env=env, seed=args.seed,
            env_by_case=env_by_case,
        )
        if args.stage == "overfit":
            return
    joint_root = root / "training/joint_scene01_scene02"
    if args.stage in {"joint", "full"}:
        _train(
            name="train_joint_scene01_scene02", case_paths=list(cases.values()),
            output=joint_root, steps=int(config["training"]["joint_steps"]),
            python=args.inspatio_python, repo=repo, root=root, env=env, seed=args.seed,
        )
        _evaluate_checkpoint(
            label="adapter_patch_only", checkpoint=joint_root / "checkpoint/adapter.pt",
            cases=cases, python=args.inspatio_python, repo=repo, root=root,
            env=env, seed=args.seed, env_by_case=env_by_case,
        )
        if args.stage == "joint":
            return
    refine_root = root / "training/joint_patch_middle"
    run_refine = args.stage == "refine"
    if args.stage == "full":
        patch_only = evaluate_memory_adapter(root)
        run_refine = not patch_only["decisions"][
            "joint_adapter_identity_metric_better_than_episode_wre"
        ]
    if run_refine:
        _train(
            name="train_joint_patch_middle", case_paths=list(cases.values()),
            output=refine_root, steps=int(config["training"]["joint_steps"]),
            python=args.inspatio_python, repo=repo, root=root, env=env,
            seed=args.seed, middle=True,
        )
        _evaluate_checkpoint(
            label="adapter_patch_middle",
            checkpoint=refine_root / "checkpoint/adapter.pt",
            cases=cases, python=args.inspatio_python, repo=repo, root=root,
            env=env, seed=args.seed, middle=True, env_by_case=env_by_case,
        )
        if args.stage == "refine":
            return
    if args.stage in {"evaluate", "full"}:
        evaluate_memory_adapter(root)
        if args.stage == "evaluate":
            return
    if args.stage in {"report", "full"}:
        print(build_memory_adapter_report(root))


if __name__ == "__main__":
    main()
