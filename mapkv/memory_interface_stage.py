from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fast_pipeline import _base_inference_command, _run
from .memory_interface_evaluation import METHOD_ROOTS, evaluate_memory_interfaces
from .memory_interface_report import build_report
from .reentry_refinement_stage import _link_path, _phases, _transcode
from .source_protected_stage import _asset_root, _environment


STAGES = {"prepare", "upper_bound", "interfaces", "all4", "evaluate", "report", "full"}


def _prepare(root: Path, reuse: Path, case: Path) -> None:
    for name in ("baseline", "cut3r", "surfel", "kv", "surfel_rgb_options"):
        _link_path(reuse / name, root / name)
    generation = root / "generation"
    generation.mkdir(parents=True, exist_ok=True)
    _link_path(reuse / "generation/episode_continuous", generation / "episode_wre")
    trajectory = root / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    for name in (
        "target_poses.npy", "yaw_pitch_roll.npy", "phase_labels.json",
        "trajectory_manifest.json", "pose_validation.json",
    ):
        _link_path(case / name, trajectory / name)


def _interface_command(
    *, name: str, interface: str, interface_steps: tuple[int, ...],
    python: str, repo: Path, case: Path, root: Path, source_chunk: int,
    observation_start_chunk: int, seed: int,
) -> list[str]:
    return _base_inference_command(
        python=python,
        repo=repo,
        asset_root=_asset_root(repo),
        case_dir=case,
        output_dir=root / "generation" / name,
        noise_bundle=root / "kv/noise_bundle.pt",
        bank_root=root / "kv/unused",
        seed=seed,
        memory_layers="all",
    ) + [
        "--run_name", name,
        "--mode", "oracle",
        "--source_chunk", str(source_chunk),
        "--selected_steps", "0", "1", "2", "3",
        "--alpha", "1",
        "--injection_mode", "replace_recent_delta",
        "--gate_mode", "global",
        "--memory_interface", interface,
        "--memory_interface_steps", *[str(value) for value in interface_steps],
        "--warp_history_representation", "rgb_warp_vae",
        "--warp_source_latents", str(root / "baseline/pred_latents.pt"),
        "--warp_intrinsics_path", str(case / "intrinsics.txt"),
        "--warp_surfel_index", str(root / "surfel/surfel_index.npz"),
        "--warp_surfel_sequence", str(root / "cut3r/sequence.json"),
        "--warp_memory_dilation_kernel", "3",
        "--warp_query_feather_kernel", "3",
        "--source_protected_memory",
        "--warp_reference_protection_kernel", "3",
        "--warp_generated_only_threshold", "0.5",
        "--reentry_memory",
        "--reentry_observation_start_chunk", str(observation_start_chunk),
        "--reentry_absent_blocks", "2",
        "--reentry_refresh_policy", "episode_continuous",
        "--compare_latents_to", str(root / "baseline/pred_latents.pt"),
    ]


def _commands(
    *, python: str, repo: Path, case: Path, root: Path, source_chunk: int,
    observation_start_chunk: int, seed: int,
) -> dict[str, list[str]]:
    common = dict(
        python=python, repo=repo, case=case, root=root,
        source_chunk=source_chunk,
        observation_start_chunk=observation_start_chunk,
        seed=seed,
    )
    return {
        "masked_hard_x0": _interface_command(
            name="masked_hard_x0", interface="masked_hard_x0",
            interface_steps=(0, 1, 2), **common,
        ),
        "dual_branch_recent": _interface_command(
            name="dual_branch_recent", interface="dual_branch_recent",
            interface_steps=(0, 1, 2), **common,
        ),
        "memory_render": _interface_command(
            name="memory_render", interface="native_render",
            interface_steps=(0, 1, 2), **common,
        ),
        "latent_anchor012": _interface_command(
            name="latent_anchor012", interface="latent_anchor",
            interface_steps=(0, 1, 2), **common,
        ),
        "latent_anchor_all4": _interface_command(
            name="latent_anchor_all4", interface="latent_anchor",
            interface_steps=(0, 1, 2, 3), **common,
        ),
    }


def _run_method(name: str, command: list[str], *, root: Path, env: dict[str, str]) -> None:
    if (root / "generation" / name / "run_metadata.json").exists():
        print(f"[MapKV] reuse {name}")
        return
    _run(name, command, root, env)


def _report_videos(root: Path, phases: dict[str, dict], fps: float = 24.0) -> None:
    original = root / "videos/original"
    report = root / "videos/report"
    original.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    reentry_start = int(phases["Leave_to_B2"]["rgb_start"])
    reentry_stop = int(phases["B2_hold"]["rgb_stop_exclusive"])
    for method, relative in METHOD_ROOTS.items():
        source = root / relative / "pred.mp4"
        if not source.exists():
            continue
        _link_path(source, original / f"{method}.mp4")
        outputs = (
            (report / f"full_revisit_{method}.mp4", None, None),
            (
                report / f"reentry_{method}.mp4",
                reentry_start / fps,
                (reentry_stop - reentry_start) / fps,
            ),
        )
        for destination, start, duration in outputs:
            if not destination.exists():
                _transcode(
                    source=source, destination=destination,
                    start_seconds=start, duration_seconds=duration,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="MapKV frozen memory-interface ladder")
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case_dir", default="artifacts/control/yaw45m20to35_scene01")
    parser.add_argument(
        "--reuse_root",
        default="results/mapkv_fast/yaw45m20to35_scene01_seed0_reentry_refresh",
    )
    parser.add_argument(
        "--output_root",
        default="results/mapkv_fast/yaw45m20to35_scene01_seed0_memory_interface",
    )
    parser.add_argument(
        "--inspatio_python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    case = Path(args.case_dir).resolve()
    reuse = Path(args.reuse_root).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest, phases = _phases(case)
    source_chunk = int(manifest["source_chunk"])
    observation_start = int(phases["B1_hold"]["start_block"])
    _prepare(root, reuse, case)
    commands = _commands(
        python=args.inspatio_python, repo=repo, case=case, root=root,
        source_chunk=source_chunk, observation_start_chunk=observation_start,
        seed=args.seed,
    )
    (root / "run_contract.json").write_text(
        json.dumps(
            {
                "focus_zh": "冻结同一历史记忆，收敛 frozen InSpatio 的 identity 接口",
                "source_chunk": source_chunk,
                "target_chunk": int(manifest["target_chunk"]),
                "memory": "chunk11 RGB-Warp→Wan-VAE target-aligned latent",
                "mask": "same M_need for M2–M6",
                "lifecycle": "episode_continuous, absent_blocks=2",
                "commands": commands,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    env = _environment(repo, args.gpu)
    if args.stage == "full":
        _run_method("masked_hard_x0", commands["masked_hard_x0"], root=root, env=env)
        upper = evaluate_memory_interfaces(run_root=root, case_dir=case)
        if not upper["decisions"]["hard_upper_bound_works"]:
            _report_videos(root, phases)
            print(build_report(root))
            return
        for name in ("dual_branch_recent", "memory_render", "latent_anchor012"):
            _run_method(name, commands[name], root=root, env=env)
        result = evaluate_memory_interfaces(run_root=root, case_dir=case)
        if result["decisions"]["latent_anchor012_needs_all4"]:
            _run_method(
                "latent_anchor_all4", commands["latent_anchor_all4"],
                root=root, env=env,
            )
            evaluate_memory_interfaces(run_root=root, case_dir=case)
        _report_videos(root, phases)
        print(build_report(root))
        return
    if args.stage == "prepare":
        print(root)
        return
    if args.stage == "upper_bound":
        _run_method("masked_hard_x0", commands["masked_hard_x0"], root=root, env=env)
    elif args.stage == "interfaces":
        for name in ("dual_branch_recent", "memory_render", "latent_anchor012"):
            _run_method(name, commands[name], root=root, env=env)
    elif args.stage == "all4":
        _run_method("latent_anchor_all4", commands["latent_anchor_all4"], root=root, env=env)
    elif args.stage == "evaluate":
        print(json.dumps(evaluate_memory_interfaces(run_root=root, case_dir=case), indent=2))
        return
    elif args.stage == "report":
        _report_videos(root, phases)
        print(build_report(root))
        return
    evaluate_memory_interfaces(run_root=root, case_dir=case)


if __name__ == "__main__":
    main()
