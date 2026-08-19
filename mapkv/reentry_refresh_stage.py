from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fast_pipeline import _json, _run
from .reentry_refinement_stage import (
    _link_path,
    _memory_command,
    _phases,
    _transcode,
)
from .reentry_refresh_evaluation import (
    METHOD_ROOTS,
    evaluate_reentry_refresh,
)
from .reentry_refresh_report import build_report
from .source_protected_stage import _environment


STAGES = {
    "prepare",
    "priority1",
    "priority2",
    "priority3",
    "priority4",
    "evaluate",
    "report",
    "full",
}


def _prepare(root: Path, reuse: Path, case: Path) -> None:
    for name in ("baseline", "cut3r", "surfel", "kv"):
        _link_path(reuse / name, root / name)
    _link_path(reuse / "surfel_rgb_options", root / "surfel_rgb_options")
    generation = root / "generation"
    generation.mkdir(parents=True, exist_ok=True)
    _link_path(
        reuse / "generation/current_source_protected",
        generation / "current_continuous",
    )
    _link_path(reuse / "generation/reentry_only", generation / "one_shot")
    trajectory = root / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    for name in (
        "target_poses.npy",
        "yaw_pitch_roll.npy",
        "phase_labels.json",
        "trajectory_manifest.json",
        "pose_validation.json",
    ):
        _link_path(case / name, trajectory / name)


def _commands(
    *,
    python: str,
    repo: Path,
    case: Path,
    root: Path,
    anchor_chunk: int,
    observation_start_chunk: int,
    seed: int,
) -> dict[str, list[str]]:
    common = dict(
        python=python,
        repo=repo,
        case=case,
        root=root,
        anchor_chunk=anchor_chunk,
        observation_start_chunk=observation_start_chunk,
        seed=seed,
        edge_safe=False,
        final_step_stabilized=False,
    )
    return {
        "episode_continuous": _memory_command(
            name="episode_continuous",
            view_adaptive=False,
            refresh_policy="episode_continuous",
            **common,
        ),
        "per_surface_ttl": _memory_command(
            name="per_surface_ttl",
            view_adaptive=False,
            refresh_policy="per_surface_ttl",
            refresh_ttl_blocks=2,
            **common,
        ),
        "same_surface_adaptive": _memory_command(
            name="same_surface_adaptive",
            view_adaptive=True,
            same_surface_source=True,
            refresh_policy="episode_continuous",
            **common,
        ),
        "edge_safe": _memory_command(
            name="edge_safe",
            view_adaptive=False,
            refresh_policy="episode_continuous",
            edge_safe=True,
            final_step_stabilized=False,
            **{
                key: value
                for key, value in common.items()
                if key not in {"edge_safe", "final_step_stabilized"}
            },
        ),
        "final_step": _memory_command(
            name="final_step",
            view_adaptive=False,
            refresh_policy="episode_continuous",
            edge_safe=True,
            final_step_stabilized=True,
            **{
                key: value
                for key, value in common.items()
                if key not in {"edge_safe", "final_step_stabilized"}
            },
        ),
    }


def _run_method(
    name: str,
    command: list[str],
    *,
    root: Path,
    env: dict[str, str],
) -> None:
    output = root / "generation" / name
    if (output / "run_metadata.json").exists():
        print(f"[MapKV] reuse {name}")
        return
    _run(name, command, root, env)


def _report_videos(
    *,
    root: Path,
    phases: dict[str, dict],
    fps: float = 24.0,
) -> None:
    original = root / "videos/original"
    report = root / "videos/report"
    original.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    departure_start = int(phases["B1_hold"]["rgb_start"])
    departure_stop = int(phases["B1_to_Leave"]["rgb_stop_exclusive"])
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
                report / f"departure_{method}.mp4",
                departure_start / fps,
                (departure_stop - departure_start) / fps,
            ),
            (
                report / f"reentry_{method}.mp4",
                reentry_start / fps,
                (reentry_stop - reentry_start) / fps,
            ),
        )
        for destination, start, duration in outputs:
            if destination.exists():
                continue
            _transcode(
                source=source,
                destination=destination,
                start_seconds=start,
                duration_seconds=duration,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MapKV re-entry continuous/per-surface refresh stage"
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--case_dir", default="artifacts/control/yaw45m20to35_scene01"
    )
    parser.add_argument(
        "--reuse_root",
        default=(
            "results/mapkv_fast/"
            "yaw45m20to35_scene01_seed0_reentry_refinement"
        ),
    )
    parser.add_argument(
        "--output_root",
        default=(
            "results/mapkv_fast/"
            "yaw45m20to35_scene01_seed0_reentry_refresh"
        ),
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
    manifest = _json(case / "trajectory_manifest.json")
    _, phases = _phases(case)
    anchor_chunk = int(manifest["source_chunk"])
    observation_start = int(phases["B1_hold"]["start_block"])
    _prepare(root, reuse, case)
    commands = _commands(
        python=args.inspatio_python,
        repo=repo,
        case=case,
        root=root,
        anchor_chunk=anchor_chunk,
        observation_start_chunk=observation_start,
        seed=args.seed,
    )
    contract = {
        "focus_zh": "回访 episode 持续刷新 → per-surface TTL → same-surface 选源",
        "canonical_identity_reference_chunk": anchor_chunk,
        "absence_blocks": 2,
        "per_surface_ttl_blocks": 2,
        "quality_path": "Source-Protected RGB-Warp→Wan VAE→Virtual Recent",
        "commands": commands,
    }
    (root / "run_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env = _environment(repo, args.gpu)
    if args.stage in {"prepare", "full"}:
        print("[MapKV] reused baseline/current/one-shot/CUT3R/surfel artifacts")
    if args.stage == "full":
        _run_method(
            "episode_continuous",
            commands["episode_continuous"],
            root=root,
            env=env,
        )
        result = evaluate_reentry_refresh(run_root=root, case_dir=case)
        if not result["decisions"]["priority1_episode_continuous_works"]:
            _report_videos(root=root, phases=phases)
            print(build_report(root))
            return
        _run_method(
            "per_surface_ttl",
            commands["per_surface_ttl"],
            root=root,
            env=env,
        )
        evaluate_reentry_refresh(run_root=root, case_dir=case)
        _run_method(
            "same_surface_adaptive",
            commands["same_surface_adaptive"],
            root=root,
            env=env,
        )
        evaluate_reentry_refresh(run_root=root, case_dir=case)
        for name in ("edge_safe", "final_step"):
            _run_method(name, commands[name], root=root, env=env)
        evaluate_reentry_refresh(run_root=root, case_dir=case)
        _report_videos(root=root, phases=phases)
        print(build_report(root))
        return
    if args.stage == "priority1":
        _run_method(
            "episode_continuous",
            commands["episode_continuous"],
            root=root,
            env=env,
        )
    if args.stage == "priority2":
        _run_method(
            "per_surface_ttl",
            commands["per_surface_ttl"],
            root=root,
            env=env,
        )
    if args.stage == "priority3":
        _run_method(
            "same_surface_adaptive",
            commands["same_surface_adaptive"],
            root=root,
            env=env,
        )
    if args.stage == "priority4":
        for name in ("edge_safe", "final_step"):
            _run_method(name, commands[name], root=root, env=env)
    if args.stage == "evaluate":
        result = evaluate_reentry_refresh(run_root=root, case_dir=case)
        _report_videos(root=root, phases=phases)
        print(json.dumps(result["decisions"], ensure_ascii=False))
    if args.stage == "report":
        if not (root / "metrics.json").exists():
            evaluate_reentry_refresh(run_root=root, case_dir=case)
        _report_videos(root=root, phases=phases)
        print(build_report(root))


if __name__ == "__main__":
    main()
