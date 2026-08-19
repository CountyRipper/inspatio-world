from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .fast_pipeline import _run
from .geometry_gate import evaluate_geometry_gate
from .geometry_repair_report import build_report
from .reentry_refinement_stage import _link_path
from .surfel_rgb_options import generate_options


STAGES = {
    "prepare",
    "cut3r",
    "surfel",
    "retrieval",
    "evaluate",
    "report",
    "full",
}


def _prepare(root: Path, legacy: Path, case: Path) -> None:
    _link_path(legacy / "baseline", root / "baseline")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair and gate MapKV fixed-pose surfel geometry"
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument(
        "--case_dir", default="artifacts/control/yaw45m20to35_scene01"
    )
    parser.add_argument(
        "--legacy_root",
        default=(
            "results/mapkv_fast/"
            "yaw45m20to35_scene01_seed0_reentry_refresh"
        ),
    )
    parser.add_argument(
        "--output_root",
        default=(
            "results/mapkv_fast/"
            "yaw45m20to35_scene01_seed0_geometry_repair"
        ),
    )
    parser.add_argument(
        "--translation_root",
        default="results/mapkv_fast/translate008_scene01_seed0_geometry",
    )
    parser.add_argument(
        "--cut3r_python",
        default="third_party/mapkv_cut3r_env/bin/python",
    )
    parser.add_argument(
        "--inspatio_python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--niter_initial", type=int, default=50)
    parser.add_argument("--niter_incremental", type=int, default=10)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    case = Path(args.case_dir).resolve()
    legacy = Path(args.legacy_root).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (case / "trajectory_manifest.json").read_text(encoding="utf-8")
    )
    source_chunk = int(manifest["source_chunk"])
    target_chunk = int(manifest["target_chunk"])
    prefix_target = target_chunk - 1
    _prepare(root, legacy, case)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    cut3r_root = repo / "third_party/CUT3R"
    checkpoint = cut3r_root / "src/cut3r_512_dpt_4_64.pth"
    cut3r_python = Path(args.cut3r_python)
    if not cut3r_python.is_absolute():
        cut3r_python = repo / cut3r_python
    commands = {
        "cut3r": [
            str(cut3r_python.absolute()),
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
            str(root / "cut3r"),
            "--target_chunk",
            str(prefix_target),
            "--query_source_chunk",
            str(source_chunk),
            "--query_pose_mode",
            "known_target_pose",
            "--query_target_chunk",
            str(target_chunk),
            "--alignment_mode",
            "fixed_global_incremental",
            "--known_intrinsics",
            str(case / "intrinsics.txt"),
            "--alignment_niter_initial",
            str(args.niter_initial),
            "--alignment_niter_incremental",
            str(args.niter_incremental),
            "--alignment_lr",
            "0.01",
            "--confidence_threshold",
            "0",
            "--device",
            args.device,
        ],
        "surfel": [
            args.inspatio_python,
            "-m",
            "mapkv.surfel_index",
            "--sequence",
            str(root / "cut3r/sequence.json"),
            "--output_dir",
            str(root / "surfel"),
            "--confidence_threshold",
            "0",
            "--confidence_keep_quantile",
            "0.35",
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
            "--max_reprojection_error_pixels",
            "12",
            "--min_stable_observations",
            "3",
            "--reference_mask_root",
            str(root / "baseline/masks"),
        ],
        "retrieval": [
            args.inspatio_python,
            "-m",
            "mapkv.retrieval",
            "--sequence",
            str(root / "cut3r/sequence.json"),
            "--surfel_index",
            str(root / "surfel/surfel_index.npz"),
            "--output_dir",
            str(root / "retrieval"),
            "--target_chunk",
            str(target_chunk),
            "--positive_chunks",
            "8",
            "9",
            "10",
            "11",
            "--top_k",
            "1",
            "--min_history_gap_chunks",
            "2",
            "--candidate_chunks",
            *[str(chunk) for chunk in range(20)],
        ],
    }
    (root / "run_contract.json").write_text(
        json.dumps(
            {
                "focus_zh": "真实 fixed-pose alignment 与 stable 3D address Gate",
                "generation_blocked_until_gate_pass": True,
                "commands": commands,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.stage in {"prepare", "full"}:
        print("[MapKV] reused deterministic baseline; generation remains blocked")
    if args.stage in {"cut3r", "full"}:
        if not (root / "cut3r/sequence.json").exists():
            _run("fixed_global_cut3r", commands["cut3r"], root, env)
        else:
            print("[MapKV] reuse fixed-global CUT3R")
    if args.stage in {"surfel", "full"}:
        if not (root / "cut3r/sequence.json").exists():
            raise FileNotFoundError("Run cut3r stage first")
        _run("stable_surfel_v5", commands["surfel"], root, env)
    if args.stage in {"retrieval", "full"}:
        if not (root / "surfel/surfel_index.npz").exists():
            raise FileNotFoundError("Run surfel stage first")
        _run("stable_retrieval", commands["retrieval"], root, env)
    if args.stage in {"evaluate", "full"}:
        result = evaluate_geometry_gate(
            run_root=root,
            legacy_root=legacy,
            case_dir=case,
        )
        print(json.dumps(result["checks"], indent=2))
        print(result["status"])
    translation = Path(args.translation_root).resolve()
    if args.stage == "report" or (
        args.stage == "full"
        and (translation / "translation_depth_gate.json").exists()
    ):
        for geometry_root in (root, translation):
            options = geometry_root / "surfel_rgb_options"
            generate_options(
                index_path=geometry_root / "surfel/surfel_index.npz",
                sequence_path=geometry_root / "cut3r/sequence.json",
                block_mapping_path=(
                    geometry_root / "baseline/block_mapping.json"
                ),
                target_chunk=37,
                source_chunk=9,
                output_dir=options,
                generated_only_source=True,
                stable_only_world=True,
            )
        print(build_report(yaw_root=root, translation_root=translation))


if __name__ == "__main__":
    main()
