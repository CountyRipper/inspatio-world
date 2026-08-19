from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .fast_pipeline import _run
from .source_protected_stage import _baseline_command, _environment
from .translation_depth_gate import evaluate_translation_depth_gate


STAGES = {
    "prepare",
    "baseline",
    "cut3r",
    "surfel",
    "retrieval",
    "evaluate",
    "full",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MapKV translation depth geometry Gate"
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument(
        "--case_dir", default="artifacts/control/translate008_scene01"
    )
    parser.add_argument(
        "--output_root",
        default="results/mapkv_fast/translate008_scene01_seed0_geometry",
    )
    parser.add_argument(
        "--source_case",
        default="artifacts/control/yaw45m20to35_scene01",
    )
    parser.add_argument(
        "--inspatio_python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    parser.add_argument(
        "--cut3r_python",
        default="third_party/mapkv_cut3r_env/bin/python",
    )
    parser.add_argument("--generation_gpu", default="0")
    parser.add_argument("--cut3r_device", default="cuda:2")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    case = Path(args.case_dir).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "kv").mkdir(exist_ok=True)
    cut3r_python = Path(args.cut3r_python)
    if not cut3r_python.is_absolute():
        cut3r_python = repo / cut3r_python
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    if args.stage in {"prepare", "full"} and not (
        case / "render_offline.mp4"
    ).exists():
        _run(
            "translation_case",
            [
                args.inspatio_python,
                str(repo / "scripts/build_mapkv_translation_case.py"),
                "--source_case",
                str(Path(args.source_case).resolve()),
                "--output_case",
                str(case),
                "--translation",
                "0.08",
                "--render",
                "--render_device",
                "2",
            ],
            root,
            env,
        )
    if not (case / "trajectory_manifest.json").exists():
        raise FileNotFoundError("Build translation case first")
    manifest = json.loads(
        (case / "trajectory_manifest.json").read_text(encoding="utf-8")
    )
    source_chunk = int(manifest["source_chunk"])
    target_chunk = int(manifest["target_chunk"])
    baseline = root / "baseline"
    cut3r_root = repo / "third_party/CUT3R"
    commands = {
        "baseline": _baseline_command(
            python=args.inspatio_python,
            repo=repo,
            case_dir=case,
            output=baseline,
            noise=root / "kv/noise_bundle.pt",
            seed=0,
        ),
        "cut3r": [
            str(cut3r_python.absolute()),
            "-m",
            "mapkv.cut3r_adapter",
            "--baseline_root",
            str(baseline),
            "--block_mapping",
            str(baseline / "block_mapping.json"),
            "--cut3r_root",
            str(cut3r_root),
            "--checkpoint",
            str(cut3r_root / "src/cut3r_512_dpt_4_64.pth"),
            "--output_dir",
            str(root / "cut3r"),
            "--target_chunk",
            str(target_chunk - 1),
            "--query_source_chunk",
            str(source_chunk),
            "--query_pose_mode",
            "known_target_pose",
            "--query_target_chunk",
            str(target_chunk),
            "--alignment_mode",
            "fixed_global_joint",
            "--known_intrinsics",
            str(case / "intrinsics.txt"),
            "--alignment_niter_initial",
            "300",
            "--alignment_niter_incremental",
            "0",
            "--confidence_threshold",
            "0",
            "--device",
            args.cut3r_device,
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
            str(baseline / "masks"),
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
            "--candidate_chunks",
            *[str(chunk) for chunk in range(20)],
            "--top_k",
            "1",
            "--min_history_gap_chunks",
            "2",
            "--pose_cluster_translation_tolerance",
            "0.015",
        ],
    }
    (root / "run_contract.json").write_text(
        json.dumps(
            {
                "focus_zh": "小平移视差下的真实 depth geometry Gate",
                "generation_after_gate": False,
                "commands": commands,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if args.stage in {"baseline", "full"} and not (
        baseline / "run_metadata.json"
    ).exists():
        _run(
            "translation_baseline",
            commands["baseline"],
            root,
            _environment(repo, args.generation_gpu),
        )
    if args.stage in {"cut3r", "full"} and not (
        root / "cut3r/sequence.json"
    ).exists():
        _run("translation_fixed_joint", commands["cut3r"], root, env)
    if args.stage in {"surfel", "full"}:
        _run("translation_stable_surfel", commands["surfel"], root, env)
    if args.stage in {"retrieval", "full"}:
        _run("translation_retrieval", commands["retrieval"], root, env)
    if args.stage in {"evaluate", "full"}:
        result = evaluate_translation_depth_gate(
            run_root=root, case_dir=case
        )
        print(json.dumps(result["checks"], indent=2))
        print(result["status"])


if __name__ == "__main__":
    main()
