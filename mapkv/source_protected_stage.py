from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from .fast_pipeline import _base_inference_command, _json, _link, _run
from .source_protected_evaluation import (
    METHOD_ROOTS,
    evaluate_source_protected_revisit,
)
from .source_protected_report import build_report
from .surfel_rgb_options import generate_options


STAGES = {
    "baseline",
    "geometry",
    "generation",
    "evaluate",
    "report",
    "full",
}


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


def _environment(repo: Path, gpu: str) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(repo)
            + ((":" + value["PYTHONPATH"]) if value.get("PYTHONPATH") else ""),
        }
    )
    return value


def _all_steps(repo: Path) -> list[int]:
    config = yaml.safe_load(
        (repo / "configs/inference_1.3b.yaml").read_text(encoding="utf-8")
    )
    return list(range(len(config["denoising_step_list"])))


def _phases(case_dir: Path) -> tuple[dict, dict, list[int]]:
    manifest = _json(case_dir / "trajectory_manifest.json")
    payload = _json(case_dir / "phase_labels.json")
    phases = {item["name"]: item for item in payload["phases"]}
    targets = list(
        range(
            int(phases["B2_hold"]["start_block"]),
            int(phases["B2_hold"]["stop_block_exclusive"]),
        )
    )
    return manifest, phases, targets


def _baseline_command(
    *,
    python: str,
    repo: Path,
    case_dir: Path,
    output: Path,
    noise: Path,
    seed: int,
) -> list[str]:
    return _base_inference_command(
        python=python,
        repo=repo,
        asset_root=_asset_root(repo),
        case_dir=case_dir,
        output_dir=output,
        noise_bundle=noise,
        bank_root=output.parent / "kv/unused",
        seed=seed,
        memory_layers="all",
    ) + [
        "--run_name",
        "source_protected_baseline",
        "--mode",
        "baseline",
        "--create_noise_bundle",
        "--verify_memory_off_replay",
        "--require_replay_tolerance",
    ]


def _build_geometry(
    *,
    repo: Path,
    root: Path,
    baseline: Path,
    source_chunk: int,
    first_target: int,
    inspatio_python: str,
    cut3r_python: str,
    env: dict[str, str],
) -> None:
    cut3r_root = repo / "third_party/CUT3R"
    checkpoint = cut3r_root / "src/cut3r_512_dpt_4_64.pth"
    cut3r_output = root / "cut3r"
    if not (cut3r_output / "sequence.json").exists():
        _run(
            "cut3r",
            [
                cut3r_python,
                "-m",
                "mapkv.cut3r_adapter",
                "--baseline_root",
                str(baseline),
                "--block_mapping",
                str(baseline / "block_mapping.json"),
                "--cut3r_root",
                str(cut3r_root),
                "--checkpoint",
                str(checkpoint),
                "--output_dir",
                str(cut3r_output),
                "--target_chunk",
                str(first_target),
                "--query_source_chunk",
                str(source_chunk),
                "--query_pose_mode",
                "known_target_pose",
                "--query_target_chunk",
                str(first_target),
                "--confidence_threshold",
                "1.5",
                "--device",
                "cuda",
            ],
            root,
            env,
        )
    surfel = root / "surfel"
    if not (surfel / "surfel_index.npz").exists():
        _run(
            "surfel_generated_only_tags",
            [
                inspatio_python,
                "-m",
                "mapkv.surfel_index",
                "--sequence",
                str(cut3r_output / "sequence.json"),
                "--output_dir",
                str(surfel),
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
                "--reference_mask_root",
                str(baseline / "masks"),
            ],
            root,
            env,
        )


def _memory_command(
    *,
    name: str,
    source_protected: bool,
    middle10: bool,
    python: str,
    repo: Path,
    case_dir: Path,
    output: Path,
    root: Path,
    source_chunk: int,
    seed: int,
) -> list[str]:
    memory_mode = "explicit" if middle10 else "all"
    command = _base_inference_command(
        python=python,
        repo=repo,
        asset_root=_asset_root(repo),
        case_dir=case_dir,
        output_dir=output,
        noise_bundle=root / "kv/noise_bundle.pt",
        bank_root=root / "kv/unused",
        seed=seed,
        memory_layers=memory_mode,
    )
    if middle10:
        command += [
            "--selected_layers",
            *[str(layer) for layer in range(10, 20)],
        ]
    command += [
        "--run_name",
        name,
        "--mode",
        "oracle",
        "--source_chunk",
        str(source_chunk),
        "--selected_steps",
        "0",
        "1",
        "2",
        "3",
        "--alpha",
        "1",
        "--injection_mode",
        "replace_recent_delta",
        "--gate_mode",
        "global",
        "--continuous_virtual_recent",
        "--continuous_recent_fallback",
        "raw",
        "--continuous_query_gate",
        "source_protected" if source_protected else "support_preserving",
        "--continuous_mask_policy",
        "strong_core",
        "--warp_history_representation",
        "rgb_warp_vae",
        "--warp_source_latents",
        str(root / "baseline/pred_latents.pt"),
        "--warp_intrinsics_path",
        str(case_dir / "intrinsics.txt"),
        "--warp_surfel_index",
        str(root / "surfel/surfel_index.npz"),
        "--warp_surfel_sequence",
        str(root / "cut3r/sequence.json"),
        "--warp_min_history_gap",
        "2",
        "--warp_memory_dilation_kernel",
        "3",
        "--warp_query_feather_kernel",
        "3",
        "--compare_latents_to",
        str(root / "baseline/pred_latents.pt"),
    ]
    if source_protected:
        command += [
            "--source_protected_memory",
            "--warp_reference_protection_kernel",
            "3",
            "--warp_generated_only_threshold",
            "0.5",
        ]
    return command


def _report_videos(root: Path, phases: dict, *, fps: float = 24.0) -> None:
    report = root / "videos/report"
    original = root / "videos/original"
    report.mkdir(parents=True, exist_ok=True)
    original.mkdir(parents=True, exist_ok=True)
    reentry = phases["Leave_to_B2"]
    b2 = phases["B2_hold"]
    start = float(reentry["rgb_start"]) / fps
    duration = float(
        b2["rgb_stop_exclusive"] - reentry["rgb_start"]
    ) / fps
    review_frames = [
        int(
            round(
                int(reentry["rgb_start"])
                + index
                * (
                    int(b2["rgb_stop_exclusive"])
                    - int(reentry["rgb_start"])
                    - 1
                )
                / 8
            )
        )
        for index in range(9)
    ]
    strips = []
    for method, relative in METHOD_ROOTS.items():
        source = root / relative / "pred.mp4"
        _link(source, original / f"{method}.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                "scale=-2:480",
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(report / f"full_revisit_{method}.mp4"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-i",
                str(source),
                "-t",
                str(duration),
                "-vf",
                "scale=-2:480",
                "-c:v",
                "libx264",
                "-crf",
                "27",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(report / f"reentry_{method}.mp4"),
            ],
            check=True,
        )
        strip = root / "assets/source_protected" / f"strip_{method}.jpg"
        strip.parent.mkdir(parents=True, exist_ok=True)
        expression = "+".join(
            f"eq(n\\,{frame})" for frame in review_frames
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                f"select='{expression}',scale=180:-2,tile=9x1",
                "-frames:v",
                "1",
                str(strip),
            ],
            check=True,
        )
        strips.append((method, strip))
    labels = {
        "baseline": "Baseline",
        "current_rgb_wre": "Current RGB-WRE",
        "source_protected": "Source-Protected",
        "middle10": "Middle10",
    }
    montage_args = []
    for method, strip in strips:
        montage_args += ["-label", labels[method], str(strip)]
    subprocess.run(
        [
            "montage",
            *montage_args,
            "-tile",
            "1x4",
            "-geometry",
            "+0+24",
            "-background",
            "white",
            str(root / "assets/source_protected/reentry_review.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "convert",
            str(root / "assets/source_protected/reentry_review.jpg"),
            "-resize",
            "1400x",
            "-quality",
            "82",
            str(root / "assets/source_protected/reentry_review_small.jpg"),
        ],
        check=True,
    )


def _copy_trajectory(root: Path, case: Path) -> None:
    target = root / "trajectory"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "target_poses.npy",
        "yaw_pitch_roll.npy",
        "phase_labels.json",
        "trajectory_manifest.json",
        "pose_validation.json",
    ):
        shutil.copy2(case / name, target / name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source-protected generated-history revisit memory"
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--case_dir", default="artifacts/control/yaw45m20to35_scene01"
    )
    parser.add_argument(
        "--output_root",
        default=(
            "results/mapkv_fast/"
            "yaw45m20to35_scene01_seed0_source_protected"
        ),
    )
    parser.add_argument(
        "--inspatio_python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    parser.add_argument(
        "--cut3r_python",
        default=str(
            Path(__file__).resolve().parents[1]
            / "third_party/mapkv_cut3r_env/bin/python"
        ),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    case = Path(args.case_dir).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "kv").mkdir(parents=True, exist_ok=True)
    (root / "generation").mkdir(parents=True, exist_ok=True)
    manifest, phases, targets = _phases(case)
    if (
        float(manifest["b1_theta_degrees"]) != 45.0
        or float(manifest["leave_theta_degrees"]) != -20.0
        or float(manifest["b2_theta_degrees"]) != 35.0
    ):
        raise RuntimeError("Expected the exact 0→45→-20→35 control case")
    source_chunk = int(manifest["source_chunk"])
    env = _environment(repo, args.gpu)
    baseline = root / "baseline"
    commands = {
        "baseline": _baseline_command(
            python=args.inspatio_python,
            repo=repo,
            case_dir=case,
            output=baseline,
            noise=root / "kv/noise_bundle.pt",
            seed=args.seed,
        )
    }
    for name, protected, middle in (
        ("current_rgb_wre", False, False),
        ("source_protected_rgb_wre", True, False),
        ("source_protected_middle10", True, True),
    ):
        commands[name] = _memory_command(
            name=name,
            source_protected=protected,
            middle10=middle,
            python=args.inspatio_python,
            repo=repo,
            case_dir=case,
            output=root / "generation" / name,
            root=root,
            source_chunk=source_chunk,
            seed=args.seed,
        )
    contract = {
        "focus_zh": "生成历史专用记忆：source protection 与真实回访 identity",
        "trajectory": "0°→+45° hold→−20° hold→+35° hold",
        "source_chunk_fixed": source_chunk,
        "target_chunks": targets,
        "geometry_and_retrieval_parameters_frozen": True,
        "canonical_k_paused": True,
        "quality_path": "RGB warp→Wan VAE→native Recent writer",
        "source_protection": (
            "generated-only B1 observations × current reference blind"
        ),
        "alpha": 1.0,
        "steps": _all_steps(repo),
        "commands": commands,
    }
    (root / "run_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _copy_trajectory(root, case)

    if args.stage in {"baseline", "full"}:
        if not (baseline / "run_metadata.json").exists():
            _run("baseline", commands["baseline"], root, env)
        else:
            print("[MapKV] reuse baseline")
    if args.stage in {"geometry", "full"}:
        if not (baseline / "run_metadata.json").exists():
            raise FileNotFoundError("Run baseline stage before geometry")
        _build_geometry(
            repo=repo,
            root=root,
            baseline=baseline,
            source_chunk=source_chunk,
            first_target=min(targets),
            inspatio_python=args.inspatio_python,
            cut3r_python=args.cut3r_python,
            env=env,
        )
    if args.stage in {"generation", "full"}:
        for name in (
            "current_rgb_wre",
            "source_protected_rgb_wre",
            "source_protected_middle10",
        ):
            output = root / "generation" / name
            if not (output / "run_metadata.json").exists():
                _run(name, commands[name], root, env)
            else:
                print(f"[MapKV] reuse {name}")
    if args.stage in {"evaluate", "full"}:
        result = evaluate_source_protected_revisit(
            run_root=root, case_dir=case
        )
        _report_videos(root, phases)
        options = root / "surfel_rgb_options"
        if not (options / "report.html").exists():
            generate_options(
                index_path=root / "surfel/surfel_index.npz",
                sequence_path=root / "cut3r/sequence.json",
                block_mapping_path=root / "baseline/block_mapping.json",
                target_chunk=int(manifest["target_chunk"]),
                source_chunk=source_chunk,
                output_dir=options,
                generated_only_source=True,
            )
        print(json.dumps({"status": result["status"]}, ensure_ascii=False))
    if args.stage in {"report", "full"}:
        print(build_report(root))


if __name__ == "__main__":
    main()
