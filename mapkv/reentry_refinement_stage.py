from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .fast_pipeline import _base_inference_command, _json, _link, _run
from .reentry_refinement_evaluation import (
    METHOD_ROOTS,
    evaluate_reentry_refinement,
)
from .reentry_refinement_report import build_report
from .source_protected_stage import _asset_root, _environment
from .surfel_rgb_options import generate_options


STAGES = {"prepare", "surfel", "generation", "evaluate", "report", "full"}


def _link_path(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        return
    destination.symlink_to(
        os.path.relpath(source, destination.parent.resolve()),
        target_is_directory=source.is_dir(),
    )


def _phases(case: Path) -> tuple[dict, dict[str, dict]]:
    manifest = _json(case / "trajectory_manifest.json")
    payload = _json(case / "phase_labels.json")
    return manifest, {item["name"]: item for item in payload["phases"]}


def _prepare_reuse(root: Path, reuse: Path, case: Path) -> None:
    for name in ("baseline", "cut3r"):
        _link_path(reuse / name, root / name)
    (root / "kv").mkdir(parents=True, exist_ok=True)
    _link(reuse / "kv/noise_bundle.pt", root / "kv/noise_bundle.pt")
    (root / "generation").mkdir(parents=True, exist_ok=True)
    _link_path(
        reuse / "generation/source_protected_rgb_wre",
        root / "generation/current_source_protected",
    )
    trajectory = root / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    for name in (
        "target_poses.npy",
        "yaw_pitch_roll.npy",
        "phase_labels.json",
        "trajectory_manifest.json",
        "pose_validation.json",
    ):
        _link(case / name, trajectory / name)


def _build_surfel(
    *,
    python: str,
    root: Path,
    env: dict[str, str],
) -> None:
    output = root / "surfel"
    if (output / "surfel_index.npz").exists():
        print("[MapKV] reuse versioned surfel index")
        return
    _run(
        "surfel_observation_metadata_v4",
        [
            python,
            "-m",
            "mapkv.surfel_index",
            "--sequence",
            str(root / "cut3r/sequence.json"),
            "--output_dir",
            str(output),
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
            str(root / "baseline/masks"),
        ],
        root,
        env,
    )


def _memory_command(
    *,
    name: str,
    python: str,
    repo: Path,
    case: Path,
    root: Path,
    anchor_chunk: int,
    observation_start_chunk: int,
    seed: int,
    view_adaptive: bool,
    edge_safe: bool,
    final_step_stabilized: bool,
) -> list[str]:
    command = _base_inference_command(
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
        "--run_name",
        name,
        "--mode",
        "oracle",
        "--source_chunk",
        str(anchor_chunk),
        "--selected_steps",
        *(
            ["0", "1", "2"]
            if final_step_stabilized
            else ["0", "1", "2", "3"]
        ),
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
        "edge_safe" if edge_safe else "source_protected",
        "--continuous_mask_policy",
        "strong_core",
        "--warp_history_representation",
        "rgb_warp_vae",
        "--warp_source_latents",
        str(root / "baseline/pred_latents.pt"),
        "--warp_intrinsics_path",
        str(case / "intrinsics.txt"),
        "--warp_surfel_index",
        str(root / "surfel/surfel_index.npz"),
        "--warp_surfel_sequence",
        str(root / "cut3r/sequence.json"),
        "--warp_memory_dilation_kernel",
        "3",
        "--warp_query_feather_kernel",
        "3",
        "--source_protected_memory",
        "--warp_reference_protection_kernel",
        "3",
        "--warp_generated_only_threshold",
        "0.75" if edge_safe else "0.5",
        "--reentry_memory",
        "--reentry_observation_start_chunk",
        str(observation_start_chunk),
        "--reentry_absent_blocks",
        "2",
        "--reentry_warp_valid_erosion_kernel",
        "3",
        "--compare_latents_to",
        str(root / "baseline/pred_latents.pt"),
    ]
    if view_adaptive:
        command.append("--reentry_view_adaptive_source")
    if edge_safe:
        command.append("--reentry_edge_safe")
    return command


def _transcode(
    *,
    source: Path,
    destination: Path,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> None:
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    if start_seconds is not None:
        command += ["-ss", str(start_seconds)]
    command += ["-i", str(source)]
    if duration_seconds is not None:
        command += ["-t", str(duration_seconds)]
    command += [
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
        str(destination),
    ]
    subprocess.run(command, check=True)


def _report_videos(root: Path, phases: dict[str, dict], fps: float = 24.0) -> None:
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
        _link(source, original / f"{method}.mp4")
        _transcode(
            source=source,
            destination=report / f"full_revisit_{method}.mp4",
        )
        _transcode(
            source=source,
            destination=report / f"departure_{method}.mp4",
            start_seconds=departure_start / fps,
            duration_seconds=(departure_stop - departure_start) / fps,
        )
        _transcode(
            source=source,
            destination=report / f"reentry_{method}.mp4",
            start_seconds=reentry_start / fps,
            duration_seconds=(reentry_stop - reentry_start) / fps,
        )
    review = {
        "baseline": "Baseline",
        "current_source_protected": "Current continuous",
        "reentry_only": "E2 Reentry-only",
        "edge_safe": "E4 Edge-safe",
        "final_step": "E5 Steps012",
    }
    assets = root / "assets/reentry_refinement"
    sample_sets = {
        "departure": np.rint(
            np.linspace(departure_start, departure_stop - 1, 9)
        ).astype(int),
        "reentry": np.rint(
            np.linspace(reentry_start, reentry_stop - 1, 9)
        ).astype(int),
    }
    for window, frames in sample_sets.items():
        strips = []
        expression = "+".join(f"eq(n\\,{frame})" for frame in frames)
        for method, label in review.items():
            source = root / METHOD_ROOTS[method] / "pred.mp4"
            strip = assets / f"{window}_strip_{method}.jpg"
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
            strips.append((label, strip))
        montage = []
        for label, strip in strips:
            montage += ["-label", label, str(strip)]
        subprocess.run(
            [
                "montage",
                *montage,
                "-tile",
                "1x5",
                "-geometry",
                "+0+24",
                "-background",
                "white",
                str(assets / f"{window}_review.jpg"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "convert",
                str(assets / f"{window}_review.jpg"),
                "-resize",
                "1500x",
                "-quality",
                "83",
                str(assets / f"{window}_review_small.jpg"),
            ],
            check=True,
        )
    target_chunk = (
        int(phases["B2_hold"]["start_block"])
        + int(phases["B2_hold"]["stop_block_exclusive"])
    ) // 2
    b2_items = [
        (
            "Selected history -> B2",
            assets / "selected_observation_warped_to_b2.png",
        )
    ] + [
        (
            label,
            root
            / METHOD_ROOTS[method]
            / "keyframes"
            / f"chunk_{target_chunk:04d}.png",
        )
        for method, label in review.items()
    ]
    montage = []
    for label, path in b2_items:
        montage += ["-label", label, str(path)]
    subprocess.run(
        [
            "montage",
            *montage,
            "-tile",
            "2x3",
            "-geometry",
            "520x300+10+28",
            "-background",
            "white",
            str(assets / "b2_full_review.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "convert",
            str(assets / "b2_full_review.jpg"),
            "-resize",
            "1400x",
            "-quality",
            "85",
            str(assets / "b2_full_review_small.jpg"),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MapKV re-entry lifecycle / view / edge refinement"
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
            "yaw45m20to35_scene01_seed0_source_protected"
        ),
    )
    parser.add_argument(
        "--output_root",
        default=(
            "results/mapkv_fast/"
            "yaw45m20to35_scene01_seed0_reentry_refinement"
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
    manifest, phases = _phases(case)
    if (
        float(manifest["b1_theta_degrees"]) != 45.0
        or float(manifest["leave_theta_degrees"]) != -20.0
        or float(manifest["b2_theta_degrees"]) != 35.0
    ):
        raise RuntimeError("Expected exact 0→45→-20→35 benchmark")
    anchor_chunk = int(manifest["source_chunk"])
    observation_start = int(phases["B1_hold"]["start_block"])
    env = _environment(repo, args.gpu)
    _prepare_reuse(root, reuse, case)
    commands = {
        "reentry_only": _memory_command(
            name="reentry_only",
            python=args.inspatio_python,
            repo=repo,
            case=case,
            root=root,
            anchor_chunk=anchor_chunk,
            observation_start_chunk=observation_start,
            seed=args.seed,
            view_adaptive=False,
            edge_safe=False,
            final_step_stabilized=False,
        ),
        "view_adaptive": _memory_command(
            name="view_adaptive",
            python=args.inspatio_python,
            repo=repo,
            case=case,
            root=root,
            anchor_chunk=anchor_chunk,
            observation_start_chunk=observation_start,
            seed=args.seed,
            view_adaptive=True,
            edge_safe=False,
            final_step_stabilized=False,
        ),
        "edge_safe": _memory_command(
            name="edge_safe",
            python=args.inspatio_python,
            repo=repo,
            case=case,
            root=root,
            anchor_chunk=anchor_chunk,
            observation_start_chunk=observation_start,
            seed=args.seed,
            view_adaptive=True,
            edge_safe=True,
            final_step_stabilized=False,
        ),
        "final_step": _memory_command(
            name="final_step",
            python=args.inspatio_python,
            repo=repo,
            case=case,
            root=root,
            anchor_chunk=anchor_chunk,
            observation_start_chunk=observation_start,
            seed=args.seed,
            view_adaptive=True,
            edge_safe=True,
            final_step_stabilized=True,
        ),
    }
    contract = {
        "focus_zh": "回访记忆生命周期、视角自适应 observation 与 FOV 边缘安全",
        "trajectory": "0°→+45° hold→−20° hold→+35° hold",
        "anchor_surface_group_chunk": anchor_chunk,
        "observation_start_chunk": observation_start,
        "absent_blocks": 2,
        "quality_path": "RGB warp→Wan VAE→Virtual Recent→native writer",
        "frozen": [
            "known-pose CUT3R",
            "radius-normal surfel geometry",
            "replace_recent_delta",
            "all 30 layers",
            "alpha=1",
        ],
        "commands": commands,
    }
    (root / "run_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.stage in {"prepare", "full"}:
        print("[MapKV] baseline/CUT3R/current Source-Protected artifacts reused")
    if args.stage in {"surfel", "full"}:
        _build_surfel(python=args.inspatio_python, root=root, env=env)
    if args.stage in {"generation", "full"}:
        if not (root / "surfel/surfel_index.npz").exists():
            raise FileNotFoundError("Run surfel stage first")
        for name in ("reentry_only", "view_adaptive", "edge_safe", "final_step"):
            output = root / "generation" / name
            if not (output / "run_metadata.json").exists():
                _run(name, commands[name], root, env)
            else:
                print(f"[MapKV] reuse {name}")
    if args.stage in {"evaluate", "full"}:
        result = evaluate_reentry_refinement(run_root=root, case_dir=case)
        _report_videos(root, phases)
        selected = int(result["trajectory"]["selected_source_chunk"])
        options = root / "surfel_rgb_options"
        if not (options / "report.html").exists():
            generate_options(
                index_path=root / "surfel/surfel_index.npz",
                sequence_path=root / "cut3r/sequence.json",
                block_mapping_path=root / "baseline/block_mapping.json",
                target_chunk=int(manifest["target_chunk"]),
                source_chunk=selected,
                output_dir=options,
                generated_only_source=True,
            )
        print(json.dumps({"statuses": result["statuses"]}, ensure_ascii=False))
    if args.stage in {"report", "full"}:
        print(build_report(root))


if __name__ == "__main__":
    main()
