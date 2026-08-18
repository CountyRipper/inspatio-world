from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .continuous_cavr_evaluation import evaluate_continuous_cavr
from .continuous_cavr_report import build_report
from .fast_pipeline import _base_inference_command, _json, _link, _run
from .surfel_rgb_options import generate_options


STAGES = {"generation", "evaluate", "report", "full"}
METHODS = (
    "baseline",
    "block_on_wre",
    "continuous_raw_recent",
    "masked_continuous_wre",
)


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


def _make_report_videos(
    *,
    root: Path,
    phase_payload: dict,
    target_chunk: int,
    previous_cavr: Path,
    fps: float = 24.0,
) -> None:
    report = root / "videos" / "report"
    original = root / "videos" / "original"
    posters = root / "assets" / "posters"
    strips = root / "assets" / "strips"
    for directory in (report, original, posters, strips):
        directory.mkdir(parents=True, exist_ok=True)
    ramp = next(
        item
        for item in phase_payload["phases"]
        if item["name"] == "A_to_B2"
    )
    b2 = next(
        item
        for item in phase_payload["phases"]
        if item["name"] == "B2_hold"
    )
    start_seconds = max(0.0, float(ramp["rgb_start"]) / fps)
    duration_seconds = (
        float(b2["rgb_stop_exclusive"] - ramp["rgb_start"]) / fps
    )
    method_roots = {
        "baseline": root / "baseline",
        "block_on_wre": root / "generation" / "block_on_wre",
        "continuous_raw_recent": (
            root / "generation" / "continuous_raw_recent"
        ),
        "masked_continuous_wre": (
            root / "generation" / "masked_continuous_wre"
        ),
    }
    targeted_frames = (202, 215, 225, 233, 239, 245, 255, 270)
    dense_frames = tuple(range(218, 242, 2))

    def extract_tile(source: Path, frames: tuple[int, ...], output: Path) -> None:
        expression = "+".join(f"eq(n\\,{frame})" for frame in frames)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                f"select='{expression}',scale=180:-2,tile={len(frames)}x1",
                "-frames:v",
                "1",
                str(output),
            ],
            check=True,
        )

    strip_paths = []
    for method in METHODS:
        run_root = method_roots[method]
        source = run_root / "pred.mp4"
        _link(source, original / f"{method}.mp4")
        clip = report / f"transition_window_{method}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(start_seconds),
                "-i",
                str(source),
                "-t",
                str(duration_seconds),
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
                str(clip),
            ],
            check=True,
        )
        full_video = report / f"full_revisit_{method}.mp4"
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
                str(full_video),
            ],
            check=True,
        )
        poster = run_root / "keyframes" / f"chunk_{target_chunk:04d}.png"
        subprocess.run(
            [
                "convert",
                str(poster),
                "-quality",
                "84",
                str(posters / f"{method}.jpg"),
            ],
            check=True,
        )
        strip = strips / f"{method}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(clip),
                "-vf",
                "fps=2,scale=280:-2,tile=6x1",
                "-frames:v",
                "1",
                str(strip),
            ],
            check=True,
        )
        strip_paths.append(strip)
        extract_tile(
            source, targeted_frames, root / "assets" / f"review_{method}.jpg"
        )
        extract_tile(
            source, dense_frames, root / "assets" / f"dense_{method}.jpg"
        )
    subprocess.run(
        [
            "montage",
            *[str(path) for path in strip_paths],
            "-tile",
            "1x4",
            "-geometry",
            "+0+18",
            "-background",
            "white",
            str(root / "assets" / "masked_continuous_transition.jpg"),
        ],
        check=True,
    )
    labels = {
        "baseline": "Baseline",
        "block_on_wre": "Block-on WRE",
        "continuous_raw_recent": "Continuous RawRecent",
        "masked_continuous_wre": "Masked Continuous WRE",
    }
    targeted_montage = []
    dense_montage = []
    for method in METHODS:
        targeted_montage.extend(
            ["-label", labels[method], str(root / "assets" / f"review_{method}.jpg")]
        )
        dense_montage.extend(
            ["-label", labels[method], str(root / "assets" / f"dense_{method}.jpg")]
        )
    subprocess.run(
        [
            "montage",
            *targeted_montage,
            "-tile",
            "1x4",
            "-geometry",
            "+0+22",
            "-background",
            "white",
            str(root / "assets" / "masked_continuous_targeted_review.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "montage",
            *dense_montage,
            "-tile",
            "1x4",
            "-geometry",
            "+0+22",
            "-background",
            "white",
            str(root / "assets" / "masked_continuous_dense_reentry.jpg"),
        ],
        check=True,
    )
    previous_video = (
        previous_cavr / "generation" / "continuous_cavr" / "pred.mp4"
    )
    if previous_video.exists():
        failed_tile = root / "assets" / "review_failed_cavr.jpg"
        extract_tile(previous_video, targeted_frames, failed_tile)
        subprocess.run(
            [
                "montage",
                "-label",
                "Failed CAVR (warped recent/global)",
                str(failed_tile),
                "-label",
                "Continuous RawRecent/global",
                str(root / "assets" / "review_continuous_raw_recent.jpg"),
                "-label",
                "Masked Continuous WRE",
                str(root / "assets" / "review_masked_continuous_wre.jpg"),
                "-tile",
                "1x3",
                "-geometry",
                "+0+22",
                "-background",
                "white",
                str(root / "assets" / "repair_causality_review.jpg"),
            ],
            check=True,
        )
    subprocess.run(
        [
            "convert",
            str(root / "assets" / "masked_continuous_transition.jpg"),
            "-resize",
            "1400x",
            str(
                root
                / "assets"
                / "masked_continuous_transition_small.jpg"
            ),
        ],
        check=True,
    )


def _continuous_command(
    *,
    name: str,
    query_gate: str,
    python: str,
    repo: Path,
    asset_root: Path,
    case_dir: Path,
    output: Path,
    geometry: Path,
    baseline: Path,
    seed: int,
    source_chunk: int,
    min_history_gap: int,
    feather_kernel: int,
) -> list[str]:
    return _base_inference_command(
        python=python,
        repo=repo,
        asset_root=asset_root,
        case_dir=case_dir,
        output_dir=output,
        noise_bundle=geometry / "kv" / "noise_bundle.pt",
        bank_root=geometry / "kv" / "recent_bank_all",
        seed=seed,
        memory_layers="all",
    ) + [
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
        query_gate,
        "--warp_source_latents",
        str(baseline / "pred_latents.pt"),
        "--warp_intrinsics_path",
        str(case_dir / "intrinsics.txt"),
        "--warp_surfel_index",
        str(geometry / "surfel" / "surfel_index.npz"),
        "--warp_surfel_sequence",
        str(geometry / "cut3r" / "sequence.json"),
        "--warp_min_history_gap",
        str(min_history_gap),
        "--warp_feather_kernel",
        str(feather_kernel),
        "--compare_latents_to",
        str(baseline / "pred_latents.pt"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Masked Continuous Warp-Reencode Recent stage"
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--case_dir", default="artifacts/control/yaw30to20_scene01"
    )
    parser.add_argument(
        "--geometry_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_locality",
    )
    parser.add_argument(
        "--wre_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_warp_reencode",
    )
    parser.add_argument(
        "--previous_cavr_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_continuous_cavr",
    )
    parser.add_argument(
        "--output_root",
        default=(
            "results/mapkv_fast/"
            "yaw30to20_scene01_seed0_masked_continuous_wre"
        ),
    )
    parser.add_argument(
        "--inspatio_python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    parser.add_argument("--feather_kernel", type=int, default=3)
    parser.add_argument("--min_history_gap", type=int, default=2)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case_dir).resolve()
    geometry = Path(args.geometry_root).resolve()
    wre = Path(args.wre_root).resolve()
    previous_cavr = Path(args.previous_cavr_root).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "generation").mkdir(parents=True, exist_ok=True)
    source_chunk = 8
    target_chunks = [21, 22]
    manifest = _json(case_dir / "trajectory_manifest.json")
    phase_payload = _json(case_dir / "phase_labels.json")
    if int(manifest["source_chunk"]) != source_chunk:
        raise RuntimeError("Controlled partial case no longer declares B1 chunk 8")

    _link(wre / "baseline", root / "baseline")
    _link(
        wre / "generation" / "warp_reencode",
        root / "generation" / "block_on_wre",
    )
    _link(geometry / "surfel", root / "surfel")
    _link(geometry / "retrieval", root / "retrieval")
    _link(geometry / "cut3r", root / "cut3r")

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(repo)
            + (
                ":" + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            ),
        }
    )
    commands = {}
    for name, query_gate in (
        ("continuous_raw_recent", "global"),
        ("masked_continuous_wre", "surfel"),
    ):
        output = root / "generation" / name
        commands[name] = _continuous_command(
            name=name,
            query_gate=query_gate,
            python=args.inspatio_python,
            repo=repo,
            asset_root=_asset_root(repo),
            case_dir=case_dir,
            output=output,
            geometry=geometry,
            baseline=root / "baseline",
            seed=args.seed,
            source_chunk=source_chunk,
            min_history_gap=args.min_history_gap,
            feather_kernel=args.feather_kernel,
        )
    contract = {
        "architecture": {
            "continuous_raw_recent": (
                "M_history * warp(B1->camera_t) + "
                "(1-M_history) * raw last_pred -> native writer -> global delta"
            ),
            "masked_continuous_wre": (
                "same Virtual Recent, then M_query * (A_virtual-A_base)"
            ),
        },
        "source_chunk": source_chunk,
        "fixed_target_chunks": False,
        "evaluation_target_chunks": target_chunks,
        "activation": "projected visible source-chunk surfels",
        "short_term_recent_warped": False,
        "same_mask_for_latent_and_attention": True,
        "alpha_schedule": None,
        "previous_failed_cavr_metrics": str(previous_cavr / "metrics.json"),
        "commands": commands,
    }
    (root / "run_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )

    if args.stage in {"generation", "full"}:
        for name, command in commands.items():
            output = root / "generation" / name
            if not (output / "run_metadata.json").exists():
                _run(name, command, root, environment)
            else:
                print(f"[MapKV] reuse {name} generation")
    if args.stage in {"evaluate", "full"}:
        result = evaluate_continuous_cavr(
            run_root=root,
            case_dir=case_dir,
            previous_cavr_root=previous_cavr,
            source_chunk=source_chunk,
            target_chunks=tuple(target_chunks),
        )
        _make_report_videos(
            root=root,
            phase_payload=phase_payload,
            target_chunk=max(target_chunks),
            previous_cavr=previous_cavr,
        )
        rgb_options = root / "surfel_rgb_options"
        if not (rgb_options / "report.html").exists():
            generate_options(
                index_path=root / "surfel" / "surfel_index.npz",
                sequence_path=root / "cut3r" / "sequence.json",
                block_mapping_path=root / "baseline" / "block_mapping.json",
                target_chunk=max(target_chunks),
                source_chunk=source_chunk,
                output_dir=rgb_options,
            )
        print(json.dumps({"status": result["status"]}, indent=2))
    if args.stage in {"report", "full"}:
        print(build_report(root))


if __name__ == "__main__":
    main()
