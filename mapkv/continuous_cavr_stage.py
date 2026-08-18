from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .continuous_cavr_evaluation import evaluate_continuous_cavr
from .continuous_cavr_report import build_report
from .fast_pipeline import _base_inference_command, _json, _link, _run


STAGES = {"generation", "evaluate", "report", "full"}


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
    fps: float = 24.0,
) -> None:
    methods = ("baseline", "block_on_wre", "continuous_cavr")
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
        "continuous_cavr": root / "generation" / "continuous_cavr",
    }
    strip_paths = []
    for method in methods:
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
        poster = (
            run_root / "keyframes" / f"chunk_{target_chunk:04d}.png"
        )
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
    subprocess.run(
        [
            "montage",
            *[str(path) for path in strip_paths],
            "-tile",
            "1x3",
            "-geometry",
            "+0+18",
            "-background",
            "white",
            str(root / "assets" / "cavr_transition_filmstrip.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "convert",
            str(root / "assets" / "cavr_transition_filmstrip.jpg"),
            "-resize",
            "1400x",
            str(root / "assets" / "cavr_transition_filmstrip_small.jpg"),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous Geometry-Reprojected Virtual Recent stage"
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
        "--output_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_continuous_cavr",
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
    output = root / "generation" / "continuous_cavr"
    command = _base_inference_command(
        python=args.inspatio_python,
        repo=repo,
        asset_root=_asset_root(repo),
        case_dir=case_dir,
        output_dir=output,
        noise_bundle=geometry / "kv" / "noise_bundle.pt",
        bank_root=geometry / "kv" / "recent_bank_all",
        seed=args.seed,
        memory_layers="all",
    ) + [
        "--run_name",
        "continuous_cavr",
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
        "--warp_source_latents",
        str(root / "baseline" / "pred_latents.pt"),
        "--warp_intrinsics_path",
        str(case_dir / "intrinsics.txt"),
        "--warp_surfel_index",
        str(root / "surfel" / "surfel_index.npz"),
        "--warp_surfel_sequence",
        str(root / "cut3r" / "sequence.json"),
        "--warp_min_history_gap",
        str(args.min_history_gap),
        "--warp_feather_kernel",
        str(args.feather_kernel),
        "--compare_latents_to",
        str(root / "baseline" / "pred_latents.pt"),
    ]
    contract = {
        "architecture": (
            "per-block source-surfel visibility -> warp historical and "
            "short-term recent -> Virtual Recent -> native t=0 writer -> "
            "replace_recent_delta"
        ),
        "source_chunk": source_chunk,
        "fixed_target_chunks": False,
        "evaluation_target_chunks": target_chunks,
        "activation": "projected visible source-chunk surfels",
        "alpha_schedule": None,
        "command": command,
    }
    (root / "run_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )

    if args.stage in {"generation", "full"}:
        if not (output / "run_metadata.json").exists():
            _run("continuous_cavr", command, root, environment)
        else:
            print("[MapKV] reuse Continuous CAVR generation")
    if args.stage in {"evaluate", "full"}:
        result = evaluate_continuous_cavr(
            run_root=root,
            case_dir=case_dir,
            source_chunk=source_chunk,
            target_chunks=tuple(target_chunks),
        )
        _make_report_videos(
            root=root,
            phase_payload=phase_payload,
            target_chunk=max(target_chunks),
        )
        print(json.dumps({"status": result["status"]}, indent=2))
    if args.stage in {"report", "full"}:
        print(build_report(root))


if __name__ == "__main__":
    main()
