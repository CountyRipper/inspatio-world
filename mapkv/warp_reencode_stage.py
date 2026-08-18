from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .fast_pipeline import (
    _base_inference_command,
    _json,
    _link,
    _run,
    _transcode_videos,
)
from .warp_reencode_evaluation import evaluate_warp_reencode
from .warp_reencode_report import build_report


STAGES = {"generation", "evaluate", "report", "full"}


def _make_filmstrip(root: Path) -> None:
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    strips = []
    for method in ("baseline", "hard_recentkv", "warp_reencode"):
        source = root / "videos" / "report" / f"b2_window_{method}.mp4"
        strip = assets / f"{method}_filmstrip.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                "fps=3,scale=320:-2,tile=3x1",
                "-frames:v",
                "1",
                str(strip),
            ],
            check=True,
        )
        strips.append(strip)
    full = assets / "b2_filmstrip.jpg"
    subprocess.run(
        [
            "montage",
            *[str(path) for path in strips],
            "-tile",
            "1x3",
            "-geometry",
            "+0+20",
            "-background",
            "white",
            "-set",
            "label",
            "%f",
            str(full),
        ],
        check=True,
    )
    subprocess.run(
        [
            "convert",
            str(full),
            "-resize",
            "1100x",
            "-quality",
            "78",
            str(assets / "b2_filmstrip_small.jpg"),
        ],
        check=True,
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


def _validate_controls(
    baseline: Path, hard: Path, source_chunk: int, targets: list[int]
) -> dict:
    baseline_meta = _json(baseline / "run_metadata.json")
    hard_meta = _json(hard / "run_metadata.json")
    sources = {
        int(item["source_chunk"])
        for item in hard_meta["mapkv"]["selections"]
    }
    hard_targets = {
        int(item["target_chunk"])
        for item in hard_meta["mapkv"]["selections"]
    }
    if sources != {source_chunk}:
        raise RuntimeError(f"HardKV source {sources} != fixed chunk {source_chunk}")
    if hard_targets != set(targets):
        raise RuntimeError(f"HardKV targets {hard_targets} != {set(targets)}")
    if hard_meta["gpu"] != baseline_meta["gpu"]:
        raise RuntimeError("Baseline and HardKV controls were generated on different GPUs")
    return {
        "baseline_gpu": baseline_meta["gpu"],
        "hard_source_chunks": sorted(sources),
        "hard_target_chunks": sorted(hard_targets),
        "hard_injection_mode": hard_meta["mapkv"]["injection_mode"],
        "hard_alpha": hard_meta["mapkv"]["alpha"],
        "hard_layers": hard_meta["mapkv"]["selected_layers_resolved"],
        "hard_steps": hard_meta["mapkv"]["selected_steps_resolved"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-source camera-aligned Warp-and-Reencode Recent stage"
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--case_dir", default="artifacts/control/yaw30to20_scene01"
    )
    parser.add_argument(
        "--control_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_locality",
    )
    parser.add_argument(
        "--output_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_warp_reencode",
    )
    parser.add_argument(
        "--inspatio_python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    parser.add_argument("--feather_kernel", type=int, default=3)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case_dir).resolve()
    control = Path(args.control_root).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "generation").mkdir(parents=True, exist_ok=True)
    source_chunk = 8
    targets = [21, 22]
    manifest = _json(case_dir / "trajectory_manifest.json")
    if int(manifest["source_chunk"]) != source_chunk:
        raise RuntimeError("The controlled partial case no longer declares B1 chunk 8")
    phase_payload = _json(case_dir / "phase_labels.json")
    b2 = next(
        item for item in phase_payload["phases"] if item["name"] == "B2_hold"
    )
    expected_targets = list(
        range(
            int(b2["start_block"]),
            int(b2["stop_block_exclusive"]),
        )
    )
    if expected_targets != targets:
        raise RuntimeError(f"Expected B2 chunks {targets}, got {expected_targets}")

    _link(control / "baseline", root / "baseline")
    _link(
        control / "generation" / "global_surfelkv",
        root / "generation" / "hard_recentkv",
    )
    _link(control / "surfel", root / "surfel")
    _link(control / "retrieval", root / "retrieval")
    controls = _validate_controls(
        root / "baseline",
        root / "generation" / "hard_recentkv",
        source_chunk,
        targets,
    )
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
    asset_root = _asset_root(repo)
    warp_root = root / "generation" / "warp_reencode"
    command = _base_inference_command(
        python=args.inspatio_python,
        repo=repo,
        asset_root=asset_root,
        case_dir=case_dir,
        output_dir=warp_root,
        noise_bundle=control / "kv" / "noise_bundle.pt",
        bank_root=control / "kv" / "recent_bank_all",
        seed=args.seed,
        memory_layers="all",
    ) + [
        "--run_name",
        "warp_reencode",
        "--mode",
        "oracle",
        "--source_chunk",
        str(source_chunk),
        "--target_chunks",
        *[str(chunk) for chunk in targets],
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
        "--warp_reencode_recent",
        "--warp_source_latents",
        str(control / "baseline" / "pred_latents.pt"),
        "--warp_intrinsics_path",
        str(case_dir / "intrinsics.txt"),
        "--warp_feather_kernel",
        str(args.feather_kernel),
        "--compare_latents_to",
        str(control / "baseline" / "pred_latents.pt"),
    ]
    run_contract = {
        "architecture": (
            "B1 clean latent -> exact camera warp -> Virtual Recent -> "
            "native t=0 writer -> replace_recent_delta"
        ),
        "source_chunk": source_chunk,
        "target_chunks": targets,
        "automatic_retrieval": False,
        "controls": controls,
        "command": command,
    }
    (root / "run_contract.json").write_text(
        json.dumps(run_contract, indent=2), encoding="utf-8"
    )

    if args.stage in {"generation", "full"}:
        if not (warp_root / "run_metadata.json").exists():
            _run("warp_reencode", command, root, environment)
        else:
            print("[MapKV] reuse warp-reencode generation")
    if args.stage in {"evaluate", "full"}:
        metrics = evaluate_warp_reencode(
            run_root=root,
            case_dir=case_dir,
            source_chunk=source_chunk,
            target_chunks=tuple(targets),
        )
        _transcode_videos(
            root,
            ["baseline", "hard_recentkv", "warp_reencode"],
            max(targets),
            int(manifest["target_rgb_index"]),
        )
        _make_filmstrip(root)
        print(json.dumps({"status": metrics["status"]}, indent=2))
    if args.stage in {"report", "full"}:
        print(build_report(root))


if __name__ == "__main__":
    main()
