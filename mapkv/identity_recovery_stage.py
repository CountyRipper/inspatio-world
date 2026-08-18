from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .continuous_cavr_stage import _asset_root
from .fast_pipeline import _base_inference_command, _json, _link, _run
from .identity_recovery_evaluation import (
    METHOD_ROOTS,
    evaluate_identity_recovery,
)
from .identity_recovery_report import build_report


STAGES = {"generation", "evaluate", "report", "full"}


def _command(
    *,
    method: str,
    python: str,
    repo: Path,
    asset_root: Path,
    case_dir: Path,
    output: Path,
    geometry: Path,
    baseline: Path,
    seed: int,
) -> list[str]:
    command = _base_inference_command(
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
        method,
        "--mode",
        "oracle",
        "--source_chunk",
        "8",
        "--selected_steps",
        "0",
        "1",
        "2",
        "3",
        "--alpha",
        "1",
        "--gate_mode",
        "global",
        "--warp_source_latents",
        str(baseline / "pred_latents.pt"),
        "--warp_intrinsics_path",
        str(case_dir / "intrinsics.txt"),
        "--warp_surfel_index",
        str(geometry / "surfel" / "surfel_index.npz"),
        "--warp_surfel_sequence",
        str(geometry / "cut3r" / "sequence.json"),
        "--warp_min_history_gap",
        "2",
        "--warp_memory_dilation_kernel",
        "3",
        "--warp_query_feather_kernel",
        "3",
        "--compare_latents_to",
        str(baseline / "pred_latents.pt"),
    ]
    if method == "canonical_kv":
        return command + [
            "--injection_mode",
            "canonical_recent_delta",
            "--canonical_kv_readdress",
        ]
    command += [
        "--injection_mode",
        "replace_recent_delta",
        "--continuous_virtual_recent",
        "--continuous_recent_fallback",
        "raw",
        "--continuous_query_gate",
        "support_preserving",
        "--continuous_mask_policy",
        "strong_core",
    ]
    if method == "rgb_warp_vae_wre":
        command += ["--warp_history_representation", "rgb_warp_vae"]
    return command


def _make_report_videos(root: Path, phase_payload: dict, fps: float = 24.0) -> None:
    report_root = root / "videos" / "report"
    original_root = root / "videos" / "original"
    report_root.mkdir(parents=True, exist_ok=True)
    original_root.mkdir(parents=True, exist_ok=True)
    ramp = next(item for item in phase_payload["phases"] if item["name"] == "A_to_B2")
    b2 = next(item for item in phase_payload["phases"] if item["name"] == "B2_hold")
    start = float(ramp["rgb_start"]) / fps
    duration = float(b2["rgb_stop_exclusive"] - ramp["rgb_start"]) / fps
    for method, relative in METHOD_ROOTS.items():
        source = root / relative / "pred.mp4"
        _link(source, original_root / f"{method}.mp4")
        _link(source, report_root / f"full_revisit_{method}.mp4")
        clip = report_root / f"reentry_{method}.mp4"
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
                "28",
                "-preset",
                "veryfast",
                "-movflags",
                "+faststart",
                "-an",
                str(clip),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MapKV identity recovery stage")
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case_dir", default="artifacts/control/yaw30to20_scene01")
    parser.add_argument(
        "--geometry_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_locality",
    )
    parser.add_argument(
        "--masked_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_masked_continuous_wre",
    )
    parser.add_argument(
        "--output_root",
        default="results/mapkv_fast/yaw30to20_scene01_seed0_identity_recovery",
    )
    parser.add_argument(
        "--inspatio_python",
        default="/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case_dir).resolve()
    geometry = Path(args.geometry_root).resolve()
    masked = Path(args.masked_root).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "generation").mkdir(parents=True, exist_ok=True)
    manifest = _json(case_dir / "trajectory_manifest.json")
    if int(manifest["source_chunk"]) != 8:
        raise RuntimeError("Identity stage requires fixed B1 source chunk 8")
    _link(masked / "baseline", root / "baseline")
    _link(
        masked / "generation" / "masked_continuous_wre",
        root / "generation" / "current_masked_wre",
    )
    for name in ("surfel", "cut3r", "retrieval"):
        _link(geometry / name, root / name)
    _link(masked / "surfel_rgb_options", root / "surfel_rgb_options")

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(repo)
            + ((":" + environment["PYTHONPATH"]) if environment.get("PYTHONPATH") else ""),
        }
    )
    commands = {
        method: _command(
            method=method,
            python=args.inspatio_python,
            repo=repo,
            asset_root=_asset_root(repo),
            case_dir=case_dir,
            output=root / "generation" / method,
            geometry=geometry,
            baseline=root / "baseline",
            seed=args.seed,
        )
        for method in ("strong_core_latent_wre", "rgb_warp_vae_wre", "canonical_kv")
    }
    contract = {
        "focus_zh": "历史身份恢复：强记忆内核、RGB-Warp 质量上界与 Canonical-K 重寻址",
        "source_chunk": 8,
        "retrieval_frozen": True,
        "geometry_frozen": True,
        "alpha": 1.0,
        "layers": "all",
        "steps": [0, 1, 2, 3],
        "commands": commands,
    }
    (root / "run_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.stage in {"generation", "full"}:
        for method, command in commands.items():
            output = root / "generation" / method
            if (output / "run_metadata.json").exists():
                print(f"[MapKV] reuse {method}")
            else:
                _run(method, command, root, environment)
    if args.stage in {"evaluate", "full"}:
        result = evaluate_identity_recovery(run_root=root, case_dir=case_dir)
        _make_report_videos(root, _json(case_dir / "phase_labels.json"))
        print(json.dumps({"status": result["status"]}, ensure_ascii=False))
    if args.stage in {"report", "full"}:
        print(build_report(root))


if __name__ == "__main__":
    main()
