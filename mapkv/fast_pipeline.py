from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image


STAGES = {
    "smoke", "kv_sanity", "cut3r", "surfel", "retrieval",
    "generation", "full", "report",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(name: str, command: list[str], root: Path, env: dict | None = None) -> None:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print("\n[MapKV]", name)
    print(" ".join(command), flush=True)
    with (log_dir / f"{name}.log").open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialize_run_contract(
    *, root: Path, case_dir: Path, baseline_root: Path, target_chunk: int
) -> None:
    trajectory_root = root / "trajectory"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "target_poses.npy",
        "yaw_pitch_roll.npy",
        "phase_labels.json",
        "trajectory_manifest.json",
    ):
        shutil.copy2(case_dir / name, trajectory_root / name)
    plots = root / "assets" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case_dir / "pair_contact_sheet.png", plots / "pair_contact_sheet.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ypr = np.load(case_dir / "yaw_pitch_roll.npy")
    poses = np.load(case_dir / "target_poses.npy")
    manifest = _json(case_dir / "trajectory_manifest.json")
    translation = np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(ypr[:, 0], label="yaw")
    axes[0].plot(ypr[:, 1], label="pitch")
    axes[0].plot(ypr[:, 2], label="roll")
    axes[0].axvline(manifest["source_rgb_index"], color="#e45756", label="B1")
    axes[0].axvline(manifest["target_rgb_index"], color="#54a24b", label="B2")
    axes[0].set(ylabel="degrees", title="Exact controlled trajectory")
    axes[0].legend(ncol=5)
    axes[1].plot(translation, color="#4f7cac")
    axes[1].set(xlabel="RGB frame", ylabel="relative translation")
    fig.tight_layout()
    fig.savefig(plots / "trajectory.png", dpi=150)
    plt.close(fig)

    _link(baseline_root / "pred.mp4", baseline_root / "full.mp4")
    _link(baseline_root / "block_mapping.json", baseline_root / "chunk_manifest.json")
    _link(case_dir / "mask_offline.mp4", baseline_root / "render_mask_preview.mp4")
    mapping = _json(baseline_root / "block_mapping.json")
    blocks = mapping["blocks"] if isinstance(mapping, dict) else mapping
    target = next(item for item in blocks if int(item["chunk_id"]) == target_chunk)
    first_target_latent = int(target["latent_indices"][0])
    latent_length = int(mapping["latent_length"])
    rgb_length = int(mapping["rgb_length"])
    cutoff = int(round(first_target_latent * (rgb_length - 1) / max(latent_length - 1, 1)))
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(baseline_root / "pred.mp4"),
            "-frames:v", str(cutoff), "-c:v", "libx264", "-crf", "25",
            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", str(baseline_root / "prefix.mp4"),
        ],
        check=True,
    )


def _base_inference_command(
    *,
    python: str,
    repo: Path,
    asset_root: Path,
    case_dir: Path,
    output_dir: Path,
    noise_bundle: Path,
    bank_root: Path,
    seed: int,
    memory_layers: str,
) -> list[str]:
    return [
        python,
        str(repo / "inference_mapkv_proto.py"),
        "--config_path", str(repo / "configs/inference_1.3b.yaml"),
        "--mapkv_config", str(repo / "configs/mapkv_fast.yaml"),
        "--checkpoint_path",
        str(asset_root / "checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"),
        "--wan_model_folder", str(asset_root / "checkpoints/Wan2.1-T2V-1.3B"),
        "--json_path", str(case_dir / "input.json"),
        "--data_path_root", "/",
        "--target_pose_path", str(case_dir / "target_poses.npy"),
        "--case_dir", str(case_dir),
        "--output_dir", str(output_dir),
        "--noise_bundle", str(noise_bundle),
        "--bank_root", str(bank_root),
        "--memory_layers", memory_layers,
        "--seed", str(seed),
    ]


def _transcode_videos(
    root: Path, methods: list[str], target_chunk: int, target_rgb_index: int, fps: float = 24.0
) -> None:
    original = root / "videos" / "original"
    report = root / "videos" / "report"
    posters = root / "assets" / "posters"
    original.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    posters.mkdir(parents=True, exist_ok=True)
    for method in methods:
        run_dir = root / "baseline" if method == "baseline" else root / "generation" / method
        source = run_dir / "pred.mp4"
        if not source.exists():
            continue
        _link(source, original / f"{method}.mp4")
        destination = report / f"{method}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                "-vf", "scale=-2:480", "-c:v", "libx264", "-crf", "28",
                "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", str(destination),
            ],
            check=True,
        )
        center_seconds = target_rgb_index / fps
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(max(0.0, center_seconds - 1.0)), "-i", str(source),
                "-t", "2.0", "-vf", "scale=-2:480", "-c:v", "libx264",
                "-crf", "27", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an",
                str(report / f"b2_window_{method}.mp4"),
            ],
            check=True,
        )
        keyframe = run_dir / "keyframes" / f"chunk_{target_chunk:04d}.png"
        Image.open(keyframe).convert("RGB").save(posters / f"{method}.jpg", quality=84)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached MapKV v0.4 closed-loop runner")
    parser.add_argument("--case", default="yaw30_scene01")
    parser.add_argument("--case-dir")
    parser.add_argument("--stage", choices=sorted(STAGES), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--methods", default="baseline,wrongkv,posekv,surfelkv,manualcorrect")
    parser.add_argument("--memory-alpha", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--memory-layers", choices=("uniform8", "middle8", "explicit", "all"),
        default="uniform8",
    )
    parser.add_argument("--gate", choices=("global", "ref_blind"), default="global")
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument("--relative-voxel-fraction", type=float, default=0.005)
    parser.add_argument("--reuse-baseline", action="store_true")
    parser.add_argument("--reuse-cut3r", action="store_true")
    parser.add_argument("--reuse-surfel", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--inspatio-python", default=sys.executable)
    parser.add_argument("--cut3r-python")
    parser.add_argument("--cut3r-root")
    parser.add_argument("--cut3r-checkpoint")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if args.top_k != 1 and args.stage in {"generation", "full"}:
        raise ValueError("The first whole-chunk injector is top-K=1; use retrieval stage for K>1")

    repo = Path(__file__).resolve().parents[1]
    common = Path(
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip()
    )
    asset_root = common.parent
    case_dir = (
        Path(args.case_dir).resolve()
        if args.case_dir else (repo / "artifacts" / "control" / args.case).resolve()
    )
    manifest = _json(case_dir / "trajectory_manifest.json")
    source_chunk = int(manifest["source_chunk"])
    target_chunk = int(manifest["target_chunk"])
    wrong_chunk = int(manifest["wrong_chunk"])
    root = (
        Path(args.output).resolve()
        if args.output else (repo / "results" / "mapkv_fast" / f"{args.case}_seed{args.seed}").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    methods = [item for item in args.methods.split(",") if item]
    if "baseline" not in methods:
        methods.insert(0, "baseline")
    cut3r_root = Path(args.cut3r_root or repo / "third_party" / "CUT3R").resolve()
    cut3r_python = str(
        Path(args.cut3r_python or repo / "third_party" / "mapkv_cut3r_env" / "bin" / "python").resolve()
    )
    checkpoint = Path(
        args.cut3r_checkpoint or cut3r_root / "src" / "cut3r_512_dpt_4_64.pth"
    ).resolve()
    inspatio_checkpoint = (
        asset_root / "checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"
    ).resolve()
    wan_model_folder = (asset_root / "checkpoints/Wan2.1-T2V-1.3B").resolve()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(repo) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    validation = _json(case_dir / "pair_validation.json")
    config = {
        "case": args.case,
        "case_dir": str(case_dir),
        "seed": args.seed,
        "source_chunk": source_chunk,
        "target_chunk": target_chunk,
        "wrong_chunk": wrong_chunk,
        "source_rgb_index": int(manifest["source_rgb_index"]),
        "target_rgb_index": int(manifest["target_rgb_index"]),
        "reference_blind_fraction": validation["target_reference_blind_fraction"],
        "memory_alpha": args.memory_alpha,
        "memory_layers": args.memory_layers,
        "top_k": args.top_k,
        "gate": args.gate,
        "confidence_threshold": args.confidence_threshold,
        "relative_voxel_fraction": args.relative_voxel_fraction,
        "query_pose_mode": "controlled_same_pose",
        "injection_mode": "residual_memory_attention",
        "cut3r_root": str(cut3r_root),
        "cut3r_checkpoint": str(checkpoint),
        "inspatio_checkpoint": str(inspatio_checkpoint),
        "wan_model_folder": str(wan_model_folder),
        "cuda_visible_device": str(args.gpu),
        "trajectory_sha256": _sha256(case_dir / "target_poses.npy"),
        "source_sha256": _sha256(case_dir / "static_source.mp4"),
    }
    (root / "config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    git_status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    )
    if git_status:
        (root / "git_diff_stat.txt").write_text(
            subprocess.check_output(
                ["git", "-C", str(repo), "diff", "--stat"], text=True
            ),
            encoding="utf-8",
        )

    stage = "report" if args.report_only else args.stage
    baseline_root = root / "baseline"
    bank_root = root / "kv" / "bank"
    noise_bundle = root / "kv" / "noise_bundle.pt"
    old_noise = case_dir / "baseline" / f"seed_{args.seed}" / "noise_bundle.pt"
    if not noise_bundle.exists() and old_noise.exists():
        noise_bundle.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_noise, noise_bundle)
    base = _base_inference_command(
        python=args.inspatio_python,
        repo=repo,
        asset_root=asset_root,
        case_dir=case_dir,
        output_dir=baseline_root,
        noise_bundle=noise_bundle,
        bank_root=bank_root,
        seed=args.seed,
        memory_layers=args.memory_layers,
    )

    if stage in {"kv_sanity", "full"}:
        if not (args.reuse_baseline and (baseline_root / "run_metadata.json").exists()):
            command = base + [
                "--run_name", "baseline", "--mode", "baseline", "--capture_kv",
                "--verify_memory_off_replay", "--require_replay_tolerance",
            ]
            if not noise_bundle.exists():
                command.append("--create_noise_bundle")
            _run("baseline", command, root, env)
        else:
            print("[MapKV] reuse baseline")
        bank_command = [
            args.inspatio_python, "-m", "mapkv.kv_bank",
            "--bank", str(bank_root), "--output", str(root / "kv" / "bank_stats.json"),
        ]
        if not (root / "kv" / "capture_manifest.json").exists():
            bank_command += [
                "--capture_manifest", str(root / "kv" / "capture_manifest.json")
            ]
        _run("kv_bank_stats", bank_command, root, env)
        sanity_root = root / "kv" / "alpha0"
        if not (sanity_root / "run_metadata.json").exists():
            _run(
                "alpha0",
                _base_inference_command(
                    python=args.inspatio_python, repo=repo, asset_root=asset_root,
                    case_dir=case_dir, output_dir=sanity_root,
                    noise_bundle=noise_bundle, bank_root=bank_root, seed=args.seed,
                    memory_layers=args.memory_layers,
                )
                + [
                    "--run_name", "alpha0", "--mode", "oracle",
                    "--source_chunk", str(source_chunk), "--target_chunks", str(target_chunk),
                    "--alpha", "0", "--injection_mode", "residual_memory_attention",
                    "--gate_mode", args.gate,
                    "--compare_latents_to", str(baseline_root / "pred_latents.pt"),
                    "--require_replay_tolerance",
                ],
                root,
                env,
            )

    def generation(name: str) -> None:
        destination = root / "generation" / name
        if (destination / "run_metadata.json").exists():
            print("[MapKV] reuse", name)
            return
        command = _base_inference_command(
            python=args.inspatio_python, repo=repo, asset_root=asset_root,
            case_dir=case_dir, output_dir=destination, noise_bundle=noise_bundle,
            bank_root=bank_root, seed=args.seed, memory_layers=args.memory_layers,
        ) + [
            "--run_name", name, "--target_chunks", str(target_chunk),
            "--alpha", str(args.memory_alpha),
            "--injection_mode", "residual_memory_attention",
            "--gate_mode", args.gate,
            "--compare_latents_to", str(baseline_root / "pred_latents.pt"),
        ]
        if name == "manualcorrect":
            command += ["--mode", "oracle", "--source_chunk", str(source_chunk)]
        elif name == "wrongkv":
            command += ["--mode", "wrong", "--wrong_chunk", str(wrong_chunk)]
        elif name == "posekv":
            command += [
                "--mode", "pose", "--retrieval_plan", str(root / "retrieval" / "pose_plan.json")
            ]
        elif name == "surfelkv":
            command += [
                "--mode", "geometry", "--retrieval_plan", str(root / "retrieval" / "retrieval.json")
            ]
        else:
            raise ValueError(name)
        _run(name, command, root, env)

    if stage in {"kv_sanity", "full"}:
        for method in ("manualcorrect", "wrongkv"):
            if method in methods:
                generation(method)
        alpha0_metadata = _json(root / "kv" / "alpha0" / "run_metadata.json")
        sanity = {
            "alpha0_vs_baseline": alpha0_metadata["replay"]["against_saved_latents"]["max_abs_diff"],
            "capture_type": "clean_context",
            "rope_state": "post_rope",
            "correct_run": str(root / "generation" / "manualcorrect"),
            "wrong_run": str(root / "generation" / "wrongkv"),
            "branch_active": all(
                (root / "generation" / name / "run_metadata.json").exists()
                for name in ("manualcorrect", "wrongkv") if name in methods
            ),
            "correct_target_latent_max_abs_diff": _json(
                root / "generation" / "manualcorrect" / "run_metadata.json"
            )["replay"]["against_saved_latents"]["per_chunk_max_abs_diff"][str(target_chunk)],
            "wrong_target_latent_max_abs_diff": _json(
                root / "generation" / "wrongkv" / "run_metadata.json"
            )["replay"]["against_saved_latents"]["per_chunk_max_abs_diff"][str(target_chunk)],
            "runtime_cache_unchanged": all(
                _json(root / "generation" / name / "run_metadata.json")["mapkv"]
                ["cache_audits"][str(target_chunk)]["unchanged"]
                for name in ("manualcorrect", "wrongkv")
            ),
        }
        (root / "kv" / "sanity_metrics.json").write_text(
            json.dumps(sanity, indent=2), encoding="utf-8"
        )

    if stage in {"cut3r", "full"}:
        if not (args.reuse_cut3r and (root / "cut3r" / "sequence.json").exists()):
            _run(
                "cut3r",
                [
                    cut3r_python, "-m", "mapkv.cut3r_adapter",
                    "--baseline_root", str(baseline_root),
                    "--block_mapping", str(baseline_root / "block_mapping.json"),
                    "--cut3r_root", str(cut3r_root),
                    "--checkpoint", str(checkpoint),
                    "--output_dir", str(root / "cut3r"),
                    "--target_chunk", str(target_chunk),
                    "--query_source_chunk", str(source_chunk),
                    "--confidence_threshold", str(args.confidence_threshold),
                    "--device", "cuda",
                ],
                root,
                env,
            )

    if stage in {"surfel", "full"}:
        if not (args.reuse_surfel and (root / "surfel" / "surfel_index.npz").exists()):
            _run(
                "surfel",
                [
                    args.inspatio_python, "-m", "mapkv.surfel_index",
                    "--sequence", str(root / "cut3r" / "sequence.json"),
                    "--output_dir", str(root / "surfel"),
                    "--confidence_threshold", str(args.confidence_threshold),
                    "--voxel_size_mode", "relative_scene",
                    "--relative_scene_fraction", str(args.relative_voxel_fraction),
                ],
                root,
                env,
            )

    if stage in {"retrieval", "full"}:
        _run(
            "retrieval",
            [
                args.inspatio_python, "-m", "mapkv.retrieval",
                "--sequence", str(root / "cut3r" / "sequence.json"),
                "--surfel_index", str(root / "surfel" / "surfel_index.npz"),
                "--output_dir", str(root / "retrieval"),
                "--target_chunk", str(target_chunk),
                "--top_k", str(args.top_k),
                "--min_history_gap_chunks", "2",
            ],
            root,
            env,
        )

    if stage in {"generation", "full"}:
        for method in ("posekv", "surfelkv"):
            if method in methods:
                generation(method)

    if stage in {"report", "full"}:
        bank_stats = _json(root / "kv" / "bank_stats.json")
        cut3r_stats = _json(root / "cut3r" / "stats.json")
        architecture = {
            "backbone": "InSpatio-World-1.3B",
            "num_frame_per_block": 3,
            "memory": {
                "enabled": True,
                "payload": "native_kv_post_rope",
                "granularity": "chunk",
                "storage": "cpu",
                "layers": bank_stats["selected_layers"],
            },
            "geometry": {
                "backend": "CUT3R",
                "mode": "offline_causal_prefix",
                "address": "voxel_surfel",
                "prefix_last_chunk": cut3r_stats["prefix_last_chunk"],
            },
            "retrieval": {
                "mode": "visible_surfel_chunk_vote",
                "top_k": args.top_k,
                "query_pose_mode": "controlled_same_pose",
            },
            "injection": {
                "mode": "residual_memory_attention",
                "alpha": args.memory_alpha,
                "gate": args.gate,
                "output_projection_count": 1,
                "base_cache_replaced": False,
            },
        }
        (root / "architecture_state.json").write_text(
            json.dumps(architecture, indent=2), encoding="utf-8"
        )
        _materialize_run_contract(
            root=root,
            case_dir=case_dir,
            baseline_root=baseline_root,
            target_chunk=target_chunk,
        )
        (root / "assets" / "posters").mkdir(parents=True, exist_ok=True)
        Image.open(case_dir / "source_frame.png").convert("RGB").save(
            root / "assets" / "posters" / "source.jpg", quality=88
        )
        available_methods = [
            name for name in methods
            if name == "baseline" or (root / "generation" / name / "pred.mp4").exists()
        ]
        _transcode_videos(
            root, available_methods, target_chunk, int(manifest["target_rgb_index"])
        )
        _run(
            "evaluation",
            [
                args.inspatio_python, "-m", "mapkv.evaluation",
                "--run_root", str(root), "--case_dir", str(case_dir),
                "--source_chunk", str(source_chunk), "--target_chunk", str(target_chunk),
                "--methods", ",".join(available_methods),
            ],
            root,
            env,
        )
        _run(
            "report",
            [
                args.inspatio_python, "-m", "mapkv.report",
                "--run_root", str(root), "--status", "CLOSED_LOOP_OK",
                "--conclusion",
                "The causal CUT3R-surfel-to-native-KV loop executed; inspect synchronized B2 videos before assigning GO/CONTINUE/NO_GO.",
                "--next_action",
                "Review the B2 windows, then run one partial-overlap 0-to-30-to-0-to-20 case if the same-pose signal is interpretable.",
            ],
            root,
            env,
        )
    print(json.dumps({"status": "completed", "stage": stage, "output": str(root)}, indent=2))


if __name__ == "__main__":
    main()
