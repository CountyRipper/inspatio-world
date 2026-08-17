#!/usr/bin/env python3
"""Build one exact-pose, repeated-static-frame MapKV control case."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from mapkv_proto.trajectory_builder import (
    build_control_phases,
    build_exact_c2w,
    build_yaw_samples,
    monotonic_index,
    phase_by_name,
    plateau_middle_chunk,
    rgb_length_for_latents,
    save_json,
    sha256_file,
    validate_exact_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--case_dir")
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--data_path_root", default=".")
    parser.add_argument("--source_frame_index", type=int, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--speed_deg_per_rgb_frame", type=float, default=0.5)
    parser.add_argument("--frames_per_block", type=int, default=3)
    parser.add_argument("--vae_calibration_metadata", required=True)
    parser.add_argument("--vae_time_map", default="artifacts/control/vae_time_map.json")
    parser.add_argument("--distractor", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_device", default="0")
    return parser.parse_args()


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _load_base_c2w(da3_dir: Path, frame_index: int) -> np.ndarray:
    rows = np.loadtxt(da3_dir / "extrinsic.txt")
    count = rows.shape[0] // 3
    if frame_index < 0 or frame_index >= count:
        raise IndexError(f"source frame {frame_index} outside DA3 frame count {count}")
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3] = rows[frame_index * 3 : (frame_index + 1) * 3, :4]
    return np.linalg.inv(w2c)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def _video_frame_count(path: Path) -> int:
    result = subprocess.check_output(
        [
            "/usr/bin/ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return int(result.strip())


def _copy_proxy_inputs(
    *, source_depth_root: Path, case_dir: Path, source_frame_index: int
) -> None:
    (case_dir / "depth").mkdir(parents=True, exist_ok=True)
    (case_dir / "images").mkdir(parents=True, exist_ok=True)
    depth = source_depth_root / "depth" / f"{source_frame_index:06d}.png"
    image = source_depth_root / "images" / f"{source_frame_index:06d}.png"
    if not depth.exists() or not image.exists():
        raise FileNotFoundError(f"Missing converted DA3 frame/depth for index {source_frame_index}")
    shutil.copy2(depth, case_dir / "depth" / "000000.png")
    shutil.copy2(image, case_dir / "images" / "000000.png")
    for name in ("metadata.txt", "intrinsics.txt", "extrinsics.txt"):
        source = source_depth_root / name
        if source.exists():
            shutil.copy2(source, case_dir / name)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case_dir or f"artifacts/control/{args.case_id}").resolve()
    case_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_path_root).resolve()
    source_entries = json.loads(Path(args.source_json).resolve().read_text(encoding="utf-8"))
    if len(source_entries) != 1:
        raise ValueError("Controlled quick check expects exactly one source entry")
    source_entry = source_entries[0]
    source_video = _resolve(source_entry["video_path"], data_root)
    source_depth_root = _resolve(source_entry["vggt_depth_path"], data_root)
    source_da3_dir = Path(str(source_depth_root) + "_da3_tmp")

    calibration_metadata = json.loads(
        Path(args.vae_calibration_metadata).resolve().read_text(encoding="utf-8")
    )
    latent_observed = int(calibration_metadata["latent_length"])
    decoded_observed = int(calibration_metadata["decoded_rgb_length"])
    temporal_stride = (decoded_observed - 1) / max(latent_observed - 1, 1)
    vae_time_map_path = Path(args.vae_time_map).resolve()
    vae_time_map = {
        "source": str(Path(args.vae_calibration_metadata).resolve()),
        "vae": "original Wan VAE",
        "observed_input_rgb_length": _video_frame_count(source_video),
        "observed_encoded_latent_length_before_truncation": int(
            calibration_metadata["latent_length_before_truncation"]
        ),
        "observed_selected_latent_length": latent_observed,
        "observed_decoded_rgb_length": decoded_observed,
        "empirical_temporal_stride": temporal_stride,
        "mapping": "rgb_idx = round(latent_idx * (T_rgb - 1) / max(T_latent - 1, 1))",
    }
    save_json(vae_time_map, vae_time_map_path)

    phases, ramp_blocks = build_control_phases(
        args.theta,
        temporal_stride=temporal_stride,
        frames_per_block=args.frames_per_block,
        requested_speed_degrees_per_frame=args.speed_deg_per_rgb_frame,
        distractor=args.distractor,
    )
    num_blocks = phases[-1].stop_block
    latent_length = num_blocks * args.frames_per_block
    rgb_length = rgb_length_for_latents(latent_length, temporal_stride)
    yaw, phase_labels = build_yaw_samples(phases, rgb_length)
    pitch = np.zeros_like(yaw)
    roll = np.zeros_like(yaw)
    np.save(case_dir / "yaw_pitch_roll.npy", np.stack([yaw, pitch, roll], axis=1))

    base_c2w = _load_base_c2w(source_da3_dir, args.source_frame_index)
    target_c2w = build_exact_c2w(base_c2w, yaw)
    np.save(case_dir / "target_poses.npy", target_c2w)

    b1 = phase_by_name(phases, "B1_hold")
    b2 = phase_by_name(phases, "B2_hold")
    wrong_phase = phase_by_name(
        phases, "wrong_hold" if args.distractor else "A1_distractor"
    )
    source_chunk = plateau_middle_chunk(b1)
    target_chunk = plateau_middle_chunk(b2)
    wrong_chunk = plateau_middle_chunk(wrong_phase)
    source_latent = source_chunk * args.frames_per_block + args.frames_per_block // 2
    target_latent = target_chunk * args.frames_per_block + args.frames_per_block // 2
    source_rgb_index = monotonic_index(source_latent, latent_length, rgb_length)
    target_rgb_index = monotonic_index(target_latent, latent_length, rgb_length)

    phase_payload = {
        "case_id": args.case_id,
        "columns_yaw_pitch_roll_npy": [
            "yaw_degrees",
            "pitch_degrees",
            "roll_degrees",
        ],
        "phases": phase_labels,
        "source_chunk": source_chunk,
        "target_chunk": target_chunk,
        "wrong_chunk": wrong_chunk,
        "source_rgb_index": source_rgb_index,
        "target_rgb_index": target_rgb_index,
    }
    save_json(phase_payload, case_dir / "phase_labels.json")

    pose_validation = validate_exact_case(
        target_c2w=target_c2w,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        roll_degrees=roll,
        source_chunk=source_chunk,
        target_chunk=target_chunk,
        source_rgb_index=source_rgb_index,
        target_rgb_index=target_rgb_index,
        phase_labels=phase_labels,
    )
    if not pose_validation["valid"]:
        raise RuntimeError(f"Generated pose validation failed: {pose_validation}")
    save_json(pose_validation, case_dir / "pose_validation.json")

    source_frame = case_dir / "source_frame.png"
    static_video = case_dir / "static_source.mp4"
    _run(
        [
            "/usr/bin/ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source_video),
            "-vf",
            f"select=eq(n\\,{args.source_frame_index})",
            "-frames:v",
            "1",
            str(source_frame),
        ]
    )
    _run(
        [
            "/usr/bin/ffmpeg",
            "-y",
            "-v",
            "error",
            "-loop",
            "1",
            "-framerate",
            str(args.fps),
            "-i",
            str(source_frame),
            "-frames:v",
            str(rgb_length),
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-g",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(args.fps),
            str(static_video),
        ]
    )
    _copy_proxy_inputs(
        source_depth_root=source_depth_root,
        case_dir=case_dir,
        source_frame_index=args.source_frame_index,
    )

    source_manifest = {
        "source_video": str(source_video),
        "source_frame_index": args.source_frame_index,
        "original_source_video": source_entry.get(
            "original_source_video", str(source_video)
        ),
        "original_source_frame_index": source_entry.get(
            "original_source_frame_index", args.source_frame_index
        ),
        "source_frame_path": str(source_frame),
        "fps": args.fps,
        "num_rgb_frames": rgb_length,
        "sha256": sha256_file(static_video),
        "source_frame_sha256": sha256_file(source_frame),
        "static_source": True,
    }
    save_json(source_manifest, case_dir / "source_manifest.json")

    trajectory_manifest = {
        "case_id": args.case_id,
        "decision_eligible": True,
        "trajectory_type": (
            "pure_yaw_same_view_revisit_with_distractor"
            if args.distractor
            else "pure_yaw_same_view_revisit"
        ),
        "pose_convention": (
            "absolute c2w; each pose is selected_source_c2w @ local_yaw_rotation"
        ),
        "target_pose_path": str(case_dir / "target_poses.npy"),
        "target_pose_sha256": sha256_file(case_dir / "target_poses.npy"),
        "interpolation": "exact_per_rgb_frame",
        "adaptive_frame": False,
        "theta_degrees": args.theta,
        "pitch_degrees": 0.0,
        "roll_degrees": 0.0,
        "relative_translation": False,
        "requested_speed_degrees_per_rgb_frame": args.speed_deg_per_rgb_frame,
        "ramp_blocks": ramp_blocks,
        "frames_per_block": args.frames_per_block,
        "latent_length": latent_length,
        "rgb_length": rgb_length,
        "num_blocks": num_blocks,
        "source_chunk": source_chunk,
        "target_chunk": target_chunk,
        "wrong_chunk": wrong_chunk,
        "source_rgb_index": source_rgb_index,
        "target_rgb_index": target_rgb_index,
        "vae_time_map": str(vae_time_map_path),
    }
    save_json(trajectory_manifest, case_dir / "trajectory_manifest.json")

    controlled_entry = dict(source_entry)
    controlled_entry["video_path"] = str(static_video)
    controlled_entry["vggt_depth_path"] = str(case_dir)
    controlled_entry["vggt_extrinsics_path"] = str(case_dir / "extrinsics.txt")
    save_json([controlled_entry], case_dir / "input.json")

    if args.render:
        render_dir = case_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(args.render_device)
        _run(
            [
                sys.executable,
                str(repo_root / "scripts/render_point_cloud.py"),
                "--da3_dir",
                str(source_da3_dir),
                "--target_pose_path",
                str(case_dir / "target_poses.npy"),
                "--source_frame_index",
                str(args.source_frame_index),
                "--output_dir",
                str(render_dir),
                "--width",
                "832",
                "--height",
                "480",
                "--fps",
                str(args.fps),
                "--render_backend",
                "warper",
                "--validation_pair",
                str(source_rgb_index),
                str(target_rgb_index),
                "--render_validation_path",
                str(case_dir / "render_revisit_diff.json"),
            ],
            env=environment,
        )
        for name in ("render_offline.mp4", "mask_offline.mp4"):
            destination = case_dir / name
            if destination.exists():
                destination.unlink()
            try:
                destination.hardlink_to(render_dir / name)
            except OSError:
                shutil.copy2(render_dir / name, destination)

    print(json.dumps({"case_dir": str(case_dir), **trajectory_manifest}, indent=2))


if __name__ == "__main__":
    main()
