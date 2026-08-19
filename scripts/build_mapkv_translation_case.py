from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive an exact out-and-back translation control case"
    )
    parser.add_argument(
        "--source_case",
        default="artifacts/control/yaw45m20to35_scene01",
    )
    parser.add_argument(
        "--output_case",
        default="artifacts/control/translate008_scene01",
    )
    parser.add_argument("--translation", type=float, default=0.08)
    parser.add_argument("--render_device", default="2")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source = Path(args.source_case).resolve()
    output = Path(args.output_case).resolve()
    output.mkdir(parents=True, exist_ok=True)
    phases_payload = _json(source / "phase_labels.json")
    phases = phases_payload["phases"]
    source_poses = np.load(source / "target_poses.npy")
    rgb_length = len(source_poses)
    translation = np.zeros(rgb_length, dtype=np.float64)
    endpoints = {
        "A0_hold": (0.0, 0.0),
        "A_to_B1": (0.0, args.translation),
        "B1_hold": (args.translation, args.translation),
        "B1_to_Leave": (args.translation, 0.0),
        "Leave_hold": (0.0, 0.0),
        "Leave_to_B2": (0.0, args.translation),
        "B2_hold": (args.translation, args.translation),
    }
    for phase in phases:
        start = int(phase["rgb_start"])
        stop = int(phase["rgb_stop_exclusive"])
        left, right = endpoints[phase["name"]]
        translation[start:stop] = np.linspace(
            left, right, stop - start, endpoint=True
        )
    base = np.asarray(source_poses[0], dtype=np.float64)
    poses = np.repeat(base[None], rgb_length, axis=0)
    poses[:, :3, 3] = (
        base[:3, 3][None]
        + translation[:, None] * base[:3, 0][None]
    )
    np.save(output / "target_poses.npy", poses)
    np.save(output / "translation_x.npy", translation)
    np.save(output / "yaw_pitch_roll.npy", np.zeros((rgb_length, 3)))

    for name in (
        "depth",
        "images",
        "extrinsics.txt",
        "intrinsics.txt",
        "metadata.txt",
        "static_source.mp4",
        "source_frame.png",
    ):
        _link(source / name, output / name)
    _link(source / "images", output / "frames")
    intrinsic_alias = output / "intrinsic.txt"
    if intrinsic_alias.is_symlink():
        intrinsic_alias.unlink()
    intrinsic_values = np.fromstring(
        (source / "intrinsics.txt")
        .read_text()
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " "),
        sep=" ",
    ).reshape(3, 3)
    np.savetxt(intrinsic_alias, intrinsic_values)
    extrinsic_values = np.fromstring(
        (source / "extrinsics.txt")
        .read_text()
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " "),
        sep=" ",
    ).reshape(-1, 4)
    np.savetxt(output / "extrinsic.txt", extrinsic_values)
    phase_copy = dict(phases_payload)
    phase_copy["case_id"] = output.name
    (output / "phase_labels.json").write_text(
        json.dumps(phase_copy, indent=2), encoding="utf-8"
    )
    source_manifest = _json(source / "source_manifest.json")
    source_manifest["source_video"] = str(output / "static_source.mp4")
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )
    input_payload = _json(source / "input.json")
    for item in input_payload:
        item["video_path"] = str(output / "static_source.mp4")
        item["vggt_depth_path"] = str(output)
        item["vggt_extrinsics_path"] = str(output / "extrinsics.txt")
    (output / "input.json").write_text(
        json.dumps(input_payload, indent=2), encoding="utf-8"
    )
    source_chunk = int(phases_payload["source_chunk"])
    target_chunk = int(phases_payload["target_chunk"])
    source_rgb = int(phases_payload["source_rgb_index"])
    target_rgb = int(phases_payload["target_rgb_index"])
    pose_error = float(
        np.max(np.abs(poses[source_rgb] - poses[target_rgb]))
    )
    validation = {
        "valid": pose_error < 1e-8,
        "trajectory_type": "pure_translation_out_and_back",
        "translation_axis": "initial_camera_local_x",
        "maximum_translation": float(np.max(np.abs(translation))),
        "B1_B2_pose_max_abs_error": pose_error,
        "source_chunk": source_chunk,
        "target_chunk": target_chunk,
    }
    (output / "pose_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    manifest = {
        **_json(source / "trajectory_manifest.json"),
        "case_id": output.name,
        "trajectory_type": "pure_translation_out_and_back",
        "pose_convention": (
            "absolute c2w; translation along initial camera local x"
        ),
        "target_pose_path": str(output / "target_poses.npy"),
        "target_pose_sha256": _sha256(output / "target_poses.npy"),
        "theta_degrees": 0.0,
        "b1_theta_degrees": 0.0,
        "b2_theta_degrees": 0.0,
        "leave_theta_degrees": 0.0,
        "relative_translation": True,
        "translation_axis": "initial_camera_local_x",
        "translation_distance": float(args.translation),
    }
    (output / "trajectory_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.render:
        render = output / "render"
        render.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(args.render_device)
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/render_point_cloud.py"),
                "--da3_dir",
                str(output),
                "--target_pose_path",
                str(output / "target_poses.npy"),
                "--source_frame_index",
                "0",
                "--output_dir",
                str(render),
                "--width",
                "832",
                "--height",
                "480",
                "--fps",
                "24",
                "--render_backend",
                "warper",
                "--validation_pair",
                str(source_rgb),
                str(target_rgb),
                "--render_validation_path",
                str(output / "render_revisit_diff.json"),
            ],
            check=True,
            env=environment,
        )
        for name in ("render_offline.mp4", "mask_offline.mp4"):
            shutil.copy2(render / name, output / name)
    print(json.dumps({"case": str(output), **validation}, indent=2))


if __name__ == "__main__":
    main()
