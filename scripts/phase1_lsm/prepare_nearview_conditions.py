#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase1_lsm.data_prep import (
    SOURCE_SPECS,
    _load_first_240_geometry,
    _target_poses,
    _write_videos,
    sha256_file,
)
from phase1_lsm.nearview import (
    validate_nearview_c2w,
    write_nearview_trajectory,
)
from phase1_lsm.trajectory import NUM_RGB_FRAMES
from scripts.render_point_cloud import DepthWarper


def prepare_one(
    output_root: Path,
    label: str,
    offset_degrees: float,
    device: torch.device,
) -> dict[str, object]:
    spec = SOURCE_SPECS["S0"]
    trajectory_path = write_nearview_trajectory(
        output_root / "trajectories" / f"{label}.txt", offset_degrees
    )
    condition_dir = output_root / "conditions" / label
    render_dir = condition_dir / "render"
    render_dir.mkdir(parents=True)
    frames_np, depths_np, K_cpu, source_c2w_cpu = _load_first_240_geometry(
        spec["geometry"]
    )
    source_c2w = [pose.to(device=device, dtype=torch.float32) for pose in source_c2w_cpu]
    target_c2w = _target_poses(trajectory_path, source_c2w[0], device)
    target_c2w_np = torch.stack(target_c2w).cpu().numpy().astype(np.float32)
    pose_audit = validate_nearview_c2w(target_c2w_np, offset_degrees)

    frames = torch.from_numpy(frames_np).to(device=device, dtype=torch.float32)
    depths = torch.from_numpy(depths_np).to(device=device, dtype=torch.float32)
    K = K_cpu.to(device=device, dtype=torch.float32)
    source_w2c = torch.stack([pose.inverse() for pose in source_c2w])
    target_w2c = torch.stack([pose.inverse() for pose in target_c2w])
    K_batch = K[None].expand(NUM_RGB_FRAMES, -1, -1)
    warper = DepthWarper()
    with torch.inference_mode():
        transformed = warper.compute_transformed_points(
            depths, source_w2c, target_w2c, K_batch, K_batch
        )
        coordinates = transformed[..., :2, 0] / transformed[..., 2:3, 0]
        transformed_depth = transformed[..., 2, 0]
        grid = warper.create_grid(
            NUM_RGB_FRAMES, frames.shape[-2], frames.shape[-1]
        ).to(device)
        flow = coordinates.permute(0, 3, 1, 2) - grid
        render, mask = warper.bilinear_splatting(
            frames,
            torch.ones_like(depths),
            transformed_depth,
            flow,
            None,
            is_image=True,
        )
        target_depth, depth_mask = warper.bilinear_splatting(
            transformed_depth[:, None],
            torch.ones_like(depths),
            transformed_depth,
            flow,
            None,
            is_image=False,
        )
    if not torch.equal(mask, depth_mask):
        raise AssertionError("RGB and depth occupancy disagree")

    np.save(render_dir / "target_c2w.npy", target_c2w_np)
    np.save(render_dir / "intrinsic.npy", K_cpu.numpy().astype(np.float32))
    np.save(
        render_dir / "depth_offline.npy",
        target_depth[:, 0].cpu().numpy().astype(np.float32),
    )
    _write_videos(render, mask, render_dir)
    (condition_dir / "depth").symlink_to(
        spec["geometry"] / "depth", target_is_directory=True
    )
    (condition_dir / "metadata.txt").symlink_to(spec["geometry"] / "metadata.txt")
    source_metadata = json.loads(spec["json"].read_text())[0]
    isolated_metadata = [{
        **source_metadata,
        "video_path": str(spec["video"]),
        "vggt_depth_path": str(condition_dir),
        "vggt_extrinsics_path": str(spec["geometry"] / "extrinsics.txt"),
    }]
    (condition_dir / "new.json").write_text(
        json.dumps(isolated_metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": "S0",
        "trajectory": label,
        "offset_degrees": offset_degrees,
        "a_yaw_degrees": 45.0,
        "aprime_yaw_degrees": 45.0 + offset_degrees,
        "source_video": str(spec["video"]),
        "source_video_sha256": sha256_file(spec["video"]),
        "source_json": str(spec["json"]),
        "source_json_sha256": sha256_file(spec["json"]),
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "decoded_frames_used": [0, NUM_RGB_FRAMES],
        "adaptive_frame": False,
        "rotation_only": True,
        "pose_audit": pose_audit,
        "outputs": {
            name: sha256_file(render_dir / name)
            for name in (
                "target_c2w.npy",
                "intrinsic.npy",
                "depth_offline.npy",
                "render_offline.mp4",
                "mask_offline.mp4",
            )
        },
    }
    (condition_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--offset", type=float, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    offsets = tuple(args.offset)
    if offsets not in ((5.0, -5.0), (10.0, -10.0)):
        raise ValueError("offsets must be exactly +5,-5 or +10,-10 in that order")
    output_root.mkdir(parents=True)
    labels = ("plus5", "minus5") if offsets[0] == 5.0 else ("plus10", "minus10")
    manifests = [
        prepare_one(output_root, label, offset, torch.device(args.device))
        for label, offset in zip(labels, offsets)
    ]
    a_poses = [
        np.load(output_root / "conditions" / label / "render/target_c2w.npy")[
            [60, 64, 68]
        ]
        for label in labels
    ]
    if not np.allclose(a_poses[0], a_poses[1], atol=1e-6, rtol=0):
        raise AssertionError("plus/minus trajectories do not share the same A pose")
    (output_root / "condition_audit.json").write_text(
        json.dumps({"passed": True, "conditions": manifests}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": True, "conditions": manifests}, indent=2))


if __name__ == "__main__":
    main()
