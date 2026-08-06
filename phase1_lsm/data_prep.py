from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.utils import generate_traj_txt
from phase1_lsm.trajectory import NUM_RGB_FRAMES, validate_target_c2w
from scripts.render_point_cloud import DepthWarper, open_ffmpeg_writer


SOURCE_SPECS = {
    "S0": {
        "video": Path("/data4/daixiangting/inspatio-world/test/example/cropped_source.mp4"),
        "json": Path("/data4/daixiangting/inspatio-world/test/example/new.json"),
        "geometry": Path("/data4/daixiangting/inspatio-world/test/example/new_vggt/cropped_source"),
    },
    "S1": {
        "video": Path("/data4/daixiangting/inspatio-world/test/example2/coffee_martini.mp4"),
        "json": Path("/data4/daixiangting/inspatio-world/test/example2/new.json"),
        "geometry": Path("/data4/daixiangting/inspatio-world/test/example2/new_vggt/coffee_martini"),
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_lines(path: Path) -> np.ndarray:
    return np.asarray(
        [ast.literal_eval(line) for line in path.read_text().splitlines() if line.strip()],
        dtype=np.float32,
    )


def _load_first_240_geometry(
    geometry_dir: Path,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, list[torch.Tensor]]:
    image_files = sorted((geometry_dir / "images").glob("*.png"))[:NUM_RGB_FRAMES]
    depth_files = sorted((geometry_dir / "depth").glob("*.png"))[:NUM_RGB_FRAMES]
    if len(image_files) != NUM_RGB_FRAMES or len(depth_files) != NUM_RGB_FRAMES:
        raise RuntimeError("geometry does not contain the required first 240 RGB/depth frames")

    depth_min, depth_max = [
        float(value) for value in (geometry_dir / "metadata.txt").read_text().split()
    ]
    frames = []
    depths = []
    for image_path, depth_path in zip(image_files, depth_files):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        frames.append(image.transpose(2, 0, 1))
        encoded_depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
        depth = encoded_depth.astype(np.float32) / 65535.0
        depths.append((depth * (depth_max - depth_min) + depth_min)[None])

    K = torch.from_numpy(_matrix_lines(geometry_dir / "intrinsics.txt")[0])
    source_w2c = _matrix_lines(geometry_dir / "extrinsics.txt")[:NUM_RGB_FRAMES]
    bottom = np.broadcast_to(
        np.array([0, 0, 0, 1], dtype=np.float32),
        (NUM_RGB_FRAMES, 1, 4),
    )
    source_w2c = np.concatenate((source_w2c, bottom), axis=1)
    source_c2w = [torch.from_numpy(matrix) for matrix in np.linalg.inv(source_w2c)]
    return np.stack(frames), np.stack(depths), K, source_c2w


def _target_poses(
    trajectory_path: Path,
    initial_c2w: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    controls = np.loadtxt(trajectory_path)
    if controls.shape != (3, NUM_RGB_FRAMES):
        raise AssertionError(f"invalid trajectory shape: {controls.shape}")
    relative = generate_traj_txt(
        controls[0], controls[1], controls[2], controls[2],
        NUM_RGB_FRAMES,
        is_translation=True,
    )
    initial_inverse = initial_c2w.to(device).inverse()
    return [
        initial_inverse @ torch.as_tensor(pose, device=device, dtype=torch.float32)
        for pose in relative
    ]


def _write_videos(render: torch.Tensor, mask: torch.Tensor, render_dir: Path) -> None:
    height, width = render.shape[-2:]
    rgb_writer = open_ffmpeg_writer(str(render_dir / "render_offline.mp4"), width, height, 24)
    mask_writer = open_ffmpeg_writer(str(render_dir / "mask_offline.mp4"), width, height, 24)
    try:
        rgb = ((render.cpu().numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5)
        rgb = rgb.clip(0, 255).astype(np.uint8)
        known = (mask.cpu().numpy().transpose(0, 2, 3, 1) * 255.0)
        known = known.clip(0, 255).astype(np.uint8)
        for rgb_frame, known_frame in zip(rgb, known):
            rgb_writer.stdin.write(rgb_frame.tobytes())
            mask_writer.stdin.write(np.repeat(known_frame, 3, axis=2).tobytes())
    finally:
        rgb_writer.stdin.close()
        mask_writer.stdin.close()
        rgb_writer.wait()
        mask_writer.wait()
    if rgb_writer.returncode or mask_writer.returncode:
        raise RuntimeError("ffmpeg failed while writing fixed conditions")


def prepare_condition(
    source_name: str,
    trajectory_name: str,
    trajectory_path: str | Path,
    output_root: str | Path,
    device: torch.device,
) -> Path:
    if source_name not in SOURCE_SPECS or trajectory_name not in ("P", "N"):
        raise ValueError("only S0/S1 and P/N are allowed")
    spec = SOURCE_SPECS[source_name]
    sample_dir = Path(output_root) / "conditions" / source_name / trajectory_name
    if sample_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing condition: {sample_dir}")
    render_dir = sample_dir / "render"
    render_dir.mkdir(parents=True)

    frames_np, depths_np, K_cpu, source_c2w_cpu = _load_first_240_geometry(spec["geometry"])
    source_c2w = [pose.to(device=device, dtype=torch.float32) for pose in source_c2w_cpu]
    target_c2w = _target_poses(Path(trajectory_path), source_c2w[0], device)
    target_c2w_np = torch.stack(target_c2w).cpu().numpy().astype(np.float32)
    pose_metrics = validate_target_c2w(target_c2w_np)

    frames = torch.from_numpy(frames_np).to(device=device, dtype=torch.float32)
    depths = torch.from_numpy(depths_np).to(device=device, dtype=torch.float32)
    K = K_cpu.to(device=device, dtype=torch.float32)
    source_w2c = torch.stack([pose.inverse() for pose in source_c2w])
    target_w2c = torch.stack([pose.inverse() for pose in target_c2w])
    K_batch = K[None].expand(NUM_RGB_FRAMES, -1, -1)

    warper = DepthWarper()
    with torch.no_grad():
        transformed = warper.compute_transformed_points(
            depths, source_w2c, target_w2c, K_batch, K_batch
        )
        coordinates = transformed[..., :2, 0] / transformed[..., 2:3, 0]
        transformed_depth = transformed[..., 2, 0]
        grid = warper.create_grid(NUM_RGB_FRAMES, frames.shape[-2], frames.shape[-1]).to(device)
        flow = coordinates.permute(0, 3, 1, 2) - grid
        render, mask = warper.bilinear_splatting(
            frames, torch.ones_like(depths), transformed_depth, flow, None, is_image=True
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
    np.save(render_dir / "depth_offline.npy", target_depth[:, 0].cpu().numpy().astype(np.float32))
    _write_videos(render, mask, render_dir)

    (sample_dir / "depth").symlink_to(spec["geometry"] / "depth", target_is_directory=True)
    (sample_dir / "metadata.txt").symlink_to(spec["geometry"] / "metadata.txt")
    source_metadata = json.loads(spec["json"].read_text())[0]
    isolated_metadata = [{
        **source_metadata,
        "video_path": str(spec["video"]),
        "vggt_depth_path": str(sample_dir),
        "vggt_extrinsics_path": str(spec["geometry"] / "extrinsics.txt"),
    }]
    (sample_dir / "new.json").write_text(
        json.dumps(isolated_metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": source_name,
        "trajectory": trajectory_name,
        "source_video": str(spec["video"]),
        "source_video_sha256": sha256_file(spec["video"]),
        "source_json": str(spec["json"]),
        "source_json_sha256": sha256_file(spec["json"]),
        "trajectory_path": str(Path(trajectory_path).resolve()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "decoded_frames_available": len(list((spec["geometry"] / "images").glob("*.png"))),
        "decoded_frames_used": [0, NUM_RGB_FRAMES],
        "adaptive_frame": False,
        "rotation_only": True,
        "pitch": 0,
        "radius": 0,
        "pose_validation": pose_metrics,
        "outputs": {
            name: sha256_file(render_dir / name)
            for name in (
                "target_c2w.npy", "intrinsic.npy", "depth_offline.npy",
                "render_offline.mp4", "mask_offline.mp4",
            )
        },
    }
    (sample_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return sample_dir
