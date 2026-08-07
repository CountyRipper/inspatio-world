from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from phase1_lsm.data_prep import _load_first_240_geometry, _target_poses
from phase2_memory.manifest import SceneSpec, TrajectoryGroup
from phase2_memory.trajectory import write_trajectory
from scripts.phase1_lsm.run_sharedA_hardgate_5deg import mask_video_for_vae
from scripts.render_point_cloud import DepthWarper
from utils.render_warper import convert_mask_video


@dataclass
class SceneGeometry:
    frames: torch.Tensor
    depths: torch.Tensor
    intrinsics: torch.Tensor
    source_c2w: list[torch.Tensor]
    prompt: str


@dataclass
class PreparedTrajectory:
    group: TrajectoryGroup
    ref_latent: torch.Tensor
    render_latent: torch.Tensor
    mask_latent: torch.Tensor
    target_depth: torch.Tensor
    target_c2w: torch.Tensor
    intrinsics: torch.Tensor
    conditional: dict[str, torch.Tensor]
    fov_degrees: float
    trajectory_path: Path


def load_scene_geometry(scene: SceneSpec) -> SceneGeometry:
    for required in (scene.video, scene.metadata_json, scene.geometry):
        if not required.exists():
            raise FileNotFoundError(required)
    frames, depths, intrinsics, source_c2w = _load_first_240_geometry(scene.geometry)
    metadata = json.loads(scene.metadata_json.read_text(encoding="utf-8"))
    return SceneGeometry(
        frames=torch.from_numpy(frames),
        depths=torch.from_numpy(depths),
        intrinsics=intrinsics.float(),
        source_c2w=source_c2w,
        prompt=str(metadata[0]["text"]),
    )


def commanded_poses(
    geometry: SceneGeometry,
    group: TrajectoryGroup,
    trajectory_path: Path,
    device: torch.device,
) -> torch.Tensor:
    write_trajectory(
        trajectory_path,
        list(group.stations),
        initial_yaw_degrees=group.initial_yaw_degrees,
    )
    return torch.stack(
        _target_poses(trajectory_path, geometry.source_c2w[0].to(device), device)
    )


@torch.inference_mode()
def render_condition(
    geometry: SceneGeometry,
    target_c2w: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = geometry.frames.to(device=device, dtype=torch.float32)
    depths = geometry.depths.to(device=device, dtype=torch.float32)
    intrinsics = geometry.intrinsics.to(device=device, dtype=torch.float32)
    source_w2c = torch.stack([
        pose.to(device=device, dtype=torch.float32).inverse()
        for pose in geometry.source_c2w
    ])
    target_w2c = torch.linalg.inv(target_c2w)
    count = frames.shape[0]
    K_batch = intrinsics[None].expand(count, -1, -1)
    warper = DepthWarper()
    transformed = warper.compute_transformed_points(
        depths, source_w2c, target_w2c, K_batch, K_batch
    )
    coordinates = transformed[..., :2, 0] / transformed[..., 2:3, 0]
    transformed_depth = transformed[..., 2, 0]
    grid = warper.create_grid(count, frames.shape[-2], frames.shape[-1]).to(device)
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
    return render.cpu(), mask.bool().cpu(), target_depth[:, 0].float().cpu()


@torch.inference_mode()
def prepare_trajectory(
    pipeline,
    geometry: SceneGeometry,
    group: TrajectoryGroup,
    trajectory_path: Path,
    device: torch.device,
    *,
    cached_ref_latent: torch.Tensor | None = None,
    cached_conditional: dict[str, torch.Tensor] | None = None,
) -> PreparedTrajectory:
    target_c2w = commanded_poses(geometry, group, trajectory_path, device)
    render, mask, target_depth = render_condition(geometry, target_c2w, device)
    source_video = geometry.frames[None].permute(0, 2, 1, 3, 4).to(
        device=device, dtype=torch.bfloat16
    )
    render_video = render[None].permute(0, 2, 1, 3, 4).to(
        device=device, dtype=torch.bfloat16
    )
    mask_video = mask_video_for_vae(mask).to(device=device, dtype=torch.bfloat16)
    pipeline.vae.model.clear_cache()
    ref_latent = (
        pipeline.vae.encode_to_latent(source_video).to(torch.bfloat16)
        if cached_ref_latent is None
        else cached_ref_latent
    )
    pipeline.vae.model.clear_cache()
    render_latent = pipeline.vae.encode_to_latent(render_video).to(torch.bfloat16)
    pipeline.vae.model.clear_cache()
    mask_latent = convert_mask_video(mask_video).to(torch.bfloat16)
    conditional = (
        pipeline.text_encoder([geometry.prompt])
        if cached_conditional is None
        else cached_conditional
    )
    expected = (1, 60, 16, 60, 104)
    if tuple(ref_latent.shape) != expected or tuple(render_latent.shape) != expected:
        raise AssertionError("prepared latent shape mismatch")
    if tuple(mask_latent.shape) != (1, 60, 4, 60, 104):
        raise AssertionError("prepared mask latent shape mismatch")
    width = geometry.depths.shape[-1]
    fov = math.degrees(2.0 * math.atan(width / (2.0 * float(geometry.intrinsics[0, 0]))))
    return PreparedTrajectory(
        group=group,
        ref_latent=ref_latent,
        render_latent=render_latent,
        mask_latent=mask_latent,
        target_depth=target_depth.to(device),
        target_c2w=target_c2w,
        intrinsics=geometry.intrinsics.to(device),
        conditional=conditional,
        fov_degrees=fov,
        trajectory_path=trajectory_path,
    )
