#!/usr/bin/env python3
"""Capture World A/B clean x0 blocks for the exact +40-degree revisit."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from safetensors.torch import save_file
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import CausalInferencePipeline
from scripts.world_memory.common import (
    cpu_contiguous,
    init_single_gpu_distributed,
    load_configs,
    pad_clean_latent,
    resolve_repo_path,
    write_video_tensor,
)
from utils.render_warper import convert_mask_video


class StaticPromptEncoder(nn.Module):
    def __init__(self, prompt_embeds: torch.Tensor):
        super().__init__()
        self.register_buffer("prompt_embeds", prompt_embeds)

    def forward(self, text_prompts):
        if len(text_prompts) != self.prompt_embeds.shape[0]:
            raise ValueError("captured prompt batch size changed")
        return {"prompt_embeds": self.prompt_embeds}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/world_memory/exact_identity.yaml",
    )
    parser.add_argument(
        "--prepare-render",
        action="store_true",
        help="render the dense exact-pose condition before capture",
    )
    return parser.parse_args()


def build_dense_yaw(frame_count: int, points) -> np.ndarray:
    point_frames = np.asarray([int(point[0]) for point in points])
    point_yaws = np.asarray([float(point[1]) for point in points])
    if point_frames[0] != 0 or point_frames[-1] != frame_count - 1:
        raise ValueError("trajectory yaw_points must cover the complete source")
    return np.interp(np.arange(frame_count), point_frames, point_yaws)


def prepare_render(experiment_config, metadata, artifact_dir: Path) -> None:
    import decord

    from datasets import utils as dataset_utils
    from scripts.render_point_cloud import render_point_cloud

    video_path = resolve_repo_path(metadata["video_path"])
    frame_count = len(decord.VideoReader(str(video_path)))
    expected_frames = int(experiment_config.trajectory.frame_count)
    if frame_count != expected_frames:
        raise ValueError(
            f"configured trajectory expects {expected_frames} frames, got {frame_count}"
        )

    yaw = build_dense_yaw(frame_count, experiment_config.trajectory.yaw_points)
    trajectory_path = artifact_dir / "exact_0_G_0_G.txt"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    zeros = np.zeros_like(yaw)
    with trajectory_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(f"{value:.8f}" for value in zeros) + "\n")
        handle.write(" ".join(f"{value:.8f}" for value in yaw) + "\n")
        handle.write(" ".join(f"{value:.8f}" for value in zeros) + "\n")

    original_interpolation = dataset_utils.txt_interpolation

    def preserve_dense_schedule(input_list, n, mode="smooth"):
        if len(input_list) == n:
            return np.asarray(input_list, dtype=np.float64)
        return original_interpolation(input_list, n, mode=mode)

    dataset_utils.txt_interpolation = preserve_dense_schedule
    try:
        render_point_cloud(
            da3_dir=str(resolve_repo_path(experiment_config.experiment.da3_dir)),
            traj_txt_path=str(trajectory_path),
            output_dir=str(artifact_dir / "render_conditions"),
            width=832,
            height=480,
            fps=24,
            relative_to_source=False,
            rotation_only=bool(experiment_config.experiment.rotation_only),
            render_backend="warper",
        )
    finally:
        dataset_utils.txt_interpolation = original_interpolation

    block_size = int(experiment_config.experiment.block_size)
    write_start = int(experiment_config.experiment.first_g_write_block) * block_size
    return_start = int(experiment_config.experiment.final_g_return_block) * block_size
    first_pixels = yaw[4 * write_start:4 * (write_start + block_size)]
    return_pixels = yaw[4 * return_start:4 * (return_start + block_size)]
    if not (
        np.all(first_pixels == float(experiment_config.experiment.yaw_degrees))
        and np.array_equal(first_pixels, return_pixels)
    ):
        raise AssertionError("write/read blocks are not exact-pose matches")


def load_video(path: Path, *, binary: bool = False) -> torch.Tensor:
    import decord

    reader = decord.VideoReader(str(path), width=832, height=480)
    frames = torch.from_numpy(reader.get_batch(range(len(reader))).asnumpy())
    frames = rearrange(frames, "t h w c -> 1 c t h w").float().div_(255.0)
    if binary:
        frames = (frames > 0.5).to(frames.dtype)
    return frames.mul_(2.0).sub_(1.0)


def decode_latents(vae, latent: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode_to_pixel(latent, use_cache=False)
    return (decoded * 0.5 + 0.5).clamp(0, 1)


def main():
    args = parse_args()
    experiment_config, base_config = load_configs(args.config)
    artifact_dir = resolve_repo_path(experiment_config.experiment.artifact_dir)
    captured_dir = artifact_dir / "captured_latents"
    videos_dir = artifact_dir / "videos"
    captured_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_repo_path(experiment_config.experiment.metadata_json)
    with metadata_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if len(entries) != 1:
        raise ValueError("the minimal exact-identity capture expects one scene")
    metadata = entries[0]

    if args.prepare_render:
        prepare_render(experiment_config, metadata, artifact_dir)

    render_dir = artifact_dir / "render_conditions"
    render_video_path = render_dir / "render_offline.mp4"
    mask_video_path = render_dir / "mask_offline.mp4"
    if not render_video_path.exists() or not mask_video_path.exists():
        raise FileNotFoundError("run capture with --prepare-render first")

    device = init_single_gpu_distributed()
    pipeline = CausalInferencePipeline(base_config, device=device)
    checkpoint_path = resolve_repo_path(experiment_config.experiment.checkpoint)
    state_dict = torch.load(checkpoint_path) if checkpoint_path.suffix == ".pt" else None
    if state_dict is None:
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    missing, unexpected = pipeline.generator.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    del state_dict

    pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval().requires_grad_(False)

    prompt = metadata["text"]
    with torch.no_grad():
        prompt_embeds = pipeline.text_encoder([prompt])["prompt_embeds"].detach()
    pipeline.text_encoder = StaticPromptEncoder(prompt_embeds).to(device=device)
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        source_video = load_video(resolve_repo_path(metadata["video_path"]))
        ref_latent = pipeline.vae.encode_to_latent(
            source_video.to(device=device, dtype=torch.bfloat16)
        ).to(torch.bfloat16)
        del source_video
        torch.cuda.empty_cache()

        render_video = load_video(render_video_path)
        render_latent = pipeline.vae.encode_to_latent(
            render_video.to(device=device, dtype=torch.bfloat16)
        ).to(torch.bfloat16)
        del render_video
        torch.cuda.empty_cache()

        mask_video = load_video(mask_video_path, binary=True)
        mask_latent = convert_mask_video(
            mask_video.to(device=device, dtype=torch.bfloat16)
        ).to(torch.bfloat16)
        del mask_video

    available_frames = min(
        ref_latent.shape[1],
        render_latent.shape[1],
        mask_latent.shape[1],
    )
    block_size = int(experiment_config.experiment.block_size)
    num_frames = available_frames - available_frames % block_size
    ref_latent = ref_latent[:, :num_frames]
    render_latent = render_latent[:, :num_frames]
    mask_latent = mask_latent[:, :num_frames]
    print(f"Captured condition length: {num_frames} latent frames")

    denoise_seed = int(experiment_config.experiment.denoise_seed)
    write_block = int(experiment_config.experiment.first_g_write_block)
    return_block = int(experiment_config.experiment.final_g_return_block)
    worlds = {}
    for world_name, seed in (
        ("A", int(experiment_config.experiment.world_a_seed)),
        ("B", int(experiment_config.experiment.world_b_seed)),
    ):
        noise_generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            1,
            num_frames,
            16,
            ref_latent.shape[-2],
            ref_latent.shape[-1],
            generator=noise_generator,
            device=device,
            dtype=torch.bfloat16,
        )
        blocks = {}

        def capture_block(block_index, latent_start, denoised_latent):
            blocks[block_index] = denoised_latent.cpu().contiguous()

        torch.manual_seed(denoise_seed + seed)
        torch.cuda.manual_seed_all(denoise_seed + seed)
        with torch.no_grad():
            output = pipeline.inference(
                noise=noise,
                text_prompts=[prompt],
                ref_latent=ref_latent,
                render_latent=render_latent,
                mask_latent=mask_latent,
                decode=False,
                block_output_callback=capture_block,
            )
        if write_block not in blocks or len(blocks) != num_frames // block_size:
            raise AssertionError("block_output_callback did not capture final clean x0")

        target = blocks[write_block]
        worlds[world_name] = {
            "seed": seed,
            "noise": noise.detach().cpu().contiguous(),
            "output": output.detach().cpu().contiguous(),
            "target": target,
        }
        save_file(
            cpu_contiguous({"latent": target}),
            str(captured_dir / f"M_{world_name}.safetensors"),
        )
        save_file(
            cpu_contiguous({"noise": noise, "denoised": output}),
            str(captured_dir / f"world_{world_name}.safetensors"),
        )

        with torch.no_grad():
            full_video = decode_latents(pipeline.vae, output)
            reference_video = decode_latents(
                pipeline.vae,
                target.to(device=device, dtype=torch.bfloat16),
            )
        write_video_tensor(videos_dir / f"world_{world_name}_capture.mp4", full_video)
        write_video_tensor(
            videos_dir / f"world_{world_name}_first_G_reference.mp4",
            reference_video,
        )
        pipeline.vae.model.clear_cache()
        del output, full_video, reference_video, noise
        torch.cuda.empty_cache()

    return_start = return_block * block_size
    previous_start = return_start - block_size
    query_ref = ref_latent[:, return_start:return_start + block_size]
    query_previous = worlds["A"]["output"][:, previous_start:return_start].to(
        device=device,
        dtype=torch.bfloat16,
    )
    query_context = torch.cat(
        [pad_clean_latent(query_ref), pad_clean_latent(query_previous)],
        dim=1,
    )
    query_render = torch.cat(
        [
            mask_latent[:, return_start:return_start + block_size],
            render_latent[:, return_start:return_start + block_size],
        ],
        dim=2,
    )

    torch.save(
        {
            "format_version": 1,
            "prompt": prompt,
            "prompt_embeds": prompt_embeds.detach().cpu().contiguous(),
            "context_frames": query_context.detach().cpu().contiguous(),
            "render_block": query_render.detach().cpu().contiguous(),
            "query_noise": worlds["A"]["noise"][:, return_start:return_start + block_size],
            "target_A": worlds["A"]["target"],
            "target_B": worlds["B"]["target"],
            "denoising_steps": pipeline.denoising_step_list.cpu(),
            "denoise_seed": denoise_seed,
            "write_block": write_block,
            "return_block": return_block,
        },
        captured_dir / "query_state.pt",
    )
    torch.save(
        {
            "prompt": prompt,
            "prompt_embeds": prompt_embeds.detach().cpu().contiguous(),
            "ref_latent": ref_latent.detach().cpu().contiguous(),
            "render_latent": render_latent.detach().cpu().contiguous(),
            "mask_latent": mask_latent.detach().cpu().contiguous(),
            "noise_A": worlds["A"]["noise"],
            "noise_B": worlds["B"]["noise"],
            "output_A": worlds["A"]["output"],
            "output_B": worlds["B"]["output"],
            "denoise_seed": denoise_seed,
        },
        captured_dir / "shared_trajectory.pt",
    )

    manifest = {
        "scene": str(resolve_repo_path(metadata["video_path"])),
        "pose": "+40 degree yaw, rotation-only",
        "timeline": "0 -> G -> 0 -> G",
        "latent_frames": num_frames,
        "block_size": block_size,
        "write_block": write_block,
        "return_block": return_block,
        "world_A_seed": worlds["A"]["seed"],
        "world_B_seed": worlds["B"]["seed"],
        "query_world": "A",
    }
    with (captured_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Capture complete: {captured_dir}")


if __name__ == "__main__":
    main()
