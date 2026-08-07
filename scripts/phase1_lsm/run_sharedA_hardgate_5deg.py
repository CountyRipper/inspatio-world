#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from phase1_lsm.adapter import ADAPTER_PARAMETER_COUNT
from phase1_lsm.data_prep import SOURCE_SPECS, _load_first_240_geometry, _target_poses, sha256_file
from phase1_lsm.latent_projection import project_memory_sequence
from phase1_lsm.nearview import (
    projection_displacement_statistics,
    validate_nearview_c2w,
    write_nearview_trajectory,
)
from phase1_lsm.trajectory import A_KEYFRAMES, APRIME_KEYFRAMES, NUM_RGB_FRAMES
from pipeline.causal_inference import CausalInferencePipeline, denoise_block
from scripts.render_point_cloud import DepthWarper
from utils.render_warper import convert_mask_video


LABEL_OFFSETS = (("plus5", 5.0), ("minus5", -5.0))
WAN_ROOT = Path("/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B")


def cpu_tensor(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = tensor.detach().contiguous().cpu()
    return value if dtype is None else value.to(dtype=dtype)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def clone_cache(cache: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{"k": item["k"].clone(), "v": item["v"].clone()} for item in cache]


def cache_torch_equal(
    left: list[dict[str, torch.Tensor]],
    right: list[dict[str, torch.Tensor]],
) -> bool:
    return len(left) == len(right) and all(
        torch.equal(a[key], b[key])
        for a, b in zip(left, right)
        for key in ("k", "v")
    )


def mask_video_for_vae(mask_t1hw: torch.Tensor) -> torch.Tensor:
    """Convert lossless warper mask [T,1,H,W] to VAE [B,1,T,H,W]."""
    if mask_t1hw.ndim != 4 or mask_t1hw.shape[1] != 1:
        raise ValueError(
            "lossless mask must be [T,1,H,W], got "
            f"{tuple(mask_t1hw.shape)}"
        )
    return mask_t1hw.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()


def render_lossless_condition(
    frames_cpu: torch.Tensor,
    depths_cpu: torch.Tensor,
    intrinsics_cpu: torch.Tensor,
    source_c2w_cpu: list[torch.Tensor],
    trajectory_path: Path,
    offset: float,
    device: torch.device,
    shared_prefix: dict[str, torch.Tensor | dict[str, object]] | None = None,
) -> dict[str, torch.Tensor | dict[str, object]]:
    source_c2w = [pose.to(device=device, dtype=torch.float32) for pose in source_c2w_cpu]
    target_c2w = torch.stack(_target_poses(trajectory_path, source_c2w[0], device))
    pose_audit = validate_nearview_c2w(target_c2w.cpu().numpy(), offset)
    target_c2w_cpu = cpu_tensor(target_c2w, torch.float32)
    start = 0 if shared_prefix is None else 69
    if shared_prefix is not None and not torch.equal(
        target_c2w_cpu[:start], shared_prefix["target_c2w"][:start]
    ):
        raise AssertionError("branch target c2w changed the shared A prefix")
    frames = frames_cpu[start:].to(device=device, dtype=torch.float32)
    depths = depths_cpu[start:].to(device=device, dtype=torch.float32)
    intrinsics = intrinsics_cpu.to(device=device, dtype=torch.float32)
    source_w2c = torch.stack([pose.inverse() for pose in source_c2w[start:]])
    target_w2c = torch.linalg.inv(target_c2w[start:])
    frame_count = NUM_RGB_FRAMES - start
    K_batch = intrinsics[None].expand(frame_count, -1, -1)
    warper = DepthWarper()
    with torch.inference_mode():
        transformed = warper.compute_transformed_points(depths, source_w2c, target_w2c, K_batch, K_batch)
        coordinates = transformed[..., :2, 0] / transformed[..., 2:3, 0]
        transformed_depth = transformed[..., 2, 0]
        grid = warper.create_grid(frame_count, frames.shape[-2], frames.shape[-1]).to(device)
        flow = coordinates.permute(0, 3, 1, 2) - grid
        render, mask = warper.bilinear_splatting(
            frames, torch.ones_like(depths), transformed_depth, flow, None, is_image=True
        )
        target_depth, depth_mask = warper.bilinear_splatting(
            transformed_depth[:, None], torch.ones_like(depths), transformed_depth,
            flow, None, is_image=False,
        )
    if not torch.equal(mask, depth_mask):
        raise AssertionError("RGB and depth lossless occupancies differ")
    render_cpu = cpu_tensor(render, torch.float16)
    mask_cpu = cpu_tensor(mask, torch.bool)
    depth_cpu = cpu_tensor(target_depth[:, 0], torch.float32)
    if shared_prefix is not None:
        render_cpu = torch.cat((shared_prefix["render"][:start], render_cpu))
        mask_cpu = torch.cat((shared_prefix["mask"][:start], mask_cpu))
        depth_cpu = torch.cat((shared_prefix["target_depth"][:start], depth_cpu))
    return {
        "render": render_cpu.contiguous(),
        "mask": mask_cpu.contiguous(),
        "target_depth": depth_cpu.contiguous(),
        "target_c2w": target_c2w_cpu,
        "pose_audit": pose_audit,
    }


def padded_context(ref_block: torch.Tensor, last_pred: torch.Tensor | None) -> tuple[torch.Tensor, int]:
    ref_padded = torch.cat((ref_block, torch.zeros_like(ref_block[:, :, :4]), torch.zeros_like(ref_block)), dim=2)
    if last_pred is None:
        return ref_padded, 1560 * 3
    previous_padded = torch.cat((last_pred, torch.zeros_like(last_pred[:, :, :4]), torch.zeros_like(last_pred)), dim=2)
    return torch.cat((ref_padded, previous_padded), dim=1), 1560 * 6


@torch.inference_mode()
def run_blocks(
    pipeline: CausalInferencePipeline,
    noise: torch.Tensor,
    ref_latent: torch.Tensor,
    render_latent: torch.Tensor,
    mask_latent: torch.Tensor,
    conditional: dict[str, torch.Tensor],
    cache: list[dict[str, torch.Tensor]],
    output: torch.Tensor,
    last_pred: torch.Tensor | None,
    first_block: int,
    last_block: int,
    capture_count: list[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, float]]:
    captured: dict[int, torch.Tensor] = {}
    step_inputs: dict[int, torch.Tensor] = {}
    transition_noises: dict[int, torch.Tensor] = {}
    step_timesteps: dict[int, float] = {}
    for block_index in range(first_block, last_block + 1):
        start = block_index * 3
        noisy_input = noise[:, start:start + 3]
        ref_block = ref_latent[:, start:start + 3]
        render_block = torch.cat((mask_latent[:, start:start + 3], render_latent[:, start:start + 3]), dim=2)
        context, kv_size = padded_context(ref_block, last_pred)

        def callback(*, step_index, timestep, noisy_input, transition_noise):
            if block_index != 19:
                return
            step_inputs[step_index] = cpu_tensor(noisy_input, torch.bfloat16)
            step_timesteps[step_index] = float(timestep)
            if transition_noise is not None:
                transition_noises[step_index] = cpu_tensor(transition_noise, torch.bfloat16)

        prediction, _ = denoise_block(
            pipeline.generator, pipeline.scheduler, noisy_input, conditional, cache,
            context_frames=context, context_no_grad=True, context_freqs_offset=0,
            render_block=render_block, denoising_kv_size=kv_size,
            denoising_steps=pipeline.denoising_step_list,
            step_callback=callback if block_index == 19 else None,
        )
        output[:, start:start + 3] = prediction
        last_pred = prediction.clone().detach()
        if block_index == 5:
            capture_count[0] += 1
        if block_index in (5, 13, 18, 19):
            captured[block_index] = cpu_tensor(last_pred, torch.bfloat16)
    return output, last_pred, captured, step_inputs, transition_noises, step_timesteps


def save_shared_state(
    root: Path,
    source_frames: torch.Tensor,
    shared: dict[str, torch.Tensor | dict[str, object]],
    prefix: torch.Tensor,
    z_a: torch.Tensor,
    ref_prefix: torch.Tensor,
    render_prefix: torch.Tensor,
    mask_prefix: torch.Tensor,
    last_pred: torch.Tensor,
    cache: list[dict[str, torch.Tensor]],
    remaining_noise: torch.Tensor,
    rng_cpu: torch.Tensor,
    rng_cuda: torch.Tensor,
    intrinsics: torch.Tensor,
) -> dict[str, str]:
    raw_path = root / "shared_A_raw.safetensors"
    save_file({
        "shared_source_raw_0_69": source_frames[:69].to(torch.float16).contiguous(),
        "shared_render_raw_0_69": shared["render"][:69].contiguous(),
        "shared_mask_raw_0_69": shared["mask"][:69].contiguous(),
        "shared_depth_raw_0_69": shared["target_depth"][:69].contiguous(),
        "K": intrinsics.float().contiguous(),
        "shared_c2w_0_69": shared["target_c2w"][:69].contiguous(),
    }, raw_path)
    state_tensors = {
        "latent_prefix_0_18": cpu_tensor(prefix, torch.bfloat16),
        "unique_z_A": z_a.contiguous(),
        "ref_latent_prefix_0_18": cpu_tensor(ref_prefix, torch.bfloat16),
        "render_latent_prefix_0_18": cpu_tensor(render_prefix, torch.bfloat16),
        "mask_latent_prefix_0_18": cpu_tensor(mask_prefix, torch.bfloat16),
        "last_pred": cpu_tensor(last_pred, torch.bfloat16),
        "remaining_initial_noise": cpu_tensor(remaining_noise, torch.bfloat16),
        "rng_cpu_at_fork": rng_cpu.contiguous(),
        "rng_cuda_at_fork": rng_cuda.contiguous(),
    }
    for index, item in enumerate(cache):
        state_tensors[f"kv_{index:02d}_k"] = cpu_tensor(item["k"], torch.bfloat16)
        state_tensors[f"kv_{index:02d}_v"] = cpu_tensor(item["v"], torch.bfloat16)
    state_path = root / "shared_A_causal_state.safetensors"
    save_file(state_tensors, state_path)
    return {"shared_A_raw_sha256": sha256_file(raw_path), "shared_A_causal_state_sha256": sha256_file(state_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    root = Path(args.output_root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    checkpoint_hash = sha256_file(args.checkpoint)
    init_hash = sha256_file(args.init_adapter)
    command_log = root / "COMMAND_LOG.md"
    command_log.write_text(
        "# Phase 1 shared-A hard-gate command log\n\n"
        f"- Start: `{datetime.now().astimezone().isoformat()}`\n"
        f"- Unique external GPU command: `{shlex.join([sys.executable, *sys.argv])}`\n"
        f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`\n"
        f"- Base checkpoint SHA256 before: `{checkpoint_hash}`\n"
        f"- Fixed8 projected adapter SHA256 before: `{init_hash}`\n"
        "- Scope: S0, seed0, +5/-5, one shared block-0:5 rollout then causal-state fork\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/inference_1.3b.yaml"))
    config.wan_model_folder = str(WAN_ROOT)
    config.generator.weight_list[0].path = str(WAN_ROOT)
    pipeline = CausalInferencePipeline(config, device=device)
    incompatible = pipeline.generator.load_state_dict(load_file(args.checkpoint), strict=False)
    if set(incompatible.missing_keys) != {"model.memory_adapter.proj.weight"} or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.eval().requires_grad_(False)
    adapter = pipeline.generator.model.memory_adapter
    if adapter.parameter_count != ADAPTER_PARAMETER_COUNT or torch.count_nonzero(adapter.proj.weight).item() != 0:
        raise AssertionError("base capture adapter must be zero and have 122,880 parameters")

    spec = SOURCE_SPECS["S0"]
    frames_np, depths_np, K_cpu, source_c2w_cpu = _load_first_240_geometry(spec["geometry"])
    source_frames = torch.from_numpy(frames_np)
    source_depths = torch.from_numpy(depths_np)
    conditions_cpu = {}
    for label, offset in LABEL_OFFSETS:
        trajectory = write_nearview_trajectory(root / "trajectories" / f"{label}.txt", offset)
        shared_prefix = None if label == "plus5" else conditions_cpu["plus5"]
        conditions_cpu[label] = render_lossless_condition(
            source_frames, source_depths, K_cpu, source_c2w_cpu,
            trajectory, offset, device, shared_prefix,
        )
        save_file({
            "render_raw": conditions_cpu[label]["render"],
            "mask_raw": conditions_cpu[label]["mask"],
            "target_depth_raw": conditions_cpu[label]["target_depth"],
            "K": K_cpu.float().contiguous(),
            "target_c2w": conditions_cpu[label]["target_c2w"],
        }, root / f"lossless_condition_{label}.safetensors")
    for name in ("render", "mask", "target_depth", "target_c2w"):
        if not torch.equal(conditions_cpu["plus5"][name][:69], conditions_cpu["minus5"][name][:69]):
            raise AssertionError(f"shared raw A prefix differs for {name}")

    source_video = source_frames[None].permute(0, 2, 1, 3, 4).to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        pipeline.vae.model.clear_cache()
        ref_latent = pipeline.vae.encode_to_latent(source_video).to(torch.bfloat16)
        pipeline.vae.model.clear_cache()
        render_latents = {}
        mask_latents = {}
        for label, _ in LABEL_OFFSETS:
            render_video = conditions_cpu[label]["render"][None].permute(0, 2, 1, 3, 4).to(device=device, dtype=torch.bfloat16)
            mask_video = mask_video_for_vae(conditions_cpu[label]["mask"]).to(device=device, dtype=torch.bfloat16)
            render_latents[label] = pipeline.vae.encode_to_latent(render_video).to(torch.bfloat16)
            pipeline.vae.model.clear_cache()
            mask_latents[label] = convert_mask_video(mask_video).to(torch.bfloat16)
    expected = (1, 60, 16, 60, 104)
    if tuple(ref_latent.shape) != expected or any(tuple(value.shape) != expected for value in render_latents.values()):
        raise AssertionError("lossless 240-frame VAE shape mismatch")
    if any(tuple(value.shape) != (1, 60, 4, 60, 104) for value in mask_latents.values()):
        raise AssertionError("lossless mask latent shape mismatch")
    if not torch.equal(render_latents["plus5"][:, :18], render_latents["minus5"][:, :18]):
        raise AssertionError("shared lossless render latent prefix differs")
    if not torch.equal(mask_latents["plus5"][:, :18], mask_latents["minus5"][:, :18]):
        raise AssertionError("shared lossless mask latent prefix differs")

    prompt = json.loads(spec["json"].read_text())[0]["text"]
    with torch.inference_mode():
        conditional = pipeline.text_encoder([prompt])
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    rng_cpu_before_noise = torch.get_rng_state()
    rng_cuda_before_noise = torch.cuda.get_rng_state(device)
    noise = torch.randn(expected, device=device, dtype=torch.bfloat16)
    pipeline._initialize_kv_cache(1, torch.bfloat16, device)
    output_shared = torch.zeros_like(noise)
    capture_count = [0]
    output_shared, shared_last, shared_captured, _, _, _ = run_blocks(
        pipeline, noise, ref_latent, render_latents["plus5"], mask_latents["plus5"],
        conditional, pipeline.kv_cache1, output_shared, None, 0, 5, capture_count,
    )
    if capture_count[0] != 1 or set(shared_captured) != {5}:
        raise AssertionError("A must be captured exactly once")
    unique_z_a = shared_captured[5]
    shared_prefix = output_shared[:, :18].clone()
    shared_cache = clone_cache(pipeline.kv_cache1)
    rng_cpu_at_fork = torch.get_rng_state()
    rng_cuda_at_fork = torch.cuda.get_rng_state(device)
    plus_cache = clone_cache(shared_cache)
    minus_cache = clone_cache(shared_cache)
    plus_prefix = shared_prefix.clone()
    minus_prefix = shared_prefix.clone()
    plus_last = shared_last.clone()
    minus_last = shared_last.clone()
    pre_fork_equal = (
        torch.equal(plus_prefix, minus_prefix)
        and torch.equal(plus_last, minus_last)
        and cache_torch_equal(plus_cache, minus_cache)
        and cache_torch_equal(plus_cache, shared_cache)
    )
    if not pre_fork_equal:
        raise AssertionError("causal state is not torch.equal before fork")
    shared_hashes = save_shared_state(
        root, source_frames, conditions_cpu["plus5"], shared_prefix, unique_z_a,
        ref_latent[:, :18], render_latents["plus5"][:, :18], mask_latents["plus5"][:, :18],
        shared_last, shared_cache, noise[:, 18:], rng_cpu_at_fork, rng_cuda_at_fork, K_cpu,
    )

    branch_results = {}
    for label, _ in LABEL_OFFSETS:
        torch.set_rng_state(rng_cpu_at_fork)
        torch.cuda.set_rng_state(rng_cuda_at_fork, device)
        branch_cache = plus_cache if label == "plus5" else minus_cache
        branch_output = torch.zeros_like(noise)
        branch_output[:, :18] = plus_prefix if label == "plus5" else minus_prefix
        branch_last = plus_last if label == "plus5" else minus_last
        branch_output, branch_last, captured, step_inputs, transition_noises, step_timesteps = run_blocks(
            pipeline, noise, ref_latent, render_latents[label], mask_latents[label], conditional,
            branch_cache, branch_output, branch_last, 6, 19, capture_count,
        )
        if set(captured) != {13, 18, 19} or set(step_inputs) != {0, 1, 2, 3} or set(transition_noises) != {0, 1, 2}:
            raise AssertionError(f"{label} continuation capture incomplete")
        branch_results[label] = {
            "output": branch_output,
            "captured": captured,
            "step_inputs": step_inputs,
            "transition_noises": transition_noises,
            "step_timesteps": step_timesteps,
        }
    if capture_count[0] != 1:
        raise AssertionError("branch continuation recaptured A")

    projection_samples = {}
    projection_audit = {"passed": True, "shared_source": "unique z_A/A-depth/K/A-c2w", "samples": {}}
    z_a_device = unique_z_a.to(device=device, dtype=torch.bfloat16)
    shared_depth = conditions_cpu["plus5"]["target_depth"][A_KEYFRAMES].to(device)
    shared_source_c2w = conditions_cpu["plus5"]["target_c2w"][A_KEYFRAMES].to(device)
    for label, offset in LABEL_OFFSETS:
        target_c2w = conditions_cpu[label]["target_c2w"]
        projected, memory_mask4, occupancy = project_memory_sequence(
            z_a_device, shared_depth, K_cpu.to(device), shared_source_c2w,
            target_c2w[APRIME_KEYFRAMES].to(device),
        )
        if torch.equal(projected, z_a_device) or not occupancy.any():
            raise AssertionError(f"{label} projection must be non-identity and nonempty")
        displacement = projection_displacement_statistics(
            shared_depth, K_cpu.to(device), shared_source_c2w,
            target_c2w[APRIME_KEYFRAMES].to(device), z_a_device.shape[-2:],
        )
        if displacement["mean_pixel_displacement"] <= 0:
            raise AssertionError(f"{label} projection displacement is zero")
        pose = conditions_cpu[label]["pose_audit"]
        projection_audit["samples"][label] = {
            "requested_offset_degrees": offset,
            "pose": pose,
            "occupancy_valid_fraction": float(occupancy.float().mean()),
            "projection_torch_equal_z_A": False,
            "latent_displacement": displacement,
        }
        projection_samples[label] = (projected, memory_mask4, occupancy)
    (root / "projection_audit.json").write_text(json.dumps(projection_audit, indent=2) + "\n", encoding="utf-8")

    z_hash = tensor_sha256(unique_z_a)
    shared_audit = {
        "passed": True,
        "A_capture_count": capture_count[0],
        "branches_reference_same_z_A_hash": {label: z_hash for label, _ in LABEL_OFFSETS},
        "branches_z_A_hash_equal": True,
        "branches_z_A_torch_equal": True,
        "pre_fork_prefix_torch_equal": bool(torch.equal(plus_prefix, minus_prefix)),
        "pre_fork_last_pred_torch_equal": bool(torch.equal(plus_last, minus_last)),
        "pre_fork_kv_cache_torch_equal": cache_torch_equal(plus_cache, minus_cache),
        "pre_fork_state_torch_equal": pre_fork_equal,
        "rng_state_cloned_at_fork": True,
        "remaining_initial_noise_shared": True,
        "actual_yaw_degrees": {
            label: conditions_cpu[label]["pose_audit"]["actual_signed_yaw_delta_degrees"]
            for label, _ in LABEL_OFFSETS
        },
        "max_camera_center_drift": {
            label: conditions_cpu[label]["pose_audit"]["max_camera_center_drift"]
            for label, _ in LABEL_OFFSETS
        },
        "projection_non_identity": {label: True for label, _ in LABEL_OFFSETS},
        "occupancy_nonempty": {label: True for label, _ in LABEL_OFFSETS},
        "latent_displacement": {
            label: projection_audit["samples"][label]["latent_displacement"]
            for label, _ in LABEL_OFFSETS
        },
        "shared_state_artifacts": shared_hashes,
        "rng_cpu_before_noise_sha256": tensor_sha256(rng_cpu_before_noise),
        "rng_cuda_before_noise_sha256": tensor_sha256(rng_cuda_before_noise),
    }
    if any(abs(value[0] - offset) > 0.1 for (label, offset), value in zip(LABEL_OFFSETS, shared_audit["actual_yaw_degrees"].values())):
        raise AssertionError("actual branch yaw is not ±5 degrees")
    if any(value != 0.0 for value in shared_audit["max_camera_center_drift"].values()):
        raise AssertionError("camera-center drift is nonzero")
    (root / "shared_A_audit.json").write_text(json.dumps(shared_audit, indent=2) + "\n", encoding="utf-8")

    samples_root = root / "samples"
    for label, _ in LABEL_OFFSETS:
        sample_dir = samples_root / label
        sample_dir.mkdir(parents=True)
        result = branch_results[label]
        projected, memory_mask4, occupancy = projection_samples[label]
        tensors = {
            "z_A": unique_z_a.contiguous(),
            "z_B": result["captured"][13].contiguous(),
            "latent_prefix_0_18": cpu_tensor(result["output"][:, :57], torch.bfloat16),
            "block18_previous": result["captured"][18].contiguous(),
            "block19_base_render16": cpu_tensor(render_latents[label][:, 57:60], torch.bfloat16),
            "block19_base_mask4": cpu_tensor(mask_latents[label][:, 57:60], torch.bfloat16),
            "block19_ref16": cpu_tensor(ref_latent[:, 57:60], torch.bfloat16),
            "z_Aprime_no_memory": result["captured"][19].contiguous(),
            "projected_memory_latent16": cpu_tensor(projected, torch.bfloat16),
            "projected_memory_mask4": cpu_tensor(memory_mask4, torch.bfloat16),
            "projected_occupancy1": cpu_tensor(occupancy, torch.bool),
            "denoise_step_inputs": torch.stack([result["step_inputs"][i] for i in range(4)]),
            "transition_noises": torch.stack([result["transition_noises"][i] for i in range(3)]),
            "prompt_embeds": cpu_tensor(conditional["prompt_embeds"], torch.bfloat16),
            "shared_A_depth": conditions_cpu["plus5"]["target_depth"][A_KEYFRAMES].contiguous(),
            "K": K_cpu.float().contiguous(),
            "planned_c2w": conditions_cpu[label]["target_c2w"].contiguous(),
            "rng_cpu_at_fork": rng_cpu_at_fork.contiguous(),
            "rng_cuda_at_fork": rng_cuda_at_fork.contiguous(),
        }
        sample_path = sample_dir / "sample.safetensors"
        save_file(tensors, sample_path)
        manifest = {
            "source": "S0", "seed": 0, "label": label,
            "true_shared_A": True, "A_capture_count": 1, "z_A_sha256": z_hash,
            "shared_A_depth_K_c2w": True,
            "branch_starts_after_block_5": True,
            "raw_lossless_inputs": True, "h264_training_truth": False,
            "denoising_step_indices": [1000, 750, 500, 250],
            "actual_model_timesteps": [result["step_timesteps"][i] for i in range(4)],
            "tensor_sha256": sha256_file(sample_path),
            "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        }
        (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    capture_seconds = time.perf_counter() - started
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- Shared-A capture/projection finish: `{datetime.now().astimezone().isoformat()}`\n"
            f"- Capture/projection seconds: `{capture_seconds:.3f}`\n"
            f"- Capture peak VRAM GiB: `{torch.cuda.max_memory_allocated(device) / 2**30:.6f}`\n"
        )
    del pipeline, noise, ref_latent, render_latents, mask_latents, branch_results
    del plus_cache, minus_cache, shared_cache, output_shared, conditions_cpu
    gc.collect(); torch.cuda.empty_cache()

    from scripts.phase1_lsm.train_sharedA_hardgate import main as train_main
    original_argv = sys.argv
    sys.argv = [
        "train_sharedA_hardgate.py", "--root", str(root),
        "--checkpoint", args.checkpoint, "--init-adapter", args.init_adapter,
        "--repo-root", args.repo_root, "--max-steps", "200", "--lr", "0.001",
        "--preservation-weight", "0.5", "--wan-root", str(WAN_ROOT),
    ]
    try:
        train_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
