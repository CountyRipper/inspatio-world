#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from datasets.video_dataset import VideoDataset
from phase1_lsm.adapter import ADAPTER_PARAMETER_COUNT
from phase1_lsm.data_prep import sha256_file
from phase1_lsm.latent_projection import (
    identity_reprojection_error,
    project_memory_sequence,
)
from phase1_lsm.trajectory import (
    A_KEYFRAMES,
    APRIME_KEYFRAMES,
    validate_target_c2w,
)
from pipeline.causal_inference import CausalInferencePipeline
from utils.render_warper import convert_mask_video


WAN_ROOT = Path("/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B")


def _cpu(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = tensor.detach().contiguous().cpu()
    return tensor if dtype is None else tensor.to(dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite sample: {output_dir}")
    output_dir.mkdir(parents=True)
    condition_dir = Path(args.condition_dir)
    condition_manifest = json.loads((condition_dir / "manifest.json").read_text())
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load("configs/inference_1.3b.yaml"),
    )
    config.wan_model_folder = str(WAN_ROOT)
    config.generator.weight_list[0].path = str(WAN_ROOT)
    config.dataset.json_path = str(condition_dir / "new.json")
    config.dataset.min_num_frames = 240
    config.dataset.max_num_frames = 240
    config.dataset.adaptive_frame = False
    config.dataset.rotation_only = True
    config.dataset.traj_txt_path = condition_manifest["trajectory_path"]

    pipeline = CausalInferencePipeline(config, device=device)
    incompatible = pipeline.generator.load_state_dict(load_file(args.checkpoint), strict=False)
    if set(incompatible.missing_keys) != {"model.memory_adapter.proj.weight"}:
        raise RuntimeError(f"unexpected missing checkpoint keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.eval().requires_grad_(False)
    adapter = pipeline.generator.model.memory_adapter
    if adapter.parameter_count != ADAPTER_PARAMETER_COUNT:
        raise AssertionError(adapter.parameter_count)
    if torch.count_nonzero(adapter.proj.weight).item() != 0:
        raise AssertionError("adapter is not zero initialized")

    dataset = VideoDataset(**OmegaConf.to_container(config.dataset, resolve=True))
    batch = dataset[0]
    target_c2w_np = batch["target_c2w"].numpy().astype(np.float32)
    pose_metrics = validate_target_c2w(target_c2w_np)
    if batch["source_video"].shape[0] != 240:
        raise AssertionError(f"expected 240 source frames, got {batch['source_video'].shape}")

    render_video = rearrange(batch["render_video"][None], "b t c h w -> b c t h w").to(
        device=device, dtype=torch.bfloat16
    )
    mask_video = rearrange(batch["mask_video"][None], "b t c h w -> b c t h w").to(
        device=device, dtype=torch.bfloat16
    )
    source_video = rearrange(batch["source_video"][None], "b t c h w -> b c t h w").to(
        device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        render_latent = pipeline.vae.encode_to_latent(render_video).to(torch.bfloat16)
        mask_latent = convert_mask_video(mask_video).to(torch.bfloat16)
        ref_latent = pipeline.vae.encode_to_latent(source_video).to(torch.bfloat16)
    expected = (1, 60, 16, 60, 104)
    if tuple(ref_latent.shape) != expected or tuple(render_latent.shape) != expected:
        raise AssertionError(
            f"240 RGB frames must encode to {expected}, got {ref_latent.shape}/{render_latent.shape}"
        )
    if tuple(mask_latent.shape) != (1, 60, 4, 60, 104):
        raise AssertionError(mask_latent.shape)

    rng_cpu_before_noise = torch.get_rng_state()
    rng_cuda_before_noise = torch.cuda.get_rng_state(device)
    sampled_noise = torch.randn(expected, device=device, dtype=torch.bfloat16)
    captured_blocks: dict[int, torch.Tensor] = {}
    step_inputs: dict[int, torch.Tensor] = {}
    transition_noises: dict[int, torch.Tensor] = {}
    step_timesteps: dict[int, int] = {}

    def block_output_callback(*, block_index, latent_start, denoised_latent, dit_ms):
        if block_index in (5, 13, 18, 19):
            captured_blocks[block_index] = _cpu(denoised_latent, torch.bfloat16)

    def block_step_callback(*, block_index, step_index, timestep, noisy_input, transition_noise):
        if block_index != 19:
            return
        step_inputs[step_index] = _cpu(noisy_input, torch.bfloat16)
        step_timesteps[step_index] = timestep
        if transition_noise is not None:
            transition_noises[step_index] = _cpu(transition_noise, torch.bfloat16)

    with torch.no_grad():
        result = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[batch["text"]],
            ref_latent=ref_latent,
            render_latent=render_latent,
            mask_latent=mask_latent,
            decode=False,
            block_output_callback=block_output_callback,
            block_step_callback=block_step_callback,
        )
        prompt_embeds = pipeline.text_encoder([batch["text"]])["prompt_embeds"]

    if set(captured_blocks) != {5, 13, 18, 19}:
        raise AssertionError(f"missing captured blocks: {captured_blocks.keys()}")
    if set(step_inputs) != {0, 1, 2, 3} or set(transition_noises) != {0, 1, 2}:
        raise AssertionError("did not capture the complete four-step noise contract")
    if set(step_timesteps) != {0, 1, 2, 3}:
        raise AssertionError("did not capture all actual model timesteps")
    if not torch.equal(captured_blocks[19], _cpu(result[:, 57:60], torch.bfloat16)):
        raise AssertionError("block-19 callback differs from rollout output")

    z_a = captured_blocks[5]
    z_b = captured_blocks[13]
    raw_depth = torch.from_numpy(
        np.array(np.load(condition_dir / "render/depth_offline.npy", mmap_mode="r"), copy=True)
    )
    K = batch["target_intrinsic"].float()
    target_c2w = batch["target_c2w"].float()
    projected, projected_mask4, occupancy = project_memory_sequence(
        z_a,
        raw_depth[A_KEYFRAMES],
        K,
        target_c2w[A_KEYFRAMES],
        target_c2w[APRIME_KEYFRAMES],
    )
    projection_metrics = identity_reprojection_error(z_a, projected, occupancy)
    if projection_metrics["max_abs_error"] > 1e-3:
        raise AssertionError(f"LSM identity reprojection failed: {projection_metrics}")

    direct_mask4 = torch.ones(1, 3, 4, 60, 104, dtype=torch.bfloat16)
    tensors = {
        "z_A": z_a,
        "z_B": z_b,
        "block18_previous": captured_blocks[18],
        "block19_base_render16": _cpu(render_latent[:, 57:60], torch.bfloat16),
        "block19_base_mask4": _cpu(mask_latent[:, 57:60], torch.bfloat16),
        "block19_ref16": _cpu(ref_latent[:, 57:60], torch.bfloat16),
        "z_Aprime_no_memory": captured_blocks[19],
        "direct_memory_latent16": z_a.clone(),
        "direct_memory_mask4": direct_mask4,
        "projected_memory_latent16": _cpu(projected, torch.bfloat16),
        "projected_memory_mask4": _cpu(projected_mask4, torch.bfloat16),
        "projected_occupancy1": _cpu(occupancy, torch.bool),
        "denoise_step_inputs": torch.stack([step_inputs[index] for index in range(4)]),
        "transition_noises": torch.stack([transition_noises[index] for index in range(3)]),
        "prompt_embeds": _cpu(prompt_embeds, torch.bfloat16),
        "raw_depth": raw_depth.to(torch.float16).contiguous(),
        "K": K.contiguous(),
        "planned_c2w": target_c2w.contiguous(),
        "rng_cpu_before_noise": rng_cpu_before_noise.contiguous(),
        "rng_cuda_before_noise": rng_cuda_before_noise.contiguous(),
    }
    tensor_path = output_dir / "sample.safetensors"
    save_file(tensors, tensor_path)
    elapsed = time.perf_counter() - started
    manifest = {
        "source": condition_manifest["source"],
        "trajectory": condition_manifest["trajectory"],
        "seed": args.seed,
        "sample_definition": "one complete rollout block-19 revisit record",
        "condition_manifest": str((condition_dir / "manifest.json").resolve()),
        "condition_manifest_sha256": sha256_file(condition_dir / "manifest.json"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "prompt": batch["text"],
        "denoising_step_indices": [1000, 750, 500, 250],
        "actual_model_timesteps": [step_timesteps[index] for index in range(4)],
        "teacher_forced_previous_block": 18,
        "memory_write_block": 5,
        "diagnostic_wrong_block": 13,
        "query_block": 19,
        "temporal_bindings": [[60, 228], [64, 232], [68, 236]],
        "pose_validation": pose_metrics,
        "identity_reprojection": projection_metrics,
        "adapter_parameter_count": adapter.parameter_count,
        "adapter_nonzero_at_capture": int(torch.count_nonzero(adapter.proj.weight).item()),
        "tensor_file": tensor_path.name,
        "tensor_sha256": sha256_file(tensor_path),
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "tensor_dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "capture_seconds": elapsed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
