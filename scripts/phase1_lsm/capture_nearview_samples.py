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
from phase1_lsm.latent_projection import project_memory_sequence
from phase1_lsm.nearview import (
    projection_displacement_statistics,
    validate_nearview_c2w,
)
from phase1_lsm.trajectory import A_KEYFRAMES, APRIME_KEYFRAMES
from pipeline.causal_inference import CausalInferencePipeline
from utils.render_warper import convert_mask_video


WAN_ROOT = Path("/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B")


def cpu_tensor(
    tensor: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = tensor.detach().contiguous().cpu()
    return tensor if dtype is None else tensor.to(dtype=dtype)


def capture_one(
    pipeline: CausalInferencePipeline,
    base_config,
    condition_dir: Path,
    sample_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    if sample_dir.exists():
        raise FileExistsError(f"refusing to overwrite {sample_dir}")
    sample_dir.mkdir(parents=True)
    started = time.perf_counter()
    condition_manifest = json.loads((condition_dir / "manifest.json").read_text())
    offset_degrees = float(condition_manifest["offset_degrees"])
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    config.dataset.json_path = str(condition_dir / "new.json")
    config.dataset.min_num_frames = 240
    config.dataset.max_num_frames = 240
    config.dataset.adaptive_frame = False
    config.dataset.rotation_only = True
    config.dataset.traj_txt_path = condition_manifest["trajectory_path"]

    torch.manual_seed(0)
    dataset = VideoDataset(**OmegaConf.to_container(config.dataset, resolve=True))
    batch = dataset[0]
    target_c2w_np = batch["target_c2w"].numpy().astype(np.float32)
    pose_audit = validate_nearview_c2w(target_c2w_np, offset_degrees)
    if batch["source_video"].shape[0] != 240:
        raise AssertionError("near-view capture must use exactly 240 decoded frames")

    render_video = rearrange(
        batch["render_video"][None], "b t c h w -> b c t h w"
    ).to(device=device, dtype=torch.bfloat16)
    mask_video = rearrange(
        batch["mask_video"][None], "b t c h w -> b c t h w"
    ).to(device=device, dtype=torch.bfloat16)
    source_video = rearrange(
        batch["source_video"][None], "b t c h w -> b c t h w"
    ).to(device=device, dtype=torch.bfloat16)
    pipeline.vae.model.clear_cache()
    with torch.inference_mode():
        render_latent = pipeline.vae.encode_to_latent(render_video).to(torch.bfloat16)
        mask_latent = convert_mask_video(mask_video).to(torch.bfloat16)
        ref_latent = pipeline.vae.encode_to_latent(source_video).to(torch.bfloat16)
    pipeline.vae.model.clear_cache()
    expected = (1, 60, 16, 60, 104)
    if tuple(ref_latent.shape) != expected or tuple(render_latent.shape) != expected:
        raise AssertionError("240 RGB frames did not encode to 60 latents")
    if tuple(mask_latent.shape) != (1, 60, 4, 60, 104):
        raise AssertionError(mask_latent.shape)

    rng_cpu_before_noise = torch.get_rng_state()
    rng_cuda_before_noise = torch.cuda.get_rng_state(device)
    sampled_noise = torch.randn(expected, device=device, dtype=torch.bfloat16)
    captured_blocks: dict[int, torch.Tensor] = {}
    step_inputs: dict[int, torch.Tensor] = {}
    transition_noises: dict[int, torch.Tensor] = {}
    step_timesteps: dict[int, float] = {}

    def block_output_callback(*, block_index, latent_start, denoised_latent, dit_ms):
        if block_index in (5, 13, 18, 19):
            captured_blocks[block_index] = cpu_tensor(
                denoised_latent, torch.bfloat16
            )

    def block_step_callback(
        *, block_index, step_index, timestep, noisy_input, transition_noise
    ):
        if block_index != 19:
            return
        step_inputs[step_index] = cpu_tensor(noisy_input, torch.bfloat16)
        step_timesteps[step_index] = float(timestep)
        if transition_noise is not None:
            transition_noises[step_index] = cpu_tensor(
                transition_noise, torch.bfloat16
            )

    with torch.inference_mode():
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
        raise AssertionError(f"missing blocks: {captured_blocks.keys()}")
    if set(step_inputs) != {0, 1, 2, 3} or set(transition_noises) != {0, 1, 2}:
        raise AssertionError("incomplete four-step denoising record")
    if not torch.equal(captured_blocks[19], cpu_tensor(result[:, 57:60])):
        raise AssertionError("block-19 callback differs from rollout result")

    z_a = captured_blocks[5]
    z_b = captured_blocks[13]
    raw_depth = torch.from_numpy(
        np.array(
            np.load(condition_dir / "render/depth_offline.npy", mmap_mode="r"),
            copy=True,
        )
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
    if torch.equal(projected, z_a):
        raise AssertionError("non-identity near-view projection equals z_A")
    displacement = projection_displacement_statistics(
        raw_depth[A_KEYFRAMES],
        K,
        target_c2w[A_KEYFRAMES],
        target_c2w[APRIME_KEYFRAMES],
        z_a.shape[-2:],
    )
    valid = occupancy.expand_as(z_a)
    projection_audit = {
        "label": condition_manifest["trajectory"],
        "pose": pose_audit,
        "occupancy_valid_fraction": float(occupancy.float().mean()),
        "projected_torch_equal_z_A": False,
        "valid_projected_l1_to_z_A": float(
            (projected.float() - z_a.float()).abs()[valid].mean()
        ),
        "displacement": displacement,
    }

    tensors = {
        "z_A": z_a,
        "z_B": z_b,
        "latent_prefix_0_18": cpu_tensor(result[:, :57], torch.bfloat16),
        "block18_previous": captured_blocks[18],
        "block19_base_render16": cpu_tensor(
            render_latent[:, 57:60], torch.bfloat16
        ),
        "block19_base_mask4": cpu_tensor(mask_latent[:, 57:60], torch.bfloat16),
        "block19_ref16": cpu_tensor(ref_latent[:, 57:60], torch.bfloat16),
        "z_Aprime_no_memory": captured_blocks[19],
        "projected_memory_latent16": cpu_tensor(projected, torch.bfloat16),
        "projected_memory_mask4": cpu_tensor(projected_mask4, torch.bfloat16),
        "projected_occupancy1": cpu_tensor(occupancy, torch.bool),
        "denoise_step_inputs": torch.stack(
            [step_inputs[index] for index in range(4)]
        ),
        "transition_noises": torch.stack(
            [transition_noises[index] for index in range(3)]
        ),
        "prompt_embeds": cpu_tensor(prompt_embeds, torch.bfloat16),
        "K": K.contiguous(),
        "planned_c2w": target_c2w.contiguous(),
        "rng_cpu_before_noise": rng_cpu_before_noise.contiguous(),
        "rng_cuda_before_noise": rng_cuda_before_noise.contiguous(),
    }
    tensor_path = sample_dir / "sample.safetensors"
    save_file(tensors, tensor_path)
    manifest = {
        "source": "S0",
        "trajectory": condition_manifest["trajectory"],
        "offset_degrees": offset_degrees,
        "seed": 0,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "condition_manifest": str((condition_dir / "manifest.json").resolve()),
        "condition_manifest_sha256": sha256_file(condition_dir / "manifest.json"),
        "prompt": batch["text"],
        "denoising_step_indices": [1000, 750, 500, 250],
        "actual_model_timesteps": [step_timesteps[index] for index in range(4)],
        "memory_write_block": 5,
        "query_block": 19,
        "teacher_forced_previous_block": 18,
        "temporal_bindings": [[60, 228], [64, 232], [68, 236]],
        "projection_audit": projection_audit,
        "tensor_file": tensor_path.name,
        "tensor_sha256": sha256_file(tensor_path),
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "adapter_parameter_count": ADAPTER_PARAMETER_COUNT,
        "adapter_nonzero_at_capture": int(
            torch.count_nonzero(pipeline.generator.model.memory_adapter.proj.weight)
        ),
        "capture_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (sample_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    labels = tuple(args.label)
    expected_labels = (
        ("plus5", "minus5")
        if "plus5" in labels
        else ("plus10", "minus10")
    )
    if labels != expected_labels:
        raise ValueError(f"labels must be exactly {expected_labels}")
    if (root / "samples").exists():
        raise FileExistsError(f"refusing to overwrite {root / 'samples'}")
    for label in labels:
        if not (root / "conditions" / label / "manifest.json").is_file():
            raise FileNotFoundError(f"missing condition for {label}")

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load("configs/inference_1.3b.yaml"),
    )
    config.wan_model_folder = str(WAN_ROOT)
    config.generator.weight_list[0].path = str(WAN_ROOT)
    pipeline = CausalInferencePipeline(config, device=device)
    incompatible = pipeline.generator.load_state_dict(
        load_file(args.checkpoint), strict=False
    )
    if set(incompatible.missing_keys) != {"model.memory_adapter.proj.weight"}:
        raise RuntimeError(f"unexpected missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected keys: {incompatible.unexpected_keys}")
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.eval().requires_grad_(False)
    if torch.count_nonzero(
        pipeline.generator.model.memory_adapter.proj.weight
    ).item() != 0:
        raise AssertionError("base rollout adapter is not zero initialized")

    manifests = []
    for label in labels:
        manifests.append(capture_one(
            pipeline,
            config,
            root / "conditions" / label,
            root / "samples" / label,
            Path(args.checkpoint),
            device,
        ))
    projection_audit = {
        "passed": True,
        "samples": {
            manifest["trajectory"]: manifest["projection_audit"]
            for manifest in manifests
        },
    }
    (root / "projection_audit.json").write_text(
        json.dumps(projection_audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(projection_audit, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
