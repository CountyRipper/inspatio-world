#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

from utils.wan_wrapper import WanDiffusionWrapper


def _load_selected(path: str | Path, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    repo_root = Path(args.repo_root)
    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/default_config.yaml"),
        OmegaConf.load(repo_root / "configs/inference_1.3b.yaml"),
    )
    wan_root = Path("/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B")
    config.generator.weight_list[0].path = str(wan_root)
    generator = WanDiffusionWrapper(**config.generator, is_causal=True)
    incompatible = generator.load_state_dict(load_file(args.checkpoint), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(incompatible.unexpected_keys)
    generator = generator.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    sample = _load_selected(
        args.sample,
        (
            "block18_previous", "block19_base_render16", "block19_base_mask4",
            "block19_ref16", "denoise_step_inputs", "prompt_embeds",
            "z_Aprime_no_memory",
        ),
    )
    sample = {name: tensor.to(device) for name, tensor in sample.items()}
    model = generator.model
    kv_cache = [
        {
            "k": torch.zeros(
                1, 1560 * 6, model.num_heads, model.dim // model.num_heads,
                device=device, dtype=torch.bfloat16,
            ),
            "v": torch.zeros(
                1, 1560 * 6, model.num_heads, model.dim // model.num_heads,
                device=device, dtype=torch.bfloat16,
            ),
        }
        for _ in model.blocks
    ]
    ref = sample["block19_ref16"]
    previous = sample["block18_previous"]
    ref_padded = torch.cat((ref, torch.zeros_like(ref[:, :, :4]), torch.zeros_like(ref)), dim=2)
    previous_padded = torch.cat(
        (previous, torch.zeros_like(previous[:, :, :4]), torch.zeros_like(previous)), dim=2
    )
    context = torch.cat((ref_padded, previous_padded), dim=1)
    render = torch.cat(
        (sample["block19_base_mask4"], sample["block19_base_render16"]), dim=2
    )
    conditional = {"prompt_embeds": sample["prompt_embeds"]}
    requested_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
    scheduler_steps = torch.cat(
        (generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    actual_steps = scheduler_steps[1000 - requested_steps]

    with torch.no_grad():
        generator(
            noisy_image_or_video=context,
            conditional_dict=conditional,
            timestep=torch.zeros(1, 3, device=device, dtype=torch.int64),
            kv_cache=kv_cache,
            render_latent_input=render,
            kv_size=(0, -1),
            freqs_offset=0,
        )
        final_timestep = torch.ones(1, 3, device=device, dtype=torch.int64) * actual_steps[-1]
        _, prediction = generator(
            noisy_image_or_video=sample["denoise_step_inputs"][3],
            conditional_dict=conditional,
            timestep=final_timestep,
            kv_cache=kv_cache,
            render_latent_input=render,
            kv_size=(0, 1560 * 6),
            freqs_offset=6,
        )
    baseline = sample["z_Aprime_no_memory"]
    equal = torch.equal(prediction, baseline)
    max_abs = float((prediction.float() - baseline.float()).abs().max())
    result = {
        "repo_root": str(repo_root.resolve()),
        "missing_keys": list(incompatible.missing_keys),
        "requested_step_indices": requested_steps.tolist(),
        "actual_model_timesteps": [float(value) for value in actual_steps],
        "torch_equal": equal,
        "max_abs_error": max_abs,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "seconds": time.perf_counter() - started,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not equal:
        raise AssertionError(f"base replay is not bitwise equal: max_abs={max_abs}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
