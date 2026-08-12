#!/usr/bin/env python3
"""Run fixed no-memory, memory-A, and content-swap-B full revisits."""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import CausalInferencePipeline
from scripts.world_memory.common import (
    cpu_contiguous,
    exact_memory_inputs,
    init_single_gpu_distributed,
    load_configs,
    load_frozen_generator,
    resolve_repo_path,
    write_video_tensor,
)
from utils.wan_wrapper import WanVAEWrapper
from world_memory import (
    attach_latent_memory_adapter,
    load_latent_memory_adapter,
)


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
    return parser.parse_args()


def set_denoise_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_latents(vae, latent: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode_to_pixel(latent, use_cache=False)
    return (decoded * 0.5 + 0.5).clamp(0, 1)


def representative_image(video: torch.Tensor) -> Image.Image:
    frame = video[0, video.shape[1] // 2]
    array = (
        frame.float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array).resize((416, 240), Image.Resampling.LANCZOS)


def write_montage(path: Path, labeled_images) -> None:
    label_height = 28
    width = 416 * len(labeled_images)
    canvas = Image.new("RGB", (width, 240 + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(labeled_images):
        x = index * 416
        canvas.paste(image, (x, label_height))
        draw.text((x + 8, 8), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = parse_args()
    experiment_config, base_config = load_configs(args.config)
    artifact_dir = resolve_repo_path(experiment_config.experiment.artifact_dir)
    captured_dir = artifact_dir / "captured_latents"
    videos_dir = artifact_dir / "videos"
    adapter_path = artifact_dir / "adapter.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError("train the adapter before evaluation")

    shared = torch.load(
        captured_dir / "shared_trajectory.pt",
        map_location="cpu",
        weights_only=True,
    )
    query = torch.load(
        captured_dir / "query_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    device = init_single_gpu_distributed()
    dtype = torch.bfloat16
    generator = load_frozen_generator(
        base_config,
        experiment_config.experiment.checkpoint,
        device,
    )
    adapter = load_latent_memory_adapter(
        adapter_path,
        device=device,
        dtype=dtype,
    )
    attach_latent_memory_adapter(
        generator.model,
        adapter,
        device=device,
        dtype=None,
    )
    generator.eval().requires_grad_(False)

    prompt_encoder = StaticPromptEncoder(
        shared["prompt_embeds"].to(device=device, dtype=dtype)
    ).to(device=device)
    vae = WanVAEWrapper(
        model_folder=str(resolve_repo_path(base_config.wan_model_folder))
    ).to(device=device, dtype=dtype)
    vae.eval().requires_grad_(False)
    pipeline = CausalInferencePipeline(
        base_config,
        device=device,
        generator=generator,
        text_encoder=prompt_encoder,
        vae=vae,
    )
    pipeline.eval().requires_grad_(False)

    ref_latent = shared["ref_latent"].to(device=device, dtype=dtype)
    render_latent = shared["render_latent"].to(device=device, dtype=dtype)
    mask_latent = shared["mask_latent"].to(device=device, dtype=dtype)
    fixed_noise = shared["noise_A"].to(device=device, dtype=dtype)
    prompt = shared["prompt"]
    return_block = int(query["return_block"])
    target_A = query["target_A"].to(device=device, dtype=dtype)
    target_B = query["target_B"].to(device=device, dtype=dtype)
    memory_inputs = {
        "memory_A": exact_memory_inputs(target_A),
        "memory_B": exact_memory_inputs(target_B),
    }

    results = {}
    return_outputs = {}
    for branch in ("no_memory", "memory_A", "memory_B"):
        provider = None
        if branch != "no_memory":
            condition, occupancy = memory_inputs[branch]

            def provider(
                block_index,
                latent_start,
                block_size,
                condition=condition,
                occupancy=occupancy,
            ):
                if block_index == return_block:
                    return condition, occupancy
                return None

        def callback(block_index, latent_start, denoised_latent):
            if block_index == return_block:
                return_outputs[branch] = denoised_latent.detach().cpu().contiguous()

        set_denoise_seed(int(shared["denoise_seed"]))
        with torch.no_grad():
            output = pipeline.inference(
                noise=fixed_noise,
                text_prompts=[prompt],
                ref_latent=ref_latent,
                render_latent=render_latent,
                mask_latent=mask_latent,
                decode=False,
                memory_provider=provider,
                block_output_callback=callback,
            )
        if branch not in return_outputs:
            raise AssertionError("final return block callback was not invoked")
        results[branch] = output.detach().cpu().contiguous()
        save_file(
            cpu_contiguous({"denoised": output}),
            str(captured_dir / f"eval_{branch}.safetensors"),
        )

        with torch.no_grad():
            video = decode_latents(vae, output)
        write_video_tensor(videos_dir / f"{branch}.mp4", video)
        vae.model.clear_cache()
        del output, video
        torch.cuda.empty_cache()

    no_memory = return_outputs["no_memory"].float()
    memory_A = return_outputs["memory_A"].float()
    memory_B = return_outputs["memory_B"].float()
    target_A_cpu = query["target_A"].float()
    target_B_cpu = query["target_B"].float()
    metrics = {
        "memory_A_to_A_latent_L1": float((memory_A - target_A_cpu).abs().mean()),
        "memory_B_to_A_latent_L1": float((memory_B - target_A_cpu).abs().mean()),
        "memory_B_to_B_latent_L1": float((memory_B - target_B_cpu).abs().mean()),
        "no_memory_to_A_latent_L1": float((no_memory - target_A_cpu).abs().mean()),
    }
    metrics["numeric_memory_A_beats_no_memory"] = (
        metrics["memory_A_to_A_latent_L1"]
        < metrics["no_memory_to_A_latent_L1"]
    )
    metrics["numeric_memory_B_prefers_B"] = (
        metrics["memory_B_to_B_latent_L1"]
        < metrics["memory_B_to_A_latent_L1"]
    )
    prefix_end = return_block * int(experiment_config.experiment.block_size)
    metrics["memory_off_prefix_exact"] = all(
        torch.equal(results["no_memory"][:, :prefix_end], results[branch][:, :prefix_end])
        for branch in ("memory_A", "memory_B")
    )
    with (artifact_dir / "simple_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    decoded_blocks = {}
    block_latents = {
        "A first G reference": query["target_A"],
        "B first G reference": query["target_B"],
        "final G no memory": return_outputs["no_memory"],
        "final G memory A": return_outputs["memory_A"],
        "final G memory B": return_outputs["memory_B"],
    }
    with torch.no_grad():
        for label, latent in block_latents.items():
            decoded = decode_latents(
                vae,
                latent.to(device=device, dtype=dtype),
            )
            decoded_blocks[label] = representative_image(decoded)
            vae.model.clear_cache()
    write_montage(artifact_dir / "montage.png", list(decoded_blocks.items()))

    result_lines = [
        "# Exact-pose implicit latent memory adapter 实验结果",
        "",
        "- 场景：仓库自带的白色桌面、咖啡杯与托盘视频。",
        "- 轨迹：rotation-only，`0° → +40° → 0° → +40°`。",
        "- World A/B：仅生成 seed 不同；content swap 保持 pose、occupancy、metadata 与 query state 不变。",
        f"- memory-A 到 A 的 latent L1：`{metrics['memory_A_to_A_latent_L1']:.6f}`。",
        f"- no-memory 到 A 的 latent L1：`{metrics['no_memory_to_A_latent_L1']:.6f}`。",
        "- memory-B 到 B/A 的 latent L1："
        f"`{metrics['memory_B_to_B_latent_L1']:.6f}` / "
        f"`{metrics['memory_B_to_A_latent_L1']:.6f}`。",
        "- 数值判定：memory-A 优于 no-memory = "
        f"`{metrics['numeric_memory_A_beats_no_memory']}`；"
        "memory-B 更偏向 B = "
        f"`{metrics['numeric_memory_B_prefers_B']}`。",
        "- 最终 return block 前三路 latent 逐元素一致 = "
        f"`{metrics['memory_off_prefix_exact']}`。",
        "- 人眼判定：见 `montage.png`，复核具体杯子、托盘与背景 identity。",
        "",
        "> 本实验只证明冻结InSpatio-World 1.3B能够通过122,880参数adapter读取"
        "exact-pose clean-latent identity；不证明near/6DoF、retrieval、WorldState、"
        "自然handoff、长期保持或实时性。",
        "",
    ]
    with (artifact_dir / "RESULT_ZH.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(result_lines))
    print(json.dumps(metrics, indent=2))
    print(f"Evaluation complete: {artifact_dir}")


if __name__ == "__main__":
    main()
