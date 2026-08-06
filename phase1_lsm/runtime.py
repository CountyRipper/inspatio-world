from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from utils.wan_wrapper import WanDiffusionWrapper


FRAME_TOKENS = 1560
BLOCK_LATENTS = 3


def load_generator(
    repo_root: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[WanDiffusionWrapper, object]:
    repo_root = Path(repo_root)
    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/default_config.yaml"),
        OmegaConf.load(repo_root / "configs/inference_1.3b.yaml"),
    )
    wan_root = repo_root / "checkpoints/Wan2.1-T2V-1.3B"
    if not wan_root.exists():
        wan_root = Path("/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B")
    config.wan_model_folder = str(wan_root)
    config.generator.weight_list[0].path = str(wan_root)

    generator = WanDiffusionWrapper(**config.generator, is_causal=True)
    incompatible = generator.load_state_dict(load_file(str(checkpoint_path)), strict=False)
    allowed_missing = {"model.memory_adapter.proj.weight"}
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    generator = generator.to(device=device, dtype=torch.bfloat16).eval()
    return generator, config


def allocate_kv_cache(generator: WanDiffusionWrapper, device: torch.device) -> list[dict]:
    model = generator.model
    cache_frames = 6
    cache = []
    for _ in range(len(model.blocks)):
        cache.append({
            "k": torch.zeros(
                1,
                FRAME_TOKENS * cache_frames,
                model.num_heads,
                model.dim // model.num_heads,
                device=device,
                dtype=torch.bfloat16,
            ),
            "v": torch.zeros(
                1,
                FRAME_TOKENS * cache_frames,
                model.num_heads,
                model.dim // model.num_heads,
                device=device,
                dtype=torch.bfloat16,
            ),
        })
    return cache


def teacher_context(ref_block: torch.Tensor, previous_block: torch.Tensor) -> torch.Tensor:
    if ref_block.shape != previous_block.shape or ref_block.shape[1:3] != (3, 16):
        raise ValueError("ref/previous blocks must both be [B,3,16,H,W]")
    ref_padded = torch.cat(
        (ref_block, torch.zeros_like(ref_block[:, :, :4]), torch.zeros_like(ref_block)),
        dim=2,
    )
    previous_padded = torch.cat(
        (
            previous_block,
            torch.zeros_like(previous_block[:, :, :4]),
            torch.zeros_like(previous_block),
        ),
        dim=2,
    )
    return torch.cat((ref_padded, previous_padded), dim=1)


def fill_teacher_cache(
    generator: WanDiffusionWrapper,
    conditional_dict: dict,
    kv_cache: list[dict],
    ref_block: torch.Tensor,
    previous_block: torch.Tensor,
    render_block: torch.Tensor,
) -> None:
    context = teacher_context(ref_block, previous_block)
    timestep = torch.zeros(
        context.shape[0], BLOCK_LATENTS, device=context.device, dtype=torch.int64
    )
    with torch.no_grad():
        generator(
            noisy_image_or_video=context,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            render_latent_input=render_block,
            kv_size=(0, -1),
            freqs_offset=0,
        )


def freeze_except_adapter(generator: WanDiffusionWrapper) -> list[torch.nn.Parameter]:
    generator.requires_grad_(False)
    adapter = generator.model.memory_adapter
    adapter.requires_grad_(True)
    trainable = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    if trainable != list(adapter.parameters()):
        raise AssertionError("trainable parameter set is not exactly MemoryPatchAdapter")
    return trainable
