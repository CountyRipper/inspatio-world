"""Shared experiment helpers kept outside the core adapter package."""

import atexit
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torchvision.io import write_video


REPO_ROOT = Path(__file__).resolve().parents[2]


def _destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def resolve_repo_path(path) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else REPO_ROOT / path


def load_configs(config_path) -> Tuple[object, object]:
    experiment_config = OmegaConf.load(resolve_repo_path(config_path))
    base_config = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs/default_config.yaml"),
        OmegaConf.load(resolve_repo_path(experiment_config.experiment.base_config)),
    )
    return experiment_config, base_config


def init_single_gpu_distributed() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("the exact-identity experiment requires a CUDA GPU")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29631")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
        atexit.register(_destroy_process_group)
    if dist.get_world_size() != 1:
        raise ValueError("these single-sample scripts require --nproc_per_node=1")
    return device


def load_frozen_generator(base_config, checkpoint_path, device):
    from utils.wan_wrapper import WanDiffusionWrapper

    generator = WanDiffusionWrapper(
        **getattr(base_config, "generator", {}),
        is_causal=True,
    )
    state_dict = load_file(str(resolve_repo_path(checkpoint_path)))
    missing, unexpected = generator.load_state_dict(state_dict, strict=False)
    print(
        "Loaded InSpatio checkpoint: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    del state_dict
    generator.to(device=device, dtype=torch.bfloat16)
    generator.eval().requires_grad_(False)
    return generator


def initialize_kv_cache(generator, batch_size, dtype, device):
    model = generator.model
    cache_size = 1560 * 6
    return [
        {
            "k": torch.zeros(
                batch_size,
                cache_size,
                model.num_heads,
                model.dim // model.num_heads,
                dtype=dtype,
                device=device,
            ),
            "v": torch.zeros(
                batch_size,
                cache_size,
                model.num_heads,
                model.dim // model.num_heads,
                dtype=dtype,
                device=device,
            ),
        }
        for _ in range(model.num_layers)
    ]


def pad_clean_latent(latent: torch.Tensor) -> torch.Tensor:
    """Convert external [B,F,16,H,W] clean x0 to 36-channel context."""
    zeros = torch.zeros_like(latent)
    return torch.cat([latent, zeros[:, :, :4], zeros], dim=2)


def exact_memory_inputs(memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build all-valid external condition and strict binary occupancy."""
    mask4 = torch.ones(
        *memory.shape[:2],
        4,
        *memory.shape[-2:],
        device=memory.device,
        dtype=memory.dtype,
    )
    occupancy = torch.ones(
        *memory.shape[:2],
        1,
        *memory.shape[-2:],
        device=memory.device,
        dtype=torch.float32,
    )
    return torch.cat([mask4, memory], dim=2), occupancy


def write_video_tensor(path, video: torch.Tensor, fps: int = 24) -> None:
    """Write [B,T,C,H,W] or [T,C,H,W] video in [0,1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("video writer expects batch size one")
        video = video[0]
    frames = (
        video.detach()
        .float()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    write_video(str(path), frames, fps=fps)


def cpu_contiguous(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous()
        for key, value in tensors.items()
    }
