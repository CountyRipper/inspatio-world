from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


BUNDLE_VERSION = 1


def _key(block_id: int, step_index: int) -> str:
    return f"block_{block_id:04d}/step_{step_index:02d}"


@dataclass
class DeterministicNoiseBundle:
    initial_noise: torch.Tensor
    re_noise: dict[str, torch.Tensor]
    seed: int
    num_blocks: int
    num_denoising_steps: int

    @classmethod
    def create(
        cls,
        *,
        shape: tuple[int, ...],
        num_blocks: int,
        num_denoising_steps: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "DeterministicNoiseBundle":
        if len(shape) != 5:
            raise ValueError("noise shape must be [B,T,C,H,W]")
        if shape[1] % num_blocks != 0:
            raise ValueError("latent frame count must divide evenly into num_blocks")
        device = torch.device(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        initial = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        frames_per_block = shape[1] // num_blocks
        block_shape = (shape[0], frames_per_block, *shape[2:])
        re_noise: dict[str, torch.Tensor] = {}
        for block_id in range(num_blocks):
            for step_index in range(max(num_denoising_steps - 1, 0)):
                tensor = torch.randn(
                    block_shape, generator=generator, device=device, dtype=dtype
                )
                re_noise[_key(block_id, step_index)] = tensor.cpu()
        return cls(
            initial_noise=initial.cpu(),
            re_noise=re_noise,
            seed=seed,
            num_blocks=num_blocks,
            num_denoising_steps=num_denoising_steps,
        )

    def get_initial(self, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        return self.initial_noise.to(device=device, dtype=dtype)

    def get_re_noise(
        self,
        *,
        block_id: int,
        step_index: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        key = _key(block_id, step_index)
        if key not in self.re_noise:
            raise KeyError(f"Missing deterministic re-noise tensor: {key}")
        tensor = self.re_noise[key]
        if tuple(tensor.shape) != tuple(like.shape):
            raise ValueError(
                f"Re-noise shape {tuple(tensor.shape)} does not match {tuple(like.shape)} for {key}"
            )
        return tensor.to(device=like.device, dtype=like.dtype, non_blocking=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": BUNDLE_VERSION,
                "seed": self.seed,
                "num_blocks": self.num_blocks,
                "num_denoising_steps": self.num_denoising_steps,
                "initial_noise": self.initial_noise,
                "re_noise": self.re_noise,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DeterministicNoiseBundle":
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("version") != BUNDLE_VERSION:
            raise ValueError(f"Unsupported noise bundle version: {payload.get('version')}")
        return cls(
            initial_noise=payload["initial_noise"],
            re_noise=payload["re_noise"],
            seed=int(payload["seed"]),
            num_blocks=int(payload["num_blocks"]),
            num_denoising_steps=int(payload["num_denoising_steps"]),
        )
