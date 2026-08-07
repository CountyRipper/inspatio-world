from __future__ import annotations

import torch


def masked_latent_anchor(
    scheduler,
    noisy_input: torch.Tensor,
    clean_latent: torch.Tensor,
    occupancy: torch.Tensor,
    anchor_noise: torch.Tensor,
    timestep: torch.Tensor | float,
    strength: float,
) -> torch.Tensor:
    """Re-noise a detached retrieved latent and blend it only inside occupancy."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("anchoring strength must be in [0,1]")
    if strength == 0.0:
        return noisy_input
    if noisy_input.shape != clean_latent.shape or noisy_input.shape != anchor_noise.shape:
        raise ValueError("noisy_input, clean_latent, and anchor_noise shapes must match")
    if occupancy.shape != (*noisy_input.shape[:2], 1, *noisy_input.shape[-2:]):
        raise ValueError("occupancy must be [B,F,1,H,W]")
    if not torch.all((occupancy == 0) | (occupancy == 1)):
        raise ValueError("anchoring occupancy must be binary")
    batch_frames = noisy_input.shape[0] * noisy_input.shape[1]
    timestep_value = torch.as_tensor(timestep, device=noisy_input.device).flatten()[0]
    timesteps = timestep_value * torch.ones(
        batch_frames, device=noisy_input.device, dtype=torch.long
    )
    noised_memory = scheduler.add_noise(
        clean_latent.detach().flatten(0, 1),
        anchor_noise.detach().flatten(0, 1),
        timesteps,
    ).unflatten(0, noisy_input.shape[:2])
    blended = noisy_input * (1.0 - strength) + noised_memory * strength
    return torch.where(occupancy.bool().expand_as(noisy_input), blended, noisy_input)
