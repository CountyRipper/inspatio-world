from __future__ import annotations

import torch
import torch.nn.functional as F


def exact_memory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    memory_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = memory_valid.bool().expand_as(prediction)
    if not valid.any():
        raise ValueError("memory_valid contains no valid elements")
    smooth_l1 = F.smooth_l1_loss(
        prediction.float()[valid], target.detach().float()[valid]
    )
    cosine_per_pixel = 1.0 - F.cosine_similarity(
        prediction.float(), target.detach().float(), dim=2, eps=1e-8
    )
    cosine_valid = memory_valid[:, :, 0].bool()
    cosine = cosine_per_pixel[cosine_valid].mean()
    total = smooth_l1 + 0.1 * cosine
    return total, {"smooth_l1": smooth_l1, "latent_cosine": cosine}
