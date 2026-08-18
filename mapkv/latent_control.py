from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LatentBlockIntervention:
    """Direct clean-x0 block override used only as a frozen-control upper bound."""

    target_block: int
    source_chunk: int
    clean_latent: torch.Tensor
    strength: float
    audit: dict = field(default_factory=dict)

    def apply(self, predicted: torch.Tensor) -> torch.Tensor:
        if predicted.shape != self.clean_latent.shape:
            raise ValueError(
                f"Latent control shape {tuple(self.clean_latent.shape)} "
                f"!= predicted {tuple(predicted.shape)}"
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Latent control strength must be in [0, 1]")
        memory = self.clean_latent.to(
            device=predicted.device, dtype=predicted.dtype
        )
        output = torch.lerp(predicted, memory, self.strength)
        self.audit.update(
            {
                "target_block": int(self.target_block),
                "source_chunk": int(self.source_chunk),
                "strength": float(self.strength),
                "predicted_to_memory_l1": float(
                    (predicted.float() - memory.float()).abs().mean().item()
                ),
                "output_delta_l1": float(
                    (output.float() - predicted.float()).abs().mean().item()
                ),
                "mode": "direct_clean_x0_block_override",
            }
        )
        return output
