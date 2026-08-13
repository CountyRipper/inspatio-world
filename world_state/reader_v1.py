"""Identity-preserving innovation Reader and patch-gated LoRA."""

import math

import torch
import torch.nn as nn

from .types import EncodedWorldContentV1, WorldLayerContextV1


class PatchGatedLoRA(nn.Module):
    def __init__(self, dimension: int, rank: int, alpha: float | None = None):
        super().__init__()
        self.rank = int(rank)
        self.scale = float(rank if alpha is None else alpha) / rank
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(dimension))
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(value)) * self.scale


class IdentityPreservingWorldReader(nn.Module):
    """Select memory vs null, then fuse a full-width content innovation."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        selector_width: int = 256,
        residual_scale: float = 0.075,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.selector_width = int(selector_width)

        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.selector_query = nn.Linear(hidden_dim, selector_width, bias=False)
        self.selector_timestep = nn.Linear(hidden_dim, selector_width, bias=False)
        self.null_logit = nn.Parameter(torch.zeros(()))

        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.current_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fusion_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fusion_timestep = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fusion_output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        # Preserve the already verified direct-adapter carrier at initialization,
        # while allowing training to learn a layer-specific state innovation.
        nn.init.eye_(self.value_projection.weight)
        nn.init.zeros_(self.current_projection.weight)
        nn.init.zeros_(self.fusion_hidden.weight)
        nn.init.zeros_(self.fusion_timestep.weight)
        nn.init.eye_(self.fusion_output.weight)

    def precompute(
        self,
        encoded: EncodedWorldContentV1,
        *,
        enable_lora: bool = False,
        force_memory_gate: bool = False,
    ) -> WorldLayerContextV1:
        return WorldLayerContextV1(
            selector_key=encoded.selector_key,
            projected_value=self.value_projection(encoded.content),
            memory_patch=encoded.memory_patch,
            enable_lora=enable_lora,
            force_memory_gate=force_memory_gate,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        timestep_embedding: torch.Tensor,
        context: WorldLayerContextV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[:2] != context.selector_key.shape[:2]:
            raise ValueError("world context and DiT hidden sequence do not match")
        normalized = self.hidden_norm(hidden)
        if context.force_memory_gate:
            alpha = torch.ones_like(context.memory_patch, dtype=hidden.dtype)
        else:
            query = self.selector_query(normalized)
            query = query + self.selector_timestep(timestep_embedding).unsqueeze(1)
            memory_logit = (query * context.selector_key).sum(dim=-1, keepdim=True)
            memory_logit = memory_logit * (self.selector_width ** -0.5)
            alpha = torch.sigmoid(memory_logit - self.null_logit)
        gate = context.memory_patch.to(alpha.dtype) * alpha

        current = self.current_projection(normalized)
        innovation = context.projected_value - current
        fusion_gate = torch.sigmoid(
            self.fusion_hidden(normalized)
            + self.fusion_timestep(timestep_embedding).unsqueeze(1)
        )
        delta = self.fusion_output(innovation * fusion_gate)
        update = gate * (self.residual_scale * delta)
        return hidden + update, gate
