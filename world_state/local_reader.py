"""Query-conditioned local cross-attention and conditional Q/O LoRA."""

import math

import torch
import torch.nn as nn

from .types import EncodedWorldTokens, WorldLayerContext


class ConditionalLoRA(nn.Module):
    def __init__(self, dimension: int, rank: int, alpha: float | None = None):
        super().__init__()
        self.rank = rank
        self.scale = float(rank if alpha is None else alpha) / rank
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(dimension))
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(value)) * self.scale


class LocalWorldReader(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        world_width: int = 512,
        heads: int = 8,
        residual_scale: float = 0.075,
        output_init_std: float = 1e-3,
    ):
        super().__init__()
        if world_width % heads:
            raise ValueError("world_width must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.world_width = world_width
        self.heads = heads
        self.head_dim = world_width // heads
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, world_width)
        self.timestep = nn.Linear(hidden_dim, world_width, bias=False)
        self.key = nn.Linear(world_width, world_width)
        self.value = nn.Linear(world_width, world_width)
        # The null candidate has value zero and must mean an exact patch-local
        # bypass even after training. A bias here would turn null-only attention
        # into a learned global residual.
        self.output = nn.Linear(world_width, hidden_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        nn.init.normal_(self.output.weight, std=output_init_std)

    def precompute(
        self,
        encoded: EncodedWorldTokens,
        *,
        enable_lora: bool = False,
    ) -> WorldLayerContext:
        batch, sequence, candidates, _ = encoded.tokens.shape
        key = self.key(encoded.tokens).view(
            batch, sequence, candidates, self.heads, self.head_dim
        )
        value = self.value(encoded.tokens).view(
            batch, sequence, candidates, self.heads, self.head_dim
        )
        value = value.masked_fill(encoded.is_null.view(1, 1, -1, 1, 1), 0)
        return WorldLayerContext(
            key=key,
            value=value,
            valid=encoded.valid,
            attention_bias=encoded.attention_bias,
            enable_lora=enable_lora,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        timestep_embedding: torch.Tensor,
        context: WorldLayerContext,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        if context.key.shape[:2] != (batch, sequence):
            raise ValueError("world context and DiT hidden sequence do not match")
        query = self.query(self.hidden_norm(hidden))
        query = query + self.timestep(timestep_embedding).unsqueeze(1)
        query = query.view(batch, sequence, self.heads, self.head_dim)
        logits = torch.einsum("blhd,blkhd->blhk", query, context.key)
        logits = logits * (self.head_dim ** -0.5)
        logits = logits + context.attention_bias.unsqueeze(2).to(logits.dtype)
        logits = logits.masked_fill(~context.valid.unsqueeze(2), float("-inf"))
        attention = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        update = torch.einsum("blhk,blkhd->blhd", attention, context.value)
        update = update.reshape(batch, sequence, self.world_width)
        return hidden + self.residual_scale * self.output(update)
