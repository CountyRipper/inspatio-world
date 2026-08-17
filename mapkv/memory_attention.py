from __future__ import annotations

from dataclasses import dataclass

from mapkv_proto.memory_context import (
    ActiveLayerMemory,
    MemoryContext,
    make_memory_context,
)


@dataclass(frozen=True)
class ResidualMemoryAttentionConfig:
    """Explicit architecture record for the frozen residual memory branch."""

    mode: str = "residual_memory_attention"
    alpha: float = 0.1
    gate: str = "global"
    payload: str = "native_kv_post_rope"
    granularity: str = "chunk"


__all__ = [
    "ActiveLayerMemory",
    "MemoryContext",
    "ResidualMemoryAttentionConfig",
    "make_memory_context",
]
