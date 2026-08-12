"""Opt-in exact-pose latent memory readout."""

from .latent_adapter import (
    LatentMemoryAdapter,
    add_gated_memory_residual,
    attach_latent_memory_adapter,
    gated_memory_residual,
    load_latent_memory_adapter,
    patchify_binary_occupancy,
    save_latent_memory_adapter,
)

__all__ = [
    "LatentMemoryAdapter",
    "add_gated_memory_residual",
    "attach_latent_memory_adapter",
    "gated_memory_residual",
    "load_latent_memory_adapter",
    "patchify_binary_occupancy",
    "save_latent_memory_adapter",
]
