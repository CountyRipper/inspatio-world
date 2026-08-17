"""Rapid MapKV closed-loop prototype.

Geometry is an address layer (CUT3R pointmaps -> voxel surfels -> chunk IDs);
native clean-context DiT KV remains the appearance payload.
"""

from .kv_bank import KVChunkBank, resolve_memory_layers

__all__ = ["KVChunkBank", "resolve_memory_layers"]
