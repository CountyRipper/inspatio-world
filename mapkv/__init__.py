"""Rapid MapKV closed-loop prototype.

Geometry is an address layer (CUT3R pointmaps -> voxel surfels -> chunk IDs);
native clean-context DiT KV remains the appearance payload.
"""

__all__ = ["KVChunkBank", "resolve_memory_layers"]


def __getattr__(name):
    if name in __all__:
        from .kv_bank import KVChunkBank, resolve_memory_layers

        return {"KVChunkBank": KVChunkBank, "resolve_memory_layers": resolve_memory_layers}[name]
    raise AttributeError(name)
