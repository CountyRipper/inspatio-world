from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable

import torch

from .config import resolve_indices


BANK_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class KVBankWriter:
    """Capture only selected layers from the clean previous-generated slot."""

    def __init__(
        self,
        root: str | Path,
        *,
        selected_layers: Iterable[int],
        num_layers: int,
        recent_slot_len: int,
        frames_per_block: int,
        tokens_per_frame: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.selected_layers = resolve_indices(
            tuple(selected_layers), num_layers, name="transformer layer"
        )
        self.num_layers = num_layers
        self.recent_slot_len = int(recent_slot_len)
        self.frames_per_block = int(frames_per_block)
        self.tokens_per_frame = int(tokens_per_frame)
        self.dtype = dtype
        self.chunks: dict[str, dict] = {}
        self._write_metadata()

    def __call__(self, *, block_id: int, kv_cache: list[dict], context_frames: torch.Tensor) -> None:
        if block_id <= 0:
            return
        if len(kv_cache) != self.num_layers:
            raise ValueError(f"KV cache has {len(kv_cache)} layers, expected {self.num_layers}")
        expected_context_frames = 2 * self.frames_per_block
        if context_frames.shape[1] != expected_context_frames:
            raise ValueError(
                f"Non-first writer context has {context_frames.shape[1]} frames, "
                f"expected {expected_context_frames}"
            )
        chunk_id = block_id - 1
        chunk_dir = self.root / f"chunk_{chunk_id:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        start = self.recent_slot_len
        stop = 2 * self.recent_slot_len
        layer_meta = {}
        for layer_index in self.selected_layers:
            cache = kv_cache[layer_index]
            k = cache["k"][:, start:stop]
            v = cache["v"][:, start:stop]
            if k.shape != v.shape or k.shape[1] != self.recent_slot_len:
                raise ValueError(
                    f"Invalid recent slot at layer {layer_index}: K={tuple(k.shape)} V={tuple(v.shape)}"
                )
            payload = {
                "chunk_id": chunk_id,
                "layer_index": layer_index,
                "k": k.detach().to(device="cpu", dtype=self.dtype).clone(),
                "v": v.detach().to(device="cpu", dtype=self.dtype).clone(),
                "rope_layout": "recent_slot_t3_t5",
                "source_dtype": str(k.dtype).replace("torch.", ""),
            }
            path = chunk_dir / f"layer_{layer_index:02d}.pt"
            torch.save(payload, path)
            layer_meta[str(layer_index)] = {
                "path": str(path.relative_to(self.root)),
                "shape": list(k.shape),
                "source_dtype": payload["source_dtype"],
                "sha256": _sha256_file(path),
                "memory_bytes": int(
                    k.numel() * k.element_size() + v.numel() * v.element_size()
                ),
            }
        self.chunks[str(chunk_id)] = {
            "chunk_id": chunk_id,
            "latent_frame_ids": list(
                range(chunk_id * self.frames_per_block, (chunk_id + 1) * self.frames_per_block)
            ),
            "rgb_keyframe_id": None,
            "pose_metadata": None,
            "rope_layout": "recent_slot_t3_t5",
            "rope_state": "post_rope",
            "capture_type": "clean_context",
            "layers": layer_meta,
        }
        self._write_metadata()

    def update_chunk_metadata(self, chunk_id: int, **metadata) -> None:
        key = str(chunk_id)
        if key not in self.chunks:
            return
        self.chunks[key].update(metadata)
        self._write_metadata()

    def _write_metadata(self) -> None:
        metadata = {
            "version": BANK_VERSION,
            "selected_layers": list(self.selected_layers),
            "num_layers": self.num_layers,
            "frames_per_block": self.frames_per_block,
            "tokens_per_frame": self.tokens_per_frame,
            "recent_slot_len": self.recent_slot_len,
            "rope_layout": "recent_slot_t3_t5",
            "storage_dtype": str(self.dtype).replace("torch.", ""),
            "capture_type": "clean_context",
            "rope_state": "post_rope",
            "memory_bytes": int(
                sum(
                    layer.get("memory_bytes", 0)
                    for chunk in self.chunks.values()
                    for layer in chunk.get("layers", {}).values()
                )
            ),
            "chunks": self.chunks,
        }
        path = self.root / "metadata.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tmp.replace(path)


class KVBank:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"KV bank metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("version") != BANK_VERSION:
            raise ValueError(f"Unsupported KV bank version: {self.metadata.get('version')}")

    @property
    def available_chunks(self) -> tuple[int, ...]:
        return tuple(sorted(int(x) for x in self.metadata["chunks"]))

    def materialize(
        self,
        chunk_id: int,
        *,
        selected_layers: Iterable[int],
        num_layers: int,
        device: torch.device | str,
        dtype: torch.dtype,
        pin_memory: bool = True,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        chunk_key = str(chunk_id)
        if chunk_key not in self.metadata["chunks"]:
            raise KeyError(f"Chunk {chunk_id} is not in KV bank; available={self.available_chunks}")
        resolved = resolve_indices(tuple(selected_layers), num_layers, name="transformer layer")
        result = {}
        for layer_index in resolved:
            layer_info = self.metadata["chunks"][chunk_key]["layers"].get(str(layer_index))
            if layer_info is None:
                raise KeyError(f"Chunk {chunk_id} has no captured layer {layer_index}")
            payload = torch.load(
                self.root / layer_info["path"], map_location="cpu", weights_only=True
            )
            k, v = payload["k"], payload["v"]
            if pin_memory and torch.cuda.is_available() and not k.is_pinned():
                k = k.pin_memory()
                v = v.pin_memory()
            result[layer_index] = (
                k.to(device=device, dtype=dtype, non_blocking=pin_memory),
                v.to(device=device, dtype=dtype, non_blocking=pin_memory),
            )
        return result
