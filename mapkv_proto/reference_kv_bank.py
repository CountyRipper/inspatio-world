from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

from .config import resolve_indices
from .kv_bank import BANK_VERSION, _sha256_file, tensor_statistics


class ReferenceKVBankWriter:
    """Persist clean generated blocks re-encoded in the reference t0-t2 slot."""

    def __init__(
        self,
        root: str | Path,
        *,
        selected_layers: Iterable[int],
        num_layers: int,
        slot_len: int,
        frames_per_block: int,
        tokens_per_frame: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.selected_layers = resolve_indices(
            tuple(selected_layers), num_layers, name="transformer layer"
        )
        self.num_layers = int(num_layers)
        self.slot_len = int(slot_len)
        self.frames_per_block = int(frames_per_block)
        self.tokens_per_frame = int(tokens_per_frame)
        self.dtype = dtype
        self.chunks: dict[str, dict] = {}
        self._write_metadata()

    def write_chunk(
        self,
        *,
        chunk_id: int,
        layer_payloads: dict[int, tuple[torch.Tensor, torch.Tensor]],
        metadata: dict | None = None,
    ) -> None:
        chunk_id = int(chunk_id)
        chunk_dir = self.root / f"chunk_{chunk_id:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        layer_meta = {}
        for layer_index in self.selected_layers:
            if layer_index not in layer_payloads:
                raise KeyError(f"Missing reference payload for layer {layer_index}")
            k, v = layer_payloads[layer_index]
            if k.shape != v.shape or k.shape[1] != self.slot_len:
                raise ValueError(
                    f"Invalid reference slot at layer {layer_index}: "
                    f"K={tuple(k.shape)} V={tuple(v.shape)}"
                )
            payload = {
                "chunk_id": chunk_id,
                "layer_index": layer_index,
                "k": k.detach().to(device="cpu", dtype=self.dtype).clone(),
                "v": v.detach().to(device="cpu", dtype=self.dtype).clone(),
                "rope_layout": "reference_slot_t0_t2",
                "source_dtype": str(k.dtype).replace("torch.", ""),
            }
            path = chunk_dir / f"layer_{layer_index:02d}.pt"
            torch.save(payload, path)
            layer_meta[str(layer_index)] = {
                "path": str(path.relative_to(self.root)),
                "shape": list(k.shape),
                "source_dtype": payload["source_dtype"],
                "sha256": _sha256_file(path),
                "k_stats": tensor_statistics(payload["k"]),
                "v_stats": tensor_statistics(payload["v"]),
                "memory_bytes": int(
                    k.numel() * k.element_size() + v.numel() * v.element_size()
                ),
            }
        item = {
            "chunk_id": chunk_id,
            "latent_frame_ids": list(
                range(
                    chunk_id * self.frames_per_block,
                    (chunk_id + 1) * self.frames_per_block,
                )
            ),
            "rope_layout": "reference_slot_t0_t2",
            "rope_state": "post_rope",
            "capture_type": "clean_reference_reencode",
            "layers": layer_meta,
        }
        item.update(metadata or {})
        self.chunks[str(chunk_id)] = item
        self._write_metadata()

    def _write_metadata(self) -> None:
        metadata = {
            "version": BANK_VERSION,
            "selected_layers": list(self.selected_layers),
            "num_layers": self.num_layers,
            "frames_per_block": self.frames_per_block,
            "tokens_per_frame": self.tokens_per_frame,
            "recent_slot_len": self.slot_len,
            "slot_kind": "reference",
            "rope_layout": "reference_slot_t0_t2",
            "storage_dtype": str(self.dtype).replace("torch.", ""),
            "capture_type": "clean_reference_reencode",
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
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(path)
