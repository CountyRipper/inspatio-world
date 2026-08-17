from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from mapkv_proto.kv_bank import KVBank, KVBankWriter, tensor_statistics


LAYER_MODES = {"uniform8", "middle8", "explicit", "all"}


def resolve_memory_layers(
    mode: str,
    num_layers: int,
    explicit: Iterable[int] = (),
) -> tuple[int, ...]:
    """Resolve an architecture-independent memory-layer policy."""
    if mode not in LAYER_MODES:
        raise ValueError(f"Unknown memory layer mode: {mode}")
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if mode == "all":
        return tuple(range(num_layers))
    if mode == "explicit":
        result = []
        for raw in explicit:
            layer = int(raw) + num_layers if int(raw) < 0 else int(raw)
            if layer < 0 or layer >= num_layers:
                raise IndexError(f"Layer {raw} resolves outside [0, {num_layers})")
            if layer not in result:
                result.append(layer)
        if not result:
            raise ValueError("explicit layer mode requires at least one layer")
        return tuple(result)
    count = min(8, num_layers)
    if mode == "middle8":
        start = (num_layers - count) // 2
        return tuple(range(start, start + count))
    # Rounding a linspace keeps endpoints represented without assuming 30/32 layers.
    return tuple(
        int(layer)
        for layer in dict.fromkeys(
            np.rint(np.linspace(0, num_layers - 1, count)).astype(int)
        )
    )


@dataclass(frozen=True)
class KVChunkRecord:
    chunk_id: int
    layer_ids: tuple[int, ...]
    capture_type: str
    rope_state: str
    memory_bytes: int
    metadata: dict


class KVChunkBank(KVBank):
    """Read-only facade with v0.4 stats over the existing clean-context bank."""

    def record(self, chunk_id: int) -> KVChunkRecord:
        item = self.metadata["chunks"][str(int(chunk_id))]
        layers = tuple(sorted(int(layer) for layer in item["layers"]))
        memory_bytes = sum(
            int(layer.get("memory_bytes", 0)) for layer in item["layers"].values()
        )
        if memory_bytes == 0:
            memory_bytes = sum(
                (self.root / layer["path"]).stat().st_size
                for layer in item["layers"].values()
            )
        return KVChunkRecord(
            chunk_id=int(chunk_id),
            layer_ids=layers,
            capture_type=item.get("capture_type", "clean_context"),
            rope_state=item.get("rope_state", item.get("rope_layout", "post_rope")),
            memory_bytes=memory_bytes,
            metadata=item,
        )

    def stats(self) -> dict:
        records = [self.record(chunk) for chunk in self.available_chunks]
        return {
            "num_chunks": len(records),
            "available_chunks": list(self.available_chunks),
            "selected_layers": list(self.metadata["selected_layers"]),
            "num_layers": int(self.metadata["num_layers"]),
            "capture_type": "clean_context",
            "rope_state": "post_rope",
            "memory_bytes": int(sum(record.memory_bytes for record in records)),
            "recent_slot_len": int(self.metadata["recent_slot_len"]),
            "tokens_per_frame": int(self.metadata["tokens_per_frame"]),
            "frames_per_block": int(self.metadata["frames_per_block"]),
        }

    def write_stats(self, path: str | Path) -> dict:
        result = self.stats()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def capture_manifest(self) -> dict:
        chunks = []
        for chunk_id in self.available_chunks:
            item = self.metadata["chunks"][str(chunk_id)]
            layers = {}
            for layer_id, layer in sorted(
                item["layers"].items(), key=lambda pair: int(pair[0])
            ):
                k_stats = layer.get("k_stats")
                v_stats = layer.get("v_stats")
                if k_stats is None or v_stats is None:
                    payload = torch.load(
                        self.root / layer["path"], map_location="cpu", weights_only=True
                    )
                    k_stats = tensor_statistics(payload["k"])
                    v_stats = tensor_statistics(payload["v"])
                layers[str(layer_id)] = {
                    "shape": layer["shape"],
                    "source_dtype": layer["source_dtype"],
                    "sha256": layer["sha256"],
                    "memory_bytes": int(layer.get("memory_bytes", 0)),
                    "k_stats": k_stats,
                    "v_stats": v_stats,
                }
            chunks.append(
                {
                    "chunk_id": int(chunk_id),
                    "latent_frame_ids": item.get("latent_frame_ids"),
                    "rgb_keyframe_id": item.get("rgb_keyframe_id"),
                    "capture_type": item.get("capture_type", "clean_context"),
                    "rope_state": item.get("rope_state", "post_rope"),
                    "layers": layers,
                }
            )
        return {
            "version": 1,
            "capture_type": "clean_context",
            "rope_state": "post_rope",
            "selected_layers": list(self.metadata["selected_layers"]),
            "chunks": chunks,
        }

    def write_capture_manifest(self, path: str | Path) -> dict:
        result = self.capture_manifest()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a clean-context KV chunk bank")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output")
    parser.add_argument("--capture_manifest")
    parser.add_argument("--resolve_layers", choices=sorted(LAYER_MODES))
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--explicit", nargs="*", type=int, default=[])
    args = parser.parse_args()
    if args.resolve_layers:
        if args.num_layers is None:
            raise ValueError("--num_layers is required with --resolve_layers")
        print(json.dumps(list(resolve_memory_layers(
            args.resolve_layers, args.num_layers, args.explicit
        ))))
        return
    bank = KVChunkBank(args.bank)
    stats = bank.write_stats(args.output) if args.output else bank.stats()
    if args.capture_manifest:
        bank.write_capture_manifest(args.capture_manifest)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "KVBankWriter",
    "KVChunkBank",
    "KVChunkRecord",
    "resolve_memory_layers",
]
