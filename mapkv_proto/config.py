from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


VALID_MODES = {"off", "oracle", "wrong", "random", "pose", "geometry"}
VALID_GATES = {"global", "ref_blind", "surfel_ref_blind"}
VALID_INJECTION_MODES = {"replace_recent_delta", "residual_memory_attention"}


def resolve_indices(indices: Sequence[int], size: int, *, name: str) -> tuple[int, ...]:
    """Resolve negative indices once runtime cardinality is known."""
    resolved = []
    for raw in indices:
        index = raw + size if raw < 0 else raw
        if index < 0 or index >= size:
            raise IndexError(f"{name} index {raw} resolves to {index}, outside [0, {size})")
        if index not in resolved:
            resolved.append(index)
    return tuple(resolved)


@dataclass(frozen=True)
class GateConfig:
    mode: str = "ref_blind"
    smooth_kernel: int = 3

    def validate(self) -> None:
        if self.mode not in VALID_GATES:
            raise ValueError(f"Unsupported gate mode: {self.mode}")
        if self.smooth_kernel < 1 or self.smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel must be a positive odd integer")


@dataclass(frozen=True)
class MapKVConfig:
    enabled: bool = False
    mode: str = "off"
    source_chunk: int | None = None
    target_chunks: tuple[int, ...] = ()
    wrong_chunk: int | None = None
    random_seed: int = 0
    selected_layers: tuple[int, ...] = (-4, -3, -2, -1)
    selected_step_indices: tuple[int, ...] = (-1,)
    alpha: float = 0.10
    injection_mode: str = "replace_recent_delta"
    gate: GateConfig = field(default_factory=GateConfig)
    bank_root: Path = Path("artifacts/baseline/kv_bank")
    pin_memory: bool = True

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported MapKV mode: {self.mode}")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.injection_mode not in VALID_INJECTION_MODES:
            raise ValueError(f"Unsupported memory injection mode: {self.injection_mode}")
        if not self.selected_layers:
            raise ValueError("selected_layers cannot be empty when MapKV is enabled")
        if not self.selected_step_indices:
            raise ValueError("selected_step_indices cannot be empty when MapKV is enabled")
        if any(chunk < 0 for chunk in self.target_chunks):
            raise ValueError("target chunk ids must be non-negative")
        self.gate.validate()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "MapKVConfig":
        raw = dict(raw or {})
        gate_raw = dict(raw.get("gate") or {})
        bank_raw = dict(raw.get("bank") or {})
        config = cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "off")),
            source_chunk=raw.get("source_chunk"),
            target_chunks=tuple(int(x) for x in raw.get("target_chunks", ())),
            wrong_chunk=raw.get("wrong_chunk"),
            random_seed=int(raw.get("random_seed", 0)),
            selected_layers=tuple(int(x) for x in raw.get("selected_layers", (-4, -3, -2, -1))),
            selected_step_indices=tuple(int(x) for x in raw.get("selected_step_indices", (-1,))),
            alpha=float(raw.get("alpha", 0.10)),
            injection_mode=str(raw.get("injection_mode", "replace_recent_delta")),
            gate=GateConfig(
                mode=str(gate_raw.get("mode", "ref_blind")),
                smooth_kernel=int(gate_raw.get("smooth_kernel", 3)),
            ),
            bank_root=Path(bank_raw.get("root", "artifacts/baseline/kv_bank")),
            pin_memory=bool(bank_raw.get("pin_memory", True)),
        )
        config.validate()
        return config
