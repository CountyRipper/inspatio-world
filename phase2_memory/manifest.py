from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from phase2_memory.trajectory import TrajectoryStation


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    video: Path
    metadata_json: Path
    geometry: Path


@dataclass(frozen=True)
class TrajectoryGroup:
    group_id: str
    scene_id: str
    split: str
    family: str
    seed: int
    initial_yaw_degrees: float
    stations: tuple[TrajectoryStation, ...]
    phase2a: bool

    @property
    def return_stations(self) -> tuple[TrajectoryStation, ...]:
        return tuple(
            station for station in self.stations
            if station.action in ("return_write", "return")
        )

    def validate(self) -> None:
        if self.split not in ("train", "heldout"):
            raise ValueError(f"{self.group_id}: split must be train or heldout")
        if len(self.stations) < 5:
            raise ValueError(f"{self.group_id}: too few stations")
        for station in self.stations:
            station.validate()
        blocks = [station.block for station in self.stations]
        if blocks != sorted(set(blocks)):
            raise ValueError(f"{self.group_id}: station blocks must increase")
        first_writes = [station for station in self.stations if station.action == "write"]
        if {station.memory_id for station in first_writes} < {"A", "B", "C"}:
            raise ValueError(f"{self.group_id}: must write generated A/B/C")
        return_ids = {station.memory_id for station in self.return_stations}
        if not {"A", "B"} <= return_ids:
            raise ValueError(f"{self.group_id}: must revisit both A and B")
        rewritten = {
            station.memory_id
            for station in self.stations
            if station.action == "return_write"
        }
        reread = {
            station.memory_id
            for station in self.stations
            if station.action == "return"
        }
        if not rewritten.intersection(reread):
            raise ValueError(f"{self.group_id}: a memory-conditioned writeback is never reread")


@dataclass(frozen=True)
class ExperimentManifest:
    path: Path
    scenes: dict[str, SceneSpec]
    groups: tuple[TrajectoryGroup, ...]
    heldout_unit: str

    def select(
        self,
        *,
        split: str | None = None,
        group_ids: set[str] | None = None,
        phase2a_only: bool = False,
    ) -> list[TrajectoryGroup]:
        selected = [
            group for group in self.groups
            if (split is None or group.split == split)
            and (group_ids is None or group.group_id in group_ids)
            and (not phase2a_only or group.phase2a)
        ]
        if group_ids is not None and {group.group_id for group in selected} != group_ids:
            missing = group_ids - {group.group_id for group in selected}
            raise KeyError(f"unknown or filtered group ids: {sorted(missing)}")
        return selected


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: str | Path) -> ExperimentManifest:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenes = {
        scene_id: SceneSpec(
            scene_id=scene_id,
            video=_resolve(path.parent, values["video"]),
            metadata_json=_resolve(path.parent, values["metadata_json"]),
            geometry=_resolve(path.parent, values["geometry"]),
        )
        for scene_id, values in payload["scenes"].items()
    }
    groups = []
    for values in payload["groups"]:
        stations = tuple(TrajectoryStation(**item) for item in values["stations"])
        group = TrajectoryGroup(
            group_id=values["group_id"],
            scene_id=values["scene_id"],
            split=values["split"],
            family=values["family"],
            seed=int(values["seed"]),
            initial_yaw_degrees=float(values.get("initial_yaw_degrees", 0.0)),
            stations=stations,
            phase2a=bool(values.get("phase2a", False)),
        )
        if group.scene_id not in scenes:
            raise KeyError(f"{group.group_id}: unknown scene {group.scene_id}")
        group.validate()
        groups.append(group)
    ids = [group.group_id for group in groups]
    if len(ids) != len(set(ids)):
        raise ValueError("group_id values must be unique")
    return ExperimentManifest(
        path=path,
        scenes=scenes,
        groups=tuple(groups),
        heldout_unit=str(payload.get("heldout_unit", "trajectory")),
    )
