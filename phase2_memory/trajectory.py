from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


NUM_RGB_FRAMES = 240
NUM_BLOCKS = 20


@dataclass(frozen=True)
class TrajectoryStation:
    block: int
    memory_id: str
    yaw_degrees: float
    pitch_degrees: float = 0.0
    radius: float = 0.0
    action: str = "write"

    def validate(self) -> None:
        if not 0 <= self.block < NUM_BLOCKS:
            raise ValueError(f"block must be in [0,{NUM_BLOCKS}), got {self.block}")
        if self.action not in ("write", "return_write", "return"):
            raise ValueError(f"invalid station action: {self.action}")
        if not self.memory_id:
            raise ValueError("memory_id must be non-empty")


def block_keyframes(block: int) -> np.ndarray:
    if not 0 <= block < NUM_BLOCKS:
        raise ValueError(block)
    start = block * 12
    return np.asarray([start, start + 4, start + 8], dtype=np.int64)


def controls_from_stations(
    stations: list[TrajectoryStation],
    *,
    initial_yaw_degrees: float = 0.0,
) -> np.ndarray:
    if not stations:
        raise ValueError("trajectory needs at least one station")
    for station in stations:
        station.validate()
    blocks = [station.block for station in stations]
    if blocks != sorted(set(blocks)):
        raise ValueError("station blocks must be unique and increasing")

    anchors: list[tuple[int, float, float, float]] = [
        (0, 0.0, float(initial_yaw_degrees), 0.0)
    ]
    for station in stations:
        center = station.block * 12
        hold_start = max(0, center - 3)
        hold_end = min(NUM_RGB_FRAMES - 1, center + 8)
        values = (
            float(station.pitch_degrees),
            float(station.yaw_degrees),
            float(station.radius),
        )
        anchors.append((hold_start, *values))
        anchors.append((hold_end, *values))
    anchors.append((NUM_RGB_FRAMES - 1, anchors[-1][1], anchors[-1][2], anchors[-1][3]))

    deduplicated: dict[int, tuple[float, float, float]] = {}
    for frame, pitch, yaw, radius in anchors:
        deduplicated[frame] = (pitch, yaw, radius)
    anchor_frames = np.asarray(sorted(deduplicated), dtype=np.float64)
    values = np.asarray([deduplicated[int(frame)] for frame in anchor_frames])
    frame_index = np.arange(NUM_RGB_FRAMES, dtype=np.float64)
    controls = np.stack([
        np.interp(frame_index, anchor_frames, values[:, channel])
        for channel in range(3)
    ])
    for station in stations:
        keys = block_keyframes(station.block)
        np.testing.assert_allclose(controls[1, keys], station.yaw_degrees, atol=1e-9, rtol=0)
        np.testing.assert_allclose(controls[0, keys], station.pitch_degrees, atol=1e-9, rtol=0)
        np.testing.assert_allclose(controls[2, keys], station.radius, atol=1e-9, rtol=0)
    return controls


def write_trajectory(
    path: str | Path,
    stations: list[TrajectoryStation],
    *,
    initial_yaw_degrees: float = 0.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    controls = controls_from_stations(
        stations, initial_yaw_degrees=initial_yaw_degrees
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in controls:
            handle.write(" ".join(f"{value:.10f}" for value in row) + "\n")
    return path
