from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monotonic_index(source_index: int, source_length: int, target_length: int) -> int:
    if source_length <= 1 or target_length <= 1:
        return 0
    return int(round(source_index * (target_length - 1) / (source_length - 1)))


def rotation_geodesic_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def yaw_rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.eye(4, dtype=np.float64)
    rotation[:3, :3] = np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    return rotation


@dataclass(frozen=True)
class Phase:
    name: str
    start_block: int
    stop_block: int
    start_yaw: float
    stop_yaw: float
    kind: str

    @property
    def blocks(self) -> int:
        return self.stop_block - self.start_block


def _append_phase(
    phases: list[Phase], name: str, blocks: int, start_yaw: float, stop_yaw: float
) -> None:
    if blocks <= 0:
        raise ValueError(f"Phase {name} must contain at least one block")
    start = phases[-1].stop_block if phases else 0
    phases.append(
        Phase(
            name=name,
            start_block=start,
            stop_block=start + blocks,
            start_yaw=float(start_yaw),
            stop_yaw=float(stop_yaw),
            kind="hold" if start_yaw == stop_yaw else "ramp",
        )
    )


def build_control_phases(
    theta_degrees: float,
    *,
    temporal_stride: float,
    frames_per_block: int = 3,
    requested_speed_degrees_per_frame: float = 0.5,
    distractor: bool = False,
    revisit_theta_degrees: float | None = None,
) -> tuple[list[Phase], int]:
    """Build a block-aligned exact-yaw schedule from measured VAE timing."""
    if theta_degrees <= 0:
        raise ValueError("theta_degrees must be positive")
    revisit_theta = (
        float(theta_degrees)
        if revisit_theta_degrees is None
        else float(revisit_theta_degrees)
    )
    if revisit_theta <= 0:
        raise ValueError("revisit_theta_degrees must be positive")
    if temporal_stride <= 0 or requested_speed_degrees_per_frame <= 0:
        raise ValueError("temporal stride and requested speed must be positive")
    rgb_per_block = temporal_stride * frames_per_block
    ramp_blocks = max(
        1,
        int(
            math.ceil(
                theta_degrees
                / (requested_speed_degrees_per_frame * rgb_per_block)
            )
        ),
    )
    revisit_ramp_blocks = max(
        1,
        int(
            math.ceil(
                revisit_theta
                / (requested_speed_degrees_per_frame * rgb_per_block)
            )
        ),
    )
    phases: list[Phase] = []
    _append_phase(phases, "A0_hold", 2, 0.0, 0.0)
    _append_phase(phases, "A_to_B1", ramp_blocks, 0.0, theta_degrees)
    _append_phase(phases, "B1_hold", 2, theta_degrees, theta_degrees)
    _append_phase(phases, "B1_to_A1", ramp_blocks, theta_degrees, 0.0)
    if distractor:
        _append_phase(phases, "A1_hold", 2, 0.0, 0.0)
        _append_phase(phases, "A1_to_wrong", ramp_blocks, 0.0, -theta_degrees)
        _append_phase(phases, "wrong_hold", 2, -theta_degrees, -theta_degrees)
        _append_phase(phases, "wrong_to_A2", ramp_blocks, -theta_degrees, 0.0)
        _append_phase(phases, "A2_distractor", 3, 0.0, 0.0)
    else:
        _append_phase(phases, "A1_distractor", 3, 0.0, 0.0)
    _append_phase(
        phases, "A_to_B2", revisit_ramp_blocks, 0.0, revisit_theta
    )
    _append_phase(phases, "B2_hold", 2, revisit_theta, revisit_theta)
    return phases, ramp_blocks


def rgb_length_for_latents(latent_length: int, temporal_stride: float) -> int:
    if latent_length <= 0:
        raise ValueError("latent_length must be positive")
    return int(round((latent_length - 1) * temporal_stride + 1))


def build_yaw_samples(phases: Iterable[Phase], rgb_length: int) -> tuple[np.ndarray, list[dict]]:
    phases = list(phases)
    if not phases:
        raise ValueError("At least one phase is required")
    num_blocks = phases[-1].stop_block
    yaw = np.empty(rgb_length, dtype=np.float64)
    labels: list[dict] = []
    for phase in phases:
        rgb_start = int(round(phase.start_block * rgb_length / num_blocks))
        rgb_stop = int(round(phase.stop_block * rgb_length / num_blocks))
        if rgb_stop <= rgb_start:
            raise ValueError(f"Phase {phase.name} received no RGB frames")
        if phase.kind == "hold":
            values = np.full(rgb_stop - rgb_start, phase.start_yaw, dtype=np.float64)
            realized_speed = 0.0
        else:
            values = np.linspace(
                phase.start_yaw,
                phase.stop_yaw,
                rgb_stop - rgb_start,
                endpoint=True,
                dtype=np.float64,
            )
            realized_speed = abs(phase.stop_yaw - phase.start_yaw) / max(
                rgb_stop - rgb_start - 1, 1
            )
        yaw[rgb_start:rgb_stop] = values
        labels.append(
            {
                "name": phase.name,
                "kind": phase.kind,
                "start_block": phase.start_block,
                "stop_block_exclusive": phase.stop_block,
                "blocks": phase.blocks,
                "rgb_start": rgb_start,
                "rgb_stop_exclusive": rgb_stop,
                "start_yaw_degrees": phase.start_yaw,
                "stop_yaw_degrees": phase.stop_yaw,
                "realized_speed_degrees_per_rgb_frame": realized_speed,
            }
        )
    return yaw, labels


def phase_by_name(phases: Iterable[Phase], name: str) -> Phase:
    for phase in phases:
        if phase.name == name:
            return phase
    raise KeyError(name)


def plateau_middle_chunk(phase: Phase) -> int:
    if phase.kind != "hold":
        raise ValueError(f"{phase.name} is not a plateau")
    return phase.start_block + phase.blocks // 2


def build_exact_c2w(base_c2w: np.ndarray, yaw_degrees: np.ndarray) -> np.ndarray:
    base_c2w = np.asarray(base_c2w, dtype=np.float64)
    if base_c2w.shape != (4, 4):
        raise ValueError(f"base_c2w must be [4,4], got {base_c2w.shape}")
    return np.stack([base_c2w @ yaw_rotation(float(value)) for value in yaw_degrees])


def validate_exact_case(
    *,
    target_c2w: np.ndarray,
    yaw_degrees: np.ndarray,
    pitch_degrees: np.ndarray,
    roll_degrees: np.ndarray,
    source_chunk: int,
    target_chunk: int,
    source_rgb_index: int,
    target_rgb_index: int,
    phase_labels: list[dict],
    expected_rotation_degrees: float = 0.0,
) -> dict:
    base_center = target_c2w[0, :3, 3]
    translation_norm = np.linalg.norm(target_c2w[:, :3, 3] - base_center, axis=1)
    rotation_error = rotation_geodesic_degrees(
        target_c2w[source_rgb_index], target_c2w[target_rgb_index]
    )
    translation_error = float(
        np.linalg.norm(
            target_c2w[source_rgb_index, :3, 3]
            - target_c2w[target_rgb_index, :3, 3]
        )
    )
    ramp_speeds = [
        item["realized_speed_degrees_per_rgb_frame"]
        for item in phase_labels
        if item["kind"] == "ramp"
    ]
    rotation_matches_expected = (
        abs(rotation_error - float(expected_rotation_degrees)) < 1e-6
    )
    checks = {
        "pitch_zero": float(np.max(np.abs(pitch_degrees))) < 1e-6,
        "roll_zero": float(np.max(np.abs(roll_degrees))) < 1e-6,
        "translation_static": float(translation_norm.max()) < 1e-6,
        "expected_view_rotation": rotation_matches_expected,
        "same_view_translation": translation_error < 1e-8,
        "source_target_gap": target_chunk - source_chunk >= 4,
        "source_outside_recent": source_chunk < target_chunk - 1,
        "ramp_speed_in_range": all(0.4 <= speed <= 0.6 for speed in ramp_speeds),
        "phase_boundaries_block_aligned": all(
            isinstance(item["start_block"], int)
            and isinstance(item["stop_block_exclusive"], int)
            for item in phase_labels
        ),
    }
    if abs(float(expected_rotation_degrees)) < 1e-12:
        checks["same_view_rotation"] = rotation_error < 1e-6
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "max_abs_pitch_degrees": float(np.max(np.abs(pitch_degrees))),
        "max_abs_roll_degrees": float(np.max(np.abs(roll_degrees))),
        "max_relative_translation_norm": float(translation_norm.max()),
        "B1_B2_rotation_distance_degrees": rotation_error,
        "expected_B1_B2_rotation_distance_degrees": float(
            expected_rotation_degrees
        ),
        "B1_B2_translation_distance": translation_error,
        "ramp_speeds_degrees_per_rgb_frame": ramp_speeds,
        "source_chunk": source_chunk,
        "target_chunk": target_chunk,
        "temporal_gap_chunks": target_chunk - source_chunk,
        "source_rgb_index": source_rgb_index,
        "target_rgb_index": target_rgb_index,
        "yaw_min_degrees": float(yaw_degrees.min()),
        "yaw_max_degrees": float(yaw_degrees.max()),
    }


def save_json(payload: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
