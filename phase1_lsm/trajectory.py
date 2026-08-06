from __future__ import annotations

import json
from pathlib import Path

import numpy as np


NUM_RGB_FRAMES = 240
A_RGB = slice(57, 69)
B_RGB = slice(153, 165)
APRIME_RGB = slice(225, 237)
A_KEYFRAMES = np.array([60, 64, 68])
APRIME_KEYFRAMES = np.array([228, 232, 236])


def fixed_yaw(sign: int) -> np.ndarray:
    if sign not in (-1, 1):
        raise ValueError(f"sign must be -1 or +1, got {sign}")
    yaw = np.empty(NUM_RGB_FRAMES, dtype=np.float64)
    yaw[0:58] = np.linspace(0.0, sign * 45.0, 58)
    yaw[57:69] = sign * 45.0
    yaw[68:154] = np.linspace(sign * 45.0, 0.0, 86)
    yaw[153:165] = 0.0
    yaw[164:226] = np.linspace(0.0, sign * 45.0, 62)
    yaw[225:240] = sign * 45.0
    return yaw


def fixed_controls(sign: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(NUM_RGB_FRAMES, dtype=np.float64),
        fixed_yaw(sign),
        np.zeros(NUM_RGB_FRAMES, dtype=np.float64),
    )


def write_trajectory(path: str | Path, sign: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    controls = fixed_controls(sign)
    lines = [" ".join(f"{value:.12g}" for value in row) for row in controls]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_trajectory(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) != 3:
        raise AssertionError(f"trajectory must contain exactly 3 lines, got {len(lines)}")
    controls = np.asarray([[float(value) for value in line.split()] for line in lines])
    if controls.shape != (3, NUM_RGB_FRAMES):
        raise AssertionError(
            f"trajectory must be [3,{NUM_RGB_FRAMES}], got {controls.shape}"
        )
    return controls


def _rotation_speed_deg(rotations: np.ndarray) -> np.ndarray:
    relative = np.einsum("tji,tjk->tik", rotations[:-1], rotations[1:])
    cos_angle = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


def validate_target_c2w(target_c2w: np.ndarray) -> dict[str, float]:
    target_c2w = np.asarray(target_c2w, dtype=np.float32)
    assert target_c2w.shape == (NUM_RGB_FRAMES, 4, 4)
    np.testing.assert_allclose(
        target_c2w[A_RGB], target_c2w[APRIME_RGB], atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        target_c2w[A_KEYFRAMES], target_c2w[APRIME_KEYFRAMES], atol=1e-6, rtol=0
    )
    centers = target_c2w[:, :3, 3]
    center_drift = np.linalg.norm(centers - centers[:1], axis=1)
    max_center_drift = float(center_drift.max())
    assert max_center_drift <= 1e-6, max_center_drift
    speeds = _rotation_speed_deg(target_c2w[:, :3, :3].astype(np.float64))
    max_yaw_speed = float(speeds.max(initial=0.0))
    assert max_yaw_speed <= 0.8 + 1e-6, max_yaw_speed
    return {
        "max_camera_center_drift": max_center_drift,
        "max_rotation_speed_degree_per_frame": max_yaw_speed,
    }


def fixed_sample_manifest() -> dict[str, object]:
    samples = []
    for source in ("S0", "S1"):
        for trajectory in ("P", "N"):
            for seed in (0, 1):
                samples.append(
                    {"source": source, "trajectory": trajectory, "seed": seed}
                )
    return {
        "num_rgb_frames": NUM_RGB_FRAMES,
        "num_latents": 60,
        "num_blocks": 20,
        "output_rgb_frames": 237,
        "smoke_sample": {"source": "S0", "trajectory": "P", "seed": 0},
        "samples": samples,
        "blocks": {
            "A": {"block": 5, "latent": [15, 18], "rgb": [57, 69], "keyframes": [60, 64, 68]},
            "B": {"block": 13, "latent": [39, 42], "rgb": [153, 165], "keyframes": [156, 160, 164]},
            "Aprime": {"block": 19, "latent": [57, 60], "rgb": [225, 237], "keyframes": [228, 232, 236]},
        },
    }


def write_fixed_manifest(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(fixed_sample_manifest(), indent=2) + "\n", encoding="utf-8"
    )
