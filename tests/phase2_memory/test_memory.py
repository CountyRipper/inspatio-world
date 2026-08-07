from pathlib import Path

import numpy as np
import torch

from phase2_memory.anchoring import masked_latent_anchor
from phase2_memory.bank import LatentMemoryBank, MemoryRecord
from phase2_memory.manifest import load_manifest
from phase2_memory.rollout import source_occupancy
from phase2_memory.trajectory import (
    TrajectoryStation,
    block_keyframes,
    controls_from_stations,
)
from pipeline.causal_inference import denoise_block


def c2w(yaw_degrees: float) -> torch.Tensor:
    angle = torch.deg2rad(torch.tensor(yaw_degrees))
    pose = torch.eye(4).repeat(3, 1, 1)
    pose[:, 0, 0] = torch.cos(angle)
    pose[:, 0, 2] = torch.sin(angle)
    pose[:, 2, 0] = -torch.sin(angle)
    pose[:, 2, 2] = torch.cos(angle)
    return pose


def record(memory_id: str, yaw: float, value: float) -> MemoryRecord:
    latent = torch.full((1, 3, 16, 4, 6), value, requires_grad=True)
    depth = torch.ones(3, 8, 12)
    occupancy = torch.ones(1, 3, 1, 4, 6, dtype=torch.bool)
    return MemoryRecord(
        memory_id=memory_id,
        clean_latent=latent,
        c2w=c2w(yaw),
        intrinsics=torch.tensor([
            [8.0, 0.0, 6.0],
            [0.0, 8.0, 4.0],
            [0.0, 0.0, 1.0],
        ]),
        depth=depth,
        occupancy=occupancy,
        confidence=occupancy.float(),
        fov_degrees=70.0,
    )


def test_bank_top2_writeback_detach_and_projection() -> None:
    bank = LatentMemoryBank()
    bank.write(record("A", 30.0, 1.0))
    bank.write(record("B", 0.0, 2.0))
    bank.write(record("C", -30.0, 3.0))
    matches = bank.retrieve(c2w(20.0), 70.0, top_k=2)
    assert [item.record.memory_id for item in matches] == ["A", "B"]
    assert matches[0].record.clean_latent.requires_grad is False
    projection = bank.project(matches[0], c2w(20.0))
    assert projection.condition.shape == (1, 3, 20, 4, 6)
    assert projection.occupancy.any()
    bank.write(record("A", 20.0, 4.0), replace_existing=True)
    assert bank.get("A").version == 2
    assert bank.retrieve(c2w(25.0), 70.0, top_k=1)[0].record.memory_id == "A"


def test_multimemory_trajectory_and_manifest() -> None:
    stations = [
        TrajectoryStation(2, "A", 30.0),
        TrajectoryStation(5, "B", 0.0),
        TrajectoryStation(8, "C", -30.0),
        TrajectoryStation(11, "A", 20.0, action="return_write"),
        TrajectoryStation(14, "B", 5.0, action="return_write"),
        TrajectoryStation(17, "A", 25.0, action="return"),
        TrajectoryStation(19, "B", 2.0, action="return"),
    ]
    controls = controls_from_stations(stations)
    assert controls.shape == (3, 240)
    for station in stations:
        np.testing.assert_allclose(
            controls[1, block_keyframes(station.block)],
            station.yaw_degrees,
            atol=1e-9,
            rtol=0,
        )
    manifest = load_manifest(Path("configs/phase2_memory_manifest.json"))
    assert len(manifest.groups) == 30
    assert len(manifest.select(split="train")) == 24
    assert len(manifest.select(split="heldout")) == 6
    assert len(manifest.select(phase2a_only=True)) == 2


def test_source_occupancy_shape() -> None:
    depth = torch.ones(3, 8, 12)
    depth[0, :2] = 0
    occupancy = source_occupancy(depth, (4, 6))
    assert occupancy.shape == (1, 3, 1, 4, 6)
    assert occupancy.dtype == torch.bool
    assert not occupancy[0, 0, 0, 0].any()


class AddNoiseScheduler:
    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps
        return clean + noise


def test_masked_anchor_changes_only_occupancy() -> None:
    noisy = torch.zeros(1, 3, 2, 2, 2)
    clean = torch.ones_like(noisy) * 2
    noise = torch.ones_like(noisy)
    occupancy = torch.zeros(1, 3, 1, 2, 2)
    occupancy[..., 0, 0] = 1
    result = masked_latent_anchor(
        AddNoiseScheduler(), noisy, clean, occupancy, noise, 500, 0.5
    )
    assert torch.all(result[..., 0, 0] == 1.5)
    assert torch.count_nonzero(result[..., 0, 1:]) == 0
    assert torch.count_nonzero(result[..., 1, :]) == 0


class TinyGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, *, noisy_image_or_video: torch.Tensor, **kwargs):
        del kwargs
        prediction = noisy_image_or_video * self.weight + self.weight
        return prediction, prediction


class TinyScheduler:
    def add_noise(
        self,
        prediction: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps
        return prediction + noise * 0.0


def gradient(all_steps: bool) -> float:
    generator = TinyGenerator()
    transformed = []
    prediction, _ = denoise_block(
        generator,
        TinyScheduler(),
        torch.ones(1, 1, 1, 1, 1),
        {},
        [],
        denoising_steps=torch.tensor([4, 3, 2, 1]),
        transition_noises=torch.zeros(3, 1, 1, 1, 1, 1),
        backpropagate_all_steps=all_steps,
        step_input_transform=lambda index, timestep, value: (
            transformed.append((index, int(timestep))) or value
        ),
    )
    prediction.sum().backward()
    assert transformed == [(0, 4), (1, 3), (2, 2), (3, 1)]
    return float(generator.weight.grad)


def test_four_step_backpropagation_is_not_last_step_only() -> None:
    last_only = gradient(False)
    all_steps = gradient(True)
    assert all_steps > last_only
