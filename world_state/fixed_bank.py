"""Immutable observation bank with camera-only retrieval."""

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

from .types import CameraBatch, WorldObservation, WorldReadPacket


def _mean_view_angle(observation: WorldObservation, query: CameraBatch) -> float:
    memory_rotation = observation.c2w_W0[:, :3, :3].to(query.c2w_W0)
    query_rotation = query.c2w_W0[0, :, :3, :3]
    relative = query_rotation.transpose(-1, -2) @ memory_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1)
    return float(torch.acos(cosine).mean())


@dataclass(frozen=True)
class FixedWorldBank:
    observations: Tuple[WorldObservation, ...]

    def __init__(self, observations: Iterable[WorldObservation]):
        values = tuple(observations)
        if not values:
            raise ValueError("a fixed bank requires at least one observation")
        scene_ids = {observation.scene_id for observation in values}
        world_ids = {observation.world_id for observation in values}
        if len(scene_ids) != 1 or len(world_ids) != 1:
            raise ValueError("all bank observations must share scene_id and world_id")
        object.__setattr__(self, "observations", values)

    def retrieve(self, query_camera: CameraBatch, top_observations: int = 2) -> Tuple[WorldObservation, ...]:
        if query_camera.batch_size != 1:
            raise ValueError("Teacher v0 retrieval expects batch size one")
        ranked = sorted(
            self.observations,
            key=lambda observation: (
                _mean_view_angle(observation, query_camera),
                -observation.authority,
                observation.observation_id,
            ),
        )
        return tuple(ranked[:top_observations])

    def retrieve_and_project(
        self,
        projector,
        query_camera: CameraBatch,
        top_observations: int = 2,
    ) -> WorldReadPacket:
        observations = self.retrieve(query_camera, top_observations=top_observations)
        return projector.project(observations, query_camera)
