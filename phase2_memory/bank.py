from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from phase1_lsm.latent_projection import project_memory_sequence


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    clean_latent: torch.Tensor
    c2w: torch.Tensor
    intrinsics: torch.Tensor
    depth: torch.Tensor
    occupancy: torch.Tensor
    confidence: torch.Tensor
    fov_degrees: float
    version: int = 1

    def validate(self) -> None:
        if not self.memory_id:
            raise ValueError("memory_id must be non-empty")
        if self.clean_latent.ndim != 5 or self.clean_latent.shape[:3] != (1, 3, 16):
            raise ValueError("clean_latent must be [1,3,16,h,w]")
        if tuple(self.c2w.shape) != (3, 4, 4):
            raise ValueError("c2w must be [3,4,4]")
        if tuple(self.intrinsics.shape) != (3, 3):
            raise ValueError("intrinsics must be [3,3]")
        if self.depth.ndim != 3 or self.depth.shape[0] != 3:
            raise ValueError("depth must be [3,H,W]")
        expected = (1, 3, 1, *self.clean_latent.shape[-2:])
        if tuple(self.occupancy.shape) != expected:
            raise ValueError(f"occupancy must be {expected}")
        if tuple(self.confidence.shape) != expected:
            raise ValueError(f"confidence must be {expected}")
        if not torch.all((self.occupancy == 0) | (self.occupancy == 1)):
            raise ValueError("occupancy must be binary")
        if not torch.all((self.confidence >= 0) & (self.confidence <= 1)):
            raise ValueError("confidence must be in [0,1]")

    def detached(self) -> "MemoryRecord":
        self.validate()
        return replace(
            self,
            clean_latent=self.clean_latent.detach().clone(),
            c2w=self.c2w.detach().clone(),
            intrinsics=self.intrinsics.detach().clone(),
            depth=self.depth.detach().clone(),
            occupancy=self.occupancy.detach().clone(),
            confidence=self.confidence.detach().clone(),
        )


@dataclass(frozen=True)
class MemoryMatch:
    record: MemoryRecord
    score: float
    rotation_degrees: float
    translation: float
    fov_delta_degrees: float


@dataclass(frozen=True)
class MemoryProjection:
    match: MemoryMatch
    latent: torch.Tensor
    mask4: torch.Tensor
    occupancy: torch.Tensor
    confidence: torch.Tensor

    @property
    def condition(self) -> torch.Tensor:
        return torch.cat((self.mask4, self.latent), dim=2)


def _pose_distance(
    source_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
) -> tuple[float, float]:
    source = source_c2w.detach().float().cpu()
    target = target_c2w.detach().float().cpu()
    relative = source[:, :3, :3].transpose(1, 2) @ target[:, :3, :3]
    cosine = ((relative.diagonal(dim1=1, dim2=2).sum(1) - 1.0) / 2.0).clamp(-1, 1)
    rotation = torch.rad2deg(torch.acos(cosine)).mean()
    translation = torch.linalg.vector_norm(
        source[:, :3, 3] - target[:, :3, 3], dim=1
    ).mean()
    return float(rotation), float(translation)


class LatentMemoryBank:
    """Small online bank addressed only by commanded camera pose and FOV."""

    def __init__(
        self,
        *,
        translation_weight: float = 20.0,
        fov_weight: float = 0.25,
    ) -> None:
        self.translation_weight = float(translation_weight)
        self.fov_weight = float(fov_weight)
        self._records: dict[str, MemoryRecord] = {}
        self.events: list[dict[str, object]] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def get(self, memory_id: str) -> MemoryRecord:
        return self._records[memory_id]

    def write(self, record: MemoryRecord, *, replace_existing: bool = False) -> MemoryRecord:
        if record.memory_id in self._records and not replace_existing:
            raise KeyError(f"memory already exists: {record.memory_id}")
        version = (
            self._records[record.memory_id].version + 1
            if record.memory_id in self._records
            else 1
        )
        stored = replace(record.detached(), version=version)
        self._records[stored.memory_id] = stored
        self.events.append({
            "event": "write",
            "memory_id": stored.memory_id,
            "version": stored.version,
            "replace_existing": replace_existing,
        })
        return stored

    def retrieve(
        self,
        query_c2w: torch.Tensor,
        query_fov_degrees: float,
        *,
        top_k: int = 2,
        exclude_ids: tuple[str, ...] = (),
    ) -> list[MemoryMatch]:
        if tuple(query_c2w.shape) != (3, 4, 4):
            raise ValueError("query_c2w must be [3,4,4]")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        matches = []
        for memory_id, record in self._records.items():
            if memory_id in exclude_ids:
                continue
            rotation, translation = _pose_distance(record.c2w, query_c2w)
            fov_delta = abs(record.fov_degrees - float(query_fov_degrees))
            matches.append(MemoryMatch(
                record=record,
                score=(
                    rotation
                    + self.translation_weight * translation
                    + self.fov_weight * fov_delta
                ),
                rotation_degrees=rotation,
                translation=translation,
                fov_delta_degrees=fov_delta,
            ))
        matches.sort(key=lambda item: (item.score, -item.record.version, item.record.memory_id))
        selected = matches[:top_k]
        self.events.append({
            "event": "retrieve",
            "top_k": top_k,
            "result_ids": [item.record.memory_id for item in selected],
            "result_versions": [item.record.version for item in selected],
            "scores": [item.score for item in selected],
        })
        return selected

    def project(
        self,
        match: MemoryMatch,
        target_c2w: torch.Tensor,
    ) -> MemoryProjection:
        record = match.record
        latent, mask4, occupancy = project_memory_sequence(
            record.clean_latent,
            record.depth,
            record.intrinsics,
            record.c2w,
            target_c2w,
        )
        source_confidence = float(record.confidence.float().mean())
        confidence = occupancy.float() * source_confidence
        return MemoryProjection(
            match=match,
            latent=latent,
            mask4=mask4,
            occupancy=occupancy,
            confidence=confidence,
        )
