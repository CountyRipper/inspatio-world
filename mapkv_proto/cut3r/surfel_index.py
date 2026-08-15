from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


INDEX_VERSION = 1


@dataclass
class KVSurfel:
    position: np.ndarray
    normal: np.ndarray
    radius: float
    confidence: float
    observing_chunks: list[int]
    created_chunk: int


def estimate_normals(pointmap: np.ndarray) -> np.ndarray:
    pointmap = np.asarray(pointmap, dtype=np.float32)
    right = np.roll(pointmap, -1, axis=1) - pointmap
    down = np.roll(pointmap, -1, axis=0) - pointmap
    normals = np.cross(right, down)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-8)
    normals[-1] = 0
    normals[:, -1] = 0
    return normals


def pointmap_to_surfels(
    *,
    pointmap: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    c2w: np.ndarray,
    focal: float,
    chunk_id: int,
    confidence_threshold: float = 1.0,
    radius_scale: float = 0.5,
    depth_quantile: float = 0.999,
) -> list[KVSurfel]:
    pointmap = np.asarray(pointmap, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    c2w = np.asarray(c2w, dtype=np.float32)
    if pointmap.shape[:2] != depth.shape or depth.shape != confidence.shape:
        raise ValueError(
            f"Pointmap/depth/confidence mismatch: {pointmap.shape}, {depth.shape}, {confidence.shape}"
        )
    normals = estimate_normals(pointmap)
    finite_depth = depth[np.isfinite(depth) & (depth > 0)]
    if finite_depth.size == 0:
        return []
    far_threshold = float(np.quantile(finite_depth, depth_quantile))
    normal_length = np.linalg.norm(normals, axis=-1)
    valid = (
        np.isfinite(pointmap).all(axis=-1)
        & np.isfinite(confidence)
        & (depth > 0)
        & (depth <= far_threshold)
        & (confidence >= confidence_threshold)
        & (normal_length > 1e-6)
    )
    positions = pointmap[valid]
    normals = normals[valid]
    depths = depth[valid]
    confidences = confidence[valid]
    if positions.size == 0:
        return []
    camera_position = c2w[:3, 3]
    camera_to_point = positions - camera_position[None]
    lengths = np.linalg.norm(camera_to_point, axis=1, keepdims=True)
    camera_to_point = np.divide(
        camera_to_point, lengths, out=np.zeros_like(camera_to_point), where=lengths > 1e-8
    )
    dot = np.sum(camera_to_point * normals, axis=1)
    normals[dot < 0] *= -1
    incidence = np.abs(np.sum(camera_to_point * normals, axis=1))
    adjustment = 0.2 + 0.8 * incidence
    radii = radius_scale * depths / float(focal) / adjustment
    return [
        KVSurfel(
            position=position.copy(),
            normal=normal.copy(),
            radius=float(radius),
            confidence=float(conf),
            observing_chunks=[int(chunk_id)],
            created_chunk=int(chunk_id),
        )
        for position, normal, radius, conf in zip(positions, normals, radii, confidences)
        if np.isfinite(radius) and radius > 0
    ]


class SurfelIndex:
    def __init__(self, surfels: Iterable[KVSurfel] = ()):
        self.surfels = list(surfels)

    def merge(
        self,
        new_surfels: Iterable[KVSurfel],
        *,
        position_threshold: float | None = None,
        normal_cosine: float = 0.6,
    ) -> dict:
        new_surfels = list(new_surfels)
        if not new_surfels:
            return {"added": 0, "merged": 0, "position_threshold": position_threshold}
        all_radii = np.asarray(
            [surfel.radius for surfel in self.surfels] + [surfel.radius for surfel in new_surfels],
            dtype=np.float32,
        )
        if position_threshold is None:
            position_threshold = float(all_radii.mean() + 0.5 * all_radii.std())
        existing_positions = np.asarray(
            [surfel.position for surfel in self.surfels], dtype=np.float32
        )
        tree = None
        if len(existing_positions):
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(existing_positions)
            except ImportError:
                tree = None
        added = []
        merged = 0
        for surfel in new_surfels:
            if len(existing_positions) == 0:
                candidates = []
            elif tree is not None:
                candidates = tree.query_ball_point(surfel.position, position_threshold)
            else:
                distances = np.linalg.norm(existing_positions - surfel.position[None], axis=1)
                candidates = np.flatnonzero(distances <= position_threshold).tolist()
            match = None
            best_distance = np.inf
            for index in candidates:
                existing = self.surfels[int(index)]
                if float(np.dot(existing.normal, surfel.normal)) <= normal_cosine:
                    continue
                distance = float(np.linalg.norm(existing.position - surfel.position))
                if distance < best_distance:
                    best_distance = distance
                    match = existing
            if match is None:
                added.append(surfel)
            else:
                for chunk_id in surfel.observing_chunks:
                    if chunk_id not in match.observing_chunks:
                        match.observing_chunks.append(chunk_id)
                merged += 1
        self.surfels.extend(added)
        return {
            "added": len(added),
            "merged": merged,
            "position_threshold": position_threshold,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        observations = []
        offsets = [0]
        for surfel in self.surfels:
            observations.extend(sorted(set(int(x) for x in surfel.observing_chunks)))
            offsets.append(len(observations))
        return {
            "version": np.asarray([INDEX_VERSION], dtype=np.int32),
            "positions": np.asarray([s.position for s in self.surfels], dtype=np.float32).reshape(-1, 3),
            "normals": np.asarray([s.normal for s in self.surfels], dtype=np.float32).reshape(-1, 3),
            "radii": np.asarray([s.radius for s in self.surfels], dtype=np.float32),
            "confidence": np.asarray([s.confidence for s in self.surfels], dtype=np.float32),
            "created_chunks": np.asarray([s.created_chunk for s in self.surfels], dtype=np.int32),
            "observation_offsets": np.asarray(offsets, dtype=np.int64),
            "observing_chunks": np.asarray(observations, dtype=np.int32),
        }

    def save(self, npz_path: str | Path, ply_path: str | Path | None = None) -> None:
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **self.arrays())
        if ply_path is not None:
            self.write_ply(ply_path)

    @classmethod
    def load(cls, path: str | Path) -> "SurfelIndex":
        with np.load(path) as data:
            if int(data["version"][0]) != INDEX_VERSION:
                raise ValueError(f"Unsupported surfel index version: {data['version'][0]}")
            offsets = data["observation_offsets"]
            surfels = []
            for index in range(len(data["positions"])):
                surfels.append(
                    KVSurfel(
                        position=data["positions"][index].copy(),
                        normal=data["normals"][index].copy(),
                        radius=float(data["radii"][index]),
                        confidence=float(data["confidence"][index]),
                        observing_chunks=data["observing_chunks"][
                            offsets[index]:offsets[index + 1]
                        ].astype(int).tolist(),
                        created_chunk=int(data["created_chunks"][index]),
                    )
                )
        return cls(surfels)

    def write_ply(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(self.surfels)}\n")
            for name in ("x", "y", "z", "nx", "ny", "nz", "radius", "confidence"):
                handle.write(f"property float {name}\n")
            handle.write("property int created_chunk\nend_header\n")
            for surfel in self.surfels:
                values = [
                    *surfel.position.tolist(),
                    *surfel.normal.tolist(),
                    surfel.radius,
                    surfel.confidence,
                    surfel.created_chunk,
                ]
                handle.write(" ".join(str(value) for value in values) + "\n")

    def render(
        self,
        *,
        c2w: np.ndarray,
        intrinsic: np.ndarray,
        image_hw: tuple[int, int],
        maximum_created_chunk: int | None = None,
        maximum_radius_pixels: float = 12.0,
    ) -> dict[str, np.ndarray]:
        height, width = image_hw
        depth = np.full((height, width), np.inf, dtype=np.float32)
        surfel_ids = np.full((height, width), -1, dtype=np.int32)
        cosine = np.zeros((height, width), dtype=np.float32)
        if not self.surfels:
            depth.fill(0)
            return {"depth": depth, "surfel_id": surfel_ids, "cosine": cosine}
        c2w = np.asarray(c2w, dtype=np.float64)
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        camera_position = c2w[:3, 3]
        positions = np.asarray([s.position for s in self.surfels], dtype=np.float64)
        homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        camera_points = (homogeneous @ w2c.T)[:, :3]
        for surfel_id, camera_point in enumerate(camera_points):
            surfel = self.surfels[surfel_id]
            if (
                maximum_created_chunk is not None
                and surfel.created_chunk > maximum_created_chunk
            ):
                continue
            z = float(camera_point[2])
            if not np.isfinite(z) or z <= 1e-4:
                continue
            camera_to_point = surfel.position - camera_position
            direction_norm = np.linalg.norm(camera_to_point)
            if direction_norm <= 1e-8:
                continue
            normal_cosine = float(np.dot(camera_to_point / direction_norm, surfel.normal))
            if normal_cosine <= 0:
                continue
            u = float(intrinsic[0, 0] * camera_point[0] / z + intrinsic[0, 2])
            v = float(intrinsic[1, 1] * camera_point[1] / z + intrinsic[1, 2])
            radius_pixels = (
                0.5 * (intrinsic[0, 0] + intrinsic[1, 1]) * surfel.radius / z
            )
            radius_pixels = float(np.clip(radius_pixels, 0.5, maximum_radius_pixels))
            x0 = max(0, int(np.floor(u - radius_pixels)))
            x1 = min(width - 1, int(np.ceil(u + radius_pixels)))
            y0 = max(0, int(np.floor(v - radius_pixels)))
            y1 = min(height - 1, int(np.ceil(v + radius_pixels)))
            if x0 > x1 or y0 > y1:
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            disk = (xx - u) ** 2 + (yy - v) ** 2 <= radius_pixels ** 2
            current = depth[y0:y1 + 1, x0:x1 + 1]
            update = disk & (z < current)
            current[update] = z
            surfel_ids[y0:y1 + 1, x0:x1 + 1][update] = surfel_id
            cosine[y0:y1 + 1, x0:x1 + 1][update] = normal_cosine
        depth[~np.isfinite(depth)] = 0
        return {"depth": depth, "surfel_id": surfel_ids, "cosine": cosine}

    def retrieve(
        self,
        *,
        target_chunk: int,
        c2w: np.ndarray,
        intrinsic: np.ndarray,
        image_hw: tuple[int, int],
        oracle_chunk: int | None = None,
    ) -> tuple[dict, dict[str, np.ndarray]]:
        rendered = self.render(
            c2w=c2w,
            intrinsic=intrinsic,
            image_hw=image_hw,
            maximum_created_chunk=target_chunk - 2,
        )
        candidate_chunks = sorted(
            {
                chunk
                for surfel in self.surfels
                if surfel.created_chunk <= target_chunk - 2
                for chunk in surfel.observing_chunks
                if 0 <= chunk < target_chunk - 1
            }
        )
        scores = {chunk: 0.0 for chunk in candidate_chunks}
        ids = rendered["surfel_id"]
        valid_pixels = np.argwhere(ids >= 0)
        for y, x in valid_pixels:
            surfel = self.surfels[int(ids[y, x])]
            contribution = (
                surfel.confidence
                * max(0.0, float(rendered["cosine"][y, x]))
                / (1.0 + float(rendered["depth"][y, x]))
            )
            for chunk in surfel.observing_chunks:
                if chunk in scores:
                    scores[chunk] += contribution
        nonzero = {chunk: value for chunk, value in scores.items() if value > 0}
        selected = max(nonzero, key=nonzero.get) if nonzero else None
        coverage = self.coverage_for_chunk(
            rendered,
            chunk_id=selected,
            target_chunk=target_chunk,
        )
        plan = {
            "target_chunk": int(target_chunk),
            "candidate_chunks": candidate_chunks,
            "scores": {str(chunk): float(value) for chunk, value in nonzero.items()},
            "selected_chunks": [] if selected is None else [int(selected)],
            "weights": [] if selected is None else [1.0],
            "oracle_hit": oracle_chunk is not None and selected == oracle_chunk,
        }
        rendered["coverage"] = coverage
        return plan, rendered

    def coverage_for_chunk(
        self,
        rendered: dict[str, np.ndarray],
        *,
        chunk_id: int | None,
        target_chunk: int,
    ) -> np.ndarray:
        """Visible-pixel coverage for one causally valid observing chunk."""
        ids = rendered["surfel_id"]
        coverage = np.zeros(ids.shape, dtype=np.float32)
        if chunk_id is None:
            return coverage
        chunk_id = int(chunk_id)
        if chunk_id < 0 or chunk_id >= target_chunk - 1:
            raise ValueError(
                f"Chunk {chunk_id} is not causally valid for target {target_chunk}"
            )
        for y, x in np.argwhere(ids >= 0):
            surfel = self.surfels[int(ids[y, x])]
            if chunk_id in surfel.observing_chunks:
                coverage[y, x] = 1.0
        return coverage
