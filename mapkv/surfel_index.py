from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image


INDEX_VERSION = 4


@dataclass
class SurfelCell:
    """Coarse geometry address; native KV remains in KVChunkBank."""

    voxel_key: tuple[int, int, int]
    xyz: np.ndarray
    confidence: float
    normal: np.ndarray | None
    radius: float
    rgb_preview: np.ndarray | None
    first_seen_chunk: int
    last_seen_chunk: int
    observing_chunks: list[int] = field(default_factory=list)
    chunk_weights: dict[int, float] = field(default_factory=dict)
    view_dirs: dict[int, np.ndarray] = field(default_factory=dict)
    reference_blind_at_write: dict[int, float] = field(default_factory=dict)
    source_pixels: dict[int, np.ndarray] = field(default_factory=dict)
    image_center_margins: dict[int, float] = field(default_factory=dict)
    observation_weight: float = 0.0


def _normal_grid(points: np.ndarray) -> np.ndarray:
    right = np.roll(points, -1, axis=1) - points
    down = np.roll(points, -1, axis=0) - points
    normals = np.cross(right, down)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-8)
    normals[-1] = 0
    normals[:, -1] = 0
    return normals


def _sample_grid(array: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    height, width = array.shape[:2]
    yy = np.rint(np.linspace(0, height - 1, min(grid_hw[0], height))).astype(int)
    xx = np.rint(np.linspace(0, width - 1, min(grid_hw[1], width))).astype(int)
    return array[np.ix_(yy, xx)]


def _sample_pixel_grid(
    source_hw: tuple[int, int], grid_hw: tuple[int, int]
) -> np.ndarray:
    height, width = source_hw
    yy = np.rint(
        np.linspace(0, height - 1, min(grid_hw[0], height))
    ).astype(np.float32)
    xx = np.rint(
        np.linspace(0, width - 1, min(grid_hw[1], width))
    ).astype(np.float32)
    pixel_x, pixel_y = np.meshgrid(xx, yy, indexing="xy")
    return np.stack([pixel_y, pixel_x], axis=-1)


def _scale_intrinsics(
    intrinsic: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    result = np.asarray(intrinsic, dtype=np.float64).copy()
    sy = target_hw[0] / source_hw[0]
    sx = target_hw[1] / source_hw[1]
    result[0] *= sx
    result[1] *= sy
    return result


class SurfelIndex:
    """Radius/normal surfel fusion with a voxel hash used only for acceleration."""

    def __init__(self, voxel_size: float, cells: list[SurfelCell] | None = None):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.cells = list(cells or [])
        self._rebuild_buckets()

    def _key(self, xyz: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(np.asarray(xyz) / self.voxel_size).astype(np.int64))

    def _rebuild_buckets(self) -> None:
        self._by_key: dict[tuple[int, int, int], list[int]] = {}
        for index, cell in enumerate(self.cells):
            cell.voxel_key = self._key(cell.xyz)
            self._by_key.setdefault(cell.voxel_key, []).append(index)

    def _voxel_candidates(self, xyz: np.ndarray, distance: float) -> list[int]:
        center = self._key(xyz)
        reach = max(1, int(np.ceil(distance / self.voxel_size)))
        candidates: list[int] = []
        for offset in product(range(-reach, reach + 1), repeat=3):
            key = tuple(center[axis] + offset[axis] for axis in range(3))
            candidates.extend(self._by_key.get(key, ()))
        return candidates

    def merge_observations(
        self,
        observations: list[SurfelCell],
        *,
        position_threshold: float | None = None,
        normal_cosine: float = 0.6,
    ) -> dict:
        """Merge a new view into existing surfaces, never within the same view.

        The threshold is derived from projected surfel radii. Candidate search
        crosses voxel boundaries; sharing one voxel is neither required nor
        sufficient for a merge.
        """
        if not observations:
            return {
                "added": 0,
                "merged": 0,
                "cross_voxel_merges": 0,
                "position_threshold": position_threshold,
            }
        radii = np.asarray(
            [cell.radius for cell in self.cells] + [cell.radius for cell in observations],
            dtype=np.float32,
        )
        finite_radii = radii[np.isfinite(radii) & (radii > 0)]
        if position_threshold is None:
            if finite_radii.size:
                position_threshold = float(
                    finite_radii.mean() + 0.5 * finite_radii.std()
                )
            else:
                position_threshold = self.voxel_size
        position_threshold = max(float(position_threshold), self.voxel_size)

        existing_count = len(self.cells)
        existing_positions = np.asarray(
            [cell.xyz for cell in self.cells], dtype=np.float32
        ).reshape(-1, 3)
        tree = None
        if existing_count:
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(existing_positions)
            except ImportError:
                tree = None
        existing_radius_95 = (
            float(np.quantile([cell.radius for cell in self.cells], 0.95))
            if self.cells
            else position_threshold
        )

        added: list[SurfelCell] = []
        merged = 0
        cross_voxel_merges = 0
        for observation in observations:
            search_radius = max(
                position_threshold,
                0.5 * (float(observation.radius) + existing_radius_95),
            )
            if not existing_count:
                candidates: list[int] = []
            elif tree is not None:
                candidates = tree.query_ball_point(
                    observation.xyz, search_radius
                )
            else:
                candidates = self._voxel_candidates(
                    observation.xyz, search_radius
                )

            match_index = None
            best_distance = np.inf
            for index in candidates:
                if index >= existing_count:
                    continue
                existing = self.cells[int(index)]
                if existing.normal is None or observation.normal is None:
                    continue
                if float(np.dot(existing.normal, observation.normal)) < normal_cosine:
                    continue
                allowed = max(
                    position_threshold,
                    0.5 * (float(existing.radius) + float(observation.radius)),
                )
                distance = float(np.linalg.norm(existing.xyz - observation.xyz))
                if distance <= allowed and distance < best_distance:
                    match_index = int(index)
                    best_distance = distance

            if match_index is None:
                observation.voxel_key = self._key(observation.xyz)
                added.append(observation)
                continue

            match = self.cells[match_index]
            if match.voxel_key != self._key(observation.xyz):
                cross_voxel_merges += 1
            # Preserve established surface geometry. Only address metadata is
            # accumulated, preventing later noisy views from dragging the map.
            match.last_seen_chunk = max(
                match.last_seen_chunk, observation.last_seen_chunk
            )
            match.observation_weight += observation.observation_weight
            for chunk in observation.observing_chunks:
                chunk = int(chunk)
                if chunk not in match.observing_chunks:
                    match.observing_chunks.append(chunk)
                match.chunk_weights[chunk] = (
                    match.chunk_weights.get(chunk, 0.0)
                    + observation.chunk_weights.get(chunk, 0.0)
                )
                incoming = observation.view_dirs.get(chunk)
                if incoming is not None:
                    previous = match.view_dirs.get(chunk)
                    if previous is None:
                        match.view_dirs[chunk] = incoming.copy()
                    else:
                        value = previous + incoming
                        norm = float(np.linalg.norm(value))
                        match.view_dirs[chunk] = (
                            value / norm if norm > 1e-8 else previous
                        )
                incoming_blind = observation.reference_blind_at_write.get(
                    chunk
                )
                if incoming_blind is not None:
                    match.reference_blind_at_write[chunk] = max(
                        float(incoming_blind),
                        float(match.reference_blind_at_write.get(chunk, 0.0)),
                    )
                incoming_margin = observation.image_center_margins.get(chunk)
                if incoming_margin is not None and (
                    chunk not in match.image_center_margins
                    or float(incoming_margin)
                    > float(match.image_center_margins[chunk])
                ):
                    match.image_center_margins[chunk] = float(incoming_margin)
                    incoming_pixel = observation.source_pixels.get(chunk)
                    if incoming_pixel is not None:
                        match.source_pixels[chunk] = incoming_pixel.copy()
            match.observing_chunks.sort()
            merged += 1

        self.cells.extend(added)
        self._rebuild_buckets()
        return {
            "added": len(added),
            "merged": merged,
            "cross_voxel_merges": cross_voxel_merges,
            "position_threshold": position_threshold,
        }

    def insert_frame(
        self,
        pts3d: np.ndarray,
        confidence: np.ndarray,
        camera_pose: np.ndarray,
        chunk_id: int,
        rgb: np.ndarray | None = None,
        reference_validity: np.ndarray | None = None,
        *,
        intrinsics: np.ndarray | None = None,
        confidence_threshold: float = 1.5,
        grid_hw: tuple[int, int] = (30, 52),
        radius_scale: float = 0.5,
        normal_cosine: float = 0.6,
        position_threshold: float | None = None,
    ) -> dict:
        source_hw = tuple(int(x) for x in np.asarray(confidence).shape)
        points = _sample_grid(np.asarray(pts3d, dtype=np.float32), grid_hw)
        conf = _sample_grid(np.asarray(confidence, dtype=np.float32), grid_hw)
        source_pixels = _sample_pixel_grid(source_hw, grid_hw)
        colors = None if rgb is None else _sample_grid(np.asarray(rgb), grid_hw)
        reference_valid = None
        if reference_validity is not None:
            reference_array = np.asarray(reference_validity, dtype=np.float32)
            if reference_array.ndim != 2:
                raise ValueError(
                    "reference_validity must be a 2D image-space mask"
                )
            if tuple(reference_array.shape) != source_hw:
                reference_array = np.asarray(
                    Image.fromarray(reference_array, mode="F").resize(
                        (source_hw[1], source_hw[0]),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            reference_valid = _sample_grid(reference_array, grid_hw).clip(0, 1)
        normals = _normal_grid(points)
        pose = np.asarray(camera_pose, dtype=np.float32)
        camera = pose[:3, 3]
        homogeneous = np.concatenate(
            [points.reshape(-1, 3), np.ones((points.size // 3, 1), dtype=np.float32)],
            axis=1,
        )
        camera_points = (
            homogeneous @ np.linalg.inv(pose).astype(np.float32).T
        )[:, :3].reshape(*points.shape)
        depth = camera_points[..., 2]
        finite_depth = depth[np.isfinite(depth) & (depth > 0)]
        far = float(np.quantile(finite_depth, 0.995)) if finite_depth.size else 0.0
        valid = (
            np.isfinite(points).all(-1)
            & np.isfinite(conf)
            & (conf >= confidence_threshold)
            & (depth > 0)
            & (depth <= far)
            & (np.linalg.norm(normals, axis=-1) > 1e-6)
        )
        if not np.any(valid):
            return {
                "chunk_id": int(chunk_id),
                "accepted": 0,
                "added": 0,
                "merged": 0,
                "cross_voxel_merges": 0,
            }

        points_flat = points[valid]
        conf_flat = conf[valid]
        depth_flat = depth[valid]
        normals_flat = normals[valid]
        colors_flat = None if colors is None else colors[valid]
        source_pixels_flat = source_pixels[valid]
        reference_valid_flat = (
            None if reference_valid is None else reference_valid[valid]
        )
        toward_camera = camera[None] - points_flat
        view_norm = np.linalg.norm(toward_camera, axis=-1, keepdims=True)
        view_dirs = np.divide(
            toward_camera,
            view_norm,
            out=np.zeros_like(toward_camera),
            where=view_norm > 1e-8,
        )
        flip = np.sum(normals_flat * view_dirs, axis=-1) < 0
        normals_flat[flip] *= -1
        incidence = np.abs(np.sum(normals_flat * view_dirs, axis=-1))

        if intrinsics is None:
            focal = float(max(grid_hw))
        else:
            scaled = _scale_intrinsics(
                np.asarray(intrinsics), source_hw, points.shape[:2]
            )
            focal = float(0.5 * (scaled[0, 0] + scaled[1, 1]))
        radii = (
            float(radius_scale)
            * depth_flat
            / max(focal, 1e-6)
            / (0.2 + 0.8 * incidence)
        )
        upper = float(np.quantile(conf_flat, 0.95))
        denom = max(upper - confidence_threshold, 1e-6)
        weights = np.clip(
            (conf_flat - confidence_threshold) / denom, 0.05, 1.0
        )

        observations = []
        for index, (point, normal, radius, conf_value, weight, view_dir) in enumerate(
            zip(
                points_flat,
                normals_flat,
                radii,
                conf_flat,
                weights,
                view_dirs,
            )
        ):
            if not np.isfinite(radius) or radius <= 0:
                continue
            color = (
                None
                if colors_flat is None
                else colors_flat[index].astype(np.float32)
            )
            pixel_y, pixel_x = source_pixels_flat[index]
            x_margin = min(
                float(pixel_x) + 0.5,
                float(source_hw[1]) - 0.5 - float(pixel_x),
            ) / max(0.5 * float(source_hw[1]), 1.0)
            y_margin = min(
                float(pixel_y) + 0.5,
                float(source_hw[0]) - 0.5 - float(pixel_y),
            ) / max(0.5 * float(source_hw[0]), 1.0)
            center_margin = float(np.clip(min(x_margin, y_margin), 0.0, 1.0))
            observations.append(
                SurfelCell(
                    voxel_key=self._key(point),
                    xyz=point.copy(),
                    confidence=float(conf_value),
                    normal=normal.copy(),
                    radius=float(radius),
                    rgb_preview=None if color is None else color.copy(),
                    first_seen_chunk=int(chunk_id),
                    last_seen_chunk=int(chunk_id),
                    observing_chunks=[int(chunk_id)],
                    chunk_weights={int(chunk_id): float(weight)},
                    view_dirs={int(chunk_id): view_dir.copy()},
                    reference_blind_at_write=(
                        {}
                        if reference_valid_flat is None
                        else {
                            int(chunk_id): float(
                                1.0 - reference_valid_flat[index]
                            )
                        }
                    ),
                    source_pixels={
                        int(chunk_id): source_pixels_flat[index].copy()
                    },
                    image_center_margins={
                        int(chunk_id): center_margin
                    },
                    observation_weight=float(weight),
                )
            )
        merged = self.merge_observations(
            observations,
            position_threshold=position_threshold,
            normal_cosine=normal_cosine,
        )
        return {
            "chunk_id": int(chunk_id),
            "accepted": len(observations),
            "mean_radius": (
                float(np.mean([item.radius for item in observations]))
                if observations
                else 0.0
            ),
            **merged,
        }

    def eligible_cell_indices(
        self,
        eligible_max_chunk: int | None,
        eligible_chunks: set[int] | None = None,
        eligible_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        allowed = (
            None
            if eligible_indices is None
            else {int(index) for index in np.asarray(eligible_indices).reshape(-1)}
        )
        if (
            eligible_max_chunk is None
            and eligible_chunks is None
            and allowed is None
        ):
            return np.arange(len(self.cells), dtype=np.int32)
        return np.asarray(
            [
                index
                for index, cell in enumerate(self.cells)
                if (allowed is None or index in allowed)
                and any(
                    (eligible_max_chunk is None or 0 <= int(chunk) <= int(eligible_max_chunk))
                    and (eligible_chunks is None or int(chunk) in eligible_chunks)
                    for chunk in cell.observing_chunks
                )
            ],
            dtype=np.int32,
        )

    def generated_only_cell_indices(
        self,
        chunk_id: int,
        *,
        reference_blind_threshold: float = 0.5,
    ) -> np.ndarray:
        """Return cells whose selected observation was source-blind at write."""
        chunk_id = int(chunk_id)
        selected = [
            index
            for index, cell in enumerate(self.cells)
            if chunk_id in cell.observing_chunks
            and float(
                cell.reference_blind_at_write.get(chunk_id, -1.0)
            )
            >= float(reference_blind_threshold)
        ]
        return np.asarray(selected, dtype=np.int32)

    def visible_cells(
        self,
        query_pose: np.ndarray,
        intrinsics: np.ndarray,
        image_size: tuple[int, int],
        *,
        source_image_size: tuple[int, int] | None = None,
        eligible_max_chunk: int | None = None,
        eligible_chunks: set[int] | None = None,
        eligible_indices: np.ndarray | None = None,
        use_occlusion: bool = True,
        front_facing: bool = False,
        maximum_radius_pixels: float = 12.0,
    ) -> dict[str, np.ndarray | int]:
        """Project only causally eligible surfaces, then apply the z-buffer."""
        candidates = self.eligible_cell_indices(
            eligible_max_chunk, eligible_chunks, eligible_indices
        )
        empty = {
            "indices": np.empty(0, dtype=np.int32),
            "pixels": np.empty((0, 2), dtype=np.int32),
            "depth": np.empty(0, dtype=np.float32),
            "normal_cosine": np.empty(0, dtype=np.float32),
            "num_eligible_cells": int(len(candidates)),
            "num_visible_cells": 0,
        }
        if not len(candidates):
            return empty

        pose = np.asarray(query_pose, dtype=np.float64)
        intrinsic = np.asarray(intrinsics, dtype=np.float64)
        if source_image_size is not None:
            intrinsic = _scale_intrinsics(
                intrinsic, source_image_size, image_size
            )
        positions = np.asarray(
            [self.cells[int(index)].xyz for index in candidates],
            dtype=np.float64,
        )
        homogeneous = np.concatenate(
            [positions, np.ones((len(positions), 1))], axis=1
        )
        camera_points = (
            homogeneous @ np.linalg.inv(pose).T
        )[:, :3]
        z = camera_points[:, 2]
        valid = np.isfinite(camera_points).all(-1) & (z > 1e-5)
        u = (
            intrinsic[0, 0] * camera_points[:, 0] / np.maximum(z, 1e-8)
            + intrinsic[0, 2]
        )
        v = (
            intrinsic[1, 1] * camera_points[:, 1] / np.maximum(z, 1e-8)
            + intrinsic[1, 2]
        )
        camera = pose[:3, 3]
        to_camera = camera[None] - positions
        to_camera /= np.maximum(
            np.linalg.norm(to_camera, axis=-1, keepdims=True), 1e-8
        )
        normals = np.asarray(
            [
                np.zeros(3, dtype=np.float64)
                if self.cells[int(index)].normal is None
                else self.cells[int(index)].normal
                for index in candidates
            ]
        )
        cosine = np.sum(normals * to_camera, axis=-1)
        if front_facing:
            valid &= cosine > 0

        if not use_occlusion:
            selected = np.flatnonzero(
                valid
                & (u >= 0)
                & (u < image_size[1])
                & (v >= 0)
                & (v < image_size[0])
            )
            if not len(selected):
                return empty
            indices = candidates[selected]
            return {
                "indices": indices.astype(np.int32),
                "pixels": np.stack(
                    [np.rint(v[selected]), np.rint(u[selected])], axis=-1
                ).astype(np.int32),
                "depth": z[selected].astype(np.float32),
                "normal_cosine": np.maximum(
                    cosine[selected], 0
                ).astype(np.float32),
                "num_eligible_cells": int(len(candidates)),
                "num_visible_cells": int(len(np.unique(indices))),
            }

        depth_map = np.full(image_size, np.inf, dtype=np.float32)
        id_map = np.full(image_size, -1, dtype=np.int32)
        cosine_map = np.zeros(image_size, dtype=np.float32)
        order = np.flatnonzero(valid)
        order = order[np.argsort(z[order])]
        focal = 0.5 * (intrinsic[0, 0] + intrinsic[1, 1])
        for local_index in order:
            cell_index = int(candidates[local_index])
            cell = self.cells[cell_index]
            radius_pixels = float(
                np.clip(
                    focal * max(float(cell.radius), 0.5 * self.voxel_size)
                    / float(z[local_index]),
                    0.5,
                    maximum_radius_pixels,
                )
            )
            x0 = max(0, int(np.floor(u[local_index] - radius_pixels)))
            x1 = min(
                image_size[1] - 1,
                int(np.ceil(u[local_index] + radius_pixels)),
            )
            y0 = max(0, int(np.floor(v[local_index] - radius_pixels)))
            y1 = min(
                image_size[0] - 1,
                int(np.ceil(v[local_index] + radius_pixels)),
            )
            if x0 > x1 or y0 > y1:
                continue
            yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            disk = (
                (xx - u[local_index]) ** 2
                + (yy - v[local_index]) ** 2
                <= radius_pixels**2
            )
            current = depth_map[y0 : y1 + 1, x0 : x1 + 1]
            update = disk & (z[local_index] < current)
            current[update] = z[local_index]
            id_map[y0 : y1 + 1, x0 : x1 + 1][update] = cell_index
            cosine_map[y0 : y1 + 1, x0 : x1 + 1][update] = max(
                float(cosine[local_index]), 0.0
            )
        pixels = np.argwhere(id_map >= 0)
        if not len(pixels):
            return empty
        indices = id_map[pixels[:, 0], pixels[:, 1]]
        return {
            "indices": indices.astype(np.int32),
            "pixels": pixels.astype(np.int32),
            "depth": depth_map[pixels[:, 0], pixels[:, 1]].astype(
                np.float32
            ),
            "normal_cosine": cosine_map[
                pixels[:, 0], pixels[:, 1]
            ].astype(np.float32),
            "num_eligible_cells": int(len(candidates)),
            "num_visible_cells": int(len(np.unique(indices))),
        }

    def stats(self) -> dict:
        observation_counts = [
            len(set(cell.observing_chunks)) for cell in self.cells
        ]
        radii = np.asarray([cell.radius for cell in self.cells], dtype=np.float32)
        return {
            "num_cells": len(self.cells),
            "voxel_size": self.voxel_size,
            "mean_observing_chunks_per_cell": (
                float(np.mean(observation_counts)) if observation_counts else 0.0
            ),
            "multi_view_cell_fraction": (
                float(np.mean(np.asarray(observation_counts) > 1))
                if observation_counts
                else 0.0
            ),
            "mean_radius": float(radii.mean()) if radii.size else 0.0,
            "radius_p95": (
                float(np.quantile(radii, 0.95)) if radii.size else 0.0
            ),
            "first_seen_chunk": (
                min(cell.first_seen_chunk for cell in self.cells)
                if self.cells
                else None
            ),
            "last_seen_chunk": (
                max(cell.last_seen_chunk for cell in self.cells)
                if self.cells
                else None
            ),
        }

    def save(self, path: str | Path) -> None:
        chunk_ids: list[int] = []
        chunk_weights: list[float] = []
        view_dirs: list[np.ndarray] = []
        reference_blind_at_write: list[float] = []
        source_pixels: list[np.ndarray] = []
        image_center_margins: list[float] = []
        offsets = [0]
        for cell in self.cells:
            for chunk_id in sorted(set(cell.observing_chunks)):
                chunk_ids.append(chunk_id)
                chunk_weights.append(cell.chunk_weights.get(chunk_id, 0.0))
                view_dirs.append(cell.view_dirs.get(chunk_id, np.zeros(3)))
                reference_blind_at_write.append(
                    float(cell.reference_blind_at_write.get(chunk_id, np.nan))
                )
                source_pixels.append(
                    cell.source_pixels.get(
                        chunk_id, np.full(2, np.nan, dtype=np.float32)
                    )
                )
                image_center_margins.append(
                    float(cell.image_center_margins.get(chunk_id, np.nan))
                )
            offsets.append(len(chunk_ids))
        np.savez_compressed(
            path,
            version=np.asarray([INDEX_VERSION], dtype=np.int32),
            voxel_size=np.asarray([self.voxel_size], dtype=np.float32),
            voxel_keys=np.asarray(
                [cell.voxel_key for cell in self.cells], dtype=np.int64
            ).reshape(-1, 3),
            xyz=np.asarray(
                [cell.xyz for cell in self.cells], dtype=np.float32
            ).reshape(-1, 3),
            confidence=np.asarray(
                [cell.confidence for cell in self.cells], dtype=np.float32
            ),
            normals=np.asarray(
                [
                    np.zeros(3) if cell.normal is None else cell.normal
                    for cell in self.cells
                ],
                dtype=np.float32,
            ).reshape(-1, 3),
            radii=np.asarray(
                [cell.radius for cell in self.cells], dtype=np.float32
            ),
            first_seen=np.asarray(
                [cell.first_seen_chunk for cell in self.cells],
                dtype=np.int32,
            ),
            last_seen=np.asarray(
                [cell.last_seen_chunk for cell in self.cells],
                dtype=np.int32,
            ),
            observation_weight=np.asarray(
                [cell.observation_weight for cell in self.cells],
                dtype=np.float32,
            ),
            chunk_ids=np.asarray(chunk_ids, dtype=np.int32),
            chunk_weights=np.asarray(chunk_weights, dtype=np.float32),
            view_dirs=np.asarray(view_dirs, dtype=np.float32).reshape(-1, 3),
            reference_blind_at_write=np.asarray(
                reference_blind_at_write, dtype=np.float32
            ),
            source_pixels=np.asarray(
                source_pixels, dtype=np.float32
            ).reshape(-1, 2),
            image_center_margins=np.asarray(
                image_center_margins, dtype=np.float32
            ),
            offsets=np.asarray(offsets, dtype=np.int64),
        )

    def write_ply(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(self.cells)}\n")
            for name in (
                "x",
                "y",
                "z",
                "nx",
                "ny",
                "nz",
                "radius",
                "confidence",
            ):
                handle.write(f"property float {name}\n")
            handle.write(
                "property int first_seen_chunk\n"
                "property int observation_count\nend_header\n"
            )
            for cell in self.cells:
                normal = (
                    np.zeros(3) if cell.normal is None else cell.normal
                )
                values = [
                    *cell.xyz.tolist(),
                    *normal.tolist(),
                    cell.radius,
                    cell.confidence,
                    cell.first_seen_chunk,
                    len(set(cell.observing_chunks)),
                ]
                handle.write(" ".join(str(value) for value in values) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "SurfelIndex":
        # NpzFile lazily decompresses on every ``__getitem__`` call.  Loading
        # fields inside the per-cell loop makes even a 2 MB index take minutes.
        # Materialize every compact array once, then assemble Python records.
        with np.load(path) as archive:
            payload = {name: archive[name] for name in archive.files}
        version = int(payload["version"][0])
        if version not in (1, 2, 3, INDEX_VERSION):
            raise ValueError(f"Unsupported surfel index version: {version}")
        offsets = payload["offsets"]
        voxel_size = float(payload["voxel_size"][0])
        cells = []
        for index in range(len(payload["xyz"])):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            chunks = payload["chunk_ids"][start:stop]
            weights = payload["chunk_weights"][start:stop]
            directions = payload["view_dirs"][start:stop]
            blind_values = (
                payload["reference_blind_at_write"][start:stop]
                if version >= 3
                else np.full(len(chunks), np.nan, dtype=np.float32)
            )
            source_pixels = (
                payload["source_pixels"][start:stop]
                if version >= 4
                else np.full((len(chunks), 2), np.nan, dtype=np.float32)
            )
            center_margins = (
                payload["image_center_margins"][start:stop]
                if version >= 4
                else np.full(len(chunks), np.nan, dtype=np.float32)
            )
            chunk_weights = {
                int(chunk): float(weight)
                for chunk, weight in zip(chunks, weights)
            }
            cells.append(
                SurfelCell(
                    voxel_key=tuple(
                        int(x) for x in payload["voxel_keys"][index]
                    ),
                    xyz=payload["xyz"][index].copy(),
                    confidence=float(payload["confidence"][index]),
                    normal=payload["normals"][index].copy(),
                    radius=(
                        float(payload["radii"][index])
                        if version >= 2
                        else voxel_size
                    ),
                    rgb_preview=None,
                    first_seen_chunk=int(payload["first_seen"][index]),
                    last_seen_chunk=int(payload["last_seen"][index]),
                    observing_chunks=sorted(chunk_weights),
                    chunk_weights=chunk_weights,
                    view_dirs={
                        int(chunk): direction.copy()
                        for chunk, direction in zip(chunks, directions)
                    },
                    reference_blind_at_write={
                        int(chunk): float(value)
                        for chunk, value in zip(chunks, blind_values)
                        if np.isfinite(value)
                    },
                    source_pixels={
                        int(chunk): pixel.copy()
                        for chunk, pixel in zip(chunks, source_pixels)
                        if np.isfinite(pixel).all()
                    },
                    image_center_margins={
                        int(chunk): float(value)
                        for chunk, value in zip(chunks, center_margins)
                        if np.isfinite(value)
                    },
                    observation_weight=float(
                        payload["observation_weight"][index]
                    ),
                )
            )
        return cls(voxel_size, cells)


def write_oriented_disk_preview(
    index: SurfelIndex,
    path: str | Path,
    *,
    max_disks: int = 3000,
    vertices_per_disk: int = 12,
) -> None:
    """Render an inspectable VMem-style oriented-disk view without changing geometry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_indices = [
        item
        for item, cell in enumerate(index.cells)
        if cell.normal is not None
        and np.isfinite(cell.xyz).all()
        and np.isfinite(cell.normal).all()
        and np.isfinite(cell.radius)
        and cell.radius > 0
        and np.linalg.norm(cell.normal) > 1e-8
    ]
    if len(valid_indices) > max_disks:
        sample = np.rint(
            np.linspace(0, len(valid_indices) - 1, max_disks)
        ).astype(np.int64)
        valid_indices = [valid_indices[int(item)] for item in sample]

    positive_radii = np.asarray(
        [index.cells[item].radius for item in valid_indices], dtype=np.float32
    )
    radius_cap = (
        float(np.quantile(positive_radii, 0.95))
        if positive_radii.size
        else index.voxel_size
    )
    theta = np.linspace(0.0, 2.0 * np.pi, vertices_per_disk, endpoint=False)
    polygons = []
    dominant_chunks = []
    centers = []
    for item in valid_indices:
        cell = index.cells[item]
        normal = np.asarray(cell.normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        seed_axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(normal, seed_axis))) > 0.9:
            seed_axis = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(normal, seed_axis)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        radius = min(float(cell.radius), radius_cap)
        disk = (
            cell.xyz[None]
            + radius * np.cos(theta)[:, None] * tangent[None]
            + radius * np.sin(theta)[:, None] * bitangent[None]
        )
        polygons.append(disk[:, [0, 2, 1]])
        centers.append(cell.xyz[[0, 2, 1]])
        dominant_chunks.append(max(cell.chunk_weights, key=cell.chunk_weights.get))

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    if polygons:
        dominant = np.asarray(dominant_chunks, dtype=np.float32)
        normalize = plt.Normalize(
            vmin=float(dominant.min()),
            vmax=max(float(dominant.max()), float(dominant.min()) + 1.0),
        )
        collection = Poly3DCollection(
            polygons,
            facecolors=plt.get_cmap("turbo")(normalize(dominant)),
            edgecolors="none",
            alpha=0.58,
        )
        axis.add_collection3d(collection)
        centers_array = np.asarray(centers)
        lower = np.quantile(centers_array, 0.01, axis=0)
        upper = np.quantile(centers_array, 0.99, axis=0)
        margin = np.maximum((upper - lower) * 0.04, index.voxel_size)
        axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
        axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
        axis.set_zlim(lower[2] - margin[2], upper[2] + margin[2])
        axis.set_box_aspect(np.maximum(upper - lower, index.voxel_size))
        scalar = plt.cm.ScalarMappable(norm=normalize, cmap="turbo")
        scalar.set_array(dominant)
        figure.colorbar(scalar, ax=axis, shrink=0.7, label="dominant chunk")
    axis.set(
        title=f"Oriented surfel disks (sampled {len(polygons)}/{len(index.cells)})",
        xlabel="x",
        ylabel="z",
        zlabel="y",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _relative_voxel_size(
    sequence_path: Path,
    grid_hw: tuple[int, int],
    confidence_threshold: float,
    fraction: float,
) -> tuple[float, float]:
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    samples = []
    for item in sequence["frames"]:
        payload = np.load(sequence_path.parent / item["data_path"])
        points = _sample_grid(payload["pts3d"], grid_hw)
        confidence = _sample_grid(payload["confidence"], grid_hw)
        valid = (
            np.isfinite(points).all(-1)
            & np.isfinite(confidence)
            & (confidence >= confidence_threshold)
        )
        if np.any(valid):
            samples.append(points[valid])
    if not samples:
        raise ValueError("No finite high-confidence CUT3R points")
    points = np.concatenate(samples)
    lower = np.quantile(points, 0.05, axis=0)
    upper = np.quantile(points, 0.95, axis=0)
    scene_scale = float(np.linalg.norm(upper - lower))
    return max(scene_scale * fraction, 1e-6), scene_scale


def build_from_sequence(
    *,
    sequence_path: str | Path,
    output_dir: str | Path,
    confidence_threshold: float = 1.5,
    voxel_size_mode: str = "relative_scene",
    voxel_size: float | None = None,
    relative_scene_fraction: float = 0.005,
    grid_hw: tuple[int, int] = (30, 52),
    radius_scale: float = 0.5,
    merge_normal_cosine: float = 0.6,
    reference_mask_root: str | Path | None = None,
) -> SurfelIndex:
    started = time.perf_counter()
    sequence_path = Path(sequence_path).resolve()
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if sequence.get("cut3r_predicted_pose_used_for_map", True):
        raise ValueError(
            "Core-repair surfel build requires known-pose CUT3R sequence"
        )
    if voxel_size_mode == "relative_scene":
        resolved_voxel_size, scene_scale = _relative_voxel_size(
            sequence_path,
            grid_hw,
            confidence_threshold,
            relative_scene_fraction,
        )
    elif voxel_size_mode == "explicit" and voxel_size is not None:
        resolved_voxel_size, scene_scale = float(voxel_size), None
    else:
        raise ValueError(
            "explicit mode needs voxel_size; otherwise use relative_scene"
        )
    index = SurfelIndex(resolved_voxel_size)
    insertions = []
    raw_points = 0
    reference_mask_root = (
        None
        if reference_mask_root is None
        else Path(reference_mask_root).resolve()
    )
    tagged_observations = 0
    for item in sequence["frames"]:
        payload = np.load(sequence_path.parent / item["data_path"])
        chunk_id = int(item["chunk_id"])
        reference_validity = None
        if reference_mask_root is not None:
            mask_path = (
                reference_mask_root
                / f"chunk_{chunk_id:04d}_reference_valid.png"
            )
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing reference-valid write mask: {mask_path}"
                )
            reference_validity = (
                np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
                / 255.0
            )
        raw_points += int(payload["confidence"].size)
        insertion = index.insert_frame(
            payload["pts3d"],
            payload["confidence"],
            payload["c2w"],
            chunk_id,
            reference_validity=reference_validity,
            intrinsics=payload["intrinsics"],
            confidence_threshold=confidence_threshold,
            grid_hw=grid_hw,
            radius_scale=radius_scale,
            normal_cosine=merge_normal_cosine,
        )
        insertions.append(insertion)
        if reference_validity is not None:
            tagged_observations += int(insertion["accepted"])
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "surfel_index.npz"
    index.save(index_path)
    index.write_ply(output_dir / "surfel_index.ply")
    stats = {
        **index.stats(),
        "index_version": INDEX_VERSION,
        "representation": "radius_normal_surfel_with_voxel_acceleration",
        "voxel_only_merge": False,
        "voxel_size_mode": voxel_size_mode,
        "relative_scene_fraction": relative_scene_fraction,
        "robust_scene_scale": scene_scale,
        "grid_hw": list(grid_hw),
        "radius_scale": radius_scale,
        "merge_normal_cosine": merge_normal_cosine,
        "raw_points": raw_points,
        "accepted_grid_points": int(
            sum(item["accepted"] for item in insertions)
        ),
        "merged_observations": int(
            sum(item["merged"] for item in insertions)
        ),
        "cross_voxel_merges": int(
            sum(item["cross_voxel_merges"] for item in insertions)
        ),
        "index_bytes": index_path.stat().st_size,
        "build_ms": (time.perf_counter() - started) * 1000.0,
        "insertions": insertions,
        "reference_blind_at_write": {
            "enabled": reference_mask_root is not None,
            "mask_root": (
                None
                if reference_mask_root is None
                else str(reference_mask_root)
            ),
            "tagged_observations": tagged_observations,
            "semantics": "1 - upstream reference_valid at observation pixel",
        },
        "observation_view_metadata": {
            "source_pixels": True,
            "image_center_margins": True,
            "view_directions": True,
            "chunk_confidence_weights": True,
            "purpose": "stable view-adaptive first-episode source scoring",
        },
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    coverage = {
        str(chunk): {
            "cells": sum(
                chunk in cell.observing_chunks for cell in index.cells
            ),
            "weight": float(
                sum(
                    cell.chunk_weights.get(chunk, 0.0)
                    for cell in index.cells
                )
            ),
        }
        for chunk in sorted(
            {
                chunk
                for cell in index.cells
                for chunk in cell.observing_chunks
            }
        )
    }
    (output_dir / "chunk_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.asarray([cell.xyz for cell in index.cells])
    dominant = np.asarray(
        [
            max(cell.chunk_weights, key=cell.chunk_weights.get)
            for cell in index.cells
        ]
    )
    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    if len(positions):
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 2],
            positions[:, 1],
            c=dominant,
            s=1.0,
            cmap="turbo",
            alpha=0.65,
        )
        fig.colorbar(scatter, ax=axis, label="dominant chunk")
    axis.set(
        title="Known-pose radius/normal surfel address",
        xlabel="x",
        ylabel="z",
        zlabel="y",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "surfel_preview.png", dpi=160)
    fig.savefig(output_dir / "surfel_center_preview.png", dpi=160)
    plt.close(fig)
    write_oriented_disk_preview(
        index, output_dir / "surfel_disk_preview.png"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a radius/normal surfel chunk address"
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument(
        "--voxel_size_mode",
        choices=("relative_scene", "explicit"),
        default="relative_scene",
    )
    parser.add_argument("--voxel_size", type=float)
    parser.add_argument(
        "--relative_scene_fraction", type=float, default=0.005
    )
    parser.add_argument("--grid_height", type=int, default=30)
    parser.add_argument("--grid_width", type=int, default=52)
    parser.add_argument("--radius_scale", type=float, default=0.5)
    parser.add_argument("--merge_normal_cosine", type=float, default=0.6)
    parser.add_argument(
        "--reference_mask_root",
        help=(
            "Optional baseline masks directory. When set, every observation "
            "stores reference_blind_at_write for its chunk."
        ),
    )
    args = parser.parse_args()
    build_from_sequence(
        sequence_path=args.sequence,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        voxel_size_mode=args.voxel_size_mode,
        voxel_size=args.voxel_size,
        relative_scene_fraction=args.relative_scene_fraction,
        grid_hw=(args.grid_height, args.grid_width),
        radius_scale=args.radius_scale,
        merge_normal_cosine=args.merge_normal_cosine,
        reference_mask_root=args.reference_mask_root,
    )


if __name__ == "__main__":
    main()
