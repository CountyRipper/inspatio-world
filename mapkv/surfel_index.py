from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SurfelCell:
    voxel_key: tuple[int, int, int]
    xyz: np.ndarray
    confidence: float
    normal: np.ndarray | None
    rgb_preview: np.ndarray | None
    first_seen_chunk: int
    last_seen_chunk: int
    chunk_weights: dict[int, float] = field(default_factory=dict)
    view_dirs: dict[int, np.ndarray] = field(default_factory=dict)
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
    def __init__(self, voxel_size: float, cells: list[SurfelCell] | None = None):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.cells = list(cells or [])
        self._by_key = {cell.voxel_key: cell for cell in self.cells}

    def insert_frame(
        self,
        pts3d: np.ndarray,
        confidence: np.ndarray,
        camera_pose: np.ndarray,
        chunk_id: int,
        rgb: np.ndarray | None = None,
        *,
        confidence_threshold: float = 1.5,
        grid_hw: tuple[int, int] = (20, 32),
    ) -> dict:
        points = _sample_grid(np.asarray(pts3d, dtype=np.float32), grid_hw)
        conf = _sample_grid(np.asarray(confidence, dtype=np.float32), grid_hw)
        colors = None if rgb is None else _sample_grid(np.asarray(rgb), grid_hw)
        normals = _normal_grid(points)
        camera = np.asarray(camera_pose, dtype=np.float32)[:3, 3]
        distance = np.linalg.norm(points - camera[None, None], axis=-1)
        finite_distance = distance[np.isfinite(distance) & (distance > 0)]
        far = float(np.quantile(finite_distance, 0.995)) if finite_distance.size else 0.0
        valid = (
            np.isfinite(points).all(-1)
            & np.isfinite(conf)
            & (conf >= confidence_threshold)
            & (distance > 0)
            & (distance <= far)
            & (np.linalg.norm(normals, axis=-1) > 1e-6)
        )
        if not np.any(valid):
            return {"chunk_id": int(chunk_id), "accepted": 0, "added": 0, "merged": 0}
        points_flat = points[valid]
        conf_flat = conf[valid]
        normals_flat = normals[valid]
        colors_flat = None if colors is None else colors[valid]
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
        upper = float(np.quantile(conf_flat, 0.95))
        denom = max(upper - confidence_threshold, 1e-6)
        weights = np.clip((conf_flat - confidence_threshold) / denom, 0.05, 1.0)
        added = 0
        merged = 0
        for index, (point, normal, conf_value, weight, view_dir) in enumerate(
            zip(points_flat, normals_flat, conf_flat, weights, view_dirs)
        ):
            key = tuple(np.floor(point / self.voxel_size).astype(np.int64).tolist())
            color = None if colors_flat is None else colors_flat[index].astype(np.float32)
            cell = self._by_key.get(key)
            if cell is None:
                cell = SurfelCell(
                    voxel_key=key,
                    xyz=point.copy(),
                    confidence=float(conf_value),
                    normal=normal.copy(),
                    rgb_preview=None if color is None else color.copy(),
                    first_seen_chunk=int(chunk_id),
                    last_seen_chunk=int(chunk_id),
                    chunk_weights={int(chunk_id): float(weight)},
                    view_dirs={int(chunk_id): view_dir.copy()},
                    observation_weight=float(weight),
                )
                self._by_key[key] = cell
                self.cells.append(cell)
                added += 1
                continue
            old_weight = cell.observation_weight
            total = old_weight + float(weight)
            cell.xyz = (cell.xyz * old_weight + point * float(weight)) / total
            cell.confidence = (
                cell.confidence * old_weight + float(conf_value) * float(weight)
            ) / total
            if cell.normal is not None:
                candidate = cell.normal * old_weight + normal * float(weight)
                length = np.linalg.norm(candidate)
                cell.normal = candidate / length if length > 1e-8 else cell.normal
            cell.last_seen_chunk = max(cell.last_seen_chunk, int(chunk_id))
            cell.chunk_weights[int(chunk_id)] = (
                cell.chunk_weights.get(int(chunk_id), 0.0) + float(weight)
            )
            prior_view = cell.view_dirs.get(int(chunk_id))
            if prior_view is None:
                cell.view_dirs[int(chunk_id)] = view_dir.copy()
            else:
                candidate = prior_view + view_dir
                length = np.linalg.norm(candidate)
                cell.view_dirs[int(chunk_id)] = (
                    candidate / length if length > 1e-8 else prior_view
                )
            cell.observation_weight = total
            merged += 1
        return {
            "chunk_id": int(chunk_id),
            "accepted": int(len(points_flat)),
            "added": added,
            "merged": merged,
        }

    def visible_cells(
        self,
        query_pose: np.ndarray,
        intrinsics: np.ndarray,
        image_size: tuple[int, int],
        *,
        source_image_size: tuple[int, int] | None = None,
        use_occlusion: bool = True,
        front_facing: bool = False,
    ) -> dict[str, np.ndarray]:
        if not self.cells:
            return {
                "indices": np.empty(0, dtype=np.int32),
                "pixels": np.empty((0, 2), dtype=np.int32),
                "depth": np.empty(0, dtype=np.float32),
                "normal_cosine": np.empty(0, dtype=np.float32),
            }
        pose = np.asarray(query_pose, dtype=np.float64)
        intrinsic = np.asarray(intrinsics, dtype=np.float64)
        if source_image_size is not None:
            intrinsic = _scale_intrinsics(intrinsic, source_image_size, image_size)
        positions = np.asarray([cell.xyz for cell in self.cells], dtype=np.float64)
        homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        camera_points = (homogeneous @ np.linalg.inv(pose).T)[:, :3]
        z = camera_points[:, 2]
        valid = np.isfinite(camera_points).all(-1) & (z > 1e-5)
        u = intrinsic[0, 0] * camera_points[:, 0] / np.maximum(z, 1e-8) + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera_points[:, 1] / np.maximum(z, 1e-8) + intrinsic[1, 2]
        px = np.rint(u).astype(np.int64)
        py = np.rint(v).astype(np.int64)
        valid &= (px >= 0) & (px < image_size[1]) & (py >= 0) & (py < image_size[0])
        camera = pose[:3, 3]
        to_camera = camera[None] - positions
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=-1, keepdims=True), 1e-8)
        normals = np.asarray(
            [
                np.zeros(3, dtype=np.float64) if cell.normal is None else cell.normal
                for cell in self.cells
            ]
        )
        cosine = np.sum(normals * to_camera, axis=-1)
        if front_facing:
            valid &= cosine > 0
        indices = np.flatnonzero(valid)
        if use_occlusion and len(indices):
            order = indices[np.argsort(z[indices])]
            seen = set()
            kept = []
            for index in order:
                pixel = (int(py[index]), int(px[index]))
                if pixel in seen:
                    continue
                seen.add(pixel)
                kept.append(int(index))
            indices = np.asarray(kept, dtype=np.int64)
        return {
            "indices": indices.astype(np.int32),
            "pixels": np.stack([py[indices], px[indices]], axis=-1).astype(np.int32),
            "depth": z[indices].astype(np.float32),
            "normal_cosine": np.maximum(cosine[indices], 0).astype(np.float32),
        }

    def stats(self) -> dict:
        observation_counts = [len(cell.chunk_weights) for cell in self.cells]
        return {
            "num_cells": len(self.cells),
            "voxel_size": self.voxel_size,
            "mean_observing_chunks_per_cell": (
                float(np.mean(observation_counts)) if observation_counts else 0.0
            ),
            "first_seen_chunk": (
                min(cell.first_seen_chunk for cell in self.cells) if self.cells else None
            ),
            "last_seen_chunk": (
                max(cell.last_seen_chunk for cell in self.cells) if self.cells else None
            ),
        }

    def save(self, path: str | Path) -> None:
        chunk_ids = []
        chunk_weights = []
        view_dirs = []
        offsets = [0]
        for cell in self.cells:
            for chunk_id in sorted(cell.chunk_weights):
                chunk_ids.append(chunk_id)
                chunk_weights.append(cell.chunk_weights[chunk_id])
                view_dirs.append(cell.view_dirs.get(chunk_id, np.zeros(3)))
            offsets.append(len(chunk_ids))
        np.savez_compressed(
            path,
            version=np.asarray([1], dtype=np.int32),
            voxel_size=np.asarray([self.voxel_size], dtype=np.float32),
            voxel_keys=np.asarray([cell.voxel_key for cell in self.cells], dtype=np.int64),
            xyz=np.asarray([cell.xyz for cell in self.cells], dtype=np.float32),
            confidence=np.asarray([cell.confidence for cell in self.cells], dtype=np.float32),
            normals=np.asarray(
                [
                    np.zeros(3) if cell.normal is None else cell.normal
                    for cell in self.cells
                ],
                dtype=np.float32,
            ),
            first_seen=np.asarray([cell.first_seen_chunk for cell in self.cells], dtype=np.int32),
            last_seen=np.asarray([cell.last_seen_chunk for cell in self.cells], dtype=np.int32),
            observation_weight=np.asarray(
                [cell.observation_weight for cell in self.cells], dtype=np.float32
            ),
            chunk_ids=np.asarray(chunk_ids, dtype=np.int32),
            chunk_weights=np.asarray(chunk_weights, dtype=np.float32),
            view_dirs=np.asarray(view_dirs, dtype=np.float32),
            offsets=np.asarray(offsets, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SurfelIndex":
        payload = np.load(path)
        offsets = payload["offsets"]
        cells = []
        for index in range(len(payload["xyz"])):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            chunks = payload["chunk_ids"][start:stop]
            weights = payload["chunk_weights"][start:stop]
            directions = payload["view_dirs"][start:stop]
            cells.append(
                SurfelCell(
                    voxel_key=tuple(int(x) for x in payload["voxel_keys"][index]),
                    xyz=payload["xyz"][index].copy(),
                    confidence=float(payload["confidence"][index]),
                    normal=payload["normals"][index].copy(),
                    rgb_preview=None,
                    first_seen_chunk=int(payload["first_seen"][index]),
                    last_seen_chunk=int(payload["last_seen"][index]),
                    chunk_weights={
                        int(chunk): float(weight) for chunk, weight in zip(chunks, weights)
                    },
                    view_dirs={
                        int(chunk): direction.copy()
                        for chunk, direction in zip(chunks, directions)
                    },
                    observation_weight=float(payload["observation_weight"][index]),
                )
            )
        return cls(float(payload["voxel_size"][0]), cells)


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
        valid = np.isfinite(points).all(-1) & np.isfinite(confidence) & (
            confidence >= confidence_threshold
        )
        samples.append(points[valid])
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
    grid_hw: tuple[int, int] = (20, 32),
) -> SurfelIndex:
    started = time.perf_counter()
    sequence_path = Path(sequence_path).resolve()
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if voxel_size_mode == "relative_scene":
        resolved_voxel_size, scene_scale = _relative_voxel_size(
            sequence_path, grid_hw, confidence_threshold, relative_scene_fraction
        )
    elif voxel_size_mode == "explicit" and voxel_size is not None:
        resolved_voxel_size, scene_scale = float(voxel_size), None
    else:
        raise ValueError("explicit mode needs voxel_size; otherwise use relative_scene")
    index = SurfelIndex(resolved_voxel_size)
    insertions = []
    raw_points = 0
    for item in sequence["frames"]:
        payload = np.load(sequence_path.parent / item["data_path"])
        raw_points += int(payload["confidence"].size)
        insertions.append(
            index.insert_frame(
                payload["pts3d"],
                payload["confidence"],
                payload["c2w"],
                int(item["chunk_id"]),
                confidence_threshold=confidence_threshold,
                grid_hw=grid_hw,
            )
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "surfel_index.npz"
    index.save(index_path)
    stats = {
        **index.stats(),
        "voxel_size_mode": voxel_size_mode,
        "relative_scene_fraction": relative_scene_fraction,
        "robust_scene_scale": scene_scale,
        "grid_hw": list(grid_hw),
        "raw_points": raw_points,
        "accepted_grid_points": int(sum(item["accepted"] for item in insertions)),
        "index_bytes": index_path.stat().st_size,
        "build_ms": (time.perf_counter() - started) * 1000.0,
        "insertions": insertions,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    coverage = {
        str(chunk): {
            "cells": sum(chunk in cell.chunk_weights for cell in index.cells),
            "weight": float(sum(cell.chunk_weights.get(chunk, 0.0) for cell in index.cells)),
        }
        for chunk in sorted(
            {chunk for cell in index.cells for chunk in cell.chunk_weights}
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
        [max(cell.chunk_weights, key=cell.chunk_weights.get) for cell in index.cells]
    )
    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    if len(positions):
        scatter = axis.scatter(
            positions[:, 0], positions[:, 2], positions[:, 1],
            c=dominant, s=1.0, cmap="turbo", alpha=0.65
        )
        fig.colorbar(scatter, ax=axis, label="dominant chunk")
    axis.set(title="Voxel-surfel address", xlabel="x", ylabel="z", zlabel="y")
    fig.tight_layout()
    fig.savefig(output_dir / "surfel_preview.png", dpi=160)
    plt.close(fig)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a coarse voxel-surfel chunk address")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument(
        "--voxel_size_mode", choices=("relative_scene", "explicit"), default="relative_scene"
    )
    parser.add_argument("--voxel_size", type=float)
    parser.add_argument("--relative_scene_fraction", type=float, default=0.005)
    parser.add_argument("--grid_height", type=int, default=20)
    parser.add_argument("--grid_width", type=int, default=32)
    args = parser.parse_args()
    build_from_sequence(
        sequence_path=args.sequence,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        voxel_size_mode=args.voxel_size_mode,
        voxel_size=args.voxel_size,
        relative_scene_fraction=args.relative_scene_fraction,
        grid_hw=(args.grid_height, args.grid_width),
    )


if __name__ == "__main__":
    main()
