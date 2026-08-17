from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Cut3RFrame:
    frame_id: int
    chunk_id: int
    image_path: str
    data_path: str
    shape: tuple[int, int]
    camera_pose: np.ndarray
    intrinsics: np.ndarray


@dataclass(frozen=True)
class Cut3RSequence:
    frames: tuple[Cut3RFrame, ...]
    coordinate_frame: str
    pose_convention: str
    prefix_last_chunk: int
    target_chunk: int
    query_source_chunk: int


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def _prepare_views(image_paths: list[Path], image_size: int) -> list[dict]:
    from dust3r.utils.image import load_images

    images = load_images([str(path) for path in image_paths], size=image_size)
    views = []
    for index, item in enumerate(images):
        image = item["img"]
        views.append(
            {
                "img": image,
                "ray_map": torch.full(
                    (image.shape[0], 6, image.shape[-2], image.shape[-1]),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(item["true_shape"]),
                "idx": index,
                "instance": str(index),
                "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                "img_mask": torch.tensor([True]),
                "ray_mask": torch.tensor([False]),
                "update": torch.tensor([True]),
                "reset": torch.tensor([False]),
            }
        )
    return views


def _intrinsics_from_pointmap(points_self: torch.Tensor) -> torch.Tensor:
    from dust3r.post_process import estimate_focal_knowing_depth

    batch, height, width, _ = points_self.shape
    principal = torch.tensor(
        [width / 2.0, height / 2.0],
        dtype=points_self.dtype,
        device=points_self.device,
    ).repeat(batch, 1)
    focal = estimate_focal_knowing_depth(
        points_self, principal, focal_mode="weiszfeld"
    )
    intrinsics = torch.eye(3, dtype=points_self.dtype).repeat(batch, 1, 1)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = principal[:, 0]
    intrinsics[:, 1, 2] = principal[:, 1]
    return intrinsics


def _write_diagnostics(
    *,
    output_dir: Path,
    frames: list[dict],
    confidence_threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_confidence = np.concatenate(
        [np.load(output_dir / item["data_path"])["confidence"].reshape(-1) for item in frames]
    )
    finite_confidence = all_confidence[np.isfinite(all_confidence)]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(finite_confidence, bins=80, color="#4f7cac")
    axis.axvline(confidence_threshold, color="#e45756", linestyle="--")
    axis.set(title="CUT3R confidence", xlabel="confidence", ylabel="points")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_histogram.png", dpi=150)
    plt.close(fig)

    poses = np.asarray([item["camera_pose"] for item in frames])
    centers = poses[:, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([item["chunk_id"] for item in frames], centers)
    axes[0].set(title="Predicted camera center", xlabel="chunk", ylabel="CUT3R units")
    axes[0].legend(["x", "y", "z"])
    axes[1].plot(centers[:, 0], centers[:, 2], marker="o", markersize=2)
    for item, center in zip(frames, centers):
        axes[1].annotate(str(item["chunk_id"]), (center[0], center[2]), fontsize=6)
    axes[1].set(title="Camera trajectory x-z", xlabel="x", ylabel="z")
    axes[1].axis("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "camera_trajectory.png", dpi=150)
    plt.close(fig)

    points = []
    for item in frames:
        payload = np.load(output_dir / item["data_path"])
        pts = payload["pts3d"]
        conf = payload["confidence"]
        valid = np.isfinite(pts).all(-1) & np.isfinite(conf) & (
            conf >= confidence_threshold
        )
        sample = pts[valid]
        if len(sample) > 2500:
            sample = sample[np.linspace(0, len(sample) - 1, 2500).astype(int)]
        points.append(sample)
    points = np.concatenate(points) if points else np.empty((0, 3))
    fig = plt.figure(figsize=(7, 6))
    axis = fig.add_subplot(111, projection="3d")
    if len(points):
        axis.scatter(points[:, 0], points[:, 2], points[:, 1], s=0.15, alpha=0.35)
    axis.set(title="Causal prefix point-cloud preview", xlabel="x", ylabel="z", zlabel="y")
    fig.tight_layout()
    fig.savefig(output_dir / "pointcloud_preview.png", dpi=160)
    plt.close(fig)


class Cut3RAdapter:
    """External-process adapter for official CUT3R feed-forward prefix inference."""

    def __init__(
        self,
        cut3r_root: str | Path,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        image_size: int = 512,
    ):
        self.root = Path(cut3r_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.image_size = int(image_size)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"CUT3R checkpoint not found: {self.checkpoint}")

    def reconstruct_prefix(
        self,
        *,
        baseline_root: str | Path,
        block_mapping: str | Path,
        output_dir: str | Path,
        target_chunk: int,
        query_source_chunk: int,
        confidence_threshold: float = 1.5,
    ) -> Cut3RSequence:
        baseline_root = Path(baseline_root).resolve()
        mapping_payload = json.loads(Path(block_mapping).read_text(encoding="utf-8"))
        mapping = (
            mapping_payload["blocks"]
            if isinstance(mapping_payload, dict)
            else mapping_payload
        )
        history = [
            item for item in mapping
            if 0 <= int(item["chunk_id"]) < int(target_chunk)
        ]
        if not history:
            raise ValueError("No generated history precedes the target chunk")
        if max(int(item["chunk_id"]) for item in history) >= int(target_chunk):
            raise AssertionError("Future leakage in CUT3R prefix")
        if int(query_source_chunk) not in {int(item["chunk_id"]) for item in history}:
            raise ValueError("controlled_same_pose source is absent from the causal prefix")
        image_paths = [(baseline_root / item["png_path"]).resolve() for item in history]
        for path in image_paths:
            if not path.is_file():
                raise FileNotFoundError(path)

        sys.path.insert(0, str(self.root))
        sys.path.insert(0, str(self.root / "src"))
        from dust3r.inference import inference
        from dust3r.model import ARCroco3DStereo
        from dust3r.utils.camera import pose_encoding_to_camera
        from dust3r.utils.geometry import geotrf

        views = _prepare_views(image_paths, self.image_size)
        model = ARCroco3DStereo.from_pretrained(str(self.checkpoint)).to(self.device)
        model.eval()
        if len(views) < 4 or len(views) > 64:
            raise ValueError(f"Final CUT3R checkpoint expects 4-64 views, got {len(views)}")
        started = time.perf_counter()
        with torch.inference_mode():
            outputs, _ = inference(views, model, self.device, verbose=True)
        runtime_seconds = time.perf_counter() - started
        predictions = outputs["pred"]
        if len(predictions) != len(history):
            raise RuntimeError(
                f"CUT3R returned {len(predictions)} predictions for {len(history)} views"
            )

        output_dir = Path(output_dir).resolve()
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_entries = []
        raw_points = 0
        accepted_points = 0
        for frame_id, (item, image_path, prediction) in enumerate(
            zip(history, image_paths, predictions)
        ):
            points_self = prediction["pts3d_in_self_view"].float()
            confidence = prediction["conf_self"].float()
            c2w = pose_encoding_to_camera(prediction["camera_pose"].float())
            points_world = geotrf(c2w, points_self)
            intrinsics = _intrinsics_from_pointmap(points_self)
            points_np = points_world[0].cpu().numpy().astype(np.float32)
            confidence_np = confidence[0].cpu().numpy().astype(np.float32)
            c2w_np = c2w[0].cpu().numpy().astype(np.float32)
            intrinsics_np = intrinsics[0].cpu().numpy().astype(np.float32)
            raw_points += int(confidence_np.size)
            accepted_points += int(
                np.count_nonzero(
                    np.isfinite(points_np).all(-1)
                    & np.isfinite(confidence_np)
                    & (confidence_np >= confidence_threshold)
                )
            )
            relative_data = Path("frames") / f"frame_{frame_id:04d}.npz"
            np.savez_compressed(
                output_dir / relative_data,
                pts3d=points_np,
                confidence=confidence_np,
                c2w=c2w_np,
                intrinsics=intrinsics_np,
            )
            frame_entries.append(
                {
                    "frame_id": frame_id,
                    "chunk_id": int(item["chunk_id"]),
                    "rgb_frame_index": int(item["rgb_center_index"]),
                    "image_path": str(image_path),
                    "data_path": str(relative_data),
                    "shape": list(confidence_np.shape),
                    "camera_pose": c2w_np.tolist(),
                    "intrinsics": intrinsics_np.tolist(),
                }
            )

        query = next(
            item for item in frame_entries
            if item["chunk_id"] == int(query_source_chunk)
        )
        sequence_payload = {
            "version": 1,
            "backend": "official_CUT3R",
            "cut3r_commit": _git(self.root, "rev-parse", "HEAD"),
            "cut3r_dirty": bool(_git(self.root, "status", "--short")),
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "coordinate_frame": "CUT3R_first_view_world",
            "pose_convention": "c2w; camera x-right y-down z-forward",
            "scale_behavior": "arbitrary learned scene scale",
            "confidence_interpretation": "exp confidence; larger is more reliable",
            "prefix_last_chunk": max(item["chunk_id"] for item in frame_entries),
            "target_chunk": int(target_chunk),
            "future_leakage": False,
            "query_pose_mode": "controlled_same_pose",
            "query_source_chunk": int(query_source_chunk),
            "query_pose": query["camera_pose"],
            "query_intrinsics": query["intrinsics"],
            "frames": frame_entries,
        }
        (output_dir / "sequence.json").write_text(
            json.dumps(sequence_payload, indent=2), encoding="utf-8"
        )
        stats = {
            "backend": "official_CUT3R",
            "frames": len(frame_entries),
            "raw_points": raw_points,
            "accepted_points": accepted_points,
            "accepted_point_ratio": accepted_points / max(raw_points, 1),
            "confidence_threshold": confidence_threshold,
            "runtime_seconds": runtime_seconds,
            "prefix_last_chunk": sequence_payload["prefix_last_chunk"],
            "target_chunk": int(target_chunk),
            "future_leakage": False,
            "query_pose_mode": "controlled_same_pose",
        }
        (output_dir / "stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        (output_dir / "coordinate_convention.md").write_text(
            "# CUT3R coordinate convention\n\n"
            "- Provider: official CUT3R feed-forward inference.\n"
            "- Stored points: world-space pointmaps obtained by applying predicted c2w "
            "to each self-view pointmap.\n"
            "- Camera pose: c2w; camera coordinates use x-right, y-down, z-forward.\n"
            "- World frame: CUT3R persistent state anchored by the first generated view.\n"
            "- Scale: arbitrary learned scene scale; voxel size is therefore relative.\n"
            "- Query: B2 reuses the predicted B1 pose in controlled_same_pose mode.\n"
            "- Causality: only generated chunks strictly before B2 were provided.\n",
            encoding="utf-8",
        )
        _write_diagnostics(
            output_dir=output_dir,
            frames=frame_entries,
            confidence_threshold=confidence_threshold,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Cut3RSequence(
            frames=tuple(
                Cut3RFrame(
                    frame_id=item["frame_id"],
                    chunk_id=item["chunk_id"],
                    image_path=item["image_path"],
                    data_path=item["data_path"],
                    shape=tuple(item["shape"]),
                    camera_pose=np.asarray(item["camera_pose"], dtype=np.float32),
                    intrinsics=np.asarray(item["intrinsics"], dtype=np.float32),
                )
                for item in frame_entries
            ),
            coordinate_frame=sequence_payload["coordinate_frame"],
            pose_convention=sequence_payload["pose_convention"],
            prefix_last_chunk=sequence_payload["prefix_last_chunk"],
            target_chunk=int(target_chunk),
            query_source_chunk=int(query_source_chunk),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official CUT3R on a causal generated prefix")
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--block_mapping", required=True)
    parser.add_argument("--cut3r_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_chunk", type=int, required=True)
    parser.add_argument("--query_source_chunk", type=int, required=True)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    adapter = Cut3RAdapter(
        args.cut3r_root,
        args.checkpoint,
        device=args.device,
        image_size=args.image_size,
    )
    adapter.reconstruct_prefix(
        baseline_root=args.baseline_root,
        block_mapping=args.block_mapping,
        output_dir=args.output_dir,
        target_chunk=args.target_chunk,
        query_source_chunk=args.query_source_chunk,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
