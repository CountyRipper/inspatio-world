from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapkv_proto.pose_utils import pose_distance, to_cut3r_c2w


@dataclass(frozen=True)
class Cut3RFrame:
    frame_id: int
    chunk_id: int
    image_path: str
    data_path: str
    shape: tuple[int, int]
    camera_pose: np.ndarray
    predicted_camera_pose: np.ndarray
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_intrinsics(path: str | Path) -> np.ndarray:
    path = Path(path).resolve()
    try:
        value = np.asarray(json.loads(path.read_text()), dtype=np.float64)
    except (json.JSONDecodeError, ValueError):
        cleaned = (
            path.read_text()
            .replace("[", " ")
            .replace("]", " ")
            .replace(",", " ")
        )
        value = np.fromstring(cleaned, sep=" ", dtype=np.float64)
        if value.size == 9:
            value = value.reshape(3, 3)
    if value.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics at {path}, got {value.shape}")
    return value


def _intrinsics_after_cut3r_resize_crop(
    intrinsic: np.ndarray,
    image_path: str | Path,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Apply CUT3R's long-edge resize and centered crop to known K."""
    with Image.open(image_path) as image:
        source_w, source_h = image.size
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    scale = max(target_w / source_w, target_h / source_h)
    resized_w = int(round(source_w * scale))
    resized_h = int(round(source_h * scale))
    crop_left = 0.5 * (resized_w - target_w)
    crop_top = 0.5 * (resized_h - target_h)
    result = np.asarray(intrinsic, dtype=np.float64).copy()
    result[0] *= scale
    result[1] *= scale
    result[0, 2] -= crop_left
    result[1, 2] -= crop_top
    return result


def _pairwise_star_output(outputs: dict, prefix_length: int) -> dict:
    """Build the view-0 star graph expected by CUT3R's global aligner."""
    from dust3r.utils.device import collate_with_cat

    if prefix_length < 2:
        raise ValueError("Global alignment needs at least two views")
    pairwise = {"view1": [], "view2": [], "pred1": [], "pred2": []}
    for view_id in range(1, int(prefix_length)):
        pairwise["view1"].append(outputs["views"][0])
        pairwise["view2"].append(outputs["views"][view_id])
        pairwise["pred1"].append(outputs["pred"][0])
        pairwise["pred2"].append(outputs["pred"][view_id])
    return {
        key: collate_with_cat(values)
        for key, values in pairwise.items()
    }


def _depth_to_world(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project aligned depth with exact known K and pose."""
    depth = np.asarray(depth, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    height, width = depth.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    camera = np.stack(
        [
            (xx - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (yy - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ],
        axis=-1,
    ).astype(np.float32)
    homogeneous = np.concatenate(
        [camera.reshape(-1, 3), np.ones((height * width, 1), np.float32)],
        axis=1,
    )
    world = (
        homogeneous @ np.asarray(c2w, dtype=np.float32).T
    )[:, :3].reshape(height, width, 3)
    return camera, world


def _reuse_previous_depths(
    scene,
    previous_depths: list[np.ndarray],
):
    """Initialize and freeze previous rows of the stacked depth parameter.

    This is the adapter-side equivalent of VMem's preset_depth extension.
    The alignment flow follows runjiali-rl/vmem@39291e4; VMem is MIT-licensed,
    while CUT3R's optimizer retains its upstream non-commercial license.
    """
    if not previous_depths:
        return None
    with torch.no_grad():
        for index, depth in enumerate(previous_depths):
            flattened = torch.as_tensor(
                depth,
                device=scene.im_depthmaps.device,
                dtype=scene.im_depthmaps.dtype,
            ).reshape(-1)
            scene.im_depthmaps.data[index, : len(flattened)] = (
                flattened.clamp_min(1e-6).log()
            )
    gradient_mask = torch.ones_like(scene.im_depthmaps)
    gradient_mask[: len(previous_depths)] = 0
    return scene.im_depthmaps.register_hook(
        lambda gradient: gradient * gradient_mask
    )


def _fixed_pose_incremental_alignment(
    *,
    outputs: dict,
    known_poses: np.ndarray,
    known_intrinsics: list[np.ndarray],
    device: str,
    niter_initial: int,
    niter_incremental: int,
    lr: float,
    min_views: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Causally align each prefix with fixed pose/K and frozen prior depths."""
    from cloud_opt.dust3r_opt import global_aligner, GlobalAlignerMode

    num_views = len(outputs["pred"])
    if num_views < min_views:
        raise ValueError(
            f"Fixed global alignment needs at least {min_views} views"
        )
    previous_depths: list[np.ndarray] = []
    aligned: list[dict | None] = [None] * num_views
    audits: list[dict] = []
    for prefix_length in range(min_views, num_views + 1):
        pairwise = _pairwise_star_output(outputs, prefix_length)
        scene = global_aligner(
            pairwise,
            device=device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=False,
            optimize_pp=True,
        )
        scene.compute_global_alignment(
            init="mst", niter=0, schedule="linear", lr=lr
        )
        poses = np.asarray(known_poses[:prefix_length], dtype=np.float32)
        intrinsics = known_intrinsics[:prefix_length]
        focals = [
            float(np.sqrt(value[0, 0] * value[1, 1]))
            for value in intrinsics
        ]
        principal_points = np.asarray(
            [value[:2, 2] for value in intrinsics], dtype=np.float32
        )
        scene.preset_pose(poses)
        scene.preset_focal(focals)
        scene.preset_principal_point(principal_points)
        depth_hook = _reuse_previous_depths(scene, previous_depths)
        iterations = (
            int(niter_initial)
            if prefix_length == min_views
            else int(niter_incremental)
        )
        started = time.perf_counter()
        loss = scene.compute_global_alignment(
            init=None,
            niter=iterations,
            schedule="linear",
            lr=float(lr),
        )
        optimization_seconds = time.perf_counter() - started
        if depth_hook is not None:
            depth_hook.remove()
        scene.clean_pointcloud()
        depths = [
            value.detach().cpu().numpy().astype(np.float32)
            for value in scene.get_depthmaps()
        ]
        confidences = [
            value.detach().cpu().numpy().astype(np.float32)
            for value in scene.get_conf(mode="none")
        ]
        previous_change = (
            max(
                float(np.max(np.abs(depths[index] - previous_depths[index])))
                for index in range(len(previous_depths))
            )
            if previous_depths
            else 0.0
        )
        if previous_change > 1e-6:
            raise RuntimeError(
                "Previous aligned depth changed despite the freeze mask: "
                f"{previous_change}"
            )
        newly_materialized = (
            range(prefix_length)
            if prefix_length == min_views
            else (prefix_length - 1,)
        )
        for index in newly_materialized:
            camera_points, world_points = _depth_to_world(
                depths[index], known_intrinsics[index], known_poses[index]
            )
            aligned[index] = {
                "pts3d": world_points,
                "pts3d_self": camera_points,
                "depth": depths[index],
                "confidence": confidences[index],
                "intrinsics": known_intrinsics[index].astype(np.float32),
            }
        audits.append(
            {
                "prefix_length": int(prefix_length),
                "iterations": iterations,
                "loss": float(loss),
                "optimization_seconds": optimization_seconds,
                "previous_depth_max_abs_change": previous_change,
                "poses_frozen": not bool(scene.im_poses.requires_grad),
                "focals_frozen": not bool(scene.im_focals.requires_grad),
                "principal_points_frozen": not bool(
                    scene.im_pp.requires_grad
                ),
            }
        )
        previous_depths = depths
        del scene, pairwise
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if any(value is None for value in aligned):
        raise RuntimeError("Fixed alignment did not materialize every frame")
    return [value for value in aligned if value is not None], audits


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
    predicted = np.asarray(
        [item["predicted_camera_pose_aligned"] for item in frames]
    )
    centers = poses[:, :3, 3]
    predicted_centers = predicted[:, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    chunks = [item["chunk_id"] for item in frames]
    for axis, label in enumerate(("x", "y", "z")):
        axes[0].plot(chunks, centers[:, axis], label=f"known {label}")
        axes[0].plot(
            chunks,
            predicted_centers[:, axis],
            linestyle="--",
            alpha=0.6,
            label=f"pred {label}",
        )
    axes[0].set(
        title="Known map pose vs aligned CUT3R prediction",
        xlabel="chunk",
        ylabel="CUT3R units",
    )
    axes[0].legend(ncol=2, fontsize=7)
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
    """Official CUT3R depth/pointmap adapter with known-pose map placement.

    CUT3R's predicted camera pose is retained only as a diagnostic. Every
    self-view pointmap is transformed into the map with the controlled c2w
    from block_mapping.json.
    """

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
        query_pose_mode: str = "controlled_same_pose_known",
        query_target_chunk: int | None = None,
        confidence_threshold: float = 1.5,
        alignment_mode: str = "rigid_self_pointmap",
        known_intrinsics_path: str | Path | None = None,
        niter_initial: int = 100,
        niter_incremental: int = 20,
        alignment_lr: float = 0.01,
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
        if query_pose_mode not in {
            "controlled_same_pose_known",
            "known_target_pose",
        }:
            raise ValueError(f"Unsupported query pose mode: {query_pose_mode}")
        if alignment_mode not in {
            "rigid_self_pointmap",
            "fixed_global_incremental",
            "fixed_global_joint",
        }:
            raise ValueError(f"Unsupported alignment mode: {alignment_mode}")
        if (
            alignment_mode in {
                "fixed_global_incremental",
                "fixed_global_joint",
            }
            and known_intrinsics_path is None
        ):
            raise ValueError(
                "fixed_global_incremental requires known_intrinsics_path"
            )
        query_target_chunk = (
            int(target_chunk)
            if query_target_chunk is None
            else int(query_target_chunk)
        )
        query_target_mapping = next(
            (
                item
                for item in mapping
                if int(item["chunk_id"]) == query_target_chunk
            ),
            None,
        )
        if query_pose_mode == "known_target_pose" and query_target_mapping is None:
            raise ValueError(
                f"Known target query chunk {query_target_chunk} is absent from block mapping"
            )
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
        cut3r_gpu = (
            torch.cuda.get_device_name(torch.device(self.device))
            if str(self.device).startswith("cuda") and torch.cuda.is_available()
            else str(self.device)
        )
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
        known_poses = np.asarray(
            [
                to_cut3r_c2w(np.asarray(item["c2w"], dtype=np.float64))
                for item in history
            ],
            dtype=np.float32,
        )
        predicted_poses = np.asarray(
            [
                pose_encoding_to_camera(prediction["camera_pose"].float())[0]
                .cpu()
                .numpy()
                for prediction in predictions
            ],
            dtype=np.float32,
        )
        # CUT3R predictions have their own world gauge. Align the first pose
        # only for a readable drift diagnostic; aligned predictions never
        # place pointmaps or answer retrieval queries.
        predicted_alignment = known_poses[0] @ np.linalg.inv(predicted_poses[0])
        predicted_aligned = np.asarray(
            [predicted_alignment @ pose for pose in predicted_poses],
            dtype=np.float32,
        )
        pose_errors = [
            pose_distance(known, predicted)
            for known, predicted in zip(known_poses, predicted_aligned)
        ]
        alignment_audits: list[dict] = []
        aligned_frames: list[dict] | None = None
        known_intrinsics: list[np.ndarray] | None = None
        if alignment_mode in {
            "fixed_global_incremental",
            "fixed_global_joint",
        }:
            base_intrinsics = _load_intrinsics(known_intrinsics_path)
            known_intrinsics = [
                _intrinsics_after_cut3r_resize_crop(
                    base_intrinsics,
                    image_path,
                    tuple(
                        int(value)
                        for value in prediction["conf_self"].shape[-2:]
                    ),
                )
                for image_path, prediction in zip(image_paths, predictions)
            ]
            aligned_frames, alignment_audits = (
                _fixed_pose_incremental_alignment(
                    outputs=outputs,
                    known_poses=known_poses,
                    known_intrinsics=known_intrinsics,
                    device=self.device,
                    niter_initial=niter_initial,
                    niter_incremental=niter_incremental,
                    lr=alignment_lr,
                    min_views=(
                        len(predictions)
                        if alignment_mode == "fixed_global_joint"
                        else 4
                    ),
                )
            )

        frame_entries = []
        raw_points = 0
        accepted_points = 0
        for frame_id, (item, image_path, prediction) in enumerate(
            zip(history, image_paths, predictions)
        ):
            if aligned_frames is not None:
                aligned = aligned_frames[frame_id]
                points_np = aligned["pts3d"]
                points_self_np = aligned["pts3d_self"]
                depth_np = aligned["depth"]
                confidence_np = aligned["confidence"]
                intrinsics_np = aligned["intrinsics"]
            else:
                points_self = prediction["pts3d_in_self_view"].float()
                confidence = prediction["conf_self"].float()
                known_c2w = torch.as_tensor(
                    known_poses[frame_id],
                    dtype=points_self.dtype,
                    device=points_self.device,
                ).unsqueeze(0)
                points_world = geotrf(known_c2w, points_self)
                intrinsics = _intrinsics_from_pointmap(points_self)
                points_np = (
                    points_world[0].cpu().numpy().astype(np.float32)
                )
                points_self_np = (
                    points_self[0].cpu().numpy().astype(np.float32)
                )
                depth_np = points_self_np[..., 2]
                confidence_np = (
                    confidence[0].cpu().numpy().astype(np.float32)
                )
                intrinsics_np = (
                    intrinsics[0].cpu().numpy().astype(np.float32)
                )
            c2w_np = known_poses[frame_id]
            predicted_c2w_np = predicted_poses[frame_id]
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
                pts3d_self=points_self_np,
                depth=depth_np,
                confidence=confidence_np,
                c2w=c2w_np,
                predicted_c2w=predicted_c2w_np,
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
                    "camera_pose_source": "known_control_c2w",
                    "alignment_mode": alignment_mode,
                    "known_intrinsics_used": bool(
                        known_intrinsics is not None
                    ),
                    "predicted_camera_pose": predicted_c2w_np.tolist(),
                    "predicted_camera_pose_aligned": predicted_aligned[
                        frame_id
                    ].tolist(),
                    "predicted_pose_error": {
                        "combined": float(pose_errors[frame_id][0]),
                        "translation": float(pose_errors[frame_id][1]),
                        "rotation_radians": float(pose_errors[frame_id][2]),
                    },
                    "intrinsics": intrinsics_np.tolist(),
                }
            )

        query_source = next(
            item for item in frame_entries
            if item["chunk_id"] == int(query_source_chunk)
        )
        if query_pose_mode == "known_target_pose":
            query_pose = to_cut3r_c2w(
                np.asarray(query_target_mapping["c2w"], dtype=np.float64)
            ).astype(np.float32)
        else:
            query_pose = np.asarray(
                query_source["camera_pose"], dtype=np.float32
            )
        cut3r_commit = _git(self.root, "rev-parse", "HEAD")
        cut3r_dirty = bool(_git(self.root, "status", "--short"))
        checkpoint_sha256 = _sha256(self.checkpoint)
        backend_name = (
            "official_CUT3R_fixed_global_incremental"
            if alignment_mode == "fixed_global_incremental"
            else (
                "official_CUT3R_fixed_global_joint"
                if alignment_mode == "fixed_global_joint"
                else "official_CUT3R_rigid_self_pointmap"
            )
        )
        sequence_payload = {
            "version": 3,
            "backend": backend_name,
            "cut3r_commit": cut3r_commit,
            "cut3r_dirty": cut3r_dirty,
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
            "gpu": cut3r_gpu,
            "coordinate_frame": "known_control_world",
            "pose_convention": "c2w; camera x-right y-down z-forward",
            "scale_behavior": "arbitrary learned scene scale",
            "confidence_interpretation": "exp confidence; larger is more reliable",
            "map_pose_source": "known_control_c2w",
            "cut3r_predicted_pose_used_for_map": False,
            "reconstruction_mode": alignment_mode,
            "known_intrinsics_path": (
                None
                if known_intrinsics_path is None
                else str(Path(known_intrinsics_path).resolve())
            ),
            "fixed_global_alignment": (
                alignment_mode
                in {"fixed_global_incremental", "fixed_global_joint"}
            ),
            "previous_depths_reused": (
                alignment_mode == "fixed_global_incremental"
            ),
            "alignment_audits": alignment_audits,
            "prefix_last_chunk": max(item["chunk_id"] for item in frame_entries),
            "target_chunk": int(target_chunk),
            "future_leakage": False,
            "query_pose_mode": query_pose_mode,
            "query_source_chunk": int(query_source_chunk),
            "query_target_chunk": query_target_chunk,
            "query_pose": query_pose.tolist(),
            "query_intrinsics": query_source["intrinsics"],
            "frames": frame_entries,
        }
        (output_dir / "sequence.json").write_text(
            json.dumps(sequence_payload, indent=2), encoding="utf-8"
        )
        stats = {
            "backend": backend_name,
            "cut3r_commit": cut3r_commit,
            "cut3r_dirty": cut3r_dirty,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "gpu": cut3r_gpu,
            "frames": len(frame_entries),
            "raw_points": raw_points,
            "accepted_points": accepted_points,
            "accepted_point_ratio": accepted_points / max(raw_points, 1),
            "confidence_threshold": confidence_threshold,
            "runtime_seconds": runtime_seconds,
            "prefix_last_chunk": sequence_payload["prefix_last_chunk"],
            "target_chunk": int(target_chunk),
            "future_leakage": False,
            "query_pose_mode": query_pose_mode,
            "map_pose_source": "known_control_c2w",
            "cut3r_predicted_pose_used_for_map": False,
            "alignment_mode": alignment_mode,
            "known_intrinsics_fixed": bool(known_intrinsics is not None),
            "previous_depths_reused": (
                alignment_mode == "fixed_global_incremental"
            ),
            "alignment_niter_initial": (
                int(niter_initial)
                if alignment_mode
                in {"fixed_global_incremental", "fixed_global_joint"}
                else None
            ),
            "alignment_niter_incremental": (
                int(niter_incremental)
                if alignment_mode == "fixed_global_incremental"
                else None
            ),
            "alignment_runtime_seconds": float(
                sum(
                    item["optimization_seconds"]
                    for item in alignment_audits
                )
            ),
            "maximum_previous_depth_change": float(
                max(
                    (
                        item["previous_depth_max_abs_change"]
                        for item in alignment_audits
                    ),
                    default=0.0,
                )
            ),
            "maximum_predicted_translation_drift": float(
                max(error[1] for error in pose_errors)
            ),
            "maximum_predicted_rotation_drift_degrees": float(
                np.degrees(max(error[2] for error in pose_errors))
            ),
        }
        (output_dir / "stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        stored_points_description = (
            "- Stored points: globally aligned depth back-projected with fixed "
            "known K and fixed controlled c2w; prior prefix depths are reused "
            "and gradient-frozen when a new view is added.\n"
            if alignment_mode == "fixed_global_incremental"
            else "- Stored points: legacy independent CUT3R self-view pointmaps "
            "rigidly transformed by controlled c2w; no global alignment.\n"
        )
        convention = "".join(
            [
                "# CUT3R coordinate convention\n\n",
                "- Provider: official CUT3R feed-forward inference.\n",
                stored_points_description,
                "- Camera pose: c2w; camera coordinates use x-right, y-down, "
                "z-forward.\n",
                "- World frame: the exact InSpatio control trajectory after "
                "the VMem Y/Z camera-basis conversion.\n",
                "- Scale: arbitrary learned scene scale; voxel size is "
                "therefore relative.\n",
                "- CUT3R predicted poses are diagnostics only and never place "
                "geometry.\n",
                f"- Query: {query_pose_mode}; fixed known intrinsics are used "
                "when alignment mode is fixed-global, and target chunk "
                f"{query_target_chunk} provides the known pose when "
                "known_target_pose is selected.\n",
                "- Causality: only generated chunks strictly before B2 were "
                "provided.\n",
            ]
        )
        (output_dir / "coordinate_convention.md").write_text(
            convention, encoding="utf-8"
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
                    predicted_camera_pose=np.asarray(
                        item["predicted_camera_pose"], dtype=np.float32
                    ),
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
    parser.add_argument(
        "--query_pose_mode",
        choices=("controlled_same_pose_known", "known_target_pose"),
        default="controlled_same_pose_known",
    )
    parser.add_argument("--query_target_chunk", type=int)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument(
        "--alignment_mode",
        choices=(
            "rigid_self_pointmap",
            "fixed_global_incremental",
            "fixed_global_joint",
        ),
        default="rigid_self_pointmap",
    )
    parser.add_argument("--known_intrinsics")
    parser.add_argument("--alignment_niter_initial", type=int, default=100)
    parser.add_argument("--alignment_niter_incremental", type=int, default=20)
    parser.add_argument("--alignment_lr", type=float, default=0.01)
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
        query_pose_mode=args.query_pose_mode,
        query_target_chunk=args.query_target_chunk,
        confidence_threshold=args.confidence_threshold,
        alignment_mode=args.alignment_mode,
        known_intrinsics_path=args.known_intrinsics,
        niter_initial=args.alignment_niter_initial,
        niter_incremental=args.alignment_niter_incremental,
        alignment_lr=args.alignment_lr,
    )


if __name__ == "__main__":
    main()
