#!/usr/bin/env python3
"""Replay an InSpatio prediction through the causal SLAM3R v6 baseline.

Every latent block follows the future online contract: render the map produced
by earlier blocks at the unchanged planned cameras, fuse with strict reference
priority, and only then register this block's generated latent keyframes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.historical_point_memory import (  # noqa: E402
    IncrementalVoxelSurfelMemory,
    VideoStreamWriter,
    fuse_reference_and_history,
    latent_block_to_pixel_span,
    latent_keyframe_indices,
    scale_adaptive_voxel_size,
)
from utils.overlap_da3_registration import (  # noqa: E402
    apply_similarity,
    estimate_similarity_registration,
    select_v4_runtime_points,
)
from utils.slam3r_incremental import (  # noqa: E402
    IncrementalSLAM3RState,
    ReferenceGeometry,
    Slam3RFrameOutput,
    prepare_reference_geometry,
    prepare_slam3r_frame,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peaks(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    gib = float(1024**3)
    return (
        float(torch.cuda.max_memory_allocated(device) / gib),
        float(torch.cuda.max_memory_reserved(device) / gib),
    )


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


class SequentialVideoReader:
    """Checked sequential OpenCV reader returning RGB uint8 tensors."""

    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        self.frame_count = int(round(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.width = int(round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.height = int(round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.position = 0
        if self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"Video has invalid metadata: {self.path}")

    def read(self, count: int) -> torch.Tensor:
        frames = []
        for _ in range(count):
            success, frame = self.capture.read()
            if not success:
                raise RuntimeError(
                    f"Unexpected end of {self.path} at frame {self.position}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
            self.position += 1
        return torch.stack(frames, dim=0)

    def close(self) -> None:
        self.capture.release()


class FrozenSimilarity:
    """Fit one checked SLAM3R-world to canonical-world Sim(3), then freeze it."""

    def __init__(self, args, device: torch.device):
        self.device = device
        self.confidence_threshold = float(args.sim3_confidence_threshold)
        self.max_normalized_rmse = float(args.sim3_max_normalized_rmse)
        self.max_validation_p90 = float(args.sim3_max_validation_p90)
        self.max_candidates = int(args.sim3_max_candidates)
        self.max_correspondences = int(args.sim3_max_correspondences)
        self.pending: list[tuple[Slam3RFrameOutput, ReferenceGeometry]] = []
        self.attempts: list[dict] = []
        self.scale: Optional[float] = None
        self.rotation: Optional[torch.Tensor] = None
        self.translation: Optional[torch.Tensor] = None

    @property
    def frozen(self) -> bool:
        return self.scale is not None

    @property
    def exhausted(self) -> bool:
        return not self.frozen and len(self.pending) >= self.max_candidates

    def append(
        self,
        output: Slam3RFrameOutput,
        reference: ReferenceGeometry,
    ) -> None:
        if self.frozen:
            raise RuntimeError("Cannot append alignment candidates after Sim(3) freezes")
        self.pending.append((output, reference))

    def maybe_fit(self, minimum_candidates: int) -> bool:
        if self.frozen or len(self.pending) < minimum_candidates:
            return self.frozen

        source = torch.stack([item[0].points_world for item in self.pending])
        target = torch.stack([item[1].points for item in self.pending])
        confidence = torch.stack([item[0].confidence for item in self.pending])
        reference_valid = torch.stack(
            [item[1].valid & item[1].mask for item in self.pending]
        )
        valid = (
            (confidence > self.confidence_threshold)
            & reference_valid
            & torch.isfinite(source).all(dim=-1)
            & (torch.linalg.norm(source, dim=-1) > 1e-8)
        )
        yy, xx = torch.meshgrid(
            torch.arange(source.shape[1], device=self.device),
            torch.arange(source.shape[2], device=self.device),
            indexing="ij",
        )
        fit_checkerboard = ((xx + yy) % 2 == 0).unsqueeze(0)
        fit_valid = valid & fit_checkerboard
        validation_valid = valid & ~fit_checkerboard
        min_correspondences = min(
            4096,
            max(
                128,
                int(source.shape[0] * source.shape[1] * source.shape[2] * 0.005),
            ),
        )
        attempt = {
            "candidate_count": len(self.pending),
            "frame_indices": [item[0].frame_index for item in self.pending],
            "valid_correspondences": int(valid.sum().item()),
            "fit_correspondences": int(fit_valid.sum().item()),
            "validation_correspondences": int(validation_valid.sum().item()),
            "accepted": False,
        }
        try:
            registration = estimate_similarity_registration(
                source,
                target,
                fit_valid,
                min_correspondences=min_correspondences,
                max_correspondences=self.max_correspondences,
            )
            aligned = apply_similarity(
                source,
                registration.scale,
                registration.rotation,
                registration.translation,
            )
            validation_residual = torch.linalg.norm(
                aligned[validation_valid] - target[validation_valid], dim=-1
            )
            if validation_residual.numel() < 128:
                raise ValueError("Too few held-out Sim(3) validation correspondences")
            validation_target = target[validation_valid]
            target_radius = torch.median(
                torch.linalg.norm(
                    validation_target - validation_target.mean(dim=0), dim=-1
                )
            ).clamp_min(1e-6)
            normalized_validation = validation_residual / target_radius
            validation_p50 = float(torch.quantile(normalized_validation, 0.50).item())
            validation_p90 = float(torch.quantile(normalized_validation, 0.90).item())
            accepted = bool(
                registration.normalized_rmse <= self.max_normalized_rmse
                and validation_p90 <= self.max_validation_p90
            )
            attempt.update({
                "scale": registration.scale,
                "rotation": registration.rotation.detach().cpu().tolist(),
                "translation": registration.translation.detach().cpu().tolist(),
                "correspondence_count": registration.correspondence_count,
                "sampled_count": registration.sampled_count,
                "inlier_count": registration.inlier_count,
                "inlier_ratio": registration.inlier_ratio,
                "rmse": registration.rmse,
                "normalized_rmse": registration.normalized_rmse,
                "validation_p50": validation_p50,
                "validation_p90": validation_p90,
                "accepted": accepted,
            })
            if accepted:
                self.scale = float(registration.scale)
                self.rotation = registration.rotation.detach()
                self.translation = registration.translation.detach()
        except ValueError as error:
            attempt["error"] = str(error)
        self.attempts.append(attempt)
        return self.frozen

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        if not self.frozen:
            raise RuntimeError("Sim(3) is not frozen")
        assert self.rotation is not None and self.translation is not None
        return apply_similarity(points, self.scale, self.rotation, self.translation)

    def save(self, path: Path) -> None:
        if not self.frozen:
            raise RuntimeError("Cannot save an unfitted Sim(3)")
        assert self.rotation is not None and self.translation is not None
        np.savez(
            path,
            scale=np.float32(self.scale),
            rotation=self.rotation.detach().cpu().numpy().astype(np.float32),
            translation=self.translation.detach().cpu().numpy().astype(np.float32),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-video", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slam3r-root", required=True)
    parser.add_argument("--i2p-model", default="siyan824/slam3r_i2p")
    parser.add_argument("--l2w-model", default="siyan824/slam3r_l2w")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents-per-block", type=int, default=3)
    parser.add_argument("--max-keyframes", type=int, default=0)
    parser.add_argument("--initial-winsize", type=int, default=5)
    parser.add_argument("--win-r", type=int, default=3)
    parser.add_argument("--num-scene-frame", type=int, default=10)
    parser.add_argument("--buffer-size", type=int, default=30)
    parser.add_argument("--conf-thres-i2p", type=float, default=1.5)
    parser.add_argument("--conf-thres-l2w", type=float, default=12.0)
    parser.add_argument("--frame-mean-conf-thres", type=float, default=10.0)
    parser.add_argument("--sim3-confidence-threshold", type=float, default=1.5)
    parser.add_argument("--sim3-max-candidates", type=int, default=11)
    parser.add_argument("--sim3-max-correspondences", type=int, default=60000)
    parser.add_argument("--sim3-max-normalized-rmse", type=float, default=0.08)
    parser.add_argument("--sim3-max-validation-p90", type=float, default=0.20)
    parser.add_argument("--target-pixel-spacing", type=float, default=3.0)
    parser.add_argument("--min-voxel-size", type=float, default=0.003)
    parser.add_argument("--max-voxel-size", type=float, default=0.012)
    parser.add_argument("--max-points", type=int, default=3_000_000)
    parser.add_argument("--point-size", type=int, default=3)
    parser.add_argument("--geometry-voxel-factor", type=float, default=2.0)
    parser.add_argument("--geometry-depth-ratio", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args) -> None:
    if args.latents_per_block <= 0:
        raise ValueError("latents-per-block must be positive")
    if args.initial_winsize < 3:
        raise ValueError("initial-winsize must be at least three")
    if args.sim3_max_candidates < args.initial_winsize:
        raise ValueError("sim3-max-candidates must cover the initial window")
    if args.max_points <= 0 or args.point_size <= 0:
        raise ValueError("max-points and point-size must be positive")


@torch.inference_mode()
def run(args) -> dict:
    _validate_args(args)
    slam3r_root = Path(args.slam3r_root).resolve()
    if not (slam3r_root / "slam3r" / "models.py").is_file():
        raise FileNotFoundError(f"Invalid SLAM3R root: {slam3r_root}")
    sys.path.insert(0, str(slam3r_root))

    reference_dir = Path(args.reference_dir).resolve()
    required_reference = {
        "rgb": reference_dir / "render_offline.mp4",
        "mask": reference_dir / "mask_offline.mp4",
        "depth": reference_dir / "depth_offline.npy",
        "intrinsic": reference_dir / "intrinsic.npy",
        "poses": reference_dir / "target_c2w.npy",
    }
    missing = [str(path) for path in required_reference.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reference artifacts: {missing}")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite to reuse it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", vars(args))

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The SLAM3R v6 baseline requires a CUDA device")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    pred_reader = SequentialVideoReader(args.pred_video)
    reference_reader = SequentialVideoReader(str(required_reference["rgb"]))
    mask_reader = SequentialVideoReader(str(required_reference["mask"]))
    readers = [pred_reader, reference_reader, mask_reader]
    writers: dict[str, VideoStreamWriter] = {}
    active_error = False
    try:
        dimensions = {(reader.height, reader.width) for reader in readers}
        if len(dimensions) != 1:
            raise ValueError(f"Pred/reference/mask video dimensions differ: {dimensions}")
        height, width = dimensions.pop()
        depth = np.load(required_reference["depth"], mmap_mode="r")
        poses = np.load(required_reference["poses"], mmap_mode="r")
        intrinsic_np = np.load(required_reference["intrinsic"])
        if depth.ndim != 3 or depth.shape[1:] != (height, width):
            raise ValueError(f"Unexpected reference depth shape: {depth.shape}")
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"Unexpected target pose shape: {poses.shape}")
        if intrinsic_np.shape != (3, 3):
            raise ValueError(f"Unexpected intrinsic shape: {intrinsic_np.shape}")

        available_frames = min(
            pred_reader.frame_count,
            reference_reader.frame_count,
            mask_reader.frame_count,
            depth.shape[0],
            poses.shape[0],
        )
        if available_frames <= 0:
            raise ValueError("No aligned input frames are available")
        available_keyframes = (available_frames - 1) // 4 + 1
        keyframe_count = available_keyframes
        if args.max_keyframes > 0:
            keyframe_count = min(keyframe_count, args.max_keyframes)
        if keyframe_count < args.initial_winsize:
            raise ValueError(
                f"Need {args.initial_winsize} keyframes, only {keyframe_count} available"
            )
        usable_frames = 4 * (keyframe_count - 1) + 1
        block_count = (
            keyframe_count + args.latents_per_block - 1
        ) // args.latents_per_block
        intrinsic = torch.from_numpy(np.asarray(intrinsic_np).copy()).to(
            device=device, dtype=torch.float32
        )

        sampled_depth = torch.from_numpy(
            np.asarray(depth[:usable_frames:4, ::8, ::8]).copy()
        )
        voxel_size, voxel_details = scale_adaptive_voxel_size(
            sampled_depth,
            intrinsic.detach().cpu(),
            target_pixel_spacing=args.target_pixel_spacing,
            min_voxel_size=args.min_voxel_size,
            max_voxel_size=args.max_voxel_size,
        )
        memory = IncrementalVoxelSurfelMemory(
            height=height,
            width=width,
            device=device,
            K=intrinsic,
            voxel_size=voxel_size,
            max_points=args.max_points,
            point_size=args.point_size,
        )

        _sync(device)
        model_load_start = time.perf_counter()
        slam_state = IncrementalSLAM3RState.from_pretrained(
            args.i2p_model,
            args.l2w_model,
            device=device,
            initial_winsize=args.initial_winsize,
            win_r=args.win_r,
            num_scene_frame=args.num_scene_frame,
            buffer_size=args.buffer_size,
            conf_thres_i2p=args.conf_thres_i2p,
            seed=args.seed,
        )
        _sync(device)
        model_load_seconds = time.perf_counter() - model_load_start

        fps = max(1, int(round(pred_reader.fps if pred_reader.fps > 0 else 16)))
        diagnostic_names = (
            "reference_causal",
            "historical_causal",
            "historical_mask_causal",
            "fused_causal",
            "fused_mask_causal",
        )
        writers = {
            name: VideoStreamWriter(
                str(output_dir / f"{name}.mp4"), width, height, fps
            )
            for name in diagnostic_names
        }
        writers["slam3r_keyframes_224"] = VideoStreamWriter(
            str(output_dir / "slam3r_keyframes_224.mp4"),
            224,
            224,
            max(1, int(round(fps / 4))),
        )

        alignment = FrozenSimilarity(args, device)
        reference_by_frame: dict[int, ReferenceGeometry] = {}
        block_metrics: list[dict] = []
        frame_metrics: list[dict] = []
        rejected_frame_count = 0

        def write_registered(
            output: Slam3RFrameOutput,
            reference: ReferenceGeometry,
        ) -> None:
            nonlocal rejected_frame_count
            points_before = memory.point_count
            metric = {
                "frame_index": output.frame_index,
                "initial_frame": output.initial_frame,
                "confidence_source": (
                    "i2p_initial" if output.initial_frame else "l2w"
                ),
                "i2p_confidence_mean": output.i2p_confidence_mean,
                "l2w_confidence_mean": output.l2w_confidence_mean,
                "retrieved_frame_indices": list(output.retrieved_frame_indices),
                "buffer_frame_indices": list(output.buffer_frame_indices),
                "points_before": points_before,
                "accepted": False,
            }
            if output.l2w_confidence_mean < args.frame_mean_conf_thres:
                rejected_frame_count += 1
                metric.update({
                    "rejection_reason": "low_frame_mean_registration_confidence",
                    "points_after": points_before,
                })
                frame_metrics.append(metric)
                return
            canonical = alignment.transform(output.points_world)
            slam_valid = (
                (output.confidence > args.conf_thres_l2w)
                & torch.isfinite(output.points_world).all(dim=-1)
                & (torch.linalg.norm(output.points_world, dim=-1) > 1e-8)
            )
            point_keep, admission = select_v4_runtime_points(
                canonical,
                reference.points,
                slam_valid,
                reference.depth,
                reference.valid,
                reference.mask,
                voxel_size=memory.voxel_size,
                voxel_factor=args.geometry_voxel_factor,
                relative_depth_ratio=args.geometry_depth_ratio,
            )
            update = memory.update_points(
                canonical,
                output.rgb_crop.permute(1, 2, 0),
                point_keep,
            )
            renamed_admission = {
                key.replace("da3", "slam3r"): value
                for key, value in admission.items()
            }
            metric.update({
                "accepted": True,
                "crop": output.crop.to_dict(),
                **renamed_admission,
                **update,
            })
            frame_metrics.append(metric)

        processed_pixel_frames = 0
        for block_index in range(block_count):
            latent_start = block_index * args.latents_per_block
            latent_count = min(
                args.latents_per_block, keyframe_count - latent_start
            )
            pixel_start, pixel_end = latent_block_to_pixel_span(
                latent_start, latent_count
            )
            if pixel_start != processed_pixel_frames:
                raise AssertionError(
                    f"Non-contiguous pixel replay: {pixel_start} != {processed_pixel_frames}"
                )
            frame_count = pixel_end - pixel_start
            pred_block_u8 = pred_reader.read(frame_count)
            reference_block_u8 = reference_reader.read(frame_count)
            mask_block_u8 = mask_reader.read(frame_count)
            processed_pixel_frames = pixel_end
            pred_block = pred_block_u8.float().div(255.0)
            reference_block = reference_block_u8.float().div(255.0).to(device)
            reference_mask = (
                mask_block_u8.float().mean(dim=1, keepdim=True).to(device) > 127.5
            )
            block_poses = torch.from_numpy(
                np.asarray(poses[pixel_start:pixel_end]).copy()
            ).to(device=device, dtype=torch.float32)

            points_before_read = memory.point_count
            _sync(device)
            render_start = time.perf_counter()
            historical_rgb, historical_mask = memory.render(block_poses, intrinsic)
            _sync(device)
            render_ms = (time.perf_counter() - render_start) * 1000.0
            fused_rgb, fused_mask, historical_only = fuse_reference_and_history(
                reference_block,
                reference_mask,
                historical_rgb,
                historical_mask,
            )
            if not torch.equal(fused_mask, reference_mask | historical_mask):
                raise AssertionError("Strict fusion mask contract failed")
            reference_pixels = reference_mask.expand_as(reference_block)
            if (fused_rgb[reference_pixels] != reference_block[reference_pixels]).any():
                raise AssertionError("History overwrote reference-valid pixels")
            writers["reference_causal"].write(reference_block)
            writers["historical_causal"].write(historical_rgb)
            writers["historical_mask_causal"].write(historical_mask.float())
            writers["fused_causal"].write(fused_rgb)
            writers["fused_mask_causal"].write(fused_mask.float())

            keyframe_start = time.perf_counter()
            produced_outputs = 0
            for frame_index in latent_keyframe_indices(latent_start, latent_count):
                local_index = frame_index - pixel_start
                if local_index < 0 or local_index >= frame_count:
                    raise AssertionError("Latent keyframe lies outside current RGB block")
                prepared = prepare_slam3r_frame(
                    pred_block[local_index], device=device
                )
                writers["slam3r_keyframes_224"].write(
                    prepared.rgb_crop.unsqueeze(0)
                )
                frame_reference = prepare_reference_geometry(
                    torch.from_numpy(np.asarray(depth[frame_index]).copy()),
                    reference_mask[local_index, 0],
                    intrinsic,
                    block_poses[local_index],
                    prepared.crop,
                    device=device,
                )
                reference_by_frame[frame_index] = frame_reference
                outputs = slam_state.push_prepared(prepared, frame_index)
                produced_outputs += len(outputs)
                for output in outputs:
                    output_reference = reference_by_frame.pop(output.frame_index)
                    if alignment.frozen:
                        write_registered(output, output_reference)
                        continue
                    alignment.append(output, output_reference)
                    if alignment.maybe_fit(args.initial_winsize):
                        pending = list(alignment.pending)
                        alignment.pending.clear()
                        for pending_output, pending_reference in pending:
                            write_registered(pending_output, pending_reference)
                    elif alignment.exhausted:
                        _write_json(
                            output_dir / "sim3_attempts.json", alignment.attempts
                        )
                        raise RuntimeError(
                            "SLAM3R-to-canonical Sim(3) failed closed after "
                            f"{len(alignment.pending)} candidates; last attempt: "
                            f"{alignment.attempts[-1]}"
                        )
            _sync(device)
            keyframe_ms = (time.perf_counter() - keyframe_start) * 1000.0
            allocated_peak, reserved_peak = _cuda_peaks(device)
            block_metric = {
                "block_index": block_index,
                "latent_start": latent_start,
                "latent_count": latent_count,
                "pixel_start": pixel_start,
                "pixel_end": pixel_end,
                "points_before_read": points_before_read,
                "points_after_write": memory.point_count,
                "registered_outputs": produced_outputs,
                "sim3_frozen_after_write": alignment.frozen,
                "reference_coverage": float(reference_mask.float().mean().item()),
                "historical_coverage": float(historical_mask.float().mean().item()),
                "history_only_coverage": float(historical_only.float().mean().item()),
                "fused_coverage": float(fused_mask.float().mean().item()),
                "render_ms": render_ms,
                "keyframe_reconstruction_ms": keyframe_ms,
                "cuda_allocated_peak_gb": allocated_peak,
                "cuda_reserved_peak_gb": reserved_peak,
                "read_before_write": True,
                "planned_trajectory_unchanged": True,
            }
            if block_index > 0:
                expected = block_metrics[-1]["points_after_write"]
                if points_before_read != expected:
                    raise AssertionError(
                        f"Map continuity failed: read {points_before_read}, expected {expected}"
                    )
            block_metrics.append(block_metric)
            _write_json(output_dir / "block_metrics.json", block_metrics)
            _write_json(output_dir / "frame_metrics.json", frame_metrics)
            _write_json(output_dir / "sim3_attempts.json", alignment.attempts)
            print(
                f"[v6 offline] block {block_index + 1}/{block_count}: "
                f"frames [{pixel_start},{pixel_end}), map {points_before_read} -> "
                f"{memory.point_count}, sim3={alignment.frozen}",
                flush=True,
            )

        if not alignment.frozen:
            raise RuntimeError(
                "Replay ended before a valid SLAM3R-to-canonical Sim(3) was found"
            )
        if reference_by_frame:
            raise AssertionError(
                f"Unregistered reference frames remain: {sorted(reference_by_frame)}"
            )
        map_npz, map_ply = memory.save(str(output_dir / "canonical_map"))
        alignment.save(output_dir / "slam3r_to_canonical_sim3.npz")
        summary = {
            "baseline": "inspatio_world_slam3r_offline_v6",
            "status": "complete",
            "pred_video": str(Path(args.pred_video).resolve()),
            "reference_dir": str(reference_dir),
            "i2p_model": args.i2p_model,
            "l2w_model": args.l2w_model,
            "usable_pixel_frames": usable_frames,
            "latent_keyframes": keyframe_count,
            "block_count": block_count,
            "model_load_seconds": model_load_seconds,
            "final_map_points": memory.point_count,
            "accepted_frame_count": sum(item["accepted"] for item in frame_metrics),
            "rejected_frame_count": rejected_frame_count,
            "sim3_attempt_count": len(alignment.attempts),
            "sim3": {
                "scale": alignment.scale,
                "accepted_attempt": alignment.attempts[-1],
            },
            "adaptive_voxel": voxel_details,
            "map_npz": map_npz,
            "map_ply": map_ply,
            "trajectory_source": str(required_reference["poses"]),
            "trajectory_modified": False,
            "slam_input": "generated_latent_keyframes_only",
            "internal_keyframe_stride": 1,
            "causal_order": "render_previous_map_then_register_current_block",
            "reference_priority": "strict",
        }
        _write_json(output_dir / "summary.json", summary)
        return summary
    except BaseException:
        active_error = True
        raise
    finally:
        for reader in readers:
            reader.close()
        close_errors = []
        for writer in writers.values():
            try:
                writer.close()
            except BaseException as error:
                close_errors.append(error)
        if close_errors and not active_error:
            raise close_errors[0]


def main() -> None:
    args = _build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
