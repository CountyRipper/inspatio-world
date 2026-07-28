"""Block condition/output controller for the RGB point-memory baseline."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import numpy as np
import torch

from utils.historical_point_memory import (
    DenseGeneratedPointMemory,
    IncrementalVoxelSurfelMemory,
    RGBPointMemory,
    VideoStreamWriter,
    calibrate_depth_scale,
    compute_depth_confidence,
    fuse_reference_and_history,
    latent_block_to_pixel_span,
    latent_keyframe_indices,
)
from utils.overlap_da3_registration import (
    apply_similarity,
    backproject_world_grid,
    estimate_similarity_registration,
    pose_residual,
    transform_da3_c2w,
)
from utils.render_warper import convert_mask_video


_OVERLAP_VOXEL_MODES = {"overlap_voxel_v3", "overlap_voxel_v3_1"}


def _synchronize(device: torch.device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak_gb(device: torch.device) -> tuple[float, float]:
    device = torch.device(device)
    if device.type != "cuda":
        return 0.0, 0.0
    gib = float(1024 ** 3)
    return (
        float(torch.cuda.max_memory_allocated(device) / gib),
        float(torch.cuda.max_memory_reserved(device) / gib),
    )


class WanVAEBlockDecoder:
    """Stateful block decoder backed by a dedicated Wan-VAE instance."""

    def __init__(self, vae_wrapper):
        self.vae = vae_wrapper
        self.device = next(vae_wrapper.model.parameters()).device
        self.vae.model.clear_cache()

    @torch.inference_mode()
    def decode(self, latent_btchw: torch.Tensor) -> torch.Tensor:
        latent = latent_btchw.to(self.device)
        latent_bcthw = latent.permute(0, 2, 1, 3, 4)
        scale = [
            self.vae.mean.to(device=self.device, dtype=latent.dtype),
            (1.0 / self.vae.std).to(device=self.device, dtype=latent.dtype),
        ]
        video = self.vae.model.cached_decode(latent_bcthw, scale)
        video = video.float().clamp_(-1, 1).permute(0, 2, 1, 3, 4)
        return (video * 0.5 + 0.5).clamp_(0, 1)


class TAEBlockDecoder:
    """Stateful block decoder using the repository's StreamingTAEHV helper."""

    def __init__(self, tae_model):
        from utils.taehv import StreamingTAEHV

        self.streaming = StreamingTAEHV(tae_model)
        self.device = next(tae_model.parameters()).device

    @torch.inference_mode()
    def decode(self, latent_btchw: torch.Tensor) -> torch.Tensor:
        latent = latent_btchw.to(self.device, dtype=torch.float16)
        frames = []
        frame = self.streaming.decode(latent)
        while frame is not None:
            frames.append(frame.float())
            frame = self.streaming.decode()
        if not frames:
            raise RuntimeError("Streaming TAE produced no RGB frames for a latent block")
        return torch.cat(frames, dim=1)


class HistoricalMemoryController:
    """Synchronous read-generate-write controller at STAR block boundaries."""

    def __init__(
        self,
        reference_rgb_bcthw: torch.Tensor,
        reference_mask_bcthw: torch.Tensor,
        target_c2w: torch.Tensor,
        K: torch.Tensor,
        encode_video: Callable[[torch.Tensor], torch.Tensor],
        block_decoder,
        depth_estimator,
        memory: RGBPointMemory | DenseGeneratedPointMemory | IncrementalVoxelSurfelMemory,
        output_dir: str,
        output_prefix: str,
        rank: int,
        reference_map_path: str,
        memory_update_mode: str = "keyframe",
        memory_map_mode: str = "bounded_voxel",
        reference_depth_thw: Optional[torch.Tensor] = None,
        memory_anchor_count: int = 1,
        adaptive_voxel_details: Optional[dict] = None,
        fps: int = 24,
        save_diagnostics: bool = True,
    ):
        if reference_rgb_bcthw.shape[0] != 1:
            raise ValueError("Historical memory baseline currently requires batch size 1")
        if target_c2w.ndim == 4:
            if target_c2w.shape[0] != 1:
                raise ValueError("Historical memory baseline currently requires batch size 1")
            target_c2w = target_c2w[0]
        if K.ndim == 3:
            if K.shape[0] != 1:
                raise ValueError("Historical memory baseline currently requires batch size 1")
            K = K[0]

        self.reference_rgb = reference_rgb_bcthw
        self.reference_mask = reference_mask_bcthw
        if reference_depth_thw is not None:
            if reference_depth_thw.ndim == 4:
                if reference_depth_thw.shape[0] != 1:
                    raise ValueError("Historical memory requires batch size 1 reference depth")
                reference_depth_thw = reference_depth_thw[0]
            if reference_depth_thw.ndim != 3:
                raise ValueError(
                    f"Expected reference depth [T,H,W], got {tuple(reference_depth_thw.shape)}"
                )
        self.reference_depth = reference_depth_thw
        self.target_c2w = target_c2w
        self.K = K
        self.encode_video = encode_video
        self.block_decoder = block_decoder
        self.depth_estimator = depth_estimator
        self.depth_backend = getattr(depth_estimator, "backend_name", "da3")
        self.memory = memory
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.rank = rank
        self.reference_map_path = reference_map_path
        if memory_update_mode not in {"keyframe", "latent_keyframe", "full_block"}:
            raise ValueError(f"Unsupported memory update mode: {memory_update_mode}")
        if memory_map_mode not in {
            "bounded_voxel", "dense_two_layer", *_OVERLAP_VOXEL_MODES
        }:
            raise ValueError(f"Unsupported memory map mode: {memory_map_mode}")
        if memory_map_mode == "dense_two_layer" and memory_update_mode not in {
            "latent_keyframe", "full_block"
        }:
            raise ValueError(
                "dense_two_layer requires memory_update_mode=latent_keyframe or full_block"
            )
        if memory_update_mode == "latent_keyframe" and memory_map_mode not in {
            "dense_two_layer", *_OVERLAP_VOXEL_MODES
        }:
            raise ValueError(
                "latent_keyframe requires dense_two_layer or overlap_voxel_v3"
            )
        if memory_map_mode in {"dense_two_layer", *_OVERLAP_VOXEL_MODES} \
                and self.reference_depth is None:
            raise ValueError(f"{memory_map_mode} requires aligned reference_depth_thw")
        if memory_map_mode in _OVERLAP_VOXEL_MODES:
            if memory_update_mode != "latent_keyframe":
                raise ValueError("overlap_voxel_v3 requires latent_keyframe updates")
            if self.depth_backend != "da3":
                raise ValueError("overlap_voxel_v3 currently requires the DA3 backend")
            if memory_anchor_count != 1:
                raise NotImplementedError(
                    "Multi-anchor DA3 windows are reserved but not implemented"
                )
            if memory_map_mode == "overlap_voxel_v3_1" \
                    and adaptive_voxel_details is None:
                raise ValueError("overlap_voxel_v3_1 requires adaptive voxel metadata")
        self.memory_update_mode = memory_update_mode
        self.memory_map_mode = memory_map_mode
        self.save_diagnostics = save_diagnostics
        self.metrics = []
        self.closed = False
        self.previous_depth_scale = None
        self._historical_depth_by_block = {}
        self.memory_anchor_count = int(memory_anchor_count)
        self.adaptive_voxel_details = adaptive_voxel_details
        self.anchor_rgb = None
        self.anchor_world_points = None
        self.anchor_valid = None
        self.anchor_frame_index = None
        self.previous_registration_scale = None
        self.previous_da3_focal = None
        self.observed_keyframes = {}

        _, _, _, height, width = reference_rgb_bcthw.shape
        self.height = height
        self.width = width
        os.makedirs(output_dir, exist_ok=True)

        paths = {
            "pred": f"{output_prefix}-pred_video_rank{rank}.mp4",
        }
        if save_diagnostics:
            paths.update({
                "reference": f"{output_prefix}-reference_render_rank{rank}.mp4",
                "historical": f"{output_prefix}-historical_render_rank{rank}.mp4",
                "fused": f"{output_prefix}-fused_render_rank{rank}.mp4",
                "reference_mask": f"{output_prefix}-reference_mask_rank{rank}.mp4",
                "historical_mask": f"{output_prefix}-historical_mask_rank{rank}.mp4",
                "fused_mask": f"{output_prefix}-fused_mask_rank{rank}.mp4",
            })
        self.writers = {
            name: VideoStreamWriter(os.path.join(output_dir, filename), width, height, fps)
            for name, filename in paths.items()
        }

    def _metric_for_block(self, block_index: int) -> dict:
        while len(self.metrics) <= block_index:
            self.metrics.append({"block_index": len(self.metrics)})
        return self.metrics[block_index]

    @torch.inference_mode()
    def condition_provider(
        self,
        block_index: int,
        latent_start: int,
        latent_count: int,
    ):
        pixel_start, pixel_end = latent_block_to_pixel_span(latent_start, latent_count)
        if pixel_end > self.reference_rgb.shape[2]:
            raise RuntimeError(
                f"Block {block_index} requires RGB frames [{pixel_start}, {pixel_end}), "
                f"but only {self.reference_rgb.shape[2]} are available"
            )

        reference_rgb = (
            self.reference_rgb[0, :, pixel_start:pixel_end]
            .permute(1, 0, 2, 3).float().add(1).mul(0.5).clamp(0, 1)
        )
        reference_mask = (
            self.reference_mask[0, :1, pixel_start:pixel_end]
            .permute(1, 0, 2, 3) > 0
        )
        poses = self.target_c2w[pixel_start:pixel_end]
        points_before_read = self.memory.point_count
        points_after_previous_block = 0
        if self.memory_map_mode == "overlap_voxel_v3_1":
            if block_index > 0:
                previous_metric = self._metric_for_block(block_index - 1)
                if "points_after" not in previous_metric:
                    raise AssertionError(
                        "V3.1 memory read occurred before the previous block write"
                    )
                points_after_previous_block = int(previous_metric["points_after"])
            if points_before_read != points_after_previous_block:
                raise AssertionError(
                    "V3.1 GPU map continuity failed: "
                    f"read={points_before_read}, previous_write={points_after_previous_block}"
                )

        _synchronize(self.memory.device)
        t_render = time.perf_counter()
        if self.memory_map_mode == "dense_two_layer":
            historical_rgb, historical_mask, historical_depth = self.memory.render_with_depth(
                poses, self.K
            )
            self._historical_depth_by_block[block_index] = historical_depth
        else:
            historical_rgb, historical_mask = self.memory.render(poses, self.K)
            historical_depth = None
        _synchronize(self.memory.device)
        hist_render_ms = (time.perf_counter() - t_render) * 1000.0

        t_merge = time.perf_counter()
        fused_rgb, fused_mask, hist_only = fuse_reference_and_history(
            reference_rgb.to(self.memory.device),
            reference_mask.to(self.memory.device),
            historical_rgb,
            historical_mask,
        )
        merge_ms = (time.perf_counter() - t_merge) * 1000.0

        historical_pixels = int(historical_mask.sum().item())
        history_injected_pixels = int(hist_only.sum().item())
        if self.memory_map_mode == "overlap_voxel_v3_1" \
                and block_index == 0 and historical_pixels != 0:
            raise AssertionError("V3.1 block zero must not read historical pixels")
        fused_rgb_l1_from_reference = float(
            (fused_rgb - reference_rgb.to(self.memory.device)).abs().mean().item()
        )

        if not torch.equal((fused_rgb != reference_rgb.to(self.memory.device)).any(dim=1, keepdim=True) & reference_mask.to(self.memory.device),
                           torch.zeros_like(reference_mask.to(self.memory.device))):
            raise AssertionError("Historical render modified a reference-valid pixel")

        if self.save_diagnostics:
            self.writers["reference"].write(reference_rgb)
            self.writers["historical"].write(historical_rgb)
            self.writers["fused"].write(fused_rgb)
            self.writers["reference_mask"].write(reference_mask.float())
            self.writers["historical_mask"].write(historical_mask.float())
            self.writers["fused_mask"].write(fused_mask.float())

        fused_video = (
            fused_rgb.permute(1, 0, 2, 3).unsqueeze(0).mul(2).sub(1)
            .to(dtype=self.reference_rgb.dtype)
        )
        fused_mask_video = fused_mask.float().permute(1, 0, 2, 3).unsqueeze(0).mul(2).sub(1)

        encode_device = fused_video.device
        _synchronize(encode_device)
        t_encode = time.perf_counter()
        render_latent = self.encode_video(fused_video)
        mask_latent = convert_mask_video(fused_mask_video)
        _synchronize(render_latent.device)
        condition_encode_ms = (time.perf_counter() - t_encode) * 1000.0

        if render_latent.shape[1] != latent_count or render_latent.shape[2] != 16:
            raise RuntimeError(f"Unexpected render latent shape: {tuple(render_latent.shape)}")
        if mask_latent.shape[1] != latent_count or mask_latent.shape[2] != 4:
            raise RuntimeError(f"Unexpected mask latent shape: {tuple(mask_latent.shape)}")

        metric = self._metric_for_block(block_index)
        cuda_allocated_peak_gb, cuda_reserved_peak_gb = _cuda_peak_gb(self.memory.device)
        metric.update({
            "latent_start": latent_start,
            "pixel_start": pixel_start,
            "pixel_end": pixel_end,
            "pixel_frames": pixel_end - pixel_start,
            "hist_render_ms": hist_render_ms,
            "merge_ms": merge_ms,
            "condition_encode_ms": condition_encode_ms,
            "reference_coverage": float(reference_mask.float().mean().item()),
            "historical_coverage": float(historical_mask.float().mean().item()),
            "hist_only_coverage": float(hist_only.float().mean().item()),
            "fused_coverage": float(fused_mask.float().mean().item()),
            "historical_pixels": historical_pixels,
            "history_injected_pixels": history_injected_pixels,
            "fused_rgb_l1_from_reference": fused_rgb_l1_from_reference,
            "render_latent_l1": float(render_latent.float().abs().mean().item()),
            "points_before_read": points_before_read,
            "points_after_previous_block": points_after_previous_block,
            "memory_read_point_continuity": (
                points_before_read == points_after_previous_block
                if self.memory_map_mode == "overlap_voxel_v3_1" else None
            ),
            "memory_read_contract": (
                "gpu_voxel_render+offline_reference_fuse+vae_encode"
                if self.memory_map_mode == "overlap_voxel_v3_1" else None
            ),
            "memory_render_uses_planned_c2w": (
                True if self.memory_map_mode == "overlap_voxel_v3_1" else None
            ),
            "memory_ply_roundtrip": (
                False if self.memory_map_mode == "overlap_voxel_v3_1" else None
            ),
            "memory_map_mode": self.memory_map_mode,
            "cuda_allocated_peak_gb": cuda_allocated_peak_gb,
            "cuda_reserved_peak_gb": cuda_reserved_peak_gb,
        })
        return render_latent, mask_latent

    @torch.inference_mode()
    def output_callback(
        self,
        *,
        block_index: int,
        latent_start: int,
        denoised_latent: torch.Tensor,
        dit_ms: float,
    ) -> None:
        pixel_start, pixel_end = latent_block_to_pixel_span(latent_start, denoised_latent.shape[1])

        _synchronize(self.block_decoder.device)
        t_decode = time.perf_counter()
        block_rgb = self.block_decoder.decode(denoised_latent)
        _synchronize(self.block_decoder.device)
        decode_ms = (time.perf_counter() - t_decode) * 1000.0
        expected_frames = pixel_end - pixel_start
        if block_rgb.shape[1] != expected_frames:
            raise RuntimeError(
                f"Block decoder returned {block_rgb.shape[1]} frames, expected {expected_frames}"
            )

        self.writers["pred"].write(block_rgb[0])

        if self.memory_map_mode in _OVERLAP_VOXEL_MODES:
            self._output_callback_overlap_v3(
                block_index=block_index,
                latent_start=latent_start,
                denoised_latent=denoised_latent,
                block_rgb=block_rgb,
                pixel_start=pixel_start,
                pixel_end=pixel_end,
                decode_ms=decode_ms,
                dit_ms=dit_ms,
            )
            return

        _synchronize(self.depth_estimator.device)
        t_depth = time.perf_counter()
        update_frame_indices = None
        update_local_indices = None
        if self.memory_update_mode == "keyframe":
            keyframe_index = pixel_end - 1
            keyframe_rgb = block_rgb[0, -1]
            depth = self.depth_estimator.estimate(
                keyframe_rgb,
                output_size=(self.height, self.width),
                output_device=self.memory.device,
            )
            block_depths = None
            block_intrinsics = None
        else:
            if self.memory_update_mode == "latent_keyframe":
                update_frame_indices = latent_keyframe_indices(
                    latent_start, denoised_latent.shape[1]
                )
                update_local_indices = tuple(
                    frame_index - pixel_start for frame_index in update_frame_indices
                )
                if any(
                    local_index < 0 or local_index >= expected_frames
                    for local_index in update_local_indices
                ):
                    raise RuntimeError(
                        f"Latent keyframes {update_frame_indices} are outside pixel span "
                        f"[{pixel_start}, {pixel_end})"
                    )
                depth_input_rgb = block_rgb[0, list(update_local_indices)]
            else:
                update_frame_indices = tuple(range(pixel_start, pixel_end))
                update_local_indices = tuple(range(expected_frames))
                depth_input_rgb = block_rgb[0]
            (
                block_reconstruction_rgb,
                block_depths,
                block_intrinsics,
                _,
            ) = self.depth_estimator.estimate_block(
                depth_input_rgb,
                output_size=(self.height, self.width),
                output_device=self.memory.device,
            )
        _synchronize(self.depth_estimator.device)
        _synchronize(self.memory.device)
        depth_estimation_ms = (time.perf_counter() - t_depth) * 1000.0

        _synchronize(self.memory.device)
        t_update = time.perf_counter()
        scale_details = {}
        confidence_details = {}
        if self.memory_map_mode == "dense_two_layer":
            if update_frame_indices is None or update_local_indices is None:
                raise AssertionError("Dense update frame selection was not initialized")
            reference_depth = self.reference_depth[list(update_frame_indices)].to(
                self.memory.device, dtype=torch.float32
            )
            reference_valid = (
                self.reference_mask[0, 0, list(update_frame_indices)]
                .to(self.memory.device) > 0
            )
            scale, scale_stats = calibrate_depth_scale(
                block_depths,
                reference_depth,
                reference_valid,
                previous_scale=self.previous_depth_scale,
            )
            self.previous_depth_scale = scale
            scaled_depths = block_depths * scale
            historical_depth = self._historical_depth_by_block.pop(block_index, None)
            if historical_depth is not None:
                historical_depth = historical_depth[list(update_local_indices)]
            confidence = compute_depth_confidence(
                scaled_depths,
                scale_reliable=scale_stats["scale_reliable"],
                log_mad=scale_stats["log_mad"],
                historical_depth=historical_depth,
            )
            confidence_sample = confidence.flatten()
            sample_stride = max(1, confidence_sample.numel() // 200_000)
            quantiles = torch.quantile(
                confidence_sample[::sample_stride].float(),
                torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=confidence.device),
            ).cpu().tolist()
            memory_metadata = {
                "frame_indices": list(update_frame_indices),
                "frame_start": update_frame_indices[0],
                "frame_end": update_frame_indices[-1] + 1,
                "depth_backend": self.depth_backend,
                **getattr(self.depth_estimator, "last_backend_metadata", {}),
                **scale_stats,
                "confidence_quantiles": quantiles,
            }
            update_stats = self.memory.update_block(
                block_reconstruction_rgb,
                scaled_depths,
                self.target_c2w[list(update_frame_indices)],
                torch.ones_like(scaled_depths, dtype=torch.bool),
                K=self.K,
                confidence=confidence,
                metadata=memory_metadata,
            )
            scale_details = scale_stats
            confidence_details = {
                "confidence_min": quantiles[0],
                "confidence_p25": quantiles[1],
                "confidence_median": quantiles[2],
                "confidence_p75": quantiles[3],
                "confidence_max": quantiles[4],
            }
            update_details = {
                "keyframe_index": None,
                "update_frame_indices": list(update_frame_indices),
                "update_frame_start": update_frame_indices[0],
                "update_frame_end": update_frame_indices[-1] + 1,
                "update_frames": len(update_frame_indices),
                "depth_backend": self.depth_backend,
                "depth_processed_shape": getattr(self.depth_estimator, "last_processed_shape", None),
                "depth_intrinsics_shape": getattr(self.depth_estimator, "last_intrinsics_shape", None),
                "depth_extrinsics_shape": getattr(self.depth_estimator, "last_extrinsics_shape", None),
                **getattr(self.depth_estimator, "last_backend_metadata", {}),
                "da3_processed_shape": getattr(self.depth_estimator, "last_processed_shape", None),
                "da3_intrinsics_shape": getattr(self.depth_estimator, "last_intrinsics_shape", None),
                "da3_extrinsics_shape": getattr(self.depth_estimator, "last_extrinsics_shape", None),
                **scale_details,
                **confidence_details,
                **update_stats,
            }
        elif self.memory_update_mode == "keyframe":
            reference_valid = self.reference_mask[0, 0, keyframe_index] > 0
            update_stats = self.memory.update(
                keyframe_rgb.to(self.memory.device),
                depth,
                self.target_c2w[keyframe_index],
                ~reference_valid,
            )
            update_details = {
                "keyframe_index": keyframe_index,
                "update_frames": 1,
                **update_stats,
            }
        else:
            reference_valid = self.reference_mask[0, 0, pixel_start:pixel_end] > 0
            update_stats = self.memory.update_block(
                block_reconstruction_rgb,
                block_depths,
                self.target_c2w[pixel_start:pixel_end],
                ~reference_valid,
                K=block_intrinsics,
            )
            update_details = {
                "keyframe_index": None,
                "update_frame_start": pixel_start,
                "update_frame_end": pixel_end,
                "update_frames": pixel_end - pixel_start,
                "da3_processed_shape": getattr(self.depth_estimator, "last_processed_shape", None),
                "da3_intrinsics_shape": getattr(self.depth_estimator, "last_intrinsics_shape", None),
                "da3_extrinsics_shape": getattr(self.depth_estimator, "last_extrinsics_shape", None),
                **update_stats,
            }
        _synchronize(self.memory.device)
        update_ms = (time.perf_counter() - t_update) * 1000.0

        metric = self._metric_for_block(block_index)
        cuda_allocated_peak_gb, cuda_reserved_peak_gb = _cuda_peak_gb(self.memory.device)
        metric.update({
            "dit_ms": float(dit_ms),
            "decode_ms": decode_ms,
            "depth_backend": self.depth_backend,
            "depth_estimation_ms": depth_estimation_ms,
            "da3_ms": depth_estimation_ms if self.depth_backend == "da3" else 0.0,
            "align3r_ms": depth_estimation_ms if self.depth_backend == "align3r" else 0.0,
            "memory_update_ms": update_ms,
            "memory_update_mode": self.memory_update_mode,
            "memory_map_mode": self.memory_map_mode,
            "depth_native_shape": self.depth_estimator.last_native_shape,
            "da3_peak_memory_gb": (
                float(getattr(self.depth_estimator, "last_peak_memory_gb", 0.0))
                if self.depth_backend == "da3" else 0.0
            ),
            "depth_backend_peak_memory_gb": float(
                getattr(self.depth_estimator, "last_peak_memory_gb", 0.0)
            ),
            "cuda_allocated_peak_gb": cuda_allocated_peak_gb,
            "cuda_reserved_peak_gb": cuda_reserved_peak_gb,
            **update_details,
        })

    @torch.inference_mode()
    def _output_callback_overlap_v3(
        self,
        *,
        block_index: int,
        latent_start: int,
        denoised_latent: torch.Tensor,
        block_rgb: torch.Tensor,
        pixel_start: int,
        pixel_end: int,
        decode_ms: float,
        dit_ms: float,
    ) -> None:
        update_frame_indices = latent_keyframe_indices(
            latent_start, denoised_latent.shape[1]
        )
        update_local_indices = tuple(
            frame_index - pixel_start for frame_index in update_frame_indices
        )
        new_rgb = block_rgb[0, list(update_local_indices)]
        anchor_input_index = self.anchor_frame_index
        anchor_count = 0 if self.anchor_rgb is None else 1
        if anchor_count:
            window_rgb = torch.cat((self.anchor_rgb.unsqueeze(0), new_rgb), dim=0)
        else:
            window_rgb = new_rgb

        _synchronize(self.depth_estimator.device)
        t_da3 = time.perf_counter()
        (
            window_reconstruction_rgb,
            window_depth,
            window_intrinsics,
            window_w2c,
        ) = self.depth_estimator.estimate_block(
            window_rgb,
            output_size=(self.height, self.width),
            output_device=self.memory.device,
        )
        _synchronize(self.depth_estimator.device)
        _synchronize(self.memory.device)
        da3_ms = (time.perf_counter() - t_da3) * 1000.0

        t_build = time.perf_counter()
        local_point_grids = []
        local_valid_grids = []
        for depth, intrinsic, extrinsic in zip(
            window_depth, window_intrinsics, window_w2c
        ):
            points, valid = backproject_world_grid(
                depth, intrinsic, w2c=extrinsic
            )
            local_point_grids.append(points)
            local_valid_grids.append(valid)

        if anchor_count:
            registration_source = local_point_grids[0]
            registration_target = self.anchor_world_points
            registration_valid = local_valid_grids[0] & self.anchor_valid
            registration_source_name = "previous_block_last_latent"
        else:
            source_parts, target_parts, valid_parts = [], [], []
            for local_index, frame_index in enumerate(update_frame_indices):
                reference_points, reference_valid = backproject_world_grid(
                    self.reference_depth[frame_index].to(
                        self.memory.device, dtype=torch.float32
                    ),
                    self.K,
                    c2w=self.target_c2w[frame_index],
                )
                source_parts.append(local_point_grids[local_index].reshape(-1, 3))
                target_parts.append(reference_points.reshape(-1, 3))
                valid_parts.append(
                    (
                        local_valid_grids[local_index]
                        & reference_valid
                        & (self.reference_mask[0, 0, frame_index].to(
                            self.memory.device
                        ) > 0)
                    ).reshape(-1)
                )
            registration_source = torch.cat(source_parts, dim=0)
            registration_target = torch.cat(target_parts, dim=0)
            registration_valid = torch.cat(valid_parts, dim=0)
            registration_source_name = "immutable_reference_depth"
        _synchronize(self.memory.device)
        pointcloud_prebuild_ms = (time.perf_counter() - t_build) * 1000.0

        registration_error = None
        registration = None
        _synchronize(self.memory.device)
        t_registration = time.perf_counter()
        try:
            registration = estimate_similarity_registration(
                registration_source,
                registration_target,
                registration_valid,
                min_correspondences=min(
                    4096, max(128, int(self.height * self.width * 0.01))
                ),
            )
        except (RuntimeError, ValueError) as error:
            registration_error = str(error)
        _synchronize(self.memory.device)
        registration_ms = (time.perf_counter() - t_registration) * 1000.0

        scale_jump = None
        accepted = registration is not None
        if accepted and self.previous_registration_scale is not None:
            scale_jump = abs(
                registration.scale / self.previous_registration_scale - 1.0
            )
        if accepted and registration.normalized_rmse > 0.15:
            accepted = False
            registration_error = (
                f"normalized RMSE {registration.normalized_rmse:.6f} exceeds 0.15"
            )
        if accepted and scale_jump is not None and scale_jump > 0.30:
            accepted = False
            registration_error = f"scale jump {scale_jump:.6f} exceeds 0.30"

        observed_residuals = []
        update_stats = {
            "raw_valid_points": 0,
            "batch_voxels": 0,
            "points_before": self.memory.point_count,
            "points_after": self.memory.point_count,
            "evicted_voxels": 0,
        }
        pointcloud_postbuild_ms = 0.0
        voxel_fusion_ms = 0.0
        if accepted:
            new_offset = anchor_count
            t_build = time.perf_counter()
            new_world_grids = []
            new_valid_grids = []
            observed_poses = []
            for new_index, frame_index in enumerate(update_frame_indices):
                window_index = new_offset + new_index
                world_points = apply_similarity(
                    local_point_grids[window_index],
                    registration.scale,
                    registration.rotation,
                    registration.translation,
                )
                valid = local_valid_grids[window_index] & torch.isfinite(
                    world_points
                ).all(dim=-1)
                observed_pose = transform_da3_c2w(
                    window_w2c[window_index],
                    registration.scale,
                    registration.rotation,
                    registration.translation,
                )
                residual = pose_residual(
                    self.target_c2w[frame_index], observed_pose
                )
                residual["frame_index"] = int(frame_index)
                observed_residuals.append(residual)
                self.observed_keyframes[int(frame_index)] = observed_pose.detach().cpu()
                new_world_grids.append(world_points)
                new_valid_grids.append(valid)
                observed_poses.append(observed_pose)

            fused_points = torch.stack(new_world_grids, dim=0)
            fused_colors = window_reconstruction_rgb[new_offset:].permute(0, 2, 3, 1)
            fused_valid = torch.stack(new_valid_grids, dim=0)
            _synchronize(self.memory.device)
            pointcloud_postbuild_ms = (time.perf_counter() - t_build) * 1000.0

            _synchronize(self.memory.device)
            t_fusion = time.perf_counter()
            update_stats = self.memory.update_points(
                fused_points, fused_colors, fused_valid
            )
            _synchronize(self.memory.device)
            voxel_fusion_ms = (time.perf_counter() - t_fusion) * 1000.0

            self.anchor_rgb = new_rgb[-1].detach().to(self.memory.device)
            self.anchor_world_points = new_world_grids[-1].detach()
            self.anchor_valid = new_valid_grids[-1].detach()
            self.anchor_frame_index = int(update_frame_indices[-1])
            self.previous_registration_scale = registration.scale

        focal = float(window_intrinsics[anchor_count:, 0, 0].mean().item())
        focal_jump = None
        if self.previous_da3_focal is not None:
            focal_jump = abs(focal / self.previous_da3_focal - 1.0)
        if accepted:
            self.previous_da3_focal = focal

        pointcloud_build_ms = pointcloud_prebuild_ms + pointcloud_postbuild_ms
        memory_update_ms = registration_ms + pointcloud_build_ms + voxel_fusion_ms
        pointcloud_total_ms = da3_ms + memory_update_ms
        max_rotation_residual = max(
            (item["rotation_degrees"] for item in observed_residuals), default=None
        )
        max_translation_residual = max(
            (item["translation"] for item in observed_residuals), default=None
        )
        registration_details = {
            "registration_source": registration_source_name,
            "registration_accepted": bool(accepted),
            "registration_error": registration_error,
            "registration_scale": None if registration is None else registration.scale,
            "registration_rmse": None if registration is None else registration.rmse,
            "registration_normalized_rmse": (
                None if registration is None else registration.normalized_rmse
            ),
            "registration_correspondences": (
                0 if registration is None else registration.correspondence_count
            ),
            "registration_inliers": 0 if registration is None else registration.inlier_count,
            "registration_inlier_ratio": (
                0.0 if registration is None else registration.inlier_ratio
            ),
            "registration_scale_jump": scale_jump,
        }
        metric = self._metric_for_block(block_index)
        cuda_allocated_peak_gb, cuda_reserved_peak_gb = _cuda_peak_gb(
            self.memory.device
        )
        metric.update({
            "dit_ms": float(dit_ms),
            "decode_ms": decode_ms,
            "depth_backend": self.depth_backend,
            "depth_estimation_ms": da3_ms,
            "da3_ms": da3_ms,
            "registration_ms": registration_ms,
            "pointcloud_build_ms": pointcloud_build_ms,
            "voxel_fusion_ms": voxel_fusion_ms,
            "memory_update_ms": memory_update_ms,
            "pointcloud_total_ms": pointcloud_total_ms,
            "memory_update_mode": self.memory_update_mode,
            "memory_map_mode": self.memory_map_mode,
            "update_frame_indices": list(update_frame_indices),
            "update_frames": len(update_frame_indices),
            "anchor_count": anchor_count,
            "anchor_frame_index_input": anchor_input_index,
            "anchor_frame_index_output": self.anchor_frame_index,
            "multi_anchor_retry_implemented": False,
            "da3_window_frames": int(window_rgb.shape[0]),
            "da3_focal_mean": focal,
            "da3_focal_jump": focal_jump,
            "observed_pose_residuals": observed_residuals,
            "max_plan_observed_rotation_degrees": max_rotation_residual,
            "max_plan_observed_translation": max_translation_residual,
            "depth_native_shape": self.depth_estimator.last_native_shape,
            "da3_peak_memory_gb": float(
                getattr(self.depth_estimator, "last_peak_memory_gb", 0.0)
            ),
            "cuda_allocated_peak_gb": cuda_allocated_peak_gb,
            "cuda_reserved_peak_gb": cuda_reserved_peak_gb,
            **registration_details,
            **update_stats,
        })

    def close(self) -> dict:
        if self.closed:
            return self.summary()
        self.closed = True
        for writer in self.writers.values():
            writer.close()

        if self.memory_map_mode in _OVERLAP_VOXEL_MODES:
            camera_indices = sorted(self.observed_keyframes)
            observed = np.stack([
                self.observed_keyframes[index].numpy() for index in camera_indices
            ]) if camera_indices else np.empty((0, 4, 4), dtype=np.float32)
            planned = self.target_c2w[camera_indices].detach().cpu().numpy() \
                if camera_indices else np.empty((0, 4, 4), dtype=np.float32)
            np.savez_compressed(
                os.path.join(
                    self.output_dir,
                    f"{self.output_prefix}-v3_keyframe_cameras_rank{self.rank}.npz",
                ),
                frame_indices=np.asarray(camera_indices, dtype=np.int64),
                planned_c2w=planned.astype(np.float32),
                observed_c2w=observed.astype(np.float32),
            )

        map_prefix = os.path.join(
            self.output_dir,
            f"{self.output_prefix}-historical_memory_final_rank{self.rank}",
        )
        map_index_path, ply_path = self.memory.save(map_prefix)
        manifest = {
            "reference_map_path": self.reference_map_path,
            "historical_map_ply": ply_path,
            "maps_are_kept_separate": True,
            "memory_update_mode": self.memory_update_mode,
            "memory_map_mode": self.memory_map_mode,
            "depth_backend": self.depth_backend,
        }
        if self.adaptive_voxel_details is not None:
            manifest["adaptive_voxel"] = self.adaptive_voxel_details
        if self.memory_map_mode == "dense_two_layer":
            manifest["historical_map_chunk_manifest"] = map_index_path
        else:
            manifest["historical_map_npz"] = map_index_path
        with open(os.path.join(
            self.output_dir,
            f"{self.output_prefix}-memory_manifest_rank{self.rank}.json",
        ), "w") as handle:
            json.dump(manifest, handle, indent=2)

        summary = self.summary()
        payload = {"summary": summary, "blocks": self.metrics}
        with open(os.path.join(
            self.output_dir,
            f"{self.output_prefix}-memory_timing_rank{self.rank}.json",
        ), "w") as handle:
            json.dump(payload, handle, indent=2)
        return summary

    def summary(self) -> dict:
        stage_names = [
            "hist_render_ms", "merge_ms", "condition_encode_ms", "dit_ms",
            "decode_ms", "depth_estimation_ms", "memory_update_ms",
        ]
        stage_totals = {
            name: float(sum(float(block.get(name, 0.0)) for block in self.metrics))
            for name in stage_names
        }
        online_ms = sum(stage_totals.values())
        output_frames = int(sum(int(block.get("pixel_frames", 0)) for block in self.metrics))
        extra_v3_totals = {
            name: float(sum(float(block.get(name, 0.0)) for block in self.metrics))
            for name in (
                "registration_ms", "pointcloud_build_ms", "voxel_fusion_ms",
                "pointcloud_total_ms",
            )
        }
        return {
            "depth_backend": self.depth_backend,
            "memory_update_mode": self.memory_update_mode,
            "memory_map_mode": self.memory_map_mode,
            "num_blocks": len(self.metrics),
            "output_frames": output_frames,
            "final_point_count": self.memory.point_count,
            "voxel_size": getattr(self.memory, "voxel_size", None),
            "splat_diameter": getattr(self.memory, "splat_diameter", None),
            "adaptive_voxel": self.adaptive_voxel_details,
            "da3_peak_memory_gb": max(
                (float(block.get("da3_peak_memory_gb", 0.0)) for block in self.metrics),
                default=0.0,
            ),
            "depth_backend_peak_memory_gb": max(
                (
                    float(block.get("depth_backend_peak_memory_gb", 0.0))
                    for block in self.metrics
                ),
                default=0.0,
            ),
            "da3_ms": float(sum(float(block.get("da3_ms", 0.0)) for block in self.metrics)),
            "align3r_ms": float(
                sum(float(block.get("align3r_ms", 0.0)) for block in self.metrics)
            ),
            "cuda_allocated_peak_gb": max(
                (float(block.get("cuda_allocated_peak_gb", 0.0)) for block in self.metrics),
                default=0.0,
            ),
            "cuda_reserved_peak_gb": max(
                (float(block.get("cuda_reserved_peak_gb", 0.0)) for block in self.metrics),
                default=0.0,
            ),
            "online_total_ms": online_ms,
            "online_fps": output_frames / (online_ms / 1000.0) if online_ms > 0 else 0.0,
            "accepted_blocks": int(sum(
                bool(block.get("registration_accepted", False))
                for block in self.metrics
            )),
            **stage_totals,
            **extra_v3_totals,
        }
