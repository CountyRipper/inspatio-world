"""Block condition/output controller for the RGB point-memory baseline."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import numpy as np
import torch

import torch.nn.functional as F
from utils.historical_point_memory import (
    DenseGeneratedPointMemory,
    IncrementalVoxelSurfelMemory,
    RGBPointMemory,
    VideoStreamWriter,
    calibrate_depth_scale,
    compute_depth_confidence,
    fuse_reference_and_history,
    fuse_reference_and_history_v4,
    latent_block_to_pixel_span,
    latent_keyframe_indices,
)
from utils.overlap_da3_registration import (
    apply_similarity,
    backproject_world_grid,
    estimate_similarity_registration,
    pose_residual,
    select_v4_runtime_points,
    transform_da3_c2w,
)
from utils.render_warper import convert_mask_video


_OVERLAP_VOXEL_V3_MODES = {
    "overlap_voxel_v3", "overlap_voxel_v3_1", "overlap_voxel_v3_2",
}
_OVERLAP_VOXEL_MODES = {*_OVERLAP_VOXEL_V3_MODES, "overlap_voxel_v4", "overlap_voxel_v5"}
_ADAPTIVE_OVERLAP_VOXEL_MODES = {
    "overlap_voxel_v3_1", "overlap_voxel_v3_2",
    "overlap_voxel_v4", "overlap_voxel_v5",
}
_CONTINUOUS_V3_READ_MODES = {"overlap_voxel_v3_1", "overlap_voxel_v3_2"}


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

    @torch.inference_mode()
    def decode_prefix(self, latent_btchw: torch.Tensor) -> torch.Tensor:
        video = self.vae.decode_to_pixel(
            latent_btchw.to(self.device), use_cache=False
        )
        return (video * 0.5 + 0.5).clamp_(0, 1)


class TAEBlockDecoder:
    """Stateful block decoder using the repository's StreamingTAEHV helper."""

    def __init__(self, tae_model):
        from utils.taehv import StreamingTAEHV

        self.streaming = StreamingTAEHV(tae_model)
        self.tae_model = tae_model
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

    @torch.inference_mode()
    def decode_prefix(self, latent_btchw: torch.Tensor) -> torch.Tensor:
        return self.tae_model.decode_video(
            latent_btchw.to(self.device, dtype=torch.float16),
            show_progress_bar=False,
        ).float()


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
        geometry_voxel_factor: float = 2.0,
        geometry_depth_ratio: float = 0.03,
        fps: int = 24,
        save_diagnostics: bool = True,
        source_rgb_bcthw: Optional[torch.Tensor] = None,
        source_c2w: Optional[torch.Tensor] = None,
        mapanything_min_consistent_ratio: float = 0.01,
        memory_single_keyframe_index: Optional[int] = None,
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
        if source_rgb_bcthw is not None and source_rgb_bcthw.shape[0] != 1:
            raise ValueError("MapAnything source RGB requires batch size 1")
        if source_c2w is not None:
            if source_c2w.ndim == 4:
                if source_c2w.shape[0] != 1:
                    raise ValueError("MapAnything source poses require batch size 1")
                source_c2w = source_c2w[0]
            if source_c2w.ndim != 3 or source_c2w.shape[-2:] != (4, 4):
                raise ValueError(
                    f"Expected source_c2w [T,4,4], got {tuple(source_c2w.shape)}"
                )


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
        self.source_rgb = source_rgb_bcthw
        self.source_c2w = source_c2w
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
                raise ValueError("overlap-voxel modes require latent_keyframe updates")
            if memory_map_mode == "overlap_voxel_v5":
                if self.depth_backend != "mapanything":
                    raise ValueError("overlap_voxel_v5 requires MapAnything")
                if self.source_rgb is None or self.source_c2w is None:
                    raise ValueError("overlap_voxel_v5 requires source RGB and c2w")
            elif self.depth_backend != "da3":
                raise ValueError("overlap_voxel_v3/v4 currently requires DA3")
            if memory_map_mode in _OVERLAP_VOXEL_V3_MODES and memory_anchor_count != 1:
                raise NotImplementedError(
                    "Multi-anchor DA3 windows are reserved but not implemented"
                )
            if memory_map_mode in _ADAPTIVE_OVERLAP_VOXEL_MODES \
                    and adaptive_voxel_details is None:
                raise ValueError(f"{memory_map_mode} requires adaptive voxel metadata")
            if memory_map_mode in {"overlap_voxel_v4", "overlap_voxel_v5"}:
                if not isinstance(memory, IncrementalVoxelSurfelMemory):
                    raise TypeError("overlap_voxel_v4/v5 requires voxel surfel memory")
                if memory.splat_diameter != 3:
                    raise ValueError("overlap_voxel_v4/v5 requires a 3x3 point splat")
                if geometry_voxel_factor <= 0 or geometry_depth_ratio <= 0:
                    raise ValueError("V4/V5 geometry thresholds must be positive")
        self.memory_update_mode = memory_update_mode
        self.memory_map_mode = memory_map_mode
        self.save_diagnostics = save_diagnostics
        self.fps = int(fps)
        self.metrics = []
        self.closed = False
        self.previous_depth_scale = None
        self._historical_depth_by_block = {}
        self.memory_anchor_count = int(memory_anchor_count)
        self.adaptive_voxel_details = adaptive_voxel_details
        self.geometry_voxel_factor = float(geometry_voxel_factor)
        self.geometry_depth_ratio = float(geometry_depth_ratio)
        self.anchor_rgb = None
        self.anchor_world_points = None
        self.anchor_valid = None
        self.anchor_frame_index = None
        self.previous_registration_scale = None
        self.previous_da3_focal = None
        self.observed_keyframes = {}
        self.generated_latent_blocks = []
        self.memory_single_keyframe_index = memory_single_keyframe_index
        self.single_keyframe_attempted = False
        self.single_keyframe_written = False
        if self.memory_map_mode == "overlap_voxel_v3_2":
            if memory_single_keyframe_index is None:
                raise ValueError(
                    "overlap_voxel_v3_2 requires memory_single_keyframe_index"
                )
            if not 0 <= memory_single_keyframe_index < target_c2w.shape[0]:
                raise ValueError(
                    "memory_single_keyframe_index must address a target RGB frame"
                )

        if not 0 <= mapanything_min_consistent_ratio <= 1:
            raise ValueError("MapAnything minimum consistent ratio must be in [0,1]")
        self.mapanything_min_consistent_ratio = float(mapanything_min_consistent_ratio)
        self.mapanything_chunks = {}

        _, _, _, height, width = reference_rgb_bcthw.shape
        self.height = height
        self.width = width
        os.makedirs(output_dir, exist_ok=True)
        self.v4_output_dir = os.path.join(
            output_dir, f"{output_prefix}-{memory_map_mode}-rank{rank}"
        )

        paths = {
            "pred": f"{output_prefix}-pred_video_rank{rank}.mp4",
        }
        if save_diagnostics and memory_map_mode not in {"overlap_voxel_v4", "overlap_voxel_v5"}:
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

    def _v4_block_dir(self, block_index: int) -> str:
        path = os.path.join(self.v4_output_dir, f"block_{block_index:03d}")
        os.makedirs(path, exist_ok=True)
        return path

    def _write_v4_video(
        self,
        block_index: int,
        name: str,
        video_tchw: torch.Tensor,
        *,
        signed: bool = False,
    ) -> None:
        if not self.save_diagnostics:
            return
        video = video_tchw
        if signed:
            video = (video.float() * 0.5 + 0.5).clamp(0, 1)
        writer = VideoStreamWriter(
            os.path.join(self._v4_block_dir(block_index), name),
            self.width,
            self.height,
            self.fps,
        )
        writer.write(video)
        writer.close()

    def _write_v4_metrics(self, block_index: int, metric: dict) -> None:
        path = os.path.join(self._v4_block_dir(block_index), "metrics.json")
        temporary_path = path + ".tmp"
        with open(temporary_path, "w") as handle:
            json.dump(metric, handle, indent=2)
        os.replace(temporary_path, path)

    @torch.inference_mode()
    def _condition_provider_v4(
        self,
        block_index: int,
        latent_start: int,
        latent_count: int,
        pixel_start: int,
        pixel_end: int,
    ):
        reference_rgb = self.reference_rgb[0, :, :pixel_end].permute(1, 0, 2, 3).float()
        reference_mask = self.reference_mask[0, :1, :pixel_end].permute(1, 0, 2, 3) > 0
        points_before_read = self.memory.point_count

        _synchronize(self.memory.device)
        t_render = time.perf_counter()
        historical_rgb, historical_mask = self.memory.render(
            self.target_c2w[:pixel_end], self.K
        )
        _synchronize(self.memory.device)
        hist_render_ms = (time.perf_counter() - t_render) * 1000.0
        historical_rgb = historical_rgb.mul(2).sub(1)

        t_merge = time.perf_counter()
        fused_rgb, fused_mask, historical_add = fuse_reference_and_history_v4(
            reference_rgb.to(self.memory.device),
            reference_mask.to(self.memory.device),
            historical_rgb,
            historical_mask,
        )
        merge_ms = (time.perf_counter() - t_merge) * 1000.0
        if not torch.equal(fused_mask, reference_mask.to(self.memory.device) | historical_mask):
            raise AssertionError("V4 fused mask differs from reference | history")

        self._write_v4_video(
            block_index, "pre_reference.mp4", reference_rgb, signed=True
        )
        self._write_v4_video(
            block_index, "pre_historical.mp4", historical_rgb, signed=True
        )
        self._write_v4_video(block_index, "pre_fused.mp4", fused_rgb, signed=True)
        self._write_v4_video(
            block_index, "pre_reference_mask.mp4", reference_mask.float()
        )
        self._write_v4_video(
            block_index, "pre_historical_mask.mp4", historical_mask.float()
        )
        self._write_v4_video(
            block_index, "pre_fused_mask.mp4", fused_mask.float()
        )

        fused_video = fused_rgb.permute(1, 0, 2, 3).unsqueeze(0).to(
            dtype=self.reference_rgb.dtype
        )
        fused_mask_video = (
            fused_mask.float().permute(1, 0, 2, 3).unsqueeze(0).mul(2).sub(1)
        )
        _synchronize(fused_video.device)
        t_encode = time.perf_counter()
        render_prefix = self.encode_video(fused_video)
        mask_prefix = convert_mask_video(fused_mask_video)
        _synchronize(render_prefix.device)
        condition_encode_ms = (time.perf_counter() - t_encode) * 1000.0
        latent_end = latent_start + latent_count
        if render_prefix.shape[1] < latent_end or mask_prefix.shape[1] < latent_end:
            raise RuntimeError("V4 prefix condition encode returned too few latents")
        render_block = render_prefix[:, latent_start:latent_end]
        mask_block = mask_prefix[:, latent_start:latent_end]

        current_slice = slice(pixel_start, pixel_end)
        current_reference_mask = reference_mask[current_slice]
        current_historical_mask = historical_mask[current_slice]
        current_historical_add = historical_add[current_slice]
        current_fused_mask = fused_mask[current_slice]
        metric = self._metric_for_block(block_index)
        metric.update({
            "latent_start": latent_start,
            "pixel_start": pixel_start,
            "pixel_end": pixel_end,
            "pixel_frames": pixel_end - pixel_start,
            "condition_prefix_frames": pixel_end,
            "hist_render_ms": hist_render_ms,
            "merge_ms": merge_ms,
            "condition_encode_ms": condition_encode_ms,
            "reference_coverage": float(current_reference_mask.float().mean().item()),
            "historical_coverage": float(current_historical_mask.float().mean().item()),
            "hist_only_coverage": float(current_historical_add.float().mean().item()),
            "fused_coverage": float(current_fused_mask.float().mean().item()),
            "historical_pixels": int(current_historical_mask.sum().item()),
            "history_injected_pixels": int(current_historical_add.sum().item()),
            "points_before_read": points_before_read,
            "memory_map_mode": self.memory_map_mode,
            "memory_read_contract": "full_prefix_gpu_voxel_render+reference_priority_fuse+vae_encode",
            "memory_render_uses_planned_c2w": True,
            "memory_ply_roundtrip": False,
        })
        return render_block, mask_block

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
        if self.memory_map_mode in {"overlap_voxel_v4", "overlap_voxel_v5"}:
            return self._condition_provider_v4(
                block_index,
                latent_start,
                latent_count,
                pixel_start,
                pixel_end,
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
        if self.memory_map_mode in _CONTINUOUS_V3_READ_MODES:
            if block_index > 0:
                previous_metric = self._metric_for_block(block_index - 1)
                if "points_after" not in previous_metric:
                    raise AssertionError(
                        f"{self.memory_map_mode} memory read occurred before the previous block write"
                    )
                points_after_previous_block = int(previous_metric["points_after"])
            if points_before_read != points_after_previous_block:
                raise AssertionError(
                    f"{self.memory_map_mode} GPU map continuity failed: "
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
        if self.memory_map_mode in _CONTINUOUS_V3_READ_MODES \
                and block_index == 0 and historical_pixels != 0:
            raise AssertionError(
                f"{self.memory_map_mode} block zero must not read historical pixels"
            )
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
                if self.memory_map_mode in _CONTINUOUS_V3_READ_MODES else None
            ),
            "memory_read_contract": (
                "gpu_voxel_render+offline_reference_fuse+vae_encode"
                if self.memory_map_mode in _CONTINUOUS_V3_READ_MODES else None
            ),
            "memory_render_uses_planned_c2w": (
                True if self.memory_map_mode in _CONTINUOUS_V3_READ_MODES else None
            ),
            "memory_ply_roundtrip": (
                False if self.memory_map_mode in _CONTINUOUS_V3_READ_MODES else None
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
        if self.memory_map_mode == "overlap_voxel_v5":
            self._output_callback_overlap_v5(
                block_index=block_index,
                latent_start=latent_start,
                denoised_latent=denoised_latent,
                pixel_start=pixel_start,
                pixel_end=pixel_end,
                dit_ms=dit_ms,
            )
            return

        if self.memory_map_mode == "overlap_voxel_v4":
            self._output_callback_overlap_v4(
                block_index=block_index,
                latent_start=latent_start,
                denoised_latent=denoised_latent,
                pixel_start=pixel_start,
                pixel_end=pixel_end,
                dit_ms=dit_ms,
            )
            return

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

    def _evaluate_mapanything_candidate(
        self,
        batch,
        pred_slots: list[int],
        frame_indices: tuple[int, ...],
    ) -> tuple[float, torch.Tensor, dict, dict]:
        """Score one MapAnything branch against immutable reference geometry."""
        points = batch.points[pred_slots]
        valid = batch.valid[pred_slots]
        intrinsics = batch.intrinsics[pred_slots]
        native_height, native_width = batch.processed_size
        reference_depth = self.reference_depth[list(frame_indices)].to(
            self.memory.device, dtype=torch.float32
        )
        reference_depth = F.interpolate(
            reference_depth.unsqueeze(1),
            size=(native_height, native_width),
            mode="nearest",
        ).squeeze(1)
        reference_mask = (
            self.reference_mask[0, 0, list(frame_indices)] > 0
        ).float().unsqueeze(1)
        reference_mask = F.interpolate(
            reference_mask,
            size=(native_height, native_width),
            mode="nearest",
        ).squeeze(1).bool().to(self.memory.device)

        reference_points, reference_valid = [], []
        for local_index, frame_index in enumerate(frame_indices):
            frame_points, frame_valid = backproject_world_grid(
                reference_depth[local_index],
                intrinsics[local_index],
                c2w=self.target_c2w[frame_index],
            )
            reference_points.append(frame_points)
            reference_valid.append(frame_valid)
        reference_points = torch.stack(reference_points)
        reference_valid = torch.stack(reference_valid)

        correspondence = valid & reference_valid & reference_mask
        geometry_error = torch.linalg.norm(points - reference_points, dim=-1)
        consistent_threshold = torch.maximum(
            torch.full_like(
                reference_depth,
                self.geometry_voxel_factor * self.memory.voxel_size,
            ),
            self.geometry_depth_ratio * reference_depth,
        )
        consistent = (
            correspondence
            & torch.isfinite(geometry_error)
            & (geometry_error <= consistent_threshold)
        )
        correspondence_count = int(correspondence.sum().item())
        valid_error = geometry_error[correspondence]
        median_error = (
            float("inf")
            if valid_error.numel() == 0
            else float(valid_error.median().item())
        )
        consistent_ratio = float(
            consistent.sum().item() / max(1, correspondence_count)
        )
        point_keep, point_stats = select_v4_runtime_points(
            points,
            reference_points,
            valid,
            reference_depth,
            reference_valid,
            reference_mask,
            voxel_size=self.memory.voxel_size,
            voxel_factor=self.geometry_voxel_factor,
            relative_depth_ratio=self.geometry_depth_ratio,
        )
        point_stats = {
            key.replace("da3", "mapanything"): value
            for key, value in point_stats.items()
        }
        pose_metrics = [
            pose_residual(
                self.target_c2w[frame_index],
                batch.camera_c2w[pred_slot],
            )
            for frame_index, pred_slot in zip(frame_indices, pred_slots)
        ]
        quality = {
            "reference_correspondence_count": correspondence_count,
            "reference_consistent_count": int(consistent.sum().item()),
            "reference_consistent_ratio": consistent_ratio,
            "reference_error_median": (
                None if not np.isfinite(median_error) else median_error
            ),
            "reference_error_p90": (
                None
                if valid_error.numel() == 0
                else float(torch.quantile(valid_error, 0.9).item())
            ),
            "pose_rotation_degrees_mean": float(
                np.mean([item["rotation_degrees"] for item in pose_metrics])
            ),
            "pose_translation_mean": float(
                np.mean([item["translation"] for item in pose_metrics])
            ),
        }
        return median_error, point_keep, quality, point_stats

    @torch.inference_mode()
    def _output_callback_overlap_v5(
        self,
        *,
        block_index: int,
        latent_start: int,
        denoised_latent: torch.Tensor,
        pixel_start: int,
        pixel_end: int,
        dit_ms: float,
    ) -> None:
        """Build a bounded online map from paired source/pred MapAnything windows."""
        self.generated_latent_blocks.append(denoised_latent.detach().cpu())
        latent_prefix = torch.cat(self.generated_latent_blocks, dim=1)

        _synchronize(self.block_decoder.device)
        decode_started = time.perf_counter()
        decoded_prefix = self.block_decoder.decode_prefix(latent_prefix)
        _synchronize(self.block_decoder.device)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if decoded_prefix.shape[1] != pixel_end:
            raise RuntimeError(
                f"V5 prefix decoder returned {decoded_prefix.shape[1]} frames, "
                f"expected {pixel_end}"
            )
        self.writers["pred"].write(decoded_prefix[0, pixel_start:pixel_end])

        frame_indices = latent_keyframe_indices(
            latent_start, denoised_latent.shape[1]
        )
        pred_keyframes = decoded_prefix[0, list(frame_indices)].float()
        source_keyframes = (
            self.source_rgb[0, :, list(frame_indices)]
            .permute(1, 0, 2, 3)
            .float().add(1).mul(0.5).clamp(0, 1)
        )
        self._write_v4_video(block_index, "post_source_keyframes.mp4", source_keyframes)
        self._write_v4_video(block_index, "post_pred_keyframes.mp4", pred_keyframes)

        target_poses = self.target_c2w[list(frame_indices)]
        source_poses = self.source_c2w[list(frame_indices)].to(
            self.memory.device, dtype=torch.float32
        )
        repeated_k = self.K.unsqueeze(0).repeat(len(frame_indices), 1, 1)
        paired_rgb = torch.stack((source_keyframes, pred_keyframes), dim=1).flatten(0, 1)
        paired_poses = torch.stack((source_poses, target_poses), dim=1).flatten(0, 1)
        paired_k = self.K.unsqueeze(0).repeat(paired_rgb.shape[0], 1, 1)

        mapanything_started = time.perf_counter()
        pred_batch = self.depth_estimator.estimate_views(
            pred_keyframes,
            intrinsics_t33=repeated_k,
            camera_c2w_t44=target_poses,
            output_device=self.memory.device,
        )
        paired_batch = self.depth_estimator.estimate_views(
            paired_rgb,
            intrinsics_t33=paired_k,
            camera_c2w_t44=paired_poses,
            output_device=self.memory.device,
        )
        _synchronize(self.depth_estimator.device)
        mapanything_ms = (time.perf_counter() - mapanything_started) * 1000.0

        pred_score, pred_keep, pred_quality, pred_point_stats = (
            self._evaluate_mapanything_candidate(
                pred_batch, list(range(len(frame_indices))), frame_indices
            )
        )
        paired_slots = list(range(1, paired_rgb.shape[0], 2))
        paired_score, paired_keep, paired_quality, paired_point_stats = (
            self._evaluate_mapanything_candidate(
                paired_batch, paired_slots, frame_indices
            )
        )
        if paired_score < pred_score:
            selected_name = "paired"
            selected_batch = paired_batch
            selected_slots = paired_slots
            selected_keep = paired_keep
            selected_quality = paired_quality
            selected_point_stats = paired_point_stats
        else:
            selected_name = "pred_only"
            selected_batch = pred_batch
            selected_slots = list(range(len(frame_indices)))
            selected_keep = pred_keep
            selected_quality = pred_quality
            selected_point_stats = pred_point_stats

        accepted = (
            selected_quality["reference_consistent_ratio"]
            >= self.mapanything_min_consistent_ratio
        )
        update_stats = {
            "raw_valid_points": 0,
            "batch_voxels": 0,
            "points_before": self.memory.point_count,
            "points_after": self.memory.point_count,
            "evicted_voxels": 0,
        }
        fusion_started = time.perf_counter()
        if accepted:
            selected_points = selected_batch.points[selected_slots]
            selected_colors = selected_batch.colors[selected_slots]
            for local_index, frame_index in enumerate(frame_indices):
                keep = selected_keep[local_index]
                self.mapanything_chunks[int(frame_index)] = (
                    selected_points[local_index][keep].detach().cpu(),
                    selected_colors[local_index][keep].detach().cpu(),
                )
            chunk_points = [item[0] for item in self.mapanything_chunks.values()]
            chunk_colors = [item[1] for item in self.mapanything_chunks.values()]
            candidate_map = self.memory.empty_like()
            update_stats = candidate_map.update_points(
                torch.cat(chunk_points, dim=0),
                torch.cat(chunk_colors, dim=0),
            )
            self.memory = candidate_map
            self.observed_keyframes.update({
                int(frame_index): selected_batch.camera_c2w[pred_slot].detach().cpu()
                for frame_index, pred_slot in zip(frame_indices, selected_slots)
            })
        _synchronize(self.memory.device)
        voxel_fusion_ms = (time.perf_counter() - fusion_started) * 1000.0

        block_dir = self._v4_block_dir(block_index)
        save_started = time.perf_counter()
        self.memory.save_ply(os.path.join(block_dir, "post_point_map.ply"))
        np.savez_compressed(
            os.path.join(block_dir, "post_mapanything_cameras.npz"),
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            source_c2w=source_poses.detach().cpu().numpy().astype(np.float32),
            planned_c2w=target_poses.detach().cpu().numpy().astype(np.float32),
            pred_only_c2w=pred_batch.camera_c2w.detach().cpu().numpy().astype(np.float32),
            paired_pred_c2w=paired_batch.camera_c2w[paired_slots]
            .detach().cpu().numpy().astype(np.float32),
            selected_branch=np.asarray(selected_name),
            accepted=np.asarray(accepted),
        )
        pointcloud_save_ms = (time.perf_counter() - save_started) * 1000.0

        metric = self._metric_for_block(block_index)
        cuda_allocated_peak_gb, cuda_reserved_peak_gb = _cuda_peak_gb(
            self.memory.device
        )
        metric.update({
            "dit_ms": float(dit_ms),
            "decode_ms": decode_ms,
            "depth_backend": self.depth_backend,
            "depth_estimation_ms": mapanything_ms,
            "mapanything_ms": mapanything_ms,
            "mapanything_window_frames": list(frame_indices),
            "mapanything_input_views_pred_only": len(frame_indices),
            "mapanything_input_views_paired": paired_rgb.shape[0],
            "mapanything_selected_branch": selected_name,
            "mapanything_block_accepted": accepted,
            "mapanything_min_consistent_ratio": self.mapanything_min_consistent_ratio,
            "pred_only_quality": pred_quality,
            "paired_quality": paired_quality,
            "voxel_size": self.memory.voxel_size,
            "voxel_count": self.memory.point_count,
            "accepted_frame_chunks": len(self.mapanything_chunks),
            "voxel_fusion_ms": voxel_fusion_ms,
            "pointcloud_save_ms": pointcloud_save_ms,
            "memory_update_ms": voxel_fusion_ms,
            "pointcloud_total_ms": mapanything_ms + voxel_fusion_ms,
            "memory_update_mode": self.memory_update_mode,
            "memory_map_mode": self.memory_map_mode,
            "memory_write_contract": "bounded_current_block_source+pred_joint_pred_points_only",
            "mapanything_peak_memory_gb": float(
                getattr(self.depth_estimator, "last_peak_memory_gb", 0.0)
            ),
            "depth_backend_peak_memory_gb": float(
                getattr(self.depth_estimator, "last_peak_memory_gb", 0.0)
            ),
            "cuda_allocated_peak_gb": cuda_allocated_peak_gb,
            "cuda_reserved_peak_gb": cuda_reserved_peak_gb,
            **selected_point_stats,
            **update_stats,
        })
        self._write_v4_metrics(block_index, metric)

    @torch.inference_mode()
    def _output_callback_overlap_v4(
        self,
        *,
        block_index: int,
        latent_start: int,
        denoised_latent: torch.Tensor,
        pixel_start: int,
        pixel_end: int,
        dit_ms: float,
    ) -> None:
        self.generated_latent_blocks.append(denoised_latent.detach().cpu())
        latent_prefix = torch.cat(self.generated_latent_blocks, dim=1)

        _synchronize(self.block_decoder.device)
        t_decode = time.perf_counter()
        decoded_prefix = self.block_decoder.decode_prefix(latent_prefix)
        _synchronize(self.block_decoder.device)
        decode_ms = (time.perf_counter() - t_decode) * 1000.0
        if decoded_prefix.shape[1] != pixel_end:
            raise RuntimeError(
                f"V4 prefix decoder returned {decoded_prefix.shape[1]} frames, "
                f"expected {pixel_end}"
            )
        self.writers["pred"].write(decoded_prefix[0, pixel_start:pixel_end])

        historical_frame_indices = latent_keyframe_indices(0, latent_prefix.shape[1])
        historical_keyframes = decoded_prefix[0, list(historical_frame_indices)]
        self._write_v4_video(
            block_index, "post_keyframes.mp4", historical_keyframes
        )

        _synchronize(self.depth_estimator.device)
        t_da3 = time.perf_counter()
        (
            reconstruction_rgb,
            da3_depth,
            da3_intrinsics,
            da3_w2c,
        ) = self.depth_estimator.estimate_block(
            historical_keyframes,
            output_size=(self.height, self.width),
            output_device=self.memory.device,
        )
        _synchronize(self.depth_estimator.device)
        _synchronize(self.memory.device)
        da3_ms = (time.perf_counter() - t_da3) * 1000.0

        _synchronize(self.memory.device)
        t_rebuild = time.perf_counter()
        local_points = []
        local_valid = []
        reference_points = []
        reference_valid = []
        for history_index, frame_index in enumerate(historical_frame_indices):
            points, valid = backproject_world_grid(
                da3_depth[history_index],
                da3_intrinsics[history_index],
                w2c=da3_w2c[history_index],
            )
            local_points.append(points)
            local_valid.append(valid)
            points, valid = backproject_world_grid(
                self.reference_depth[frame_index].to(
                    self.memory.device, dtype=torch.float32
                ),
                self.K,
                c2w=self.target_c2w[frame_index],
            )
            reference_points.append(points)
            reference_valid.append(valid)

        local_points = torch.stack(local_points)
        local_valid = torch.stack(local_valid)
        reference_points = torch.stack(reference_points)
        reference_valid = torch.stack(reference_valid)
        reference_depth = self.reference_depth[list(historical_frame_indices)].to(
            self.memory.device, dtype=torch.float32
        )
        reference_mask = (
            self.reference_mask[0, 0, list(historical_frame_indices)]
            .to(self.memory.device) > 0
        )
        correspondence_valid = local_valid & reference_valid & reference_mask
        correspondence_count = int(correspondence_valid.sum().item())

        registration = None
        registration_error = None
        canonical_points = None
        canonical_c2w = torch.empty(
            (0, 4, 4), device=self.memory.device, dtype=torch.float32
        )
        point_stats = {
            "da3_valid_points": int(local_valid.sum().item()),
            "reference_uncovered_points": 0,
            "reference_covered_points": 0,
            "reference_geometry_consistent_points": 0,
            "reference_geometry_rejected_points": 0,
            "kept_points": 0,
        }
        update_stats = {
            "raw_valid_points": 0,
            "batch_voxels": 0,
            "points_before": self.memory.point_count,
            "points_after": self.memory.point_count,
            "evicted_voxels": 0,
        }
        voxel_fusion_ms = 0.0
        try:
            registration = estimate_similarity_registration(
                local_points,
                reference_points,
                correspondence_valid,
                min_correspondences=min(
                    4096,
                    max(
                        128,
                        int(len(historical_frame_indices) * self.height * self.width * 0.01),
                    ),
                ),
            )
            canonical_points = apply_similarity(
                local_points,
                registration.scale,
                registration.rotation,
                registration.translation,
            )
            point_keep, point_stats = select_v4_runtime_points(
                canonical_points,
                reference_points,
                local_valid,
                reference_depth,
                reference_valid,
                reference_mask,
                voxel_size=self.memory.voxel_size,
                voxel_factor=self.geometry_voxel_factor,
                relative_depth_ratio=self.geometry_depth_ratio,
            )
            canonical_c2w = torch.stack([
                transform_da3_c2w(
                    extrinsic,
                    registration.scale,
                    registration.rotation,
                    registration.translation,
                )
                for extrinsic in da3_w2c
            ])
            candidate_map = self.memory.empty_like()
            _synchronize(self.memory.device)
            t_fusion = time.perf_counter()
            update_stats = candidate_map.update_points(
                canonical_points,
                reconstruction_rgb.permute(0, 2, 3, 1),
                point_keep,
            )
            _synchronize(self.memory.device)
            voxel_fusion_ms = (time.perf_counter() - t_fusion) * 1000.0

            self.memory = candidate_map
            self.observed_keyframes = {
                int(frame_index): pose.detach().cpu()
                for frame_index, pose in zip(historical_frame_indices, canonical_c2w)
            }
        except (RuntimeError, ValueError) as error:
            registration_error = str(error)
        _synchronize(self.memory.device)
        pointcloud_rebuild_ms = (time.perf_counter() - t_rebuild) * 1000.0

        block_dir = self._v4_block_dir(block_index)
        _synchronize(self.memory.device)
        t_save = time.perf_counter()
        self.memory.save_ply(os.path.join(block_dir, "post_point_map.ply"))
        np.savez_compressed(
            os.path.join(block_dir, "post_da3_cameras.npz"),
            frame_indices=np.asarray(historical_frame_indices, dtype=np.int64),
            da3_intrinsics=da3_intrinsics.detach().cpu().numpy().astype(np.float32),
            da3_w2c=da3_w2c.detach().cpu().numpy().astype(np.float32),
            planned_c2w=self.target_c2w[list(historical_frame_indices)]
            .detach().cpu().numpy().astype(np.float32),
            canonical_da3_c2w=canonical_c2w.detach().cpu().numpy().astype(np.float32),
            sim3_scale=np.float32(np.nan if registration is None else registration.scale),
            sim3_rotation=(
                np.full((3, 3), np.nan, dtype=np.float32)
                if registration is None
                else registration.rotation.detach().cpu().numpy().astype(np.float32)
            ),
            sim3_translation=(
                np.full((3,), np.nan, dtype=np.float32)
                if registration is None
                else registration.translation.detach().cpu().numpy().astype(np.float32)
            ),
        )
        pointcloud_save_ms = (time.perf_counter() - t_save) * 1000.0

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
            "historical_keyframe_count": len(historical_frame_indices),
            "historical_frame_indices": list(historical_frame_indices),
            "update_frame_indices": list(
                latent_keyframe_indices(latent_start, denoised_latent.shape[1])
            ),
            "update_frames": int(denoised_latent.shape[1]),
            "registration_accepted": registration is not None and registration_error is None,
            "rebuild_succeeded": registration is not None and registration_error is None,
            "rebuild_error": registration_error,
            "sim3_correspondence_count": correspondence_count,
            "sim3_sampled_correspondence_count": (
                0 if registration is None else registration.sampled_count
            ),
            "sim3_inlier_count": 0 if registration is None else registration.inlier_count,
            "sim3_inlier_ratio": 0.0 if registration is None else registration.inlier_ratio,
            "sim3_rmse": None if registration is None else registration.rmse,
            "sim3_normalized_rmse": (
                None if registration is None else registration.normalized_rmse
            ),
            "sim3_scale": None if registration is None else registration.scale,
            "voxel_size": self.memory.voxel_size,
            "voxel_count": self.memory.point_count,
            "pointcloud_rebuild_ms": pointcloud_rebuild_ms,
            "voxel_fusion_ms": voxel_fusion_ms,
            "pointcloud_save_ms": pointcloud_save_ms,
            "memory_update_ms": pointcloud_rebuild_ms,
            "pointcloud_total_ms": da3_ms + pointcloud_rebuild_ms,
            "memory_update_mode": self.memory_update_mode,
            "memory_map_mode": self.memory_map_mode,
            "geometry_voxel_factor": self.geometry_voxel_factor,
            "geometry_depth_ratio": self.geometry_depth_ratio,
            "da3_peak_memory_gb": float(
                getattr(self.depth_estimator, "last_peak_memory_gb", 0.0)
            ),
            "cuda_allocated_peak_gb": cuda_allocated_peak_gb,
            "cuda_reserved_peak_gb": cuda_reserved_peak_gb,
            **point_stats,
            **update_stats,
        })
        self._write_v4_metrics(block_index, metric)

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
        single_keyframe_selected = False
        if self.memory_map_mode == "overlap_voxel_v3_2":
            target_index = int(self.memory_single_keyframe_index)
            single_keyframe_selected = (
                not self.single_keyframe_attempted
                and pixel_start <= target_index < pixel_end
            )
            if not single_keyframe_selected:
                metric = self._metric_for_block(block_index)
                cuda_allocated_peak_gb, cuda_reserved_peak_gb = _cuda_peak_gb(
                    self.memory.device
                )
                point_count = self.memory.point_count
                metric.update({
                    "dit_ms": float(dit_ms),
                    "decode_ms": decode_ms,
                    "depth_backend": self.depth_backend,
                    "depth_estimation_ms": 0.0,
                    "da3_ms": 0.0,
                    "registration_ms": 0.0,
                    "pointcloud_build_ms": 0.0,
                    "voxel_fusion_ms": 0.0,
                    "memory_update_ms": 0.0,
                    "pointcloud_total_ms": 0.0,
                    "memory_update_mode": self.memory_update_mode,
                    "memory_map_mode": self.memory_map_mode,
                    "update_frame_indices": [],
                    "update_frames": 0,
                    "anchor_count": 0,
                    "anchor_frame_index_input": self.anchor_frame_index,
                    "anchor_frame_index_output": self.anchor_frame_index,
                    "multi_anchor_retry_implemented": False,
                    "da3_window_frames": 0,
                    "registration_source": None,
                    "registration_accepted": False,
                    "registration_error": None,
                    "registration_scale": None,
                    "registration_rmse": None,
                    "registration_normalized_rmse": None,
                    "registration_correspondences": 0,
                    "registration_inliers": 0,
                    "registration_inlier_ratio": 0.0,
                    "registration_scale_jump": None,
                    "registration_normalized_rmse_limit": 0.20,
                    "raw_valid_points": 0,
                    "batch_voxels": 0,
                    "points_before": point_count,
                    "points_after": point_count,
                    "evicted_voxels": 0,
                    "single_keyframe_target_index": target_index,
                    "single_keyframe_selected": False,
                    "single_keyframe_attempted_total": self.single_keyframe_attempted,
                    "single_keyframe_written_total": self.single_keyframe_written,
                    "da3_peak_memory_gb": float(
                        getattr(self.depth_estimator, "last_peak_memory_gb", 0.0)
                    ),
                    "cuda_allocated_peak_gb": cuda_allocated_peak_gb,
                    "cuda_reserved_peak_gb": cuda_reserved_peak_gb,
                })
                return
            self.single_keyframe_attempted = True
            update_frame_indices = (target_index,)
        else:
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
        # V3.2 has one mandated single-frame build and no anchor retry. Keep its
        # quality gate bounded, but slightly wider than the multi-frame V3 gate.
        normalized_rmse_limit = (
            0.20 if self.memory_map_mode == "overlap_voxel_v3_2" else 0.15
        )
        if accepted and registration.normalized_rmse > normalized_rmse_limit:
            accepted = False
            registration_error = (
                f"normalized RMSE {registration.normalized_rmse:.6f} exceeds "
                f"{normalized_rmse_limit:.2f}"
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

        if self.memory_map_mode == "overlap_voxel_v3_2":
            self.single_keyframe_written = bool(accepted)

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
            "registration_normalized_rmse_limit": normalized_rmse_limit,
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
            "single_keyframe_target_index": self.memory_single_keyframe_index,
            "single_keyframe_selected": single_keyframe_selected,
            "single_keyframe_attempted_total": self.single_keyframe_attempted,
            "single_keyframe_written_total": self.single_keyframe_written,
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

        if self.memory_map_mode in {"overlap_voxel_v4", "overlap_voxel_v5"}:
            return self.summary()

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
            "single_keyframe_index": self.memory_single_keyframe_index,
            "single_keyframe_attempted": self.single_keyframe_attempted,
            "single_keyframe_written": self.single_keyframe_written,
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
                bool(block.get(
                    "mapanything_block_accepted",
                    block.get("registration_accepted", False),
                ))
                for block in self.metrics
            )),
            "mapanything_ms": float(sum(float(block.get("mapanything_ms", 0.0)) for block in self.metrics)),
            **stage_totals,
            **extra_v3_totals,
        }
