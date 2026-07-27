"""Block condition/output controller for the RGB point-memory baseline."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import torch

from utils.historical_point_memory import (
    DenseGeneratedPointMemory,
    RGBPointMemory,
    VideoStreamWriter,
    calibrate_depth_scale,
    compute_depth_confidence,
    fuse_reference_and_history,
    latent_block_to_pixel_span,
    latent_keyframe_indices,
)
from utils.render_warper import convert_mask_video


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
        memory: RGBPointMemory | DenseGeneratedPointMemory,
        output_dir: str,
        output_prefix: str,
        rank: int,
        reference_map_path: str,
        memory_update_mode: str = "keyframe",
        memory_map_mode: str = "bounded_voxel",
        reference_depth_thw: Optional[torch.Tensor] = None,
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
        if memory_map_mode not in {"bounded_voxel", "dense_two_layer"}:
            raise ValueError(f"Unsupported memory map mode: {memory_map_mode}")
        if memory_map_mode == "dense_two_layer" and memory_update_mode not in {
            "latent_keyframe", "full_block"
        }:
            raise ValueError(
                "dense_two_layer requires memory_update_mode=latent_keyframe or full_block"
            )
        if memory_update_mode == "latent_keyframe" and memory_map_mode != "dense_two_layer":
            raise ValueError("latent_keyframe currently requires memory_map_mode=dense_two_layer")
        if memory_map_mode == "dense_two_layer" and self.reference_depth is None:
            raise ValueError("dense_two_layer requires aligned reference_depth_thw")
        self.memory_update_mode = memory_update_mode
        self.memory_map_mode = memory_map_mode
        self.save_diagnostics = save_diagnostics
        self.metrics = []
        self.closed = False
        self.previous_depth_scale = None
        self._historical_depth_by_block = {}

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
            "points_before_read": self.memory.point_count,
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

    def close(self) -> dict:
        if self.closed:
            return self.summary()
        self.closed = True
        for writer in self.writers.values():
            writer.close()

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
        return {
            "depth_backend": self.depth_backend,
            "memory_update_mode": self.memory_update_mode,
            "memory_map_mode": self.memory_map_mode,
            "num_blocks": len(self.metrics),
            "output_frames": output_frames,
            "final_point_count": self.memory.point_count,
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
            **stage_totals,
        }
