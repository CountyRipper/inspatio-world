"""In-memory MapAnything point-map inference for InSpatio experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MapAnythingBatch:
    """Canonicalized dense outputs for one ordered view list."""

    points: torch.Tensor
    colors: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    intrinsics: torch.Tensor
    camera_c2w: torch.Tensor
    processed_size: tuple[int, int]
    inference_ms: float


def resize_for_mapanything(
    rgb_tchw: torch.Tensor,
    intrinsics_t33: Optional[torch.Tensor] = None,
    *,
    target_width: int = 518,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Resize RGB/K with MapAnything's 14-pixel alignment convention."""
    if rgb_tchw.ndim != 4 or rgb_tchw.shape[1] != 3:
        raise ValueError(f"Expected RGB [T,3,H,W], got {tuple(rgb_tchw.shape)}")
    _, _, height, width = rgb_tchw.shape
    resized_height = round(height * (target_width / width) / 14) * 14
    resized_height = max(14, resized_height)
    crop_top = max(0, (resized_height - 518) // 2)
    output_height = min(resized_height, 518)

    rgb = F.interpolate(
        rgb_tchw.float(),
        size=(resized_height, target_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    if resized_height > 518:
        rgb = rgb[:, :, crop_top : crop_top + 518]

    if intrinsics_t33 is None:
        return rgb, None
    intrinsics = intrinsics_t33.detach().float().clone()
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(rgb.shape[0], -1, -1).clone()
    if intrinsics.shape != (rgb.shape[0], 3, 3):
        raise ValueError(
            f"Expected intrinsics {(rgb.shape[0], 3, 3)}, got {tuple(intrinsics.shape)}"
        )
    intrinsics[:, 0, :] *= target_width / width
    intrinsics[:, 1, :] *= resized_height / height
    intrinsics[:, 1, 2] -= crop_top
    return rgb, intrinsics


def transform_points_c2w(points: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Transform row-vector point grids from a camera/reference frame to world."""
    if points.shape[-1] != 3 or c2w.shape != (4, 4):
        raise ValueError("Expected point grid [...,3] and c2w [4,4]")
    return points @ c2w[:3, :3].T + c2w[:3, 3]


class MapAnythingPointEstimator:
    """Pose-conditioned MapAnything wrapper with a canonical world contract."""

    backend_name = "mapanything"

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        *,
        confidence_percentile: float = 10.0,
    ):
        from mapanything.models import MapAnything
        from uniception.models.encoders.image_normalizations import (
            IMAGE_NORMALIZATION_DICT,
        )

        self.device = torch.device(device)
        self.model = MapAnything.from_pretrained(model_path).to(self.device)
        self.model.eval()
        normalization = IMAGE_NORMALIZATION_DICT["dinov2"]
        self.mean = torch.as_tensor(
            normalization.mean, dtype=torch.float32
        ).detach().clone().view(1, 3, 1, 1)
        self.std = torch.as_tensor(
            normalization.std, dtype=torch.float32
        ).detach().clone().view(1, 3, 1, 1)
        self.confidence_percentile = float(confidence_percentile)
        self.last_peak_memory_gb = 0.0
        self.last_native_shape = None

    @torch.inference_mode()
    def estimate_views(
        self,
        rgb_tchw: torch.Tensor,
        *,
        intrinsics_t33: Optional[torch.Tensor] = None,
        camera_c2w_t44: Optional[torch.Tensor] = None,
        output_device: Optional[torch.device] = None,
    ) -> MapAnythingBatch:
        """Jointly estimate ordered views and return points in canonical world."""
        import time

        rgb, resized_intrinsics = resize_for_mapanything(rgb_tchw, intrinsics_t33)
        view_count = rgb.shape[0]
        if camera_c2w_t44 is not None:
            camera_c2w_t44 = camera_c2w_t44.detach().float()
            if camera_c2w_t44.shape != (view_count, 4, 4):
                raise ValueError(
                    f"Expected poses {(view_count, 4, 4)}, got {tuple(camera_c2w_t44.shape)}"
                )
            anchor_c2w = camera_c2w_t44[0].to(self.device)
        else:
            anchor_c2w = torch.eye(4, device=self.device, dtype=torch.float32)

        rgb = rgb.to(self.device)
        normalized = (rgb - self.mean.to(self.device)) / self.std.to(self.device)
        views = []
        for view_index in range(view_count):
            view = {
                "img": normalized[view_index : view_index + 1],
                "true_shape": np.int32([normalized.shape[-2:]]),
                "idx": view_index,
                "instance": str(view_index),
                "data_norm_type": ["dinov2"],
            }
            if resized_intrinsics is not None:
                view["intrinsics"] = resized_intrinsics[
                    view_index : view_index + 1
                ].to(self.device)
            if camera_c2w_t44 is not None:
                view["camera_poses"] = camera_c2w_t44[
                    view_index : view_index + 1
                ].to(self.device)
                view["is_metric_scale"] = torch.tensor(
                    [True], device=self.device
                )
            views.append(view)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        predictions = self.model.infer(
            views,
            memory_efficient_inference=view_count > 24,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=self.confidence_percentile > 0,
            confidence_percentile=self.confidence_percentile,
            use_multiview_confidence=False,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        points, colors, valid, confidence, predicted_c2w = [], [], [], [], []
        for prediction in predictions:
            local_points = prediction["pts3d"][0].float()
            points.append(transform_points_c2w(local_points, anchor_c2w))
            colors.append(prediction["img_no_norm"][0].float().clamp(0, 1))
            mask = prediction.get("mask")
            if mask is None:
                mask = torch.ones_like(prediction["conf"][0], dtype=torch.bool)
            else:
                mask = mask[0, ..., 0].bool()
            mask &= torch.isfinite(local_points).all(dim=-1)
            valid.append(mask)
            confidence.append(prediction["conf"][0].float())
            local_c2w = prediction.get("camera_poses")
            if local_c2w is None:
                local_c2w = torch.eye(
                    4, device=self.device, dtype=torch.float32
                ).unsqueeze(0)
            predicted_c2w.append(anchor_c2w @ local_c2w[0].float())

        output_device = self.device if output_device is None else torch.device(output_device)
        result = MapAnythingBatch(
            points=torch.stack(points).to(output_device),
            colors=torch.stack(colors).to(output_device),
            valid=torch.stack(valid).to(output_device),
            confidence=torch.stack(confidence).to(output_device),
            intrinsics=(
                torch.stack([prediction["intrinsics"][0].float() for prediction in predictions])
                if resized_intrinsics is None
                else resized_intrinsics.to(self.device)
            ).to(output_device),
            camera_c2w=torch.stack(predicted_c2w).to(output_device),
            processed_size=(int(rgb.shape[-2]), int(rgb.shape[-1])),
            inference_ms=inference_ms,
        )
        self.last_native_shape = tuple(result.points.shape)
        if self.device.type == "cuda":
            self.last_peak_memory_gb = float(
                torch.cuda.max_memory_allocated(self.device) / 1024**3
            )
        return result
