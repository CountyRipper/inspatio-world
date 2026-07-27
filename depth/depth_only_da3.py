"""In-memory DA3 RGB-D estimation for historical point-cloud memory."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DA3DepthOnlyEstimator:
    """Run DA3 on a generated keyframe or a jointly estimated RGB block.

    The keyframe method returns depth only. The full-block method also returns
    DA3's processed RGB, intrinsics, and extrinsics. Historical point placement
    remains anchored by the known target pose supplied by InSpatio so that each
    independently estimated block cannot introduce a new global gauge.
    """

    def __init__(self, model_path: str, device: torch.device):
        from depth_anything_3.api import DepthAnything3

        self.device = torch.device(device)
        self.model = DepthAnything3.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.last_native_shape = None
        self.last_peak_memory_gb = 0.0
        self.last_processed_shape = None
        self.last_intrinsics_shape = None
        self.last_extrinsics_shape = None

    @torch.inference_mode()
    def estimate(
        self,
        rgb_chw: torch.Tensor,
        output_size: Tuple[int, int],
        output_device: torch.device,
    ) -> torch.Tensor:
        """Estimate depth for RGB in [0,1] and resize it to (height, width)."""
        if rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3:
            raise ValueError(f"Expected RGB [3,H,W], got {tuple(rgb_chw.shape)}")

        frame = (
            rgb_chw.detach().float().clamp(0, 1)
            .permute(1, 2, 0)
            .mul(255).round().to(torch.uint8).cpu().numpy()
        )
        result = self.model.inference([frame], use_ray_pose=False)
        depth = result.depth[0]
        if torch.is_tensor(depth):
            depth = depth.detach().float().cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)
        self.last_native_shape = tuple(depth.shape)

        height, width = output_size
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(np.ascontiguousarray(depth)).to(output_device, dtype=torch.float32)

    @torch.inference_mode()
    def estimate_block(
        self,
        rgb_tchw: torch.Tensor,
        output_size: Tuple[int, int],
        output_device: torch.device,
    ):
        """Jointly estimate a generated RGB block's full DA3 geometry.

        Returns processed RGB, resized depth maps, correspondingly scaled DA3
        intrinsics, and raw DA3 w2c extrinsics. Global point placement remains
        anchored by the caller's known InSpatio target c2w trajectory.
        """
        if rgb_tchw.ndim != 4 or rgb_tchw.shape[1] != 3:
            raise ValueError(f"Expected RGB [T,3,H,W], got {tuple(rgb_tchw.shape)}")

        frames = (
            rgb_tchw.detach().float().clamp(0, 1)
            .permute(0, 2, 3, 1)
            .mul(255).round().to(torch.uint8).cpu().numpy()
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        result = self.model.inference(list(frames), use_ray_pose=False)

        depth = result.depth
        processed_images = result.processed_images
        intrinsics = result.intrinsics
        extrinsics = result.extrinsics
        if torch.is_tensor(depth):
            depth = depth.detach().float().cpu().numpy()
        if torch.is_tensor(processed_images):
            processed_images = processed_images.detach().cpu().numpy()
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.detach().float().cpu().numpy()
        if torch.is_tensor(extrinsics):
            extrinsics = extrinsics.detach().float().cpu().numpy()

        depth = np.asarray(depth, dtype=np.float32)
        processed_images = np.asarray(processed_images)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        extrinsics = np.asarray(extrinsics, dtype=np.float32)
        if depth.ndim != 3 or depth.shape[0] != rgb_tchw.shape[0]:
            raise RuntimeError(f"Unexpected DA3 block depth shape: {depth.shape}")
        if processed_images.shape != (*depth.shape, 3):
            raise RuntimeError(
                f"Unexpected DA3 processed RGB shape: {processed_images.shape}"
            )
        if intrinsics.shape != (depth.shape[0], 3, 3):
            raise RuntimeError(f"Unexpected DA3 block intrinsic shape: {intrinsics.shape}")
        if extrinsics.shape != (depth.shape[0], 3, 4):
            raise RuntimeError(f"Unexpected DA3 block extrinsic shape: {extrinsics.shape}")

        native_height, native_width = depth.shape[-2:]
        self.last_native_shape = tuple(depth.shape)
        self.last_processed_shape = tuple(processed_images.shape)
        self.last_intrinsics_shape = tuple(intrinsics.shape)
        self.last_extrinsics_shape = tuple(extrinsics.shape)
        if self.device.type == "cuda":
            self.last_peak_memory_gb = (
                torch.cuda.max_memory_allocated(self.device) / 1024**3
            )

        height, width = output_size
        rgb_tensor = torch.from_numpy(
            np.ascontiguousarray(processed_images)
        ).to(output_device, dtype=torch.float32)
        if rgb_tensor.max().item() > 1.5:
            rgb_tensor = rgb_tensor / 255.0
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2).clamp(0, 1)
        if (native_height, native_width) != (height, width):
            rgb_tensor = F.interpolate(
                rgb_tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).to(
            output_device, dtype=torch.float32
        )
        if (native_height, native_width) != (height, width):
            depth_tensor = F.interpolate(
                depth_tensor.unsqueeze(1),
                size=(height, width),
                mode="nearest",
            ).squeeze(1)

        intrinsics = intrinsics.copy()
        intrinsics[:, 0, :] *= width / native_width
        intrinsics[:, 1, :] *= height / native_height
        intrinsic_tensor = torch.from_numpy(intrinsics).to(
            output_device, dtype=torch.float32
        )
        extrinsic_tensor = torch.from_numpy(extrinsics).to(
            output_device, dtype=torch.float32
        )
        return rgb_tensor, depth_tensor, intrinsic_tensor, extrinsic_tensor
