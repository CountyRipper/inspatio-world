"""Causal tensor adapter for incremental SLAM3R reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from utils.overlap_da3_registration import backproject_world_grid


@dataclass(frozen=True)
class CenterCropTransform:
    source_height: int
    source_width: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    crop_size: int = 224

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedSlam3RFrame:
    rgb_crop: torch.Tensor
    model_image: torch.Tensor
    crop: CenterCropTransform


@dataclass
class ReferenceGeometry:
    points: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor
    mask: torch.Tensor
    intrinsic: torch.Tensor


@dataclass
class Slam3RFrameOutput:
    frame_index: int
    rgb_crop: torch.Tensor
    points_world: torch.Tensor
    confidence: torch.Tensor
    i2p_confidence_mean: float
    l2w_confidence_mean: float
    retrieved_frame_indices: tuple[int, ...]
    buffer_frame_indices: tuple[int, ...]
    crop: CenterCropTransform
    initial_frame: bool


def _resize_shape(height: int, width: int, size: int) -> tuple[int, int]:
    if height <= 0 or width <= 0 or size <= 0:
        raise ValueError("Image dimensions and target size must be positive")
    if width >= height:
        return size, int(round(size * width / height))
    return int(round(size * height / width)), size


def prepare_slam3r_frame(
    rgb_chw: torch.Tensor,
    *,
    device: torch.device | str,
    size: int = 224,
) -> PreparedSlam3RFrame:
    """Apply SLAM3R's PIL-Lanczos short-edge resize and center crop."""
    if rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3:
        raise ValueError(f"Expected RGB [3,H,W], got {tuple(rgb_chw.shape)}")
    device = torch.device(device)
    rgb = rgb_chw.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
    height, width = rgb.shape[-2:]
    resized_height, resized_width = _resize_shape(height, width, size)
    crop_top = (resized_height - size) // 2
    crop_left = (resized_width - size) // 2
    image = PIL.Image.fromarray(
        (rgb.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    )
    image = image.resize(
        (resized_width, resized_height), PIL.Image.Resampling.LANCZOS
    )
    image = image.crop(
        (crop_left, crop_top, crop_left + size, crop_top + size)
    )
    crop = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    crop = crop.to(device=device, dtype=torch.float32).div(255.0).contiguous()
    if crop.shape[-2:] != (size, size):
        raise RuntimeError(f"SLAM3R crop has unexpected shape {tuple(crop.shape)}")
    transform = CenterCropTransform(
        source_height=height,
        source_width=width,
        resized_height=resized_height,
        resized_width=resized_width,
        crop_top=crop_top,
        crop_left=crop_left,
        crop_size=size,
    )
    return PreparedSlam3RFrame(
        rgb_crop=crop,
        model_image=crop.mul(2).sub(1).unsqueeze(0),
        crop=transform,
    )


def crop_intrinsic(
    intrinsic: torch.Tensor,
    crop: CenterCropTransform,
) -> torch.Tensor:
    """Map a full-resolution pinhole intrinsic into the SLAM3R crop."""
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic [3,3], got {tuple(intrinsic.shape)}")
    sx = crop.resized_width / crop.source_width
    sy = crop.resized_height / crop.source_height
    transform = torch.tensor(
        [
            [sx, 0.0, -float(crop.crop_left)],
            [0.0, sy, -float(crop.crop_top)],
            [0.0, 0.0, 1.0],
        ],
        device=intrinsic.device,
        dtype=torch.float32,
    )
    return transform @ intrinsic.float()


@torch.inference_mode()
def prepare_reference_geometry(
    depth_hw: torch.Tensor,
    mask_hw: torch.Tensor,
    intrinsic: torch.Tensor,
    c2w: torch.Tensor,
    crop: CenterCropTransform,
    *,
    device: torch.device | str,
) -> ReferenceGeometry:
    """Build a canonical point grid matching one SLAM3R RGB crop."""
    device = torch.device(device)
    depth = depth_hw.detach().to(device=device, dtype=torch.float32)
    mask = mask_hw.detach().to(device=device)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth [H,W], got {tuple(depth.shape)}")
    if mask.ndim == 3 and mask.shape[0] in (1, 3):
        mask = mask[0]
    if mask.shape != depth.shape:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} differs from depth {tuple(depth.shape)}"
        )
    if depth.shape != (crop.source_height, crop.source_width):
        raise ValueError(
            f"Reference shape {tuple(depth.shape)} differs from crop source "
            f"{(crop.source_height, crop.source_width)}"
        )

    resized_depth = F.interpolate(
        depth[None, None],
        size=(crop.resized_height, crop.resized_width),
        mode="nearest",
    )[0, 0]
    resized_mask = F.interpolate(
        mask.float()[None, None],
        size=(crop.resized_height, crop.resized_width),
        mode="nearest",
    )[0, 0] > 0.5
    row = slice(crop.crop_top, crop.crop_top + crop.crop_size)
    col = slice(crop.crop_left, crop.crop_left + crop.crop_size)
    cropped_depth = resized_depth[row, col].contiguous()
    cropped_mask = resized_mask[row, col].contiguous()
    cropped_intrinsic = crop_intrinsic(
        intrinsic.detach().to(device=device, dtype=torch.float32), crop
    )
    points, valid = backproject_world_grid(
        cropped_depth,
        cropped_intrinsic,
        c2w=c2w.detach().to(device=device, dtype=torch.float32),
    )
    return ReferenceGeometry(
        points=points,
        depth=cropped_depth,
        valid=valid,
        mask=cropped_mask,
        intrinsic=cropped_intrinsic,
    )


class IncrementalSLAM3RState:
    """Minimal causal I2P/L2W state shared by offline and future online paths."""

    def __init__(
        self,
        i2p_model,
        l2w_model,
        *,
        device: torch.device | str,
        initial_winsize: int = 5,
        win_r: int = 3,
        num_scene_frame: int = 10,
        buffer_size: int = 30,
        conf_thres_i2p: float = 1.5,
        seed: int = 42,
    ):
        if initial_winsize < 3:
            raise ValueError("SLAM3R initialization requires at least three frames")
        if win_r < 0 or num_scene_frame <= 0:
            raise ValueError("Invalid SLAM3R window/reference settings")
        if 0 < buffer_size < initial_winsize:
            raise ValueError("buffer_size must hold the complete initial window")
        self.i2p_model = i2p_model
        self.l2w_model = l2w_model
        self.device = torch.device(device)
        self.initial_winsize = int(initial_winsize)
        self.win_r = int(win_r)
        self.num_scene_frame = int(num_scene_frame)
        self.buffer_size = int(buffer_size)
        self.conf_thres_i2p = float(conf_thres_i2p)
        self.rng = np.random.default_rng(seed)
        self.input_views: list[dict] = []
        self.prepared_frames: list[PreparedSlam3RFrame] = []
        self.frame_indices: list[int] = []
        self.buffering_set_ids: list[int] = []
        self.initialized = False

        from slam3r.utils.recon_utils import (
            i2p_inference_batch,
            l2w_inference,
            normalize_views,
        )

        self._i2p_inference_batch = i2p_inference_batch
        self._l2w_inference = l2w_inference
        self._normalize_views = normalize_views

    @classmethod
    def from_pretrained(
        cls,
        i2p_model_path: str,
        l2w_model_path: str,
        **kwargs,
    ) -> "IncrementalSLAM3RState":
        from slam3r.models import Image2PointsModel, Local2WorldModel

        device = torch.device(kwargs["device"])
        i2p_model = Image2PointsModel.from_pretrained(i2p_model_path)
        l2w_model = Local2WorldModel.from_pretrained(l2w_model_path)
        i2p_model.to(device).eval()
        l2w_model.to(device).eval()
        return cls(i2p_model, l2w_model, **kwargs)

    @torch.inference_mode()
    def _encode(self, prepared: PreparedSlam3RFrame, frame_index: int) -> dict:
        raw_view = {
            "img": prepared.model_image,
            "true_shape": torch.tensor(
                [[prepared.crop.crop_size, prepared.crop.crop_size]],
                device=self.device,
                dtype=torch.int32,
            ),
        }
        _, features, positions = self.i2p_model._encode_multiview(
            [raw_view],
            view_batchsize=1,
            normalize=False,
            silent=True,
        )
        return {
            "label": f"frame_{frame_index:06d}",
            "img_tokens": features[0],
            "true_shape": raw_view["true_shape"],
            "img_pos": positions[0],
        }

    @torch.inference_mode()
    def push_prepared(
        self,
        prepared: PreparedSlam3RFrame,
        frame_index: int,
    ) -> list[Slam3RFrameOutput]:
        if self.frame_indices and frame_index <= self.frame_indices[-1]:
            raise ValueError("SLAM3R frames must be pushed in strictly increasing order")
        self.input_views.append(self._encode(prepared, frame_index))
        self.prepared_frames.append(prepared)
        self.frame_indices.append(int(frame_index))
        current_id = len(self.input_views) - 1
        if not self.initialized:
            if len(self.input_views) < self.initial_winsize:
                return []
            return self._initialize()
        return [self._register_one(current_id)]

    @torch.inference_mode()
    def _initialize(self) -> list[Slam3RFrameOutput]:
        window = self.input_views[:self.initial_winsize]
        best_ref_id = 0
        best_confidence = -float("inf")
        for ref_id in range(self.initial_winsize):
            result = self._i2p_inference_batch(
                [window],
                self.i2p_model,
                ref_id=ref_id,
                tocpu=True,
                unsqueeze=False,
            )
            mean_confidence = torch.stack(
                [prediction["conf"].float().mean() for prediction in result["preds"]]
            ).mean().item()
            if mean_confidence > best_confidence:
                best_confidence = mean_confidence
                best_ref_id = ref_id

        result = self._i2p_inference_batch(
            [window],
            self.i2p_model,
            ref_id=best_ref_id,
            tocpu=False,
            unsqueeze=False,
        )["preds"]
        pointmaps, confidences, valid_masks = [], [], []
        for view_id, prediction in enumerate(result):
            point_key = "pts3d" if view_id == best_ref_id else "pts3d_in_other_view"
            pointmaps.append(prediction[point_key])
            confidences.append(prediction["conf"])
            valid_masks.append(prediction["conf"] > self.conf_thres_i2p)
        pointmaps = self._normalize_views(pointmaps, valid_masks)

        self.buffering_set_ids = list(range(self.initial_winsize))
        self.initialized = True
        outputs = []
        for view_id in range(self.initial_winsize):
            pointmap = pointmaps[view_id]
            pointmap[~valid_masks[view_id]] = 0
            self.input_views[view_id]["pts3d_world"] = pointmap
            confidence = confidences[view_id][0]
            outputs.append(
                self._make_output(
                    view_id,
                    pointmap[0],
                    confidence,
                    float(confidence.mean().item()),
                    (),
                    initial_frame=True,
                )
            )
        return outputs

    @torch.inference_mode()
    def _retrieve_buffer_ids(self, current_id: int) -> list[int]:
        candidates = list(self.buffering_set_ids)
        if len(candidates) <= self.num_scene_frame:
            return candidates
        views = [self.input_views[current_id]] + [
            self.input_views[index] for index in candidates
        ]
        scores = self.i2p_model.get_corr_score(views, ref_id=0, depth=2)
        scores = scores.float().mean(dim=(1, 2))
        selected_positions = torch.argsort(scores, descending=True)[
            :self.num_scene_frame
        ].detach().cpu().tolist()
        # Retrieval scores are candidate-local positions, not global frame IDs.
        return [candidates[position] for position in selected_positions]

    def _reference_ids(self, current_id: int) -> list[int]:
        ref_ids = self._retrieve_buffer_ids(current_id)
        for distance in range(1, self.win_r + 1):
            adjacent_id = current_id - distance
            if adjacent_id >= 0 and adjacent_id not in ref_ids:
                ref_ids.append(adjacent_id)
        return ref_ids

    @torch.inference_mode()
    def _register_one(self, current_id: int) -> Slam3RFrameOutput:
        ref_ids = self._reference_ids(current_id)
        local_views = [self.input_views[current_id]] + [
            self.input_views[index] for index in ref_ids
        ]
        i2p_prediction = self._i2p_inference_batch(
            [local_views],
            self.i2p_model,
            ref_id=0,
            tocpu=False,
            unsqueeze=False,
        )["preds"][0]
        i2p_confidence = i2p_prediction["conf"]
        valid = i2p_confidence > self.conf_thres_i2p
        local_points = self._normalize_views(
            [i2p_prediction["pts3d"]], [valid]
        )[0]
        local_points[~valid] = 0
        self.input_views[current_id]["pts3d_cam"] = local_points

        ref_views = [self.input_views[index] for index in ref_ids]
        l2w_output = self._l2w_inference(
            ref_views + [self.input_views[current_id]],
            self.l2w_model,
            ref_ids=list(range(len(ref_views))),
            normalize=False,
            device=str(self.device),
        )[-1]
        world_points = l2w_output["pts3d_in_other_view"]
        confidence = l2w_output["conf"][0]
        self.input_views[current_id]["pts3d_world"] = world_points
        self._update_buffer(current_id)
        return self._make_output(
            current_id,
            world_points[0],
            confidence,
            float(i2p_confidence.mean().item()),
            tuple(self.frame_indices[index] for index in ref_ids),
            initial_frame=False,
        )

    def _update_buffer(self, current_id: int) -> None:
        if self.buffer_size <= 0 or len(self.buffering_set_ids) < self.buffer_size:
            self.buffering_set_ids.append(current_id)
            return
        seen_count = current_id + 1
        if self.rng.random() < self.buffer_size / seen_count:
            replace_position = int(self.rng.integers(0, self.buffer_size))
            self.buffering_set_ids[replace_position] = current_id

    def _make_output(
        self,
        internal_id: int,
        points_world: torch.Tensor,
        confidence: torch.Tensor,
        i2p_confidence_mean: float,
        retrieved_frame_indices: tuple[int, ...],
        *,
        initial_frame: bool,
    ) -> Slam3RFrameOutput:
        return Slam3RFrameOutput(
            frame_index=self.frame_indices[internal_id],
            rgb_crop=self.prepared_frames[internal_id].rgb_crop,
            points_world=points_world,
            confidence=confidence,
            i2p_confidence_mean=float(i2p_confidence_mean),
            l2w_confidence_mean=float(confidence.mean().item()),
            retrieved_frame_indices=retrieved_frame_indices,
            buffer_frame_indices=tuple(
                self.frame_indices[index] for index in self.buffering_set_ids
            ),
            crop=self.prepared_frames[internal_id].crop,
            initial_frame=initial_frame,
        )
