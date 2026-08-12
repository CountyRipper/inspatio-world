"""Visibility-driven pure-rotation projection for Teacher v0."""

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

from .types import CameraBatch, Provenance, WorldObservation, WorldReadPacket


def _axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((rotation.trace() - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vector = torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    sine = torch.sin(angle)
    small = angle.abs() < 1e-5
    axis = torch.where(small, vector * 0.5, vector / (2.0 * sine.clamp_min(1e-7)))
    return axis * angle


class RotationProjector:
    """Project independent observations without any angle/coverage switch."""

    def __init__(self, *, exact_tolerance: float = 1e-5):
        self.exact_tolerance = exact_tolerance

    def _project_frame(
        self,
        latent: torch.Tensor,
        valid: torch.Tensor,
        confidence: torch.Tensor,
        memory_K: torch.Tensor,
        memory_c2w: torch.Tensor,
        query_K: torch.Tensor,
        query_c2w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = latent.shape[-2:]
        exact = (
            torch.allclose(memory_K, query_K, atol=self.exact_tolerance, rtol=0)
            and torch.allclose(memory_c2w, query_c2w, atol=self.exact_tolerance, rtol=0)
        )
        relative = torch.linalg.inv(query_c2w) @ memory_c2w
        rotation = relative[:3, :3]
        pose = torch.cat((relative[:3, 3], _axis_angle(rotation)))
        angle = torch.acos(((rotation.trace() - 1.0) * 0.5).clamp(-1.0, 1.0)).view(1)

        if exact:
            subpixel = torch.zeros(2, height, width, device=latent.device, dtype=latent.dtype)
            return latent, valid.bool(), confidence, pose, angle, subpixel

        y, x = torch.meshgrid(
            torch.arange(height, device=latent.device, dtype=torch.float32),
            torch.arange(width, device=latent.device, dtype=torch.float32),
            indexing="ij",
        )
        query_pixels = torch.stack((x, y, torch.ones_like(x)), dim=0).reshape(3, -1)

        # Requirement-facing forward homography: K_q R_q<-m inv(K_m).
        forward = query_K.float() @ rotation.float() @ torch.linalg.inv(memory_K.float())
        memory_pixels = torch.linalg.inv(forward) @ query_pixels
        denominator = memory_pixels[2]
        finite = torch.isfinite(memory_pixels).all(dim=0) & (denominator > 1e-7)
        memory_xy = memory_pixels[:2] / denominator.clamp_min(1e-7)
        mx = memory_xy[0].reshape(height, width)
        my = memory_xy[1].reshape(height, width)
        in_bounds = finite.reshape(height, width)
        in_bounds &= (mx >= 0) & (mx <= width - 1) & (my >= 0) & (my <= height - 1)

        grid = torch.stack(
            (
                2.0 * mx / max(width - 1, 1) - 1.0,
                2.0 * my / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        ).unsqueeze(0)
        projected = F.grid_sample(
            latent.unsqueeze(0).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].to(latent.dtype)
        projected_valid = F.grid_sample(
            valid.unsqueeze(0).float(),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0] > 0.5
        projected_confidence = F.grid_sample(
            confidence.unsqueeze(0).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].to(confidence.dtype)
        projected_valid &= in_bounds.unsqueeze(0)
        projected = projected * projected_valid.to(projected.dtype)
        projected_confidence = projected_confidence * projected_valid.to(projected_confidence.dtype)
        subpixel = torch.stack((mx - mx.round(), my - my.round()), dim=0).to(latent.dtype)
        subpixel = subpixel * projected_valid.to(subpixel.dtype)
        return projected, projected_valid, projected_confidence, pose, angle, subpixel

    def project(
        self,
        observations: Iterable[WorldObservation],
        query_camera: CameraBatch,
    ) -> WorldReadPacket:
        observations = tuple(observations)
        if not observations:
            raise ValueError("project requires at least one observation")
        if query_camera.batch_size != 1:
            raise ValueError("Teacher v0 projection expects batch size one")

        frames = query_camera.frames
        if any(observation.clean_latent.shape[0] != frames for observation in observations):
            raise ValueError("observation and query frame counts must match")

        candidate_values = []
        candidate_valid = []
        candidate_confidence = []
        candidate_pose = []
        candidate_angle = []
        candidate_subpixel = []
        for observation in observations:
            values, validity, confidence, poses, angles, offsets = [], [], [], [], [], []
            observation_confidence = (
                observation.static_confidence * observation.geometry_confidence
            )
            for frame in range(frames):
                result = self._project_frame(
                    observation.clean_latent[frame],
                    observation.valid[frame],
                    observation_confidence[frame],
                    observation.K[frame],
                    observation.c2w_W0[frame],
                    query_camera.K[0, frame],
                    query_camera.c2w_W0[0, frame],
                )
                value, valid, conf, pose, angle, offset = result
                values.append(value)
                validity.append(valid)
                confidence.append(conf)
                poses.append(pose)
                angles.append(angle)
                offsets.append(offset)
            candidate_values.append(torch.stack(values))
            candidate_valid.append(torch.stack(validity))
            candidate_confidence.append(torch.stack(confidence))
            candidate_pose.append(torch.stack(poses))
            candidate_angle.append(torch.stack(angles))
            candidate_subpixel.append(torch.stack(offsets))

        values = torch.stack(candidate_values, dim=0).unsqueeze(0)
        valid = torch.stack(candidate_valid, dim=0).unsqueeze(0).bool()
        confidence = torch.stack(candidate_confidence, dim=0).unsqueeze(0)
        relative_pose = torch.stack(candidate_pose, dim=0).unsqueeze(0)
        view_angle = torch.stack(candidate_angle, dim=0).unsqueeze(0)
        subpixel = torch.stack(candidate_subpixel, dim=0).unsqueeze(0)

        # Trusted static source owns conflicts. Generated candidates remain
        # independent everywhere else; no raw-latent averaging occurs.
        source_valid = torch.zeros_like(valid[:, :1])
        for index, observation in enumerate(observations):
            if int(observation.provenance) == int(Provenance.SOURCE):
                source_valid |= valid[:, index:index + 1]
        for index, observation in enumerate(observations):
            if int(observation.provenance) == int(Provenance.GENERATED):
                valid[:, index:index + 1] &= ~source_valid
                values[:, index:index + 1] *= valid[:, index:index + 1].to(values.dtype)
                confidence[:, index:index + 1] *= valid[:, index:index + 1].to(confidence.dtype)

        mask4 = valid.to(values.dtype).expand(-1, -1, -1, 4, -1, -1)
        condition = torch.cat((mask4, values), dim=3)
        authority = torch.zeros_like(valid, dtype=torch.long)
        provenance = torch.empty(
            1, len(observations), dtype=torch.long, device=values.device
        )
        for index, observation in enumerate(observations):
            authority[:, index].fill_(observation.authority)
            authority[:, index] *= valid[:, index].to(authority.dtype)
            provenance[:, index].fill_(int(observation.provenance))

        return WorldReadPacket(
            candidate_20ch=condition,
            valid=valid,
            authority=authority,
            confidence=confidence,
            relative_pose=relative_pose,
            view_angle=view_angle,
            subpixel_offset=subpixel,
            provenance=provenance,
            observation_ids=tuple(observation.observation_id for observation in observations),
        )
