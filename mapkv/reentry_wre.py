from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from mapkv_proto.pose_utils import (
    rotation_geodesic,
    scale_intrinsics,
    to_cut3r_c2w,
)

from .reentry_memory import ReentryMemoryLifecycle, erode_binary_coverage
from .surfel_index import SurfelIndex
from .warp_reencode import (
    WarpReencodePlan,
    _surfel_coverage_for_pose,
    build_rotation_target_to_source_grid,
    infer_intrinsic_image_hw,
    latent_to_rgb_index,
    load_intrinsics,
    reference_protected_coverage,
    strong_memory_coverage,
    warp_latent,
)


def _block_rgb_indices(
    chunk: int,
    *,
    frames_per_block: int,
    latent_length: int,
    rgb_length: int,
) -> tuple[int, ...]:
    start = int(chunk) * int(frames_per_block)
    return tuple(
        latent_to_rgb_index(index, latent_length, rgb_length)
        for index in range(start, start + int(frames_per_block))
    )


def _candidate_chunks(
    index: SurfelIndex,
    anchor_indices: np.ndarray,
    *,
    start_chunk: int,
    stop_chunk_exclusive: int,
    reference_blind_threshold: float,
) -> list[int]:
    # Lifecycle is anchored to ``anchor_indices``, but source selection must
    # not depend on successful cross-view fusion into those exact cells.  A
    # good first-pass observation can contain the same surface in a nearby,
    # not-yet-merged cell.  Keep the parameter to make this distinction
    # explicit and verify that the anchor group itself is non-empty.
    if not len(np.asarray(anchor_indices).reshape(-1)):
        return []
    candidates: set[int] = set()
    for cell in index.cells:
        for chunk in cell.observing_chunks:
            chunk = int(chunk)
            if not start_chunk <= chunk < stop_chunk_exclusive:
                continue
            if (
                float(cell.reference_blind_at_write.get(chunk, -1.0))
                >= float(reference_blind_threshold)
            ):
                candidates.add(chunk)
    return sorted(candidates)


def score_view_adaptive_observations(
    *,
    surfel_index: SurfelIndex,
    candidate_chunks: Iterable[int],
    surface_group_indices: np.ndarray,
    target_chunk: int,
    query_pose: np.ndarray,
    intrinsics: np.ndarray,
    source_image_hw: tuple[int, int],
    target_hw: tuple[int, int],
    poses: np.ndarray,
    latent_length: int,
    rgb_length: int,
    frames_per_block: int,
    reference_blind_threshold: float,
) -> list[dict]:
    """Score first-episode observations and return a deterministic ranking."""
    query_pose = np.asarray(query_pose, dtype=np.float64)
    camera = query_pose[:3, 3]
    ranking = []
    for chunk in sorted({int(item) for item in candidate_chunks}):
        generated = surfel_index.generated_only_cell_indices(
            chunk,
            reference_blind_threshold=reference_blind_threshold,
        )
        eligible = generated.astype(np.int32)
        shared = np.intersect1d(
            np.asarray(surface_group_indices, dtype=np.int32),
            generated,
            assume_unique=False,
        ).astype(np.int32)
        visible = surfel_index.visible_cells(
            query_pose,
            intrinsics,
            target_hw,
            source_image_size=source_image_hw,
            eligible_max_chunk=int(target_chunk) - 2,
            eligible_chunks={chunk},
            eligible_indices=eligible,
            use_occlusion=True,
        )
        pixels = np.asarray(visible["pixels"], dtype=np.int32)
        unique = np.unique(
            np.asarray(visible["indices"], dtype=np.int32)
        )
        coverage = float(len(pixels) / max(target_hw[0] * target_hw[1], 1))
        alignments = []
        qualities = []
        margins = []
        for cell_index in unique:
            cell = surfel_index.cells[int(cell_index)]
            source_view = cell.view_dirs.get(chunk)
            if source_view is not None:
                target_view = camera - np.asarray(cell.xyz, dtype=np.float64)
                target_norm = float(np.linalg.norm(target_view))
                source_norm = float(np.linalg.norm(source_view))
                if target_norm > 1e-8 and source_norm > 1e-8:
                    alignments.append(
                        max(
                            0.0,
                            float(
                                np.dot(
                                    target_view / target_norm,
                                    np.asarray(source_view) / source_norm,
                                )
                            ),
                        )
                    )
            qualities.append(float(cell.chunk_weights.get(chunk, 0.0)))
            margins.append(float(cell.image_center_margins.get(chunk, 0.0)))
        surface_alignment = float(np.mean(alignments)) if alignments else 1.0
        observation_quality = float(np.mean(qualities)) if qualities else 0.0
        center_margin = float(np.mean(margins)) if margins else 0.0
        source_rgb = _block_rgb_indices(
            chunk,
            frames_per_block=frames_per_block,
            latent_length=latent_length,
            rgb_length=rgb_length,
        )
        target_rgb = _block_rgb_indices(
            target_chunk,
            frames_per_block=frames_per_block,
            latent_length=latent_length,
            rgb_length=rgb_length,
        )
        source_pose = poses[int(source_rgb[len(source_rgb) // 2])]
        target_pose = poses[int(target_rgb[len(target_rgb) // 2])]
        rotation_degrees = float(
            np.degrees(
                rotation_geodesic(
                    source_pose[:3, :3], target_pose[:3, :3]
                )
            )
        )
        # Under pure rotation, cell→camera rays are nearly invariant because
        # the optical center is fixed.  Camera-orientation alignment is the
        # meaningful view term; a 10° Gaussian gives a strong but smooth
        # preference to an observation near the re-entry view.
        camera_alignment = float(
            np.exp(-0.5 * (rotation_degrees / 10.0) ** 2)
        )
        view_alignment = camera_alignment * surface_alignment
        score = (
            coverage
            * view_alignment
            * observation_quality
            * center_margin
        )
        ranking.append(
            {
                "chunk_id": chunk,
                "score": score,
                "visible_coverage": coverage,
                "visible_support_pixels": int(len(pixels)),
                "visible_surfels": int(len(unique)),
                "view_alignment": view_alignment,
                "camera_orientation_alignment": camera_alignment,
                "surface_view_alignment": surface_alignment,
                "observation_quality": observation_quality,
                "image_center_margin": center_margin,
                "rotation_distance_degrees": rotation_degrees,
                "eligible_generated_only_surfels": int(len(eligible)),
                "shared_anchor_surfels": int(len(shared)),
            }
        )
    ranking.sort(
        key=lambda item: (
            -float(item["score"]),
            float(item["rotation_distance_degrees"]),
            int(item["chunk_id"]),
        )
    )
    return ranking


def build_reentry_virtual_recent_plans(
    *,
    source_latents_path: str | Path,
    source_chunk: int,
    observation_start_chunk: int,
    target_pose_path: str | Path,
    intrinsics_path: str | Path,
    surfel_index_path: str | Path,
    surfel_sequence_path: str | Path,
    latent_length: int,
    rgb_length: int,
    frames_per_block: int,
    latent_hw: tuple[int, int],
    image_hw: tuple[int, int],
    selected_layers: Iterable[int],
    selected_step_indices: Iterable[int],
    alpha: float,
    feather_kernel: int,
    device: torch.device,
    dtype: torch.dtype,
    vae,
    reference_mask_latent: torch.Tensor,
    absent_blocks: int = 2,
    view_adaptive_source: bool = False,
    edge_safe: bool = False,
    reference_protection_dilation_kernel: int = 3,
    generated_only_threshold: float = 0.5,
    memory_dilation_kernel: int = 3,
    query_feather_kernel: int = 3,
    warp_valid_erosion_kernel: int = 3,
) -> tuple[dict[int, WarpReencodePlan], list[dict]]:
    """Build one-shot, lifecycle-aware RGB-Warp→VAE Recent corrections."""
    if vae is None:
        raise ValueError("Re-entry WRE requires the native Wan VAE")
    if feather_kernel < 1 or feather_kernel % 2 == 0:
        raise ValueError("feather_kernel must be a positive odd integer")
    if reference_mask_latent.ndim != 5:
        raise ValueError("reference_mask_latent must be [B,F,C,H,W]")
    if int(reference_mask_latent.shape[1]) != int(latent_length):
        raise ValueError("reference mask latent length does not match generation")
    if observation_start_chunk > int(source_chunk):
        raise ValueError("observation_start_chunk must include the anchor source")

    source_path = Path(source_latents_path).resolve()
    all_latents = torch.load(source_path, map_location="cpu", weights_only=True)
    if isinstance(all_latents, dict):
        all_latents = all_latents["pred_latents"]
    if all_latents.ndim != 5 or int(all_latents.shape[1]) != int(latent_length):
        raise ValueError("Source latent sequence shape is incompatible")
    num_blocks = int(latent_length) // int(frames_per_block)
    poses = np.load(Path(target_pose_path).resolve()).astype(np.float64)
    if poses.shape != (rgb_length, 4, 4):
        raise ValueError(
            f"Pose artifact {poses.shape} != expected {(rgb_length, 4, 4)}"
        )
    raw_intrinsics = load_intrinsics(intrinsics_path)
    intrinsic_source_hw = infer_intrinsic_image_hw(raw_intrinsics)
    image_intrinsics = scale_intrinsics(
        raw_intrinsics, source_hw=intrinsic_source_hw, target_hw=image_hw
    )
    latent_intrinsics = scale_intrinsics(
        image_intrinsics, source_hw=image_hw, target_hw=latent_hw
    )
    sequence_path = Path(surfel_sequence_path).resolve()
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if sequence.get("cut3r_predicted_pose_used_for_map", True):
        raise ValueError("Re-entry WRE requires known-pose CUT3R geometry")
    sequence_frames = {
        int(item["chunk_id"]): item for item in sequence["frames"]
    }
    if int(source_chunk) not in sequence_frames:
        raise ValueError(f"Anchor source chunk {source_chunk} is absent")
    geometry_frame = sequence_frames[int(source_chunk)]
    surfel_intrinsics = np.asarray(
        geometry_frame["intrinsics"], dtype=np.float64
    )
    surfel_source_hw = tuple(int(value) for value in geometry_frame["shape"])
    surfel_index = SurfelIndex.load(Path(surfel_index_path).resolve())
    anchor_indices = surfel_index.generated_only_cell_indices(
        source_chunk,
        reference_blind_threshold=generated_only_threshold,
    )
    if not len(anchor_indices):
        raise RuntimeError(
            "The anchor has no generated-only surfels at the requested threshold"
        )

    def source_indices(chunk: int) -> tuple[int, ...]:
        return _block_rgb_indices(
            chunk,
            frames_per_block=frames_per_block,
            latent_length=latent_length,
            rgb_length=rgb_length,
        )

    anchor_rgb = source_indices(source_chunk)
    anchor_poses = poses[np.asarray(anchor_rgb, dtype=np.int64)]
    first_eligible = int(source_chunk) + 2
    target_info: dict[int, dict] = {}
    for target_chunk in range(first_eligible, num_blocks):
        target_start = target_chunk * frames_per_block
        target_stop = target_start + frames_per_block
        target_rgb = source_indices(target_chunk)
        target_poses = poses[np.asarray(target_rgb, dtype=np.int64)]
        raw_surfel_masks = []
        anchor_valid = []
        for anchor_pose, target_pose in zip(anchor_poses, target_poses):
            _, valid, _ = build_rotation_target_to_source_grid(
                anchor_pose,
                target_pose,
                latent_intrinsics,
                latent_intrinsics,
                latent_hw,
            )
            surfel_mask, _ = _surfel_coverage_for_pose(
                surfel_index=surfel_index,
                source_chunk=int(source_chunk),
                target_chunk=target_chunk,
                query_pose=to_cut3r_c2w(target_pose),
                intrinsics=surfel_intrinsics,
                source_image_hw=surfel_source_hw,
                target_hw=latent_hw,
                generated_only=False,
                eligible_indices=anchor_indices,
            )
            anchor_valid.append(valid)
            raw_surfel_masks.append(surfel_mask)
        raw_history = torch.stack(raw_surfel_masks).unsqueeze(0).to(device)
        warp_valid = torch.stack(anchor_valid).unsqueeze(0).to(device)
        target_reference_mask = reference_mask_latent[
            :, target_start:target_stop
        ].to(device=device)
        ref_valid = (
            ((target_reference_mask.float() + 1.0) * 0.5)
            .clamp(0, 1)
            .mean(dim=2)
        )
        ref_protected = reference_protected_coverage(
            target_reference_mask,
            dilation_kernel=reference_protection_dilation_kernel,
        ).to(device=device)
        anchor_need = (
            raw_history * warp_valid * (1.0 - ref_protected)
        )
        target_info[target_chunk] = {
            "target_start": target_start,
            "target_stop": target_stop,
            "target_rgb": target_rgb,
            "target_poses": target_poses,
            "raw_history": raw_history,
            "warp_valid": warp_valid,
            "reference_valid": ref_valid,
            "reference_protected": ref_protected,
            "anchor_need": anchor_need,
        }

    absent_run = 0
    first_absence_start = None
    for target_chunk, info in target_info.items():
        if bool(info["raw_history"].any()):
            absent_run = 0
        else:
            absent_run += 1
            if absent_run == int(absent_blocks):
                first_absence_start = target_chunk - int(absent_blocks) + 1
                break
    if first_absence_start is None:
        raise RuntimeError("Historical surface group never becomes absent")
    candidates = _candidate_chunks(
        surfel_index,
        anchor_indices,
        start_chunk=int(observation_start_chunk),
        stop_chunk_exclusive=int(first_absence_start),
        reference_blind_threshold=generated_only_threshold,
    )
    if int(source_chunk) not in candidates:
        raise RuntimeError("Anchor source is missing from first-episode candidates")

    lifecycle = ReentryMemoryLifecycle(absent_blocks=absent_blocks)
    decisions: dict[int, dict] = {}
    for target_chunk, info in target_info.items():
        visible = bool(info["raw_history"].any())
        read_support = bool(info["anchor_need"].any())
        decision = lifecycle.step(
            visible=visible,
            read_support=read_support,
        )
        decisions[target_chunk] = {
            "target_chunk": int(target_chunk),
            "state_before": decision.state_before.value,
            "state_after": decision.state_after.value,
            "historical_visible": visible,
            "historical_visibility_fraction": float(
                info["raw_history"].float().mean().item()
            ),
            "absence_count": decision.absence_count,
            "episode_id": decision.episode_id,
            "read_support": read_support,
            "read_long_term": decision.read_long_term,
            "anchor_need_fraction": float(
                info["anchor_need"].float().mean().item()
            ),
            "first_absence_start_chunk": int(first_absence_start),
            "first_episode_candidate_chunks": candidates,
        }

    source_cache: dict[int, dict] = {}

    def load_source(chunk: int) -> dict:
        chunk = int(chunk)
        if chunk in source_cache:
            return source_cache[chunk]
        start = chunk * frames_per_block
        stop = start + frames_per_block
        latent = all_latents[:, start:stop].to(device=device, dtype=dtype)
        with torch.no_grad():
            rgb = (
                vae.decode_to_pixel(latent, use_cache=False) * 0.5 + 0.5
            ).clamp(0, 1)
        rgb_indices = source_indices(chunk)
        result = {
            "chunk": chunk,
            "latent": latent,
            "rgb": rgb,
            "rgb_indices": rgb_indices,
            "poses": poses[np.asarray(rgb_indices, dtype=np.int64)],
        }
        source_cache[chunk] = result
        return result

    plans: dict[int, WarpReencodePlan] = {}
    selections: list[dict] = []
    selected_for_episode: dict[int, int] = {}
    selected_layers = tuple(int(layer) for layer in selected_layers)
    selected_steps = tuple(int(step) for step in selected_step_indices)
    for target_chunk, info in target_info.items():
        timeline = dict(decisions[target_chunk])
        timeline.update(
            {
                "activation_policy": "reentry_once_then_native_recent_handoff",
                "absent_blocks_required": int(absent_blocks),
                "view_adaptive_source": bool(view_adaptive_source),
                "edge_safe_support": bool(edge_safe),
                "generated_only_threshold": float(
                    generated_only_threshold
                ),
                "status": "memory_off_lifecycle",
                "selected_source_chunk": None,
                "read_coverage_fraction": 0.0,
            }
        )
        if not timeline["read_long_term"]:
            selections.append(timeline)
            continue
        episode_id = int(timeline["episode_id"])
        ranking = []
        if view_adaptive_source:
            target_pose = to_cut3r_c2w(
                info["target_poses"][len(info["target_poses"]) // 2]
            )
            ranking = score_view_adaptive_observations(
                surfel_index=surfel_index,
                candidate_chunks=candidates,
                surface_group_indices=anchor_indices,
                target_chunk=target_chunk,
                query_pose=target_pose,
                intrinsics=surfel_intrinsics,
                source_image_hw=surfel_source_hw,
                target_hw=latent_hw,
                poses=poses,
                latent_length=latent_length,
                rgb_length=rgb_length,
                frames_per_block=frames_per_block,
                reference_blind_threshold=generated_only_threshold,
            )
            if not ranking or float(ranking[0]["score"]) <= 0:
                raise RuntimeError("No valid first-episode observation for re-entry")
            selected_source = int(ranking[0]["chunk_id"])
        else:
            selected_source = int(source_chunk)
        selected_for_episode.setdefault(episode_id, selected_source)
        if selected_for_episode[episode_id] != selected_source:
            raise RuntimeError("Historical source switched within a re-entry episode")
        source = load_source(selected_source)
        selected_generated = surfel_index.generated_only_cell_indices(
            selected_source,
            reference_blind_threshold=generated_only_threshold,
        )
        selected_group = selected_generated.astype(np.int32)
        if not len(selected_group):
            raise RuntimeError("Selected source has no shared generated-only surfels")

        historical_grids = []
        raw_masks = []
        warp_valid_masks = []
        homographies = []
        rotations = []
        translations = []
        geometry_audits = []
        for source_pose, target_pose in zip(
            source["poses"], info["target_poses"]
        ):
            grid, valid, homography = build_rotation_target_to_source_grid(
                source_pose,
                target_pose,
                latent_intrinsics,
                latent_intrinsics,
                latent_hw,
            )
            surfel_mask, geometry = _surfel_coverage_for_pose(
                surfel_index=surfel_index,
                source_chunk=selected_source,
                target_chunk=target_chunk,
                query_pose=to_cut3r_c2w(target_pose),
                intrinsics=surfel_intrinsics,
                source_image_hw=surfel_source_hw,
                target_hw=latent_hw,
                generated_only=False,
                eligible_indices=selected_group,
            )
            historical_grids.append(grid)
            raw_masks.append(surfel_mask)
            warp_valid_masks.append(valid)
            homographies.append(homography)
            rotations.append(
                float(
                    np.degrees(
                        rotation_geodesic(
                            source_pose[:3, :3], target_pose[:3, :3]
                        )
                    )
                )
            )
            translations.append(
                float(
                    np.linalg.norm(
                        source_pose[:3, 3] - target_pose[:3, 3]
                    )
                )
            )
            geometry_audits.append(geometry)
        raw_history = torch.stack(raw_masks).unsqueeze(0).to(device)
        warp_valid = torch.stack(warp_valid_masks).unsqueeze(0).to(device)
        accepted_warp_valid = (
            erode_binary_coverage(
                warp_valid, kernel_size=warp_valid_erosion_kernel
            )
            if edge_safe
            else warp_valid
        )
        history_coverage = raw_history * accepted_warp_valid
        hard_read = (
            history_coverage
            * (1.0 - info["reference_protected"])
        )
        if edge_safe:
            memory_coverage = (hard_read > 0).to(torch.float32)
            query_gate_mode = "surfel_edge_safe_source_protected"
        else:
            memory_coverage = strong_memory_coverage(
                hard_read, memory_dilation_kernel
            )
            memory_coverage = (
                memory_coverage * (1.0 - info["reference_protected"])
            )
            query_gate_mode = "surfel_source_protected"
        if not bool(memory_coverage.any()):
            raise RuntimeError("Lifecycle scheduled an empty re-entry read")

        rgb_frames = int(source["rgb"].shape[1])
        source_dense = np.rint(
            np.linspace(
                source["rgb_indices"][0],
                source["rgb_indices"][-1],
                rgb_frames,
            )
        ).astype(np.int64)
        target_dense = np.rint(
            np.linspace(
                info["target_rgb"][0],
                info["target_rgb"][-1],
                rgb_frames,
            )
        ).astype(np.int64)
        rgb_grids = []
        rgb_valid = []
        for source_index, target_index in zip(source_dense, target_dense):
            grid, valid, _ = build_rotation_target_to_source_grid(
                poses[int(source_index)],
                poses[int(target_index)],
                image_intrinsics,
                image_intrinsics,
                image_hw,
            )
            rgb_grids.append(grid)
            rgb_valid.append(valid)
        rgb_padding_mode = "border" if edge_safe else "zeros"
        warped_rgb = warp_latent(
            source["rgb"],
            torch.stack(rgb_grids).to(device=device),
            padding_mode=rgb_padding_mode,
        )
        with torch.no_grad():
            historical_target = vae.encode_to_latent(
                (warped_rgb * 2.0 - 1.0)
                .to(dtype=dtype)
                .permute(0, 2, 1, 3, 4)
            ).to(device=device, dtype=dtype)
        if historical_target.shape != source["latent"].shape:
            raise RuntimeError(
                "RGB-warp VAE encode changed historical block shape"
            )
        preview_index = rgb_frames // 2
        plan = WarpReencodePlan(
            target_block=target_chunk,
            source_chunk=selected_source,
            historical_latent=historical_target,
            target_to_source_grid=torch.stack(historical_grids).to(device),
            coverage=memory_coverage,
            selected_layers=selected_layers,
            selected_step_indices=selected_steps,
            alpha=float(alpha),
            source_rgb_indices=tuple(source["rgb_indices"]),
            target_rgb_indices=tuple(info["target_rgb"]),
            source_to_target_homographies=tuple(homographies),
            source_target_rotation_degrees=tuple(rotations),
            source_target_translation=tuple(translations),
            hard_coverage=hard_read,
            history_coverage=history_coverage,
            reference_valid_coverage=info["reference_valid"],
            reference_protected_coverage=info["reference_protected"],
            need_coverage=hard_read,
            warp_valid_coverage=accepted_warp_valid,
            reentry_coverage=raw_history,
            safe_coverage=(hard_read if edge_safe else None),
            query_coverage=memory_coverage,
            query_feather_kernel=int(query_feather_kernel),
            reference_protection_kernel=int(
                reference_protection_dilation_kernel
            ),
            historical_representation="rgb_warp_vae",
            historical_is_target_aligned=True,
            rgb_padding_mode=rgb_padding_mode,
            rgb_preview_source=source["rgb"][:, preview_index],
            rgb_preview_target=warped_rgb[:, preview_index],
            rgb_warp_coverage_preview=torch.stack(rgb_valid)[preview_index],
            query_gate_mode=query_gate_mode,
            mode=(
                "edge_safe_view_adaptive_reentry_rgb_wre"
                if edge_safe
                else (
                    "view_adaptive_reentry_rgb_wre"
                    if view_adaptive_source
                    else "reentry_only_rgb_wre"
                )
            ),
            geometry_audit={
                "coordinate_frame": sequence.get("coordinate_frame"),
                "pose_source": "known_control_c2w_to_cut3r_c2w",
                "surface_group_anchor_chunk": int(source_chunk),
                "surface_group_cells": int(len(anchor_indices)),
                "selected_observation_cells": int(len(selected_group)),
                "generated_only_threshold": float(
                    generated_only_threshold
                ),
                "edge_safe": bool(edge_safe),
                "warp_valid_erosion_kernel": (
                    int(warp_valid_erosion_kernel) if edge_safe else None
                ),
                "rgb_padding_mode": rgb_padding_mode,
                "per_frame": geometry_audits,
            },
            audit={
                "memory_lifecycle": "reentry_once_then_native_recent_handoff",
                "episode_id": episode_id,
                "selected_source_locked_for_episode": True,
                "view_adaptive_source": bool(view_adaptive_source),
                "observation_ranking": ranking,
                "first_episode_candidate_chunks": candidates,
                "first_absence_start_chunk": int(first_absence_start),
            },
        )
        plans[target_chunk] = plan
        timeline.update(
            {
                "status": "scheduled_reentry_read",
                "selected_source_chunk": selected_source,
                "source_locked_for_episode": True,
                "candidate_scores": ranking,
                "read_coverage_fraction": float(
                    memory_coverage.float().mean().item()
                ),
                "safe_support_fraction": float(
                    hard_read.float().mean().item()
                ),
                "selected_source_rotation_degrees": rotations,
                "rgb_padding_mode": rgb_padding_mode,
            }
        )
        selections.append(timeline)
    return plans, selections


__all__ = [
    "build_reentry_virtual_recent_plans",
    "score_view_adaptive_observations",
]
