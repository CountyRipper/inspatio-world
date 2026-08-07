#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from phase1_lsm.adapter import ADAPTER_PARAMETER_COUNT
from phase1_lsm.data_prep import (
    SOURCE_SPECS,
    _load_first_240_geometry,
    _target_poses,
    sha256_file,
)
from phase1_lsm.latent_projection import identity_reprojection_error, project_memory_sequence
from phase1_lsm.nearview import (
    MIN_WIDE_OCCUPANCY,
    choose_wide_offset,
    projection_displacement_statistics,
    write_nearview_trajectory,
)
from phase1_lsm.trajectory import A_KEYFRAMES, APRIME_KEYFRAMES
from pipeline.causal_inference import CausalInferencePipeline
from scripts.phase1_lsm.run_sharedA_hardgate_5deg import (
    WAN_ROOT,
    cache_torch_equal,
    clone_cache,
    cpu_tensor,
    mask_video_for_vae,
    render_lossless_condition,
    run_blocks,
    save_shared_state,
    tensor_sha256,
)
from utils.render_warper import convert_mask_video


SCENES = ("S0", "S1")
BASE_QUERIES = (
    {"query": "exact", "offset": 0.0, "view_class": "exact"},
    {"query": "plus10", "offset": 10.0, "view_class": "medium"},
    {"query": "minus10", "offset": -10.0, "view_class": "medium"},
)


def encode_condition(
    pipeline: CausalInferencePipeline,
    condition: dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        render_video = condition["render"][None].permute(0, 2, 1, 3, 4).to(
            device=device, dtype=torch.bfloat16
        )
        mask_video = mask_video_for_vae(condition["mask"]).to(
            device=device, dtype=torch.bfloat16
        )
        pipeline.vae.model.clear_cache()
        render_latent = pipeline.vae.encode_to_latent(render_video).to(torch.bfloat16)
        pipeline.vae.model.clear_cache()
        mask_latent = convert_mask_video(mask_video).to(torch.bfloat16)
    if tuple(render_latent.shape) != (1, 60, 16, 60, 104):
        raise AssertionError(f"render latent shape: {render_latent.shape}")
    if tuple(mask_latent.shape) != (1, 60, 4, 60, 104):
        raise AssertionError(f"mask latent shape: {mask_latent.shape}")
    return render_latent, mask_latent


def target_pose_for_offset(
    scene_root: Path,
    offset: float,
    initial_c2w: torch.Tensor,
    device: torch.device,
    folder: str = "trajectories",
) -> tuple[Path, torch.Tensor]:
    sign = "plus" if offset > 0 else "minus" if offset < 0 else ""
    name = "exact" if offset == 0 else f"{sign}{abs(int(offset))}"
    trajectory = write_nearview_trajectory(
        scene_root / folder / f"{name}.txt", offset
    )
    target = torch.stack(_target_poses(trajectory, initial_c2w.to(device), device))
    return trajectory, target


def coverage_precheck(
    scene_root: Path,
    z_a: torch.Tensor,
    shared_depth: torch.Tensor,
    K: torch.Tensor,
    shared_source_c2w: torch.Tensor,
    initial_c2w: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    audit = {
        "performed_before_training": True,
        "threshold": MIN_WIDE_OCCUPANCY,
        "fallback": "signed 20 degrees to signed 15 degrees only",
        "directions": {},
    }
    for direction, name in ((1, "positive"), (-1, "negative")):
        candidate_coverages = {}
        candidate_targets = {}
        for magnitude in (20, 15):
            offset = float(direction * magnitude)
            _, target = target_pose_for_offset(
                scene_root, offset, initial_c2w, device, folder="coverage_trajectories"
            )
            projected, _, occupancy = project_memory_sequence(
                z_a,
                shared_depth,
                K,
                shared_source_c2w,
                target[APRIME_KEYFRAMES],
            )
            candidate_coverages[magnitude] = float(occupancy.float().mean())
            candidate_targets[magnitude] = target
            if torch.equal(projected, z_a):
                raise AssertionError(f"non-zero {offset} projection became identity")
        selected_offset, low_overlap = choose_wide_offset(
            direction,
            candidate_coverages[20],
            candidate_coverages[15],
        )
        selected_magnitude = abs(int(selected_offset))
        query = f"{'plus' if direction > 0 else 'minus'}{selected_magnitude}"
        rows.append({
            "query": query,
            "offset": selected_offset,
            "view_class": "wide",
            "eligible": not low_overlap,
            "low_no_overlap_diagnostic": low_overlap,
        })
        audit["directions"][name] = {
            "occupancy_at_20": candidate_coverages[20],
            "occupancy_at_15": candidate_coverages[15],
            "selected_offset": selected_offset,
            "selection_reason": (
                "20deg occupancy >=5%"
                if candidate_coverages[20] >= MIN_WIDE_OCCUPANCY
                else "20deg occupancy <5%; fixed fallback to 15deg"
            ),
            "low_no_overlap_diagnostic": low_overlap,
            "eligible_for_content_gate": not low_overlap,
        }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    root = Path(args.output_root)
    if root.exists():
        allowed_precreated = {"GPU_ATTEMPTS.md", "stdout.log", "stderr.log"}
        unexpected = {path.name for path in root.iterdir()} - allowed_precreated
        if unexpected:
            raise FileExistsError(f"refusing to overwrite {root}: {sorted(unexpected)}")
    else:
        root.mkdir(parents=True)
    (root / "samples").mkdir()
    (root / "outputs").mkdir()
    (root / "montages").mkdir()

    repo_root = Path(args.repo_root).resolve()
    expected_init = (
        repo_root
        / "artifacts/phase1_lsm/train/fixed8_projected/memory_adapter.safetensors"
    ).resolve()
    if Path(args.init_adapter).resolve() != expected_init:
        raise ValueError(f"initial adapter must be {expected_init}")
    checkpoint_hash = sha256_file(args.checkpoint)
    init_hash = sha256_file(args.init_adapter)
    command_log = root / "COMMAND_LOG.md"
    command_log.write_text(
        "# Phase 1 wide shared-A hard-gate command log\n\n"
        f"- Start: {datetime.now().astimezone().isoformat()}\n"
        f"- Unique scientific command: {shlex.join([sys.executable, *sys.argv])}\n"
        f"- CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        f"- Base checkpoint SHA256 before: {checkpoint_hash}\n"
        f"- Fixed8 projected adapter SHA256 before: {init_hash}\n"
        "- Scope: S0/S1, exact, +/-10, pre-training coverage-selected +/-20 or +/-15\n"
        "- Shared-A contract: one block-0:5 capture per scene, then five causal-state forks\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/default_config.yaml"),
        OmegaConf.load(repo_root / "configs/inference_1.3b.yaml"),
    )
    config.wan_model_folder = str(WAN_ROOT)
    config.generator.weight_list[0].path = str(WAN_ROOT)
    pipeline = CausalInferencePipeline(config, device=device)
    incompatible = pipeline.generator.load_state_dict(load_file(args.checkpoint), strict=False)
    if (
        set(incompatible.missing_keys) != {"model.memory_adapter.proj.weight"}
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.eval().requires_grad_(False)
    adapter = pipeline.generator.model.memory_adapter
    if (
        adapter.parameter_count != ADAPTER_PARAMETER_COUNT
        or torch.count_nonzero(adapter.proj.weight).item() != 0
    ):
        raise AssertionError("capture model adapter must be zero with 122,880 parameters")

    shared_audit: dict[str, object] = {
        "passed": True,
        "capture_scope": "one shared A per scene",
        "scenes": {},
    }
    projection_audit: dict[str, object] = {
        "passed": True,
        "coverage_check_before_training": True,
        "shared_projection_source": "per-scene unique z_A/A-depth/K/A-c2w",
        "scenes": {},
    }
    all_descriptors: list[dict[str, object]] = []

    for scene_index, scene in enumerate(SCENES):
        scene_started = time.perf_counter()
        scene_root = root / "scenes" / scene
        scene_root.mkdir(parents=True)
        spec = SOURCE_SPECS[scene]
        frames_np, depths_np, K_cpu, source_c2w_cpu = _load_first_240_geometry(
            spec["geometry"]
        )
        source_frames = torch.from_numpy(frames_np)
        source_depths = torch.from_numpy(depths_np)

        exact_trajectory, _ = target_pose_for_offset(
            scene_root, 0.0, source_c2w_cpu[0], device
        )
        exact_condition = render_lossless_condition(
            source_frames,
            source_depths,
            K_cpu,
            source_c2w_cpu,
            exact_trajectory,
            0.0,
            device,
        )

        source_video = source_frames[None].permute(0, 2, 1, 3, 4).to(
            device=device, dtype=torch.bfloat16
        )
        with torch.inference_mode():
            pipeline.vae.model.clear_cache()
            ref_latent = pipeline.vae.encode_to_latent(source_video).to(torch.bfloat16)
            pipeline.vae.model.clear_cache()
            exact_render_latent, exact_mask_latent = encode_condition(
                pipeline, exact_condition, device
            )
            prompt = json.loads(spec["json"].read_text())[0]["text"]
            conditional = pipeline.text_encoder([prompt])
        if tuple(ref_latent.shape) != (1, 60, 16, 60, 104):
            raise AssertionError(f"{scene}: source latent shape {ref_latent.shape}")

        torch.manual_seed(scene_index)
        torch.cuda.manual_seed_all(scene_index)
        rng_cpu_before_noise = torch.get_rng_state()
        rng_cuda_before_noise = torch.cuda.get_rng_state(device)
        noise = torch.randn(
            (1, 60, 16, 60, 104), device=device, dtype=torch.bfloat16
        )
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        output_shared = torch.zeros_like(noise)
        capture_count = [0]
        output_shared, shared_last, captured, _, _, _ = run_blocks(
            pipeline,
            noise,
            ref_latent,
            exact_render_latent,
            exact_mask_latent,
            conditional,
            pipeline.kv_cache1,
            output_shared,
            None,
            0,
            5,
            capture_count,
        )
        if capture_count[0] != 1 or set(captured) != {5}:
            raise AssertionError(f"{scene}: A capture count is not one")
        unique_z_a = captured[5]
        shared_prefix = output_shared[:, :18].clone()
        shared_cache = clone_cache(pipeline.kv_cache1)
        rng_cpu_at_fork = torch.get_rng_state()
        rng_cuda_at_fork = torch.cuda.get_rng_state(device)
        shared_depth = exact_condition["target_depth"][A_KEYFRAMES].to(device)
        shared_source_c2w = exact_condition["target_c2w"][A_KEYFRAMES].to(device)
        z_a_device = unique_z_a.to(device=device, dtype=torch.bfloat16)

        wide_queries, coverage_audit = coverage_precheck(
            scene_root,
            z_a_device,
            shared_depth,
            K_cpu.to(device),
            shared_source_c2w,
            source_c2w_cpu[0],
            device,
        )
        query_specs = [
            {**item, "eligible": True, "low_no_overlap_diagnostic": False}
            for item in BASE_QUERIES
        ] + wide_queries

        state_hashes = save_shared_state(
            scene_root,
            source_frames,
            exact_condition,
            shared_prefix,
            unique_z_a,
            ref_latent[:, :18],
            exact_render_latent[:, :18],
            exact_mask_latent[:, :18],
            shared_last,
            shared_cache,
            noise[:, 18:],
            rng_cpu_at_fork,
            rng_cuda_at_fork,
            K_cpu,
        )
        z_a_hash = tensor_sha256(unique_z_a)
        scene_shared_queries = {}
        scene_projection_queries = {}

        for query_spec in query_specs:
            query = str(query_spec["query"])
            offset = float(query_spec["offset"])
            if query == "exact":
                condition = exact_condition
                render_latent = exact_render_latent
                mask_latent = exact_mask_latent
            else:
                trajectory, _ = target_pose_for_offset(
                    scene_root, offset, source_c2w_cpu[0], device
                )
                condition = render_lossless_condition(
                    source_frames,
                    source_depths,
                    K_cpu,
                    source_c2w_cpu,
                    trajectory,
                    offset,
                    device,
                    exact_condition,
                )
                render_latent, mask_latent = encode_condition(
                    pipeline, condition, device
                )
                if not torch.equal(
                    render_latent[:, :18], exact_render_latent[:, :18]
                ):
                    raise AssertionError(f"{scene}/{query}: shared render latent changed")
                if not torch.equal(
                    mask_latent[:, :18], exact_mask_latent[:, :18]
                ):
                    raise AssertionError(f"{scene}/{query}: shared mask latent changed")

            branch_cache = clone_cache(shared_cache)
            branch_prefix = shared_prefix.clone()
            branch_last = shared_last.clone()
            pre_fork_equal = (
                torch.equal(branch_prefix, shared_prefix)
                and torch.equal(branch_last, shared_last)
                and cache_torch_equal(branch_cache, shared_cache)
            )
            if not pre_fork_equal:
                raise AssertionError(f"{scene}/{query}: pre-fork state changed")
            torch.set_rng_state(rng_cpu_at_fork)
            torch.cuda.set_rng_state(rng_cuda_at_fork, device)
            branch_output = torch.zeros_like(noise)
            branch_output[:, :18] = branch_prefix
            branch_output, _, branch_captured, step_inputs, transition_noises, step_timesteps = run_blocks(
                pipeline,
                noise,
                ref_latent,
                render_latent,
                mask_latent,
                conditional,
                branch_cache,
                branch_output,
                branch_last,
                6,
                19,
                capture_count,
            )
            if (
                set(branch_captured) != {13, 18, 19}
                or set(step_inputs) != {0, 1, 2, 3}
                or set(transition_noises) != {0, 1, 2}
            ):
                raise AssertionError(f"{scene}/{query}: continuation capture incomplete")
            if capture_count[0] != 1:
                raise AssertionError(f"{scene}: branch recaptured A")

            target_c2w = condition["target_c2w"]
            projected, memory_mask4, occupancy = project_memory_sequence(
                z_a_device,
                shared_depth,
                K_cpu.to(device),
                shared_source_c2w,
                target_c2w[APRIME_KEYFRAMES].to(device),
            )
            coverage = float(occupancy.float().mean())
            displacement = projection_displacement_statistics(
                shared_depth,
                K_cpu.to(device),
                shared_source_c2w,
                target_c2w[APRIME_KEYFRAMES].to(device),
                z_a_device.shape[-2:],
            )
            projection_equal = bool(torch.equal(projected, z_a_device))
            identity_overlap = None
            if offset == 0.0:
                identity_overlap = identity_reprojection_error(
                    z_a_device, projected, occupancy
                )
                if (
                    identity_overlap["max_abs_error"] != 0.0
                    or displacement["max_pixel_displacement"] > 1e-3
                ):
                    raise AssertionError(f"{scene}/exact: overlap identity changed")
            elif occupancy.any():
                if projection_equal or displacement["mean_pixel_displacement"] <= 0.0:
                    raise AssertionError(f"{scene}/{query}: non-zero projection is identity")
            elif not query_spec["low_no_overlap_diagnostic"]:
                raise AssertionError(f"{scene}/{query}: unexpected empty projection")
            pose = condition["pose_audit"]
            actual_yaw = float(np.mean(pose["actual_signed_yaw_delta_degrees"]))
            if abs(actual_yaw - offset) > 0.1:
                raise AssertionError(f"{scene}/{query}: actual yaw {actual_yaw}")

            sample_id = f"{scene}_{query}"
            tensors = {
                "z_A": unique_z_a.contiguous(),
                "z_B": branch_captured[13].contiguous(),
                "latent_prefix_0_18": cpu_tensor(branch_output[:, :57], torch.bfloat16),
                "block18_previous": branch_captured[18].contiguous(),
                "block19_base_render16": cpu_tensor(render_latent[:, 57:60], torch.bfloat16),
                "block19_base_mask4": cpu_tensor(mask_latent[:, 57:60], torch.bfloat16),
                "block19_ref16": cpu_tensor(ref_latent[:, 57:60], torch.bfloat16),
                "z_Aprime_no_memory": branch_captured[19].contiguous(),
                "projected_memory_latent16": cpu_tensor(projected, torch.bfloat16),
                "projected_memory_mask4": cpu_tensor(memory_mask4, torch.bfloat16),
                "projected_occupancy1": cpu_tensor(occupancy, torch.bool),
                "denoise_step_inputs": torch.stack([step_inputs[index] for index in range(4)]),
                "transition_noises": torch.stack([transition_noises[index] for index in range(3)]),
                "prompt_embeds": cpu_tensor(conditional["prompt_embeds"], torch.bfloat16),
                "shared_A_depth": cpu_tensor(shared_depth, torch.float32),
                "K": K_cpu.float().contiguous(),
                "shared_A_c2w": cpu_tensor(shared_source_c2w, torch.float32),
                "planned_c2w": condition["target_c2w"].contiguous(),
                "rng_cpu_at_fork": rng_cpu_at_fork.contiguous(),
                "rng_cuda_at_fork": rng_cuda_at_fork.contiguous(),
                "target_Aprime_render12": condition["render"][225:237].to(torch.float16).contiguous(),
            }
            sample_dir = root / "samples" / sample_id
            sample_dir.mkdir()
            sample_path = sample_dir / "sample.safetensors"
            save_file(tensors, sample_path)
            manifest = {
                "scene": scene,
                "query": query,
                "sample_id": sample_id,
                "view_class": query_spec["view_class"],
                "eligible": query_spec["eligible"],
                "low_no_overlap_diagnostic": query_spec["low_no_overlap_diagnostic"],
                "requested_yaw_degrees": offset,
                "actual_yaw_degrees": actual_yaw,
                "overlap_coverage": coverage,
                "latent_displacement": displacement,
                "true_shared_A": True,
                "A_capture_count": 1,
                "z_A_sha256": z_a_hash,
                "shared_A_depth_K_c2w": True,
                "branch_starts_after_block_5": True,
                "raw_lossless_inputs": True,
                "h264_shared_training_truth": False,
                "denoising_step_indices": [1000, 750, 500, 250],
                "actual_model_timesteps": [step_timesteps[index] for index in range(4)],
                "tensor_sha256": sha256_file(sample_path),
                "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
            }
            (sample_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            descriptor = {
                "scene": scene,
                "query": query,
                "sample_id": sample_id,
                "view_class": query_spec["view_class"],
                "requested_yaw_degrees": offset,
                "actual_yaw_degrees": actual_yaw,
                "overlap_coverage": coverage,
                "latent_displacement_mean_pixels": displacement["mean_pixel_displacement"],
                "eligible": query_spec["eligible"],
                "low_no_overlap_diagnostic": query_spec["low_no_overlap_diagnostic"],
            }
            all_descriptors.append(descriptor)
            scene_shared_queries[query] = {
                "z_A_hash": z_a_hash,
                "z_A_torch_equal_shared": True,
                "pre_fork_prefix_torch_equal": True,
                "pre_fork_last_pred_torch_equal": True,
                "pre_fork_kv_cache_torch_equal": True,
                "pre_fork_state_torch_equal": pre_fork_equal,
                "rng_continuation_state_restored": True,
                "actual_yaw_degrees": actual_yaw,
                "camera_center_drift": pose["max_camera_center_drift"],
                "occupancy": coverage,
                "latent_displacement": displacement,
                "projection_non_identity": not projection_equal,
            }
            scene_projection_queries[query] = {
                "requested_yaw_degrees": offset,
                "actual_yaw_degrees": actual_yaw,
                "occupancy_valid_fraction": coverage,
                "latent_displacement": displacement,
                "projection_torch_equal_z_A": projection_equal,
                "identity_overlap_error": identity_overlap,
                "eligible": query_spec["eligible"],
                "low_no_overlap_diagnostic": query_spec["low_no_overlap_diagnostic"],
            }
            if query != "exact":
                del condition, render_latent, mask_latent, branch_cache
                gc.collect()
                torch.cuda.empty_cache()

        if any(row["z_A_hash"] != z_a_hash for row in scene_shared_queries.values()):
            raise AssertionError(f"{scene}: branch z_A hashes differ")
        if any(row["camera_center_drift"] != 0.0 for row in scene_shared_queries.values()):
            raise AssertionError(f"{scene}: camera center moved")
        if any(
            not row["projection_non_identity"]
            for query, row in scene_shared_queries.items()
            if query != "exact"
        ):
            raise AssertionError(f"{scene}: non-zero projection identity")
        shared_audit["scenes"][scene] = {
            "A_capture_count": capture_count[0],
            "unique_z_A_hash": z_a_hash,
            "all_queries_reference_same_z_A_hash": True,
            "all_pre_fork_tensor_state_torch_equal": all(
                row["pre_fork_state_torch_equal"]
                for row in scene_shared_queries.values()
            ),
            "latent_prefix_saved_once": True,
            "last_pred_saved_once": True,
            "kv_cache_saved_once": True,
            "rng_continuation_state_saved_once": True,
            "remaining_initial_noise_saved_once": True,
            "raw_lossless_shared_inputs": True,
            "queries": scene_shared_queries,
            "shared_state_artifacts": state_hashes,
            "rng_cpu_before_noise_sha256": tensor_sha256(rng_cpu_before_noise),
            "rng_cuda_before_noise_sha256": tensor_sha256(rng_cuda_before_noise),
        }
        projection_audit["scenes"][scene] = {
            "coverage_precheck": coverage_audit,
            "queries": scene_projection_queries,
        }
        with command_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"- {scene} capture/projection finish: {datetime.now().astimezone().isoformat()}\n"
                f"- {scene} capture/projection seconds: {time.perf_counter() - scene_started:.3f}\n"
                f"- {scene} A capture count: {capture_count[0]}\n"
                f"- {scene} selected wide: {[item['query'] for item in query_specs if item['view_class'] == 'wide']}\n"
            )

        del (
            source_frames,
            source_depths,
            ref_latent,
            exact_render_latent,
            exact_mask_latent,
            exact_condition,
            conditional,
            noise,
            output_shared,
            shared_cache,
            shared_prefix,
            shared_last,
            shared_depth,
            shared_source_c2w,
            z_a_device,
        )
        gc.collect()
        torch.cuda.empty_cache()

    shared_audit["passed"] = bool(
        set(shared_audit["scenes"]) == set(SCENES)
        and all(
            item["A_capture_count"] == 1
            and item["all_queries_reference_same_z_A_hash"]
            and item["all_pre_fork_tensor_state_torch_equal"]
            for item in shared_audit["scenes"].values()
        )
    )
    projection_audit["passed"] = bool(
        set(projection_audit["scenes"]) == set(SCENES)
        and all(
            len(item["queries"]) == 5
            and all(
                query == "exact"
                or not values["projection_torch_equal_z_A"]
                for query, values in item["queries"].items()
            )
            for item in projection_audit["scenes"].values()
        )
    )
    if not shared_audit["passed"] or not projection_audit["passed"]:
        raise AssertionError("root shared/projection audit failed")
    (root / "shared_A_audit.json").write_text(
        json.dumps(shared_audit, indent=2) + "\n", encoding="utf-8"
    )
    (root / "projection_audit.json").write_text(
        json.dumps(projection_audit, indent=2) + "\n", encoding="utf-8"
    )
    experiment_manifest = {
        "scope": "S0/S1 capacity/overfit only",
        "scene_count": 2,
        "query_count": 10,
        "queries": all_descriptors,
        "training": {
            "shared_adapter": True,
            "balanced_cyclic": True,
            "optimizer": "AdamW",
            "lr": 0.001,
            "steps": 500,
            "preservation_weight": 0.5,
            "last_denoise_step_backprop_only": True,
        },
        "base_checkpoint_sha256_before_capture": checkpoint_hash,
        "initial_adapter_sha256_before_capture": init_hash,
    }
    (root / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise AssertionError("base checkpoint changed during capture")
    if sha256_file(args.init_adapter) != init_hash:
        raise AssertionError("initial adapter changed during capture")
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- All capture/projection finish: {datetime.now().astimezone().isoformat()}\n"
            f"- Capture/projection seconds: {time.perf_counter() - started:.3f}\n"
            f"- Capture peak VRAM GiB: {torch.cuda.max_memory_allocated(device) / 2**30:.6f}\n"
        )

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    from scripts.phase1_lsm.train_eval_wide_sharedA_hardgate import main as train_main

    original_argv = sys.argv
    sys.argv = [
        "train_eval_wide_sharedA_hardgate.py",
        "--root",
        str(root),
        "--checkpoint",
        args.checkpoint,
        "--init-adapter",
        args.init_adapter,
        "--repo-root",
        str(repo_root),
        "--max-steps",
        "500",
        "--lr",
        "0.001",
        "--preservation-weight",
        "0.5",
        "--wan-root",
        str(WAN_ROOT),
    ]
    try:
        train_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
