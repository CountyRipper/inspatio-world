#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from phase1_lsm.data_prep import sha256_file
from phase1_lsm.runtime import allocate_kv_cache
from phase2_memory.data import load_scene_geometry, prepare_trajectory
from phase2_memory.manifest import load_manifest
from phase2_memory.rollout import replay_return_record, run_online_rollout
from scripts.phase2_memory.common import (
    DEFAULT_CHECKPOINT,
    cpu,
    finish_distributed,
    init_distributed,
    json_dump,
    load_pipeline,
    record_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/phase2_memory_manifest.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", choices=("train", "heldout"))
    parser.add_argument("--group-id", action="append")
    parser.add_argument("--phase2a-only", action="store_true")
    parser.add_argument("--anchoring-strength", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.anchoring_strength <= 1.0:
        raise ValueError("anchoring strength must be in [0,1]")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    groups = manifest.select(
        split=args.split,
        group_ids=None if args.group_id is None else set(args.group_id),
        phase2a_only=args.phase2a_only,
    )
    if not groups:
        raise ValueError("no trajectory groups selected")

    device = init_distributed()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = Path(args.checkpoint).resolve()
    adapter = Path(args.adapter).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    adapter_hash = sha256_file(adapter)
    pipeline = load_pipeline(
        Path(args.repo_root).resolve(), checkpoint, adapter, device
    )
    scene_cache = {}
    latent_cache = {}
    completed = []
    started = time.perf_counter()

    for group in groups:
        group_root = root / "groups" / group.group_id
        tensor_path = group_root / "rollout.safetensors"
        metadata_path = group_root / "rollout.json"
        if tensor_path.is_file() and metadata_path.is_file():
            completed.append(group.group_id)
            print(f"SKIP_COMPLETE {group.group_id}", flush=True)
            continue
        if group_root.exists():
            raise FileExistsError(f"partial group output exists: {group_root}")
        group_root.mkdir(parents=True)
        tick = time.perf_counter()
        if group.scene_id not in scene_cache:
            scene_cache[group.scene_id] = load_scene_geometry(
                manifest.scenes[group.scene_id]
            )
        geometry = scene_cache[group.scene_id]
        cache = latent_cache.get(group.scene_id)
        prepared = prepare_trajectory(
            pipeline,
            geometry,
            group,
            group_root / "trajectory.txt",
            device,
            cached_ref_latent=None if cache is None else cache["ref_latent"],
            cached_conditional=None if cache is None else cache["conditional"],
        )
        latent_cache[group.scene_id] = {
            "ref_latent": prepared.ref_latent,
            "conditional": prepared.conditional,
        }
        result = run_online_rollout(
            pipeline,
            prepared,
            variant="correct",
            anchoring_strength=args.anchoring_strength,
            capture_records=True,
        )
        if len(result.records) != len(group.return_stations):
            raise AssertionError("return record count mismatch")

        kv_cache = allocate_kv_cache(pipeline.generator, device)
        replay_equal = []
        for record in result.records:
            condition = torch.cat((
                record.memory_mask4,
                record.projected_memory,
            ), dim=2)
            gate = record.occupancy.float()
            replay = replay_return_record(
                pipeline.generator,
                pipeline.scheduler,
                pipeline.denoising_step_list,
                prepared.conditional,
                record,
                memory_condition=condition,
                memory_gate=gate,
                anchoring_strength=args.anchoring_strength,
                truncate_history=False,
                kv_cache=kv_cache,
            )
            equal = bool(torch.equal(replay, record.prediction))
            replay_equal.append(equal)
            if not equal:
                max_error = float((replay.float() - record.prediction.float()).abs().max())
                raise AssertionError(
                    f"{group.group_id}/block{record.block}: replay changed ({max_error})"
                )
            record.no_memory_full = replay_return_record(
                pipeline.generator,
                pipeline.scheduler,
                pipeline.denoising_step_list,
                prepared.conditional,
                record,
                memory_condition=None,
                memory_gate=None,
                anchoring_strength=0.0,
                truncate_history=False,
                kv_cache=kv_cache,
            )
            record.no_memory_truncated = replay_return_record(
                pipeline.generator,
                pipeline.scheduler,
                pipeline.denoising_step_list,
                prepared.conditional,
                record,
                memory_condition=None,
                memory_gate=None,
                anchoring_strength=0.0,
                truncate_history=True,
                kv_cache=kv_cache,
            )

        tensors = record_tensors(
            result.output,
            prepared.conditional["prompt_embeds"],
            result.records,
        )
        save_file(tensors, tensor_path)
        observations = [{
            "block": item.block,
            "memory_id": item.memory_id,
            "retrieved_ids": item.retrieved_ids,
            "retrieved_versions": item.retrieved_versions,
            "retrieved_scores": item.retrieved_scores,
            "selected_id": item.selected_id,
            "selected_version": item.selected_version,
            "projection_non_identity": item.projection_non_identity,
            "occupancy_fraction": item.occupancy_fraction,
        } for item in result.observations]
        metadata = {
            "group_id": group.group_id,
            "scene_id": group.scene_id,
            "split": group.split,
            "family": group.family,
            "seed": group.seed,
            "stations": [station.__dict__ for station in group.stations],
            "generated_memories": ["A", "B", "C"],
            "observations": observations,
            "records": [{
                "block": item.block,
                "memory_id": item.memory_id,
                "memory_version": item.memory_version,
                "retrieved_ids": item.retrieved_ids,
                "retrieved_scores": item.retrieved_scores,
            } for item in result.records],
            "bank_events": result.bank_events,
            "history_latents_detached": all(
                not item.previous_latent.requires_grad for item in result.records
            ),
            "future_return_gt_used_as_memory": False,
            "denoise_steps_recorded": [0, 1, 2, 3],
            "correct_replay_torch_equal": replay_equal,
            "anchoring_strength": args.anchoring_strength,
            "checkpoint_sha256": checkpoint_hash,
            "adapter_sha256": adapter_hash,
            "tensor_sha256": sha256_file(tensor_path),
            "tensor_shapes": {
                name: list(value.shape) for name, value in tensors.items()
            },
            "seconds": time.perf_counter() - tick,
        }
        json_dump(metadata_path, metadata)
        completed.append(group.group_id)
        print(json.dumps({
            "group": group.group_id,
            "records": len(result.records),
            "seconds": metadata["seconds"],
        }), flush=True)
        del prepared, result, tensors, kv_cache
        gc.collect()
        torch.cuda.empty_cache()

    if sha256_file(checkpoint) != checkpoint_hash or sha256_file(adapter) != adapter_hash:
        raise AssertionError("immutable checkpoint or input adapter changed")
    json_dump(root / "capture_summary.json", {
        "manifest": str(manifest.path),
        "selected_group_count": len(groups),
        "completed_groups": completed,
        "anchoring_strength": args.anchoring_strength,
        "checkpoint_sha256": checkpoint_hash,
        "adapter_sha256": adapter_hash,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "seconds": time.perf_counter() - started,
    })
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    finish_distributed()


if __name__ == "__main__":
    main()
