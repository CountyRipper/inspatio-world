#!/usr/bin/env python3
"""Capture block-aligned paired worlds and the common return snapshot."""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from safetensors.torch import save_file
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import CausalInferencePipeline
from scripts.render_point_cloud import (
    generate_target_c2ws,
    load_extrinsic_c2w,
    load_intrinsic,
    render_point_cloud,
    read_da3_depth,
    scale_intrinsic,
)
from scripts.world_memory.common import (
    init_single_gpu_distributed,
    initialize_kv_cache,
    load_configs,
    load_frozen_generator,
    pad_clean_latent,
    resolve_repo_path,
    write_video_tensor,
)
from utils.render_warper import convert_mask_video
from world_state.source_truth import conservative_static_confidence


class StaticPromptEncoder(nn.Module):
    def __init__(self, prompt_embeds: torch.Tensor):
        super().__init__()
        self.register_buffer("prompt_embeds", prompt_embeds)

    def forward(self, text_prompts):
        return {"prompt_embeds": self.prompt_embeds.expand(len(text_prompts), -1, -1)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_teacher/teacher_v0.yaml")
    parser.add_argument("--scene", default="S0")
    parser.add_argument("--prepare-render", action="store_true")
    parser.add_argument("--skip-kv-snapshot", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--refresh-source-truth", action="store_true")
    return parser.parse_args()


def build_yaw(frame_count: int, trajectory) -> np.ndarray:
    yaw = np.zeros(frame_count, dtype=np.float64)
    value = float(trajectory.yaw_degrees)
    first = np.linspace(
        0.0,
        value,
        int(trajectory.first_ramp_end) - int(trajectory.first_ramp_start) + 1,
    )
    second_start = int(trajectory.second_ramp_start)
    second_end = int(trajectory.second_ramp_end)
    if second_end - second_start + 1 != len(first):
        raise ValueError("the two traversal ramps must have identical lengths")
    yaw[int(trajectory.first_ramp_start):int(trajectory.first_ramp_end) + 1] = first
    yaw[int(trajectory.first_ramp_end) + 1:int(trajectory.return_ramp_start)] = value
    yaw[int(trajectory.return_ramp_start):int(trajectory.return_ramp_end) + 1] = first[::-1]
    yaw[second_start:second_end + 1] = first
    yaw[second_end + 1:] = value
    if not np.array_equal(
        yaw[int(trajectory.first_ramp_start):int(trajectory.first_ramp_end) + 1],
        yaw[second_start:second_end + 1],
    ):
        raise AssertionError("first and second traversal schedules differ")
    return yaw


def write_trajectory(path: Path, yaw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    zeros = np.zeros_like(yaw)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(f"{value:.8f}" for value in zeros) + "\n")
        handle.write(" ".join(f"{value:.8f}" for value in yaw) + "\n")
        handle.write(" ".join(f"{value:.8f}" for value in zeros) + "\n")


def load_video(path: Path, *, binary=False) -> torch.Tensor:
    import decord

    reader = decord.VideoReader(str(path), width=832, height=480)
    frames = torch.from_numpy(reader.get_batch(range(len(reader))).asnumpy())
    frames = rearrange(frames, "t h w c -> 1 c t h w").float().div_(255.0)
    if binary:
        frames = (frames > 0.5).float()
    return frames.mul_(2.0).sub_(1.0)


def camera_record(da3_dir, trajectory_path, pixel_frames, latent_frames):
    device = torch.device("cpu")
    K_render = scale_intrinsic(load_intrinsic(str(da3_dir), device), 832, 480)
    initial_c2w, source_c2ws = load_extrinsic_c2w(str(da3_dir), device)
    target_c2ws = generate_target_c2ws(
        str(trajectory_path),
        initial_c2w,
        source_c2ws,
        pixel_frames,
        device,
        relative_to_source=False,
        rotation_only=True,
    )
    pixel_indices = torch.arange(latent_frames) * 4
    target = torch.stack(target_c2ws)[pixel_indices]
    source = torch.stack(source_c2ws)[pixel_indices]
    world_to_W0 = torch.linalg.inv(target_c2ws[0])
    target_W0 = world_to_W0.unsqueeze(0) @ target
    source_W0 = world_to_W0.unsqueeze(0) @ source
    scale = torch.tensor(
        [[104.0 / 832.0, 0.0, 0.0], [0.0, 60.0 / 480.0, 0.0], [0.0, 0.0, 1.0]]
    )
    K_latent = (scale @ K_render).repeat(latent_frames, 1, 1)
    return K_latent.unsqueeze(0), target_W0.unsqueeze(0), source_W0.unsqueeze(0)


def decode(vae, latent):
    return (vae.decode_to_pixel(latent, use_cache=False) * 0.5 + 0.5).clamp(0, 1)


def load_source_depth_block(da3_dir: Path, latent_start: int, block_size: int) -> torch.Tensor:
    depth_paths = sorted(glob.glob(str(da3_dir / "depth" / "*.png")))
    selected = []
    for latent_index in range(latent_start, latent_start + block_size):
        pixel_index = 4 * latent_index
        if pixel_index >= len(depth_paths):
            raise ValueError("source depth sequence is shorter than the latent camera record")
        selected.append(torch.from_numpy(read_da3_depth(depth_paths[pixel_index])).float())
    depth = torch.stack(selected).unsqueeze(1)
    return torch.nn.functional.interpolate(
        depth, size=(60, 104), mode="nearest"
    ).contiguous()


def write_record_manifest(record: dict, data_dir: Path) -> None:
    block_size = int(record["block_size"])
    manifest = {
        "scene": record["scene_id"],
        "pixel_frames": int(record["yaw_pixel_degrees"].shape[0]),
        "latent_frames": int(record["ref_latent"].shape[1]),
        "timeline": "0 -> 40 -> 0 -> hold(3 blocks) -> 40 -> hold",
        "paired_blocks": {"first": [2, 3, 4], "second": [13, 14, 15]},
        "write_block": int(record["write_block"]),
        "common_snapshot_after_block": int(record["common_prefix_last_block"]),
        "camera_translation_max": float(
            record["target_c2w_W0"][..., :3, 3].abs().max()
        ),
        "source_static_coverage": float(
            record["source_observation"]["valid"].float().mean()
        ),
    }
    with (data_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_memory_sidecars(record: dict, data_dir: Path) -> None:
    block_size = int(record["block_size"])
    start = int(record["write_block"]) * block_size
    end = start + block_size
    for identity in ("A", "B"):
        save_file(
            {
                "clean_latent": record[f"output_{identity}"][:, start:end]
                .contiguous()
            },
            str(data_dir / f"M40_{identity}.safetensors"),
        )


def materialize_saved_snapshot(config, base_config, data_dir, device) -> None:
    record = torch.load(data_dir / "paired_record.pt", map_location="cpu", weights_only=True)
    generator = load_frozen_generator(
        base_config, config.experiment.checkpoint, device
    )
    block_size = int(record["block_size"])
    snapshot_block = int(record["second_traversal_first_block"])
    start = snapshot_block * block_size
    ref = record["ref_latent"][:, start:start + block_size].to(
        device=device, dtype=torch.bfloat16
    )
    last_pred = record["output_A"][:, start - block_size:start].to(
        device=device, dtype=torch.bfloat16
    )
    context_frames = torch.cat(
        (pad_clean_latent(ref), pad_clean_latent(last_pred)), dim=1
    )
    render = torch.cat(
        (
            record["mask_latent"][:, start:start + block_size],
            record["render_latent"][:, start:start + block_size],
        ),
        dim=2,
    ).to(device=device, dtype=torch.bfloat16)
    kv_cache = initialize_kv_cache(generator, 1, torch.bfloat16, device)
    with torch.no_grad():
        generator(
            noisy_image_or_video=context_frames,
            conditional_dict={
                "prompt_embeds": record["prompt_embeds"].to(
                    device=device, dtype=torch.bfloat16
                )
            },
            timestep=torch.zeros(1, block_size, device=device, dtype=torch.int64),
            kv_cache=kv_cache,
            render_latent_input=render,
            kv_size=(0, -1),
            freqs_offset=0,
        )
    snapshot = {
        "format_version": 1,
        "scene_id": record["scene_id"],
        "next_block": snapshot_block,
        "prefix_output": record["output_A"][:, :start],
        "last_pred": record["output_A"][:, start - block_size:start],
        "next_noise": record["noise_A"][:, start:],
        "context_frames": context_frames.cpu().contiguous(),
        "transition_seed": int(record["transition_seed"]) + snapshot_block,
        "kv_cache": [
            {"k": values["k"].cpu().contiguous(), "v": values["v"].cpu().contiguous()}
            for values in kv_cache
        ],
    }
    torch.save(snapshot, data_dir / "common_snapshot.pt")
    print(f"Saved common snapshot: {data_dir / 'common_snapshot.pt'}")


def main():
    args = parse_args()
    config, base_config = load_configs(args.config)
    if args.scene not in config.scenes:
        raise ValueError(f"unknown scene {args.scene}")
    scene = config.scenes[args.scene]
    data_dir = resolve_repo_path(config.experiment.data_root) / args.scene
    artifact_dir = resolve_repo_path(config.experiment.artifact_root) / args.scene
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.snapshot_only:
        if not (data_dir / "paired_record.pt").exists():
            raise FileNotFoundError("snapshot-only requires paired_record.pt")
        materialize_saved_snapshot(
            config, base_config, data_dir, init_single_gpu_distributed()
        )
        return
    if args.refresh_source_truth:
        record_path = data_dir / "paired_record.pt"
        if not record_path.exists():
            raise FileNotFoundError("refresh-source-truth requires paired_record.pt")
        record = torch.load(record_path, map_location="cpu", weights_only=True)
        confidence = conservative_static_confidence(record["ref_latent"][0])
        start = int(record["write_block"]) * int(record["block_size"])
        end = start + int(record["block_size"])
        confidence = confidence[start:end].contiguous()
        record["source_observation"]["static_confidence"] = confidence
        record["source_observation"]["valid"] = (
            confidence >= float(config.training.source_static_threshold)
        )
        record["source_observation"]["depth"] = load_source_depth_block(
            resolve_repo_path(scene.da3_dir), start, int(record["block_size"])
        )
        record["source_observation"]["latent_indices"] = list(range(start, end))
        with resolve_repo_path(scene.metadata_json).open("r", encoding="utf-8") as handle:
            source_metadata = json.load(handle)[0]
        record["source_video_path"] = str(
            resolve_repo_path(source_metadata["video_path"])
        )
        torch.save(record, record_path)
        write_record_manifest(record, data_dir)
        write_memory_sidecars(record, data_dir)
        print(
            "Refreshed source truth: "
            f"coverage={record['source_observation']['valid'].float().mean():.6f}"
        )
        return

    with resolve_repo_path(scene.metadata_json).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)[0]
    video_path = resolve_repo_path(metadata["video_path"])
    import decord

    pixel_frames = len(decord.VideoReader(str(video_path)))
    yaw = build_yaw(pixel_frames, config.trajectory)
    trajectory_path = data_dir / "yaw_block_aligned.txt"
    write_trajectory(trajectory_path, yaw)
    render_dir = data_dir / "render_conditions"
    if args.prepare_render:
        render_point_cloud(
            da3_dir=str(resolve_repo_path(scene.da3_dir)),
            traj_txt_path=str(trajectory_path),
            output_dir=str(render_dir),
            width=832,
            height=480,
            fps=24,
            relative_to_source=False,
            rotation_only=True,
            render_backend="warper",
        )
    if not (render_dir / "render_offline.mp4").exists():
        raise FileNotFoundError("run with --prepare-render")

    device = init_single_gpu_distributed()
    pipeline = CausalInferencePipeline(base_config, device=device)
    from safetensors.torch import load_file

    state = load_file(str(resolve_repo_path(config.experiment.checkpoint)))
    missing, unexpected = pipeline.generator.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    del state
    pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device)
    pipeline.text_encoder.to(device)
    pipeline.vae.to(device)
    pipeline.eval().requires_grad_(False)

    with torch.no_grad():
        prompt_embeds = pipeline.text_encoder([metadata["text"]])["prompt_embeds"].detach()
    pipeline.text_encoder = StaticPromptEncoder(prompt_embeds).to(device)

    with torch.no_grad():
        ref_latent = pipeline.vae.encode_to_latent(
            load_video(video_path).to(device=device, dtype=torch.bfloat16)
        ).to(torch.bfloat16)
        torch.cuda.empty_cache()
        render_latent = pipeline.vae.encode_to_latent(
            load_video(render_dir / "render_offline.mp4").to(
                device=device, dtype=torch.bfloat16
            )
        ).to(torch.bfloat16)
        torch.cuda.empty_cache()
        mask_latent = convert_mask_video(
            load_video(render_dir / "mask_offline.mp4", binary=True).to(
                device=device, dtype=torch.bfloat16
            )
        ).to(torch.bfloat16)

    available = min(ref_latent.shape[1], render_latent.shape[1], mask_latent.shape[1])
    block_size = int(config.experiment.block_size)
    latent_frames = available - available % block_size
    ref_latent = ref_latent[:, :latent_frames]
    render_latent = render_latent[:, :latent_frames]
    mask_latent = mask_latent[:, :latent_frames]
    K, target_c2w_W0, source_c2w_W0 = camera_record(
        resolve_repo_path(scene.da3_dir), trajectory_path, pixel_frames, latent_frames
    )

    offset = (
        int(config.experiment.second_traversal_first_block) - 2
    ) * block_size
    first = target_c2w_W0[:, 2 * block_size:5 * block_size]
    second = target_c2w_W0[:, 2 * block_size + offset:5 * block_size + offset]
    if not torch.equal(first, second):
        raise AssertionError("latent traversal camera schedules are not exactly paired")
    if target_c2w_W0[..., :3, 3].abs().max() > 1e-5:
        raise AssertionError("rotation-only target camera translated in W0")

    outputs = {}
    noises = {}
    for identity, seed in (
        ("A", int(scene.world_a_seed)),
        ("B", int(scene.world_b_seed)),
    ):
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            1, latent_frames, 16, 60, 104,
            generator=generator, device=device, dtype=torch.bfloat16,
        )
        torch.manual_seed(int(config.experiment.transition_seed) + seed)
        torch.cuda.manual_seed_all(int(config.experiment.transition_seed) + seed)
        with torch.no_grad():
            output = pipeline.inference(
                noise=noise,
                text_prompts=[metadata["text"]],
                ref_latent=ref_latent,
                render_latent=render_latent,
                mask_latent=mask_latent,
                decode=False,
            )
        outputs[identity] = output.detach().cpu().contiguous()
        noises[identity] = noise.detach().cpu().contiguous()
        with torch.no_grad():
            write_video_tensor(artifact_dir / f"world_{identity}_memory_off.mp4", decode(pipeline.vae, output))
            write_start = int(config.experiment.write_block) * block_size
            write_video_tensor(
                artifact_dir / f"M40_{identity}_reference.mp4",
                decode(pipeline.vae, output[:, write_start:write_start + block_size]),
            )
        pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()

    static_full = conservative_static_confidence(ref_latent[0].detach().cpu())
    write_start = int(config.experiment.write_block) * block_size
    write_end = write_start + block_size
    source_confidence = static_full[write_start:write_end]
    source_observation = {
        "scene_id": args.scene,
        "world_id": "A",
        "observation_id": f"{args.scene}:source_static",
        "provenance": 0,
        "clean_latent": ref_latent[0, write_start:write_end].cpu().contiguous(),
        "K": K[0, write_start:write_end].contiguous(),
        "c2w_W0": source_c2w_W0[0, write_start:write_end].contiguous(),
        "depth": load_source_depth_block(
            resolve_repo_path(scene.da3_dir), write_start, block_size
        ),
        "latent_indices": list(range(write_start, write_end)),
        "valid": (source_confidence >= float(config.training.source_static_threshold)),
        "static_confidence": source_confidence.contiguous(),
        "geometry_confidence": torch.ones_like(source_confidence),
    }

    record = {
        "format_version": 1,
        "scene_id": args.scene,
        "prompt": metadata["text"],
        "source_video_path": str(video_path),
        "prompt_embeds": prompt_embeds.cpu().contiguous(),
        "block_size": block_size,
        "write_block": int(config.experiment.write_block),
        "common_prefix_last_block": int(config.experiment.common_prefix_last_block),
        "second_traversal_first_block": int(config.experiment.second_traversal_first_block),
        "first_exact_block": int(config.experiment.first_exact_block),
        "transition_seed": int(config.experiment.transition_seed),
        "denoising_steps": pipeline.denoising_step_list.cpu(),
        "ref_latent": ref_latent.cpu().contiguous(),
        "render_latent": render_latent.cpu().contiguous(),
        "mask_latent": mask_latent.cpu().contiguous(),
        "noise_A": noises["A"],
        "noise_B": noises["B"],
        "output_A": outputs["A"],
        "output_B": outputs["B"],
        "camera_K": K.contiguous(),
        "target_c2w_W0": target_c2w_W0.contiguous(),
        "source_c2w_W0": source_c2w_W0.contiguous(),
        "source_observation": source_observation,
        "yaw_pixel_degrees": torch.from_numpy(yaw).float(),
    }
    torch.save(record, data_dir / "paired_record.pt")
    write_memory_sidecars(record, data_dir)

    snapshot_block = int(config.experiment.second_traversal_first_block)
    start = snapshot_block * block_size
    previous = outputs["A"][:, start - block_size:start].to(device=device, dtype=torch.bfloat16)
    context_frames = torch.cat(
        (
            pad_clean_latent(ref_latent[:, start:start + block_size]),
            pad_clean_latent(previous),
        ),
        dim=1,
    )
    snapshot = {
        "format_version": 1,
        "scene_id": args.scene,
        "next_block": snapshot_block,
        "prefix_output": outputs["A"][:, :start],
        "last_pred": outputs["A"][:, start - block_size:start],
        "next_noise": noises["A"][:, start:],
        "context_frames": context_frames.cpu().contiguous(),
        "transition_seed": int(config.experiment.transition_seed) + snapshot_block,
    }
    if not args.skip_kv_snapshot:
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        for cache in pipeline.kv_cache1:
            cache["k"].zero_()
            cache["v"].zero_()
        with torch.no_grad():
            snapshot_render = torch.cat(
                (
                    mask_latent[:, start:start + block_size],
                    render_latent[:, start:start + block_size],
                ),
                dim=2,
            )
            pipeline.generator(
                noisy_image_or_video=context_frames,
                conditional_dict={"prompt_embeds": prompt_embeds},
                timestep=torch.zeros(1, block_size, device=device, dtype=torch.int64),
                kv_cache=pipeline.kv_cache1,
                render_latent_input=snapshot_render,
                kv_size=(0, -1),
                freqs_offset=0,
            )
        snapshot["kv_cache"] = [
            {
                "k": cache["k"].cpu().contiguous(),
                "v": cache["v"].cpu().contiguous(),
            }
            for cache in pipeline.kv_cache1
        ]
    torch.save(snapshot, data_dir / "common_snapshot.pt")

    write_record_manifest(record, data_dir)
    run_state = resolve_repo_path(config.experiment.artifact_root) / "RUN_STATE.md"
    run_state.parent.mkdir(parents=True, exist_ok=True)
    run_state.write_text(
        f"Stage: records {args.scene}\nRecent checkpoint: none\n"
        f"Conclusion: paired block-aligned record captured for {args.scene}.\n"
        f"Next: train Stage A exact Reader-only.\n",
        encoding="utf-8",
    )
    print(f"Saved paired record: {data_dir / 'paired_record.pt'}")


if __name__ == "__main__":
    main()
