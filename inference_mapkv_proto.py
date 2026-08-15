#!/usr/bin/env python3
"""Deterministic single-GPU runner for the CUT3R/MapKV prototype.

This intentionally follows ``inference_causal_test.py`` for model, dataset, VAE,
and scheduler behavior.  MapKV is opt-in and only changes selected denoising
attention calls; capture and deterministic noise replay are independent hooks.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.io import write_video

from datasets.video_dataset import VideoDataset
from mapkv_proto.config import MapKVConfig, resolve_indices
from mapkv_proto.deterministic_noise import DeterministicNoiseBundle
from mapkv_proto.kv_bank import KVBank, KVBankWriter
from mapkv_proto.memory_context import make_memory_context
from mapkv_proto.retrieval import RetrievalPlan
from mapkv_proto.revisit_pair import build_block_mapping
from mapkv_proto.visualization import save_gate_overlay
from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from utils.render_warper import convert_mask_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/inference_1.3b.yaml")
    parser.add_argument("--mapkv_config", default="configs/mapkv_proto.yaml")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--wan_model_folder", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument(
        "--data_path_root",
        default=".",
        help="Root used by relative paths inside json_path (upstream uses its cwd).",
    )
    parser.add_argument("--traj_txt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="mapkv")
    parser.add_argument("--video_output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise_bundle", required=True)
    parser.add_argument("--create_noise_bundle", action="store_true")
    parser.add_argument("--capture_kv", action="store_true")
    parser.add_argument("--bank_root")
    parser.add_argument(
        "--mode", choices=("off", "baseline", "oracle", "wrong", "pose", "geometry")
    )
    parser.add_argument("--source_chunk", type=int)
    parser.add_argument("--wrong_chunk", type=int)
    parser.add_argument("--target_chunks", nargs="+", type=int)
    parser.add_argument("--selected_layers", nargs="+", type=int)
    parser.add_argument("--selected_steps", nargs="+", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--gate_mode", choices=("global", "ref_blind", "surfel_ref_blind")
    )
    parser.add_argument("--retrieval_plan")
    parser.add_argument("--compare_latents_to")
    parser.add_argument("--verify_memory_off_replay", action="store_true")
    parser.add_argument("--replay_tolerance", type=float, default=0.0)
    parser.add_argument("--require_replay_tolerance", action="store_true")
    return parser.parse_args()


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _prepare_runtime_json(
    json_path: str | Path, data_path_root: str | Path, output_path: Path
) -> Path:
    source = Path(json_path).resolve()
    root = Path(data_path_root).resolve()
    entries = json.loads(source.read_text(encoding="utf-8"))
    path_keys = ("video_path", "vggt_depth_path", "vggt_extrinsics_path")
    for entry in entries:
        for key in path_keys:
            if not entry.get(key):
                continue
            value = Path(entry[key])
            if not value.is_absolute():
                value = root / value
            entry[key] = str(value.resolve())
            if not Path(entry[key]).exists():
                raise FileNotFoundError(f"Input path from JSON does not exist: {entry[key]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    shutil.copy2(source, output_path.with_name("input.original.json"))
    return output_path


def _load_configs(args: argparse.Namespace, runtime_json: Path):
    base = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(base, OmegaConf.load(args.config_path))
    config.wan_model_folder = str(Path(args.wan_model_folder).resolve())
    for item in config.generator.weight_list:
        item.path = config.wan_model_folder
    config.dataset.json_path = str(runtime_json)
    config.dataset.traj_txt_path = str(Path(args.traj_txt_path).resolve())
    config.dataset.adaptive_frame = False

    experiment = OmegaConf.load(args.mapkv_config)
    raw: dict[str, Any] = OmegaConf.to_container(
        experiment.get("mapkv", {}), resolve=True
    )
    mode = args.mode or str(raw.get("mode", "off"))
    if mode == "baseline":
        mode = "off"
    raw["mode"] = mode
    raw["enabled"] = mode != "off"
    if args.source_chunk is not None:
        raw["source_chunk"] = args.source_chunk
    if args.wrong_chunk is not None:
        raw["wrong_chunk"] = args.wrong_chunk
    if args.target_chunks is not None:
        raw["target_chunks"] = args.target_chunks
    elif mode != "off" and not raw.get("target_chunks") and args.retrieval_plan:
        retrieval_payload = json.loads(
            Path(args.retrieval_plan).read_text(encoding="utf-8")
        )
        entries = (
            retrieval_payload.get("targets", retrieval_payload)
            if isinstance(retrieval_payload, dict)
            else retrieval_payload
        )
        raw["target_chunks"] = [int(entry["target_chunk"]) for entry in entries]
    if args.selected_layers is not None:
        raw["selected_layers"] = args.selected_layers
    if args.selected_steps is not None:
        raw["selected_step_indices"] = args.selected_steps
    if args.alpha is not None:
        raw["alpha"] = args.alpha
    if args.gate_mode is not None:
        raw.setdefault("gate", {})["mode"] = args.gate_mode
    if args.bank_root is not None:
        raw.setdefault("bank", {})["root"] = args.bank_root
    mapkv = MapKVConfig.from_mapping(raw)
    if mapkv.enabled and not mapkv.target_chunks:
        raise ValueError("MapKV is enabled but target_chunks is empty")
    return config, experiment, mapkv


def _save_rgb(frame_chw: torch.Tensor, path: Path) -> None:
    array = (
        frame_chw.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        * 255.0
    ).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _save_masks(mask_latent: torch.Tensor, output_root: Path, frames_per_block: int) -> None:
    mask_root = output_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    torch.save(mask_latent.detach().cpu(), mask_root / "mask_latent.pt")
    for chunk_id, start in enumerate(range(0, mask_latent.shape[1], frames_per_block)):
        block = mask_latent[0, start:start + frames_per_block]
        valid = ((block.float() + 1.0) * 0.5).clamp(0, 1).mean(dim=(0, 1))
        generated = 1.0 - valid
        for name, value in (("reference_valid", valid), ("generated_region", generated)):
            image = Image.fromarray((value.cpu().numpy() * 255).round().astype(np.uint8))
            image = image.resize((832, 480), Image.Resampling.NEAREST)
            image.save(mask_root / f"chunk_{chunk_id:04d}_{name}.png")


def _validate_noise_bundle(
    bundle: DeterministicNoiseBundle,
    *,
    shape: tuple[int, ...],
    num_blocks: int,
    num_steps: int,
) -> None:
    if tuple(bundle.initial_noise.shape) != shape:
        raise ValueError(
            f"Noise bundle initial shape {tuple(bundle.initial_noise.shape)} != {shape}"
        )
    if bundle.num_blocks != num_blocks or bundle.num_denoising_steps != num_steps:
        raise ValueError(
            "Noise bundle schedule mismatch: "
            f"blocks={bundle.num_blocks}/{num_blocks}, "
            f"steps={bundle.num_denoising_steps}/{num_steps}"
        )


def _build_memory_contexts(
    *,
    config: MapKVConfig,
    retrieval_plan_path: str | None,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, object], list[dict]]:
    if not config.enabled or config.mode == "off":
        return {}, []
    bank = KVBank(config.bank_root)
    plan = RetrievalPlan(retrieval_plan_path) if retrieval_plan_path else None
    contexts = {}
    selections = []
    payload_cache = {}
    for target_chunk in config.target_chunks:
        coverage = None
        plan_source = None
        if plan is not None:
            plan_source = plan.selected_chunk(target_chunk)
            coverage = plan.load_coverage(target_chunk)
        if config.mode == "oracle":
            source_chunk = config.source_chunk
            if source_chunk is None:
                source_chunk = plan_source
            if plan_source is not None and source_chunk != plan_source:
                raise ValueError(
                    f"Oracle source {source_chunk} disagrees with plan source {plan_source}"
                )
        elif config.mode == "wrong":
            source_chunk = config.wrong_chunk
        else:
            if plan is None:
                raise ValueError(f"{config.mode} mode requires --retrieval_plan")
            source_chunk = plan_source
        selection = {
            "target_chunk": int(target_chunk),
            "source_chunk": None if source_chunk is None else int(source_chunk),
            "mode": config.mode,
            "coverage_fraction": None if coverage is None else float(coverage.float().mean()),
        }
        if source_chunk is None:
            selection["status"] = "memory_off_empty_retrieval"
            selections.append(selection)
            continue
        source_chunk = int(source_chunk)
        if source_chunk >= target_chunk - 1:
            raise ValueError(
                f"Source chunk {source_chunk} is not causally valid for target {target_chunk}"
            )
        if config.gate.mode == "surfel_ref_blind" and (
            coverage is None or not bool(coverage.any())
        ):
            selection["status"] = "memory_off_empty_coverage"
            selections.append(selection)
            continue
        if source_chunk not in payload_cache:
            payload_cache[source_chunk] = bank.materialize(
                source_chunk,
                selected_layers=config.selected_layers,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
                pin_memory=config.pin_memory,
            )
        context = make_memory_context(
            target_block=target_chunk,
            source_chunk=source_chunk,
            layer_payloads=payload_cache[source_chunk],
            selected_layers=config.selected_layers,
            selected_step_indices=config.selected_step_indices,
            alpha=config.alpha,
            gate_mode=config.gate.mode,
            smooth_kernel=config.gate.smooth_kernel,
            coverage=coverage,
        )
        if context is None:
            selection["status"] = "memory_off_empty_coverage"
        else:
            contexts[int(target_chunk)] = context
            selection["status"] = "scheduled"
        selections.append(selection)
    return contexts, selections


def _validate_activation_audit(
    contexts: dict[int, object], num_layers: int, num_steps: int
) -> list[dict]:
    records = []
    for target, context in contexts.items():
        audit = list(context.audit_log)
        if context.alpha == 0:
            if audit:
                raise RuntimeError("alpha=0 unexpectedly executed auxiliary attention")
        else:
            layers = resolve_indices(context.selected_layers, num_layers, name="layer")
            steps = resolve_indices(context.selected_step_indices, num_steps, name="step")
            expected = {(target, step, layer) for step in steps for layer in layers}
            actual = {
                (int(item["target_block"]), int(item["step_index"]), int(item["layer_index"]))
                for item in audit
            }
            if actual != expected or len(audit) != len(expected):
                raise RuntimeError(
                    f"Memory activation audit mismatch for target {target}: "
                    f"expected={sorted(expected)} actual={sorted(actual)}"
                )
        records.extend(audit)
    return records


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The fixed MapKV prototype requires one CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_json = _prepare_runtime_json(
        args.json_path, args.data_path_root, output_root / "input.resolved.json"
    )
    config, experiment_config, mapkv_config = _load_configs(args, runtime_json)
    OmegaConf.save(config, output_root / "inference_config.resolved.yaml")
    OmegaConf.save(experiment_config, output_root / "mapkv_config.input.yaml")

    started = time.perf_counter()
    print(f"[MapKV] Loading pipeline on {device} in bf16; compile=False, TAE=False")
    pipeline = CausalInferencePipeline(config, device=device)
    state_dict = load_file(str(Path(args.checkpoint_path).resolve()))
    incompatible = pipeline.generator.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval().requires_grad_(False)

    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    dataset = VideoDataset(**dataset_config)
    if len(dataset) != 1:
        raise ValueError(f"Quick check expects exactly one video, got {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(dataloader))

    encode_started = time.perf_counter()
    render_video = rearrange(
        batch["render_video"].to(device, dtype=dtype), "b t c h w -> b c t h w"
    )
    mask_video = rearrange(
        batch["mask_video"].to(device, dtype=dtype), "b t c h w -> b c t h w"
    )
    source_video_bcthw = rearrange(
        batch["source_video"].to(device, dtype=dtype), "b t c h w -> b c t h w"
    )
    target_video_bcthw = rearrange(
        batch.get("target_video", batch["source_video"]).to(device, dtype=dtype),
        "b t c h w -> b c t h w",
    )
    with torch.no_grad():
        render_latent = pipeline.vae.encode_to_latent(render_video).to(dtype=dtype)
        mask_latent = convert_mask_video(mask_video).to(device=device, dtype=dtype)
        target_latent = pipeline.vae.encode_to_latent(target_video_bcthw).to(dtype=dtype)
        ref_latent = pipeline.vae.encode_to_latent(source_video_bcthw).to(dtype=dtype)
    frames_per_block = int(config.num_frame_per_block)
    assert frames_per_block == 3
    latent_length = target_latent.shape[1]
    output_length = latent_length - latent_length % frames_per_block
    if output_length <= 0:
        raise ValueError(f"Encoded video has too few latent frames: {latent_length}")
    target_latent = target_latent[:, :output_length]
    render_latent = render_latent[:, :output_length]
    mask_latent = mask_latent[:, :output_length]
    noise_shape = (
        1,
        output_length,
        int(target_latent.shape[2]),
        int(target_latent.shape[3]),
        int(target_latent.shape[4]),
    )
    num_blocks = output_length // frames_per_block
    num_steps = len(pipeline.denoising_step_list)
    encode_seconds = time.perf_counter() - encode_started

    bundle_path = Path(args.noise_bundle).resolve()
    if args.create_noise_bundle:
        if bundle_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing noise bundle: {bundle_path}"
            )
        bundle = DeterministicNoiseBundle.create(
            shape=noise_shape,
            num_blocks=num_blocks,
            num_denoising_steps=num_steps,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )
        bundle.save(bundle_path)
    else:
        bundle = DeterministicNoiseBundle.load(bundle_path)
    _validate_noise_bundle(
        bundle, shape=noise_shape, num_blocks=num_blocks, num_steps=num_steps
    )
    sampled_noise = bundle.get_initial(device=device, dtype=dtype)

    layout = pipeline._runtime_layout(target_latent.shape[-2], target_latent.shape[-1])
    bank_writer = None
    if args.capture_kv:
        bank_writer = KVBankWriter(
            mapkv_config.bank_root,
            selected_layers=mapkv_config.selected_layers,
            num_layers=pipeline.num_transformer_blocks,
            recent_slot_len=layout["recent_slot_len"],
            frames_per_block=frames_per_block,
            tokens_per_frame=layout["tokens_per_frame"],
            dtype=dtype,
        )
    memory_contexts, memory_selections = _build_memory_contexts(
        config=mapkv_config,
        retrieval_plan_path=args.retrieval_plan,
        num_layers=pipeline.num_transformer_blocks,
        device=device,
        dtype=dtype,
    )

    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with torch.no_grad():
        pred_latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=batch["text"],
            ref_latent=ref_latent,
            render_latent=render_latent,
            mask_latent=mask_latent,
            decode=False,
            noise_provider=bundle,
            after_context_write=bank_writer,
            memory_contexts=memory_contexts or None,
        )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    if not bool(torch.isfinite(pred_latents).all()):
        raise FloatingPointError("Generated latents contain NaN or Inf")
    activation_audit = _validate_activation_audit(
        memory_contexts, pipeline.num_transformer_blocks, num_steps
    )
    in_process_replay = None
    if args.verify_memory_off_replay:
        if mapkv_config.enabled or memory_contexts:
            raise ValueError("--verify_memory_off_replay is valid for memory-off only")
        with torch.no_grad():
            replay_latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=batch["text"],
                ref_latent=ref_latent,
                render_latent=render_latent,
                mask_latent=mask_latent,
                decode=False,
                noise_provider=bundle,
            )
        torch.cuda.synchronize(device)
        in_process_max_abs_diff = float(
            (pred_latents.float() - replay_latents.float()).abs().max()
        )
        in_process_replay = {
            "max_abs_diff": in_process_max_abs_diff,
            "tolerance": args.replay_tolerance,
            "within_tolerance": in_process_max_abs_diff <= args.replay_tolerance,
        }
        torch.save(replay_latents.detach().cpu(), output_root / "replay_latents.pt")
        if args.require_replay_tolerance and not in_process_replay["within_tolerance"]:
            raise RuntimeError(
                f"Deterministic in-process replay failed: {in_process_replay}"
            )
    latent_path = output_root / "pred_latents.pt"
    torch.save(pred_latents.detach().cpu(), latent_path)

    decode_started = time.perf_counter()
    with torch.no_grad():
        pred_video = pipeline.vae.decode_to_pixel(pred_latents, use_cache=False)
        pred_video = (pred_video * 0.5 + 0.5).clamp(0, 1)
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started

    video_uint8 = (
        pred_video[0].permute(0, 2, 3, 1).float().cpu() * 255.0
    ).round().to(torch.uint8)
    pred_video_path = output_root / "pred.mp4"
    write_video(str(pred_video_path), video_uint8, fps=24)
    if args.video_output:
        video_output = Path(args.video_output).resolve()
        video_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred_video_path, video_output)

    source_video_01 = (source_video_bcthw * 0.5 + 0.5).clamp(0, 1)
    _save_rgb(source_video_01[0, :, 0], output_root / "anchor.png")
    target_tcw = batch.get("target_extrinsics")
    if target_tcw is None:
        target_tcw = torch.eye(4).repeat(pred_video.shape[1], 1, 1)
    mapping = build_block_mapping(
        pred_video=pred_video.detach().cpu(),
        latent_length=output_length,
        mask_latent=mask_latent.detach().cpu(),
        target_tcw=target_tcw,
        frames_per_block=frames_per_block,
        output_root=output_root,
    )
    _save_masks(mask_latent.detach().cpu(), output_root, frames_per_block)
    torch.save(torch.as_tensor(target_tcw).float().cpu(), output_root / "target_extrinsics.pt")

    if bank_writer is not None:
        for block in mapping:
            bank_writer.update_chunk_metadata(
                int(block["chunk_id"]),
                rgb_keyframe_id=int(block["rgb_center_index"]),
                png_path=block["png_path"],
                pose_metadata={"Tcw": block["Tcw"], "c2w": block["c2w"]},
                reference_valid_fraction=float(block["reference_valid_fraction"]),
            )
    for target, gate in pipeline.last_query_gates.items():
        keyframe = output_root / mapping[target]["png_path"]
        save_gate_overlay(
            keyframe,
            gate[0],
            output_root / "masks" / f"chunk_{target:04d}_query_gate_overlay.png",
        )

    replay = None
    if args.compare_latents_to:
        reference = torch.load(
            Path(args.compare_latents_to).resolve(), map_location="cpu", weights_only=True
        )
        if isinstance(reference, dict):
            reference = reference["pred_latents"]
        candidate = pred_latents.detach().cpu()
        if reference.shape != candidate.shape:
            raise ValueError(
                f"Replay latent shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
            )
        max_abs_diff = float((reference.float() - candidate.float()).abs().max())
        replay = {
            "reference": str(Path(args.compare_latents_to).resolve()),
            "max_abs_diff": max_abs_diff,
            "tolerance": args.replay_tolerance,
            "within_tolerance": max_abs_diff <= args.replay_tolerance,
        }
        if args.require_replay_tolerance and not replay["within_tolerance"]:
            raise RuntimeError(f"Deterministic replay failed: {replay}")

    run_metadata = {
        "run_name": args.run_name,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "checkpoint_load": {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        },
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "single_gpu": True,
        "torch_compile": False,
        "tae": False,
        "frames_per_block": frames_per_block,
        "latent_length_before_truncation": int(latent_length),
        "latent_length": int(output_length),
        "decoded_rgb_length": int(pred_video.shape[1]),
        "block_count": num_blocks,
        "runtime_layout": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in layout.items()
        },
        "noise_bundle": {
            "path": str(bundle_path),
            "seed": bundle.seed,
            "num_re_noise_tensors": len(bundle.re_noise),
        },
        "mapkv": {
            "enabled": mapkv_config.enabled,
            "mode": mapkv_config.mode,
            "alpha": mapkv_config.alpha,
            "selected_layers_raw": list(mapkv_config.selected_layers),
            "selected_layers_resolved": list(
                resolve_indices(
                    mapkv_config.selected_layers,
                    pipeline.num_transformer_blocks,
                    name="layer",
                )
            ),
            "selected_steps_raw": list(mapkv_config.selected_step_indices),
            "selected_steps_resolved": list(
                resolve_indices(mapkv_config.selected_step_indices, num_steps, name="step")
            ),
            "gate_mode": mapkv_config.gate.mode,
            "selections": memory_selections,
            "activation_audit": activation_audit,
        },
        "timing_seconds": {
            "encode": encode_seconds,
            "dit": inference_seconds,
            "decode": decode_seconds,
            "total": time.perf_counter() - started,
            "per_block": {
                str(key): value for key, value in pipeline.last_block_latencies.items()
            },
        },
        "replay": {
            "against_saved_latents": replay,
            "in_process_memory_off": in_process_replay,
        },
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    pipeline.vae.model.clear_cache()
    print(
        json.dumps(
            {
                "output": str(output_root),
                "replay_against_saved": replay,
                "in_process_memory_off_replay": in_process_replay,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
