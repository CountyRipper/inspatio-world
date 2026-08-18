#!/usr/bin/env python3
"""Deterministic single-GPU runner for the CUT3R/MapKV prototype.

This intentionally follows ``inference_causal_test.py`` for model, dataset, VAE,
and scheduler behavior.  MapKV is opt-in and only changes selected denoising
attention calls; capture and deterministic noise replay are independent hooks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import replace
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
from mapkv_proto.reference_kv_bank import ReferenceKVBankWriter
from mapkv_proto.memory_context import make_memory_context
from mapkv_proto.retrieval import RetrievalPlan
from mapkv_proto.revisit_pair import build_block_mapping
from mapkv_proto.visualization import save_gate_overlay
from mapkv.latent_control import LatentBlockIntervention
from mapkv.kv_bank import resolve_memory_layers
from mapkv.warp_reencode import (
    build_continuous_virtual_recent_plans,
    build_warp_reencode_plans,
    save_warp_reencode_artifacts,
)
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
    parser.add_argument("--traj_txt_path")
    parser.add_argument(
        "--target_pose_path",
        help="Exact absolute c2w [T,4,4]. When set, txt/spline generation is bypassed.",
    )
    parser.add_argument("--case_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="mapkv")
    parser.add_argument("--video_output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise_bundle", required=True)
    parser.add_argument("--create_noise_bundle", action="store_true")
    parser.add_argument("--capture_kv", action="store_true")
    parser.add_argument("--capture_chunks", nargs="+", type=int)
    parser.add_argument("--capture_ref_kv", action="store_true")
    parser.add_argument("--ref_bank_root")
    parser.add_argument("--ref_chunks", nargs="+", type=int)
    parser.add_argument("--latent_memory_path")
    parser.add_argument("--latent_source_chunks", nargs="+", type=int)
    parser.add_argument("--latent_target_chunks", nargs="+", type=int)
    parser.add_argument("--latent_strengths", nargs="+", type=float)
    parser.add_argument("--bank_root")
    parser.add_argument(
        "--mode",
        choices=("off", "baseline", "oracle", "wrong", "random", "zero", "pose", "geometry"),
    )
    parser.add_argument("--source_chunk", type=int)
    parser.add_argument("--wrong_chunk", type=int)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--target_chunks", nargs="+", type=int)
    parser.add_argument("--selected_layers", nargs="+", type=int)
    parser.add_argument(
        "--memory_layers",
        choices=("uniform8", "middle8", "explicit", "all"),
        help="Resolve memory layers from the runtime transformer depth.",
    )
    parser.add_argument("--selected_steps", nargs="+", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--injection_mode",
        choices=(
            "replace_recent_delta",
            "selected_recent_delta",
            "replace_ref_delta",
            "replace_both_delta",
            "residual_memory_attention",
        ),
    )
    parser.add_argument(
        "--gate_mode",
        choices=("global", "ref_blind", "surfel", "surfel_ref_blind"),
    )
    parser.add_argument("--retrieval_plan")
    parser.add_argument("--warp_reencode_recent", action="store_true")
    parser.add_argument("--continuous_virtual_recent", action="store_true")
    parser.add_argument(
        "--continuous_recent_fallback",
        choices=("raw", "warped"),
        default="raw",
        help="Use native last_pred (repaired) or camera-warped last_pred (legacy CAVR).",
    )
    parser.add_argument(
        "--continuous_query_gate",
        choices=("global", "surfel"),
        default="surfel",
        help="Apply Virtual Recent delta globally or only at M_history query tokens.",
    )
    parser.add_argument("--warp_source_latents")
    parser.add_argument("--warp_intrinsics_path")
    parser.add_argument("--warp_surfel_index")
    parser.add_argument("--warp_surfel_sequence")
    parser.add_argument("--warp_min_history_gap", type=int, default=2)
    parser.add_argument("--warp_feather_kernel", type=int, default=3)
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if args.target_pose_path:
        config.dataset.target_pose_path = str(Path(args.target_pose_path).resolve())
        config.dataset.traj_txt_path = None
    else:
        if not args.traj_txt_path:
            raise ValueError("Either --target_pose_path or --traj_txt_path is required")
        config.dataset.traj_txt_path = str(Path(args.traj_txt_path).resolve())
        config.dataset.target_pose_path = None
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
    if args.random_seed is not None:
        raw["random_seed"] = args.random_seed
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
    if args.injection_mode is not None:
        raw["injection_mode"] = args.injection_mode
    if args.gate_mode is not None:
        raw.setdefault("gate", {})["mode"] = args.gate_mode
    if args.bank_root is not None:
        raw.setdefault("bank", {})["root"] = args.bank_root
    mapkv = MapKVConfig.from_mapping(raw)
    if (
        mapkv.enabled
        and not mapkv.target_chunks
        and not args.continuous_virtual_recent
    ):
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
    reference_bank_root: str | None,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, object], list[dict]]:
    if not config.enabled or config.mode == "off":
        return {}, []
    bank = KVBank(config.bank_root)
    needs_reference_payload = config.injection_mode in {
        "replace_ref_delta",
        "replace_both_delta",
    }
    reference_bank = None
    if needs_reference_payload:
        if not reference_bank_root:
            raise ValueError(f"{config.injection_mode} requires --ref_bank_root")
        reference_bank = KVBank(reference_bank_root)
        if reference_bank.metadata.get("rope_layout") != "reference_slot_t0_t2":
            raise ValueError("Reference KV bank is not encoded with reference-slot RoPE")
    plan = RetrievalPlan(retrieval_plan_path) if retrieval_plan_path else None
    contexts = {}
    selections = []
    payload_cache = {}
    reference_payload_cache = {}
    for target_chunk in config.target_chunks:
        coverage = None
        selected_token_indices = None
        plan_source = None
        if plan is not None:
            plan_source = plan.selected_chunk(target_chunk)
            coverage = plan.load_coverage(target_chunk)
            selected_token_indices = plan.load_token_indices(target_chunk)
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
        elif config.mode in {"random", "zero"}:
            source_chunk = config.source_chunk
        else:
            if plan is None:
                raise ValueError(f"{config.mode} mode requires --retrieval_plan")
            source_chunk = plan_source
        selection = {
            "target_chunk": int(target_chunk),
            "source_chunk": None if source_chunk is None else int(source_chunk),
            "mode": config.mode,
            "payload_kind": (
                "zeroed_slot"
                if config.mode == "zero"
                else (
                    "token_permuted_historical" if config.mode == "random" else "native"
                )
            ),
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
        if config.gate.mode in {"surfel", "surfel_ref_blind"} and (
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
        layer_payloads = payload_cache[source_chunk]
        if config.injection_mode == "selected_recent_delta":
            if selected_token_indices is None or not len(selected_token_indices):
                raise ValueError(
                    "selected_recent_delta requires selected_token_indices_path "
                    f"for target {target_chunk}"
                )
            first_k = next(iter(layer_payloads.values()))[0]
            indices = selected_token_indices.to(
                device=first_k.device, dtype=torch.long
            )
            if int(indices.min()) < 0 or int(indices.max()) >= first_k.shape[1]:
                raise IndexError(
                    f"Selected token index outside [0,{first_k.shape[1]})"
                )
            layer_payloads = {
                layer: (
                    k.index_select(1, indices),
                    v.index_select(1, indices),
                )
                for layer, (k, v) in layer_payloads.items()
            }
            selection["selected_token_count"] = int(indices.numel())
            selection["selected_token_fraction"] = float(
                indices.numel() / first_k.shape[1]
            )
        reference_layer_payloads = None
        if reference_bank is not None:
            if source_chunk not in reference_payload_cache:
                reference_payload_cache[source_chunk] = reference_bank.materialize(
                    source_chunk,
                    selected_layers=config.selected_layers,
                    num_layers=num_layers,
                    device=device,
                    dtype=dtype,
                    pin_memory=config.pin_memory,
                )
            reference_layer_payloads = reference_payload_cache[source_chunk]
            selection["reference_rope_layout"] = "reference_slot_t0_t2"
        if config.mode == "zero":
            if config.injection_mode in {
                "replace_recent_delta",
                "replace_both_delta",
            }:
                layer_payloads = {
                    layer: (torch.zeros_like(k), torch.zeros_like(v))
                    for layer, (k, v) in layer_payloads.items()
                }
            if config.injection_mode in {
                "replace_ref_delta",
                "replace_both_delta",
            }:
                assert reference_layer_payloads is not None
                reference_layer_payloads = {
                    layer: (torch.zeros_like(k), torch.zeros_like(v))
                    for layer, (k, v) in reference_layer_payloads.items()
                }
        if config.mode == "random":
            first_payload = next(iter(layer_payloads.values()))
            token_count = int(first_payload[0].shape[1])
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(config.random_seed)
            permutation_cpu = torch.randperm(token_count, generator=cpu_generator)
            permutation_sha256 = hashlib.sha256(
                permutation_cpu.numpy().tobytes()
            ).hexdigest()
            layer_payloads = {
                layer: (
                    k.index_select(1, permutation_cpu.to(device=k.device)),
                    v.index_select(1, permutation_cpu.to(device=v.device)),
                )
                for layer, (k, v) in layer_payloads.items()
            }
            selection["random_seed"] = config.random_seed
            selection["token_permutation_sha256"] = permutation_sha256
        context = make_memory_context(
            target_block=target_chunk,
            source_chunk=source_chunk,
            layer_payloads=layer_payloads,
            reference_layer_payloads=reference_layer_payloads,
            selected_layers=config.selected_layers,
            selected_step_indices=config.selected_step_indices,
            alpha=config.alpha,
            injection_mode=config.injection_mode,
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
    if args.target_pose_path and args.traj_txt_path:
        raise ValueError("--target_pose_path and --traj_txt_path are mutually exclusive")
    if args.capture_chunks and not args.capture_kv:
        raise ValueError("--capture_chunks requires --capture_kv")
    if args.capture_ref_kv and (not args.ref_bank_root or not args.ref_chunks):
        raise ValueError(
            "--capture_ref_kv requires both --ref_bank_root and --ref_chunks"
        )
    if args.ref_chunks and not args.capture_ref_kv:
        raise ValueError("--ref_chunks requires --capture_ref_kv")
    latent_arguments = (
        args.latent_memory_path,
        args.latent_source_chunks,
        args.latent_target_chunks,
        args.latent_strengths,
    )
    if any(item is not None for item in latent_arguments) and not all(
        item is not None for item in latent_arguments
    ):
        raise ValueError(
            "Latent control requires memory path, source chunks, target chunks, "
            "and strengths together"
        )
    if args.latent_memory_path and not (
        len(args.latent_source_chunks)
        == len(args.latent_target_chunks)
        == len(args.latent_strengths)
    ):
        raise ValueError("Latent source/target/strength lists must have equal length")
    if args.warp_reencode_recent and args.continuous_virtual_recent:
        raise ValueError(
            "Block-on Warp-Reencode and Continuous CAVR are mutually exclusive"
        )
    if args.warp_reencode_recent or args.continuous_virtual_recent:
        if not args.warp_source_latents or not args.warp_intrinsics_path:
            raise ValueError(
                "Virtual Recent requires --warp_source_latents and "
                "--warp_intrinsics_path"
            )
        if not args.target_pose_path:
            raise ValueError("Warp-reencode requires the exact --target_pose_path")
        if args.retrieval_plan:
            raise ValueError(
                "Warp-reencode freezes the manual source and cannot use a retrieval plan"
            )
        if args.latent_memory_path:
            raise ValueError("Warp-reencode and direct latent control are separate runs")
    if args.continuous_virtual_recent and (
        not args.warp_surfel_index or not args.warp_surfel_sequence
    ):
        raise ValueError(
            "Continuous CAVR requires --warp_surfel_index and "
            "--warp_surfel_sequence"
        )
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
    if args.memory_layers is not None:
        mapkv_config = replace(
            mapkv_config,
            selected_layers=resolve_memory_layers(
                args.memory_layers,
                pipeline.num_transformer_blocks,
                mapkv_config.selected_layers,
            ),
        )
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
    if args.target_pose_path:
        exact_c2w = np.load(Path(args.target_pose_path).resolve())
        if exact_c2w.shape != (batch["source_video"].shape[1], 4, 4):
            raise ValueError(
                "Exact pose/source length mismatch after dataset loading: "
                f"{exact_c2w.shape} vs source T={batch['source_video'].shape[1]}"
            )

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
            capture_chunk_ids=args.capture_chunks,
        )
    virtual_recent_contexts = {}
    if args.warp_reencode_recent or args.continuous_virtual_recent:
        if not mapkv_config.enabled or mapkv_config.mode != "oracle":
            raise ValueError(
                "Warp-reencode requires enabled manual oracle mode with a fixed source"
            )
        if mapkv_config.source_chunk is None:
            raise ValueError("Warp-reencode requires --source_chunk")
        if mapkv_config.injection_mode != "replace_recent_delta":
            raise ValueError("Warp-reencode supports replace_recent_delta only")
        if mapkv_config.gate.mode != "global":
            raise ValueError(
                "Warp-reencode blends in latent space and uses a global KV delta gate"
            )
        selected_writer_layers = resolve_indices(
            mapkv_config.selected_layers,
            pipeline.num_transformer_blocks,
            name="warp-reencode layer",
        )
        common_virtual_kwargs = dict(
            source_latents_path=args.warp_source_latents,
            source_chunk=mapkv_config.source_chunk,
            target_pose_path=args.target_pose_path,
            intrinsics_path=args.warp_intrinsics_path,
            latent_length=output_length,
            rgb_length=int(batch["source_video"].shape[1]),
            frames_per_block=frames_per_block,
            latent_hw=(
                int(target_latent.shape[-2]),
                int(target_latent.shape[-1]),
            ),
            image_hw=(
                int(source_video_bcthw.shape[-2]),
                int(source_video_bcthw.shape[-1]),
            ),
            selected_layers=selected_writer_layers,
            selected_step_indices=mapkv_config.selected_step_indices,
            alpha=mapkv_config.alpha,
            feather_kernel=args.warp_feather_kernel,
            device=device,
            dtype=dtype,
        )
        if args.continuous_virtual_recent:
            virtual_recent_contexts, memory_selections = (
                build_continuous_virtual_recent_plans(
                    **common_virtual_kwargs,
                    surfel_index_path=args.warp_surfel_index,
                    surfel_sequence_path=args.warp_surfel_sequence,
                    min_history_gap_chunks=args.warp_min_history_gap,
                    warp_short_term_recent=(
                        args.continuous_recent_fallback == "warped"
                    ),
                    query_gate_mode=(
                        "surfel_exact"
                        if args.continuous_query_gate == "surfel"
                        else "global"
                    ),
                )
            )
        else:
            virtual_recent_contexts, memory_selections = (
                build_warp_reencode_plans(
                    **common_virtual_kwargs,
                    target_chunks=mapkv_config.target_chunks,
                )
            )
        memory_contexts = {}
    else:
        memory_contexts, memory_selections = _build_memory_contexts(
            config=mapkv_config,
            retrieval_plan_path=args.retrieval_plan,
            reference_bank_root=args.ref_bank_root,
            num_layers=pipeline.num_transformer_blocks,
            device=device,
            dtype=dtype,
        )
    latent_block_interventions = {}
    if args.latent_memory_path:
        if mapkv_config.enabled:
            raise ValueError("KV memory and direct latent control must run separately")
        latent_memory = torch.load(
            Path(args.latent_memory_path).resolve(),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(latent_memory, dict):
            latent_memory = latent_memory["pred_latents"]
        if tuple(latent_memory.shape) != noise_shape:
            raise ValueError(
                f"Latent memory shape {tuple(latent_memory.shape)} != {noise_shape}"
            )
        for source_chunk, target_block, strength in zip(
            args.latent_source_chunks,
            args.latent_target_chunks,
            args.latent_strengths,
        ):
            source_chunk = int(source_chunk)
            target_block = int(target_block)
            start = source_chunk * frames_per_block
            stop = start + frames_per_block
            if source_chunk < 0 or stop > output_length:
                raise IndexError(f"Latent source chunk {source_chunk} is out of range")
            if target_block <= source_chunk or target_block >= num_blocks:
                raise ValueError(
                    f"Latent target {target_block} must be after source "
                    f"{source_chunk} and inside [0, {num_blocks})"
                )
            latent_block_interventions[target_block] = LatentBlockIntervention(
                target_block=target_block,
                source_chunk=source_chunk,
                clean_latent=latent_memory[:, start:stop].to(
                    device=device, dtype=dtype
                ),
                strength=float(strength),
            )
        del latent_memory

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
            virtual_recent_contexts=virtual_recent_contexts or None,
            latent_block_interventions=latent_block_interventions or None,
        )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    if not bool(torch.isfinite(pred_latents).all()):
        raise FloatingPointError("Generated latents contain NaN or Inf")
    active_memory_contexts = dict(memory_contexts)
    active_memory_contexts.update(pipeline.last_virtual_memory_contexts)
    activation_audit = _validate_activation_audit(
        active_memory_contexts, pipeline.num_transformer_blocks, num_steps
    )
    in_process_replay = None
    if args.verify_memory_off_replay:
        if (
            mapkv_config.enabled
            or memory_contexts
            or virtual_recent_contexts
            or latent_block_interventions
        ):
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
    reference_capture = {"enabled": False, "chunks": []}
    if args.capture_ref_kv:
        if in_process_replay is not None:
            del replay_latents
        pipeline.kv_cache1 = None
        gc.collect()
        torch.cuda.empty_cache()
        selected_reference_layers = resolve_indices(
            mapkv_config.selected_layers,
            pipeline.num_transformer_blocks,
            name="reference capture layer",
        )
        reference_writer = ReferenceKVBankWriter(
            args.ref_bank_root,
            selected_layers=selected_reference_layers,
            num_layers=pipeline.num_transformer_blocks,
            slot_len=layout["recent_slot_len"],
            frames_per_block=frames_per_block,
            tokens_per_frame=layout["tokens_per_frame"],
            dtype=dtype,
        )
        reference_capture_started = time.perf_counter()
        for chunk_id in args.ref_chunks:
            chunk_id = int(chunk_id)
            start = chunk_id * frames_per_block
            stop = start + frames_per_block
            if start < 0 or stop > pred_latents.shape[1]:
                raise IndexError(
                    f"Reference capture chunk {chunk_id} is outside "
                    f"[0, {num_blocks})"
                )
            clean_block = pred_latents[:, start:stop]
            payloads = pipeline.encode_clean_latent_as_reference_slot(
                clean_block,
                batch["text"],
                selected_reference_layers,
            )
            reference_writer.write_chunk(
                chunk_id=chunk_id,
                layer_payloads=payloads,
                metadata={
                    "source": "baseline_pred_latents",
                    "source_latent_range": [start, stop],
                },
            )
            reference_capture["chunks"].append(chunk_id)
            del payloads
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        reference_capture.update(
            {
                "enabled": True,
                "bank_root": str(Path(args.ref_bank_root).resolve()),
                "rope_layout": "reference_slot_t0_t2",
                "capture_type": "clean_reference_reencode",
                "selected_layers": list(selected_reference_layers),
                "seconds": time.perf_counter() - reference_capture_started,
            }
        )
    latent_path = output_root / "pred_latents.pt"
    torch.save(pred_latents.detach().cpu(), latent_path)

    decode_started = time.perf_counter()
    with torch.no_grad():
        pred_video = pipeline.vae.decode_to_pixel(pred_latents, use_cache=False)
        pred_video = (pred_video * 0.5 + 0.5).clamp(0, 1)
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    warp_reencode_manifest = None
    if virtual_recent_contexts:
        warp_reencode_manifest = save_warp_reencode_artifacts(
            plans=virtual_recent_contexts,
            vae=pipeline.vae,
            output_root=output_root,
            device=device,
        )

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
        per_chunk_max_abs_diff = {}
        for chunk_id, start in enumerate(range(0, output_length, frames_per_block)):
            per_chunk_max_abs_diff[str(chunk_id)] = float(
                (
                    reference[:, start : start + frames_per_block].float()
                    - candidate[:, start : start + frames_per_block].float()
                )
                .abs()
                .max()
            )
        replay = {
            "reference": str(Path(args.compare_latents_to).resolve()),
            "max_abs_diff": max_abs_diff,
            "tolerance": args.replay_tolerance,
            "within_tolerance": max_abs_diff <= args.replay_tolerance,
            "per_chunk_max_abs_diff": per_chunk_max_abs_diff,
        }
        if args.require_replay_tolerance and not replay["within_tolerance"]:
            raise RuntimeError(f"Deterministic replay failed: {replay}")

    benchmark_metadata = None
    if args.target_pose_path:
        exact_path = Path(args.target_pose_path).resolve()
        exact_c2w = np.load(exact_path)
        batch_tcw = torch.as_tensor(target_tcw).squeeze(0).double().cpu().numpy()
        dataset_pose_max_abs_diff = float(
            np.max(np.abs(batch_tcw - np.linalg.inv(exact_c2w)))
        )
        if dataset_pose_max_abs_diff > 1e-6:
            raise RuntimeError(
                "Dataset target extrinsics diverged from exact pose artifact: "
                f"{dataset_pose_max_abs_diff}"
            )
        runtime_entries = json.loads(runtime_json.read_text(encoding="utf-8"))
        entry = runtime_entries[0]
        render_root = Path(entry["vggt_depth_path"]) / "render"
        prompt_digest = hashlib.sha256(
            "\n".join(str(item) for item in batch["text"]).encode("utf-8")
        ).hexdigest()
        benchmark_metadata = {
            "decision_eligible": True,
            "case_dir": None if args.case_dir is None else str(Path(args.case_dir).resolve()),
            "target_pose_path": str(exact_path),
            "target_pose_sha256": _sha256_file(exact_path),
            "dataset_pose_max_abs_diff": dataset_pose_max_abs_diff,
            "source_rgb_length": int(batch["source_video"].shape[1]),
            "render_rgb_length": int(batch["render_video"].shape[1]),
            "mask_rgb_length": int(batch["mask_video"].shape[1]),
            "input_checksums": {
                "static_source": _sha256_file(entry["video_path"]),
                "render": _sha256_file(render_root / "render_offline.mp4"),
                "mask": _sha256_file(render_root / "mask_offline.mp4"),
                "prompt": prompt_digest,
                "noise_bundle": _sha256_file(bundle_path),
            },
        }

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
        "benchmark": benchmark_metadata,
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
            "injection_mode": mapkv_config.injection_mode,
            "selected_layers_raw": list(mapkv_config.selected_layers),
            "memory_layers_mode": args.memory_layers or "explicit",
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
            "recent_bank_root": str(Path(mapkv_config.bank_root).resolve()),
            "reference_bank_root": (
                None if args.ref_bank_root is None else str(Path(args.ref_bank_root).resolve())
            ),
            "recent_capture_chunks": args.capture_chunks,
            "reference_capture": reference_capture,
            "base_runtime_cache_replaced": False,
            "selections": memory_selections,
            "activation_audit": activation_audit,
            "cache_audits": {
                str(target): context.cache_audit
                for target, context in active_memory_contexts.items()
            },
            "warp_reencode": {
                "enabled": bool(virtual_recent_contexts),
                "mode": (
                    "continuous_geometry_reprojected_virtual_recent"
                    if args.continuous_virtual_recent
                    else (
                        "block_on_warp_reencode"
                        if args.warp_reencode_recent
                        else None
                    )
                ),
                "source_latents": args.warp_source_latents,
                "intrinsics_path": args.warp_intrinsics_path,
                "surfel_index": args.warp_surfel_index,
                "surfel_sequence": args.warp_surfel_sequence,
                "min_history_gap_chunks": args.warp_min_history_gap,
                "feather_kernel": args.warp_feather_kernel,
                "short_term_recent": (
                    args.continuous_recent_fallback
                    if args.continuous_virtual_recent
                    else "raw"
                ),
                "attention_query_gate": (
                    args.continuous_query_gate
                    if args.continuous_virtual_recent
                    else "global"
                ),
                "writer_isolated_from_runtime_cache": True,
                "manifest": warp_reencode_manifest,
                "audits": {
                    str(target): audit
                    for target, audit in pipeline.last_virtual_recent_audits.items()
                },
            },
        },
        "latent_control": {
            "enabled": bool(latent_block_interventions),
            "mode": (
                "direct_clean_x0_block_override"
                if latent_block_interventions
                else None
            ),
            "memory_path": args.latent_memory_path,
            "interventions": {
                str(target): intervention.audit
                for target, intervention in latent_block_interventions.items()
            },
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
