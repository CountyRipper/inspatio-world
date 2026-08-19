from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, SequentialSampler

from datasets.video_dataset import VideoDataset
from mapkv.memory_adapter import (
    MemoryAdapterConfig,
    MemoryAdapterContext,
    MemoryPatchAdapter,
    freeze_backbone_for_adapter,
    save_adapter_checkpoint,
)
from mapkv_proto.deterministic_noise import DeterministicNoiseBundle
from pipeline import CausalInferencePipeline
from utils.render_warper import convert_mask_video


@dataclass(frozen=True)
class AdapterCaseSpec:
    case_id: str
    case_dir: Path
    baseline_root: Path
    sample_root: Path
    noise_bundle: Path


@dataclass
class AdapterSample:
    case_id: str
    block_id: int
    source_chunk: int
    memory: torch.Tensor
    recent: torch.Tensor
    baseline: torch.Tensor
    need: torch.Tensor
    source_valid: torch.Tensor
    ref: torch.Tensor
    render: torch.Tensor
    mask: torch.Tensor
    conditioning: dict[str, torch.Tensor]
    noise_by_step: tuple[torch.Tensor, ...]


def balanced_training_position(
    iteration: int,
    *,
    case_count: int,
    sample_count: int,
    denoising_step_count: int,
) -> tuple[int, int, int]:
    if min(case_count, sample_count, denoising_step_count) <= 0:
        raise ValueError("Balanced schedule cardinalities must be positive")
    case_index = int(iteration) % int(case_count)
    local_iteration = int(iteration) // int(case_count)
    sample_index = local_iteration % int(sample_count)
    step_index = (
        local_iteration // int(sample_count)
    ) % int(denoising_step_count)
    return case_index, sample_index, step_index


def _masked_l1(
    left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    value = mask.to(device=left.device, dtype=left.dtype)
    if value.ndim == 4:
        value = value.unsqueeze(2)
    if value.shape[-2:] != left.shape[-2:]:
        b, f = value.shape[:2]
        value = F.interpolate(
            value.reshape(b * f, 1, *value.shape[-2:]),
            size=left.shape[-2:],
            mode="nearest",
        ).reshape(b, f, 1, *left.shape[-2:])
    denominator = value.sum().clamp_min(1.0) * left.shape[2]
    return ((left - right).abs() * value).sum() / denominator


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float()
    if value.ndim == 5:
        value = value.squeeze(2)
    b, f, h, w = value.shape
    flat = value.reshape(b * f, 1, h, w)
    dilated = F.max_pool2d(flat, 3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - flat, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1).reshape(b, f, 1, h, w)


def _spatial_grad(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value[..., :, 1:] - value[..., :, :-1], value[..., 1:, :] - value[..., :-1, :]


def _boundary_loss(
    predicted: torch.Tensor, target: torch.Tensor, band: torch.Tensor
) -> torch.Tensor:
    dx, dy = _spatial_grad(predicted)
    tx, ty = _spatial_grad(target)
    mask_x = torch.minimum(band[..., :, 1:], band[..., :, :-1])
    mask_y = torch.minimum(band[..., 1:, :], band[..., :-1, :])
    return _masked_l1(dx, tx, mask_x) + _masked_l1(dy, ty, mask_y)


def _temporal_loss(
    predicted: torch.Tensor, target: torch.Tensor, need: torch.Tensor
) -> torch.Tensor:
    if predicted.shape[1] < 2:
        return predicted.new_zeros(())
    delta_pred = predicted[:, 1:] - predicted[:, :-1]
    delta_target = target[:, 1:] - target[:, :-1]
    union = torch.maximum(need[:, 1:], need[:, :-1])
    return _masked_l1(delta_pred, delta_target, union)


def compute_adapter_losses(
    predicted: torch.Tensor,
    *,
    memory: torch.Tensor,
    baseline: torch.Tensor,
    need: torch.Tensor,
    source_valid: torch.Tensor,
    rgb_prediction: torch.Tensor | None = None,
    rgb_memory: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    teacher = need.to(predicted.dtype) * memory + (
        1.0 - need.to(predicted.dtype)
    ) * baseline
    losses = {
        "memory_latent": _masked_l1(predicted, memory, need),
        "source": _masked_l1(predicted, baseline, source_valid),
        "boundary": _boundary_loss(predicted, teacher, _boundary(need)),
        "temporal": _temporal_loss(predicted, teacher, need),
    }
    if rgb_prediction is None or rgb_memory is None:
        losses["memory_rgb"] = predicted.new_zeros(())
    else:
        b, f, _, h, w = rgb_prediction.shape
        need_rgb = F.interpolate(
            need.permute(0, 2, 1, 3, 4).float(),
            size=(f, h, w),
            mode="trilinear",
            align_corners=False,
        ).permute(0, 2, 1, 3, 4)
        losses["memory_rgb"] = _masked_l1(
            rgb_prediction, rgb_memory, need_rgb
        )
    losses["core_total"] = (
        losses["memory_latent"]
        + losses["source"]
        + 0.25 * losses["boundary"]
        + 0.25 * losses["temporal"]
    )
    losses["total"] = losses["core_total"] + losses["memory_rgb"]
    return losses


def _pipeline_config(repo: Path, wan_model_folder: Path):
    config = OmegaConf.merge(
        OmegaConf.load(repo / "configs/default_config.yaml"),
        OmegaConf.load(repo / "configs/inference_1.3b.yaml"),
    )
    config.wan_model_folder = str(wan_model_folder)
    for item in config.generator.weight_list:
        item.path = str(wan_model_folder)
    return config


def load_frozen_pipeline(
    *, repo: Path, asset_root: Path, device: torch.device
) -> tuple[CausalInferencePipeline, object]:
    config = _pipeline_config(repo, asset_root / "checkpoints/Wan2.1-T2V-1.3B")
    pipeline = CausalInferencePipeline(config, device=device)
    state = load_file(
        str(
            asset_root
            / "checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"
        )
    )
    incompatible = pipeline.generator.load_state_dict(state, strict=False)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Backbone checkpoint mismatch: {incompatible.missing_keys}, "
            f"{incompatible.unexpected_keys}"
        )
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval().requires_grad_(False)
    return pipeline, config


def _case_batch(spec: AdapterCaseSpec, config) -> dict:
    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    dataset_config.update(
        {
            "json_path": str(spec.case_dir / "input.json"),
            "target_pose_path": str(spec.case_dir / "target_poses.npy"),
            "traj_txt_path": None,
            "adaptive_frame": False,
        }
    )
    dataset = VideoDataset(**dataset_config)
    if len(dataset) != 1:
        raise ValueError(f"Adapter case {spec.case_id} must contain one video")
    return next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                sampler=SequentialSampler(dataset),
                num_workers=0,
                drop_last=False,
            )
        )
    )


@torch.no_grad()
def build_case_samples(
    *,
    spec: AdapterCaseSpec,
    pipeline: CausalInferencePipeline,
    config,
    device: torch.device,
    minimum_coverage: float,
) -> list[AdapterSample]:
    batch = _case_batch(spec, config)
    dtype = torch.bfloat16
    source = rearrange(
        batch["source_video"].to(device, dtype=dtype), "b t c h w -> b c t h w"
    )
    render = rearrange(
        batch["render_video"].to(device, dtype=dtype), "b t c h w -> b c t h w"
    )
    mask_rgb = rearrange(
        batch["mask_video"].to(device, dtype=dtype), "b t c h w -> b c t h w"
    )
    ref_latent = pipeline.vae.encode_to_latent(source).to(dtype=dtype)
    render_latent = pipeline.vae.encode_to_latent(render).to(dtype=dtype)
    mask_latent = convert_mask_video(mask_rgb).to(device=device, dtype=dtype)
    baseline = torch.load(
        spec.baseline_root / "pred_latents.pt", map_location="cpu", weights_only=True
    ).to(device=device, dtype=dtype)
    conditioning = pipeline.text_encoder(text_prompts=batch["text"])
    bundle = DeterministicNoiseBundle.load(spec.noise_bundle)
    initial = bundle.get_initial(device=device, dtype=dtype)
    frames_per_block = int(config.num_frame_per_block)
    samples: list[AdapterSample] = []
    for block_root in sorted((spec.sample_root / "memory_adapter").glob("block_*")):
        block_id = int(block_root.name.split("_")[-1])
        memory = torch.load(
            block_root / "L_mem.pt", map_location="cpu", weights_only=True
        ).to(device=device, dtype=dtype)
        need = torch.load(
            block_root / "M_need.pt", map_location="cpu", weights_only=True
        ).to(device=device, dtype=torch.float32)
        if need.ndim == 4:
            need = need.unsqueeze(2)
        coverage = float(need.mean().item())
        if coverage < float(minimum_coverage):
            continue
        start = block_id * frames_per_block
        stop = start + frames_per_block
        recent = baseline[:, start - frames_per_block:start]
        current = baseline[:, start:stop]
        source_valid = (
            ((mask_latent[:, start:stop].float() + 1.0) * 0.5)
            .clamp(0, 1)
            .mean(dim=2, keepdim=True)
        )
        step_noises = [initial[:, start:stop]]
        for step in range(1, len(pipeline.denoising_step_list)):
            step_noises.append(
                bundle.get_re_noise(
                    block_id=block_id,
                    step_index=step - 1,
                    like=current,
                )
            )
        manifest = json.loads(
            (spec.sample_root / "memory_adapter/manifest.json").read_text()
        )
        entry = next(
            value for value in manifest["blocks"]
            if int(value["target_block"]) == block_id
        )
        samples.append(
            AdapterSample(
                case_id=spec.case_id,
                block_id=block_id,
                source_chunk=int(entry["source_chunk"]),
                memory=memory,
                recent=recent,
                baseline=current,
                need=need,
                source_valid=source_valid,
                ref=ref_latent[:, start:stop],
                render=render_latent[:, start:stop],
                mask=mask_latent[:, start:stop],
                conditioning={key: value.detach() for key, value in conditioning.items()},
                noise_by_step=tuple(step_noises),
            )
        )
    if not samples:
        raise RuntimeError(
            f"No {spec.case_id} blocks meet M_need >= {minimum_coverage:.3f}"
        )
    return samples


@torch.no_grad()
def _write_native_context(
    pipeline: CausalInferencePipeline, sample: AdapterSample
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    b, f, _, h, w = sample.recent.shape
    pipeline._initialize_kv_cache(
        batch_size=b,
        dtype=sample.recent.dtype,
        device=sample.recent.device,
        latent_height=h,
        latent_width=w,
    )
    zeros_ref = torch.zeros_like(sample.ref)
    padded_ref = torch.cat(
        [sample.ref, zeros_ref[:, :, :4], zeros_ref], dim=2
    )
    zeros_recent = torch.zeros_like(sample.recent)
    padded_recent = torch.cat(
        [sample.recent, zeros_recent[:, :, :4], zeros_recent], dim=2
    )
    context = torch.cat([padded_ref, padded_recent], dim=1)
    render_block = torch.cat([sample.mask, sample.render], dim=2)
    timestep_zero = torch.zeros(
        [b, f], device=sample.recent.device, dtype=torch.int64
    )
    pipeline.generator(
        noisy_image_or_video=context,
        conditional_dict=sample.conditioning,
        timestep=timestep_zero,
        kv_cache=pipeline.kv_cache1,
        render_latent_input=render_block,
        kv_size=(0, -1),
        freqs_offset=0,
    )
    return pipeline.kv_cache1, render_block


def _predict_sample(
    *,
    pipeline: CausalInferencePipeline,
    sample: AdapterSample,
    step_index: int,
) -> torch.Tensor:
    cache, render_block = _write_native_context(pipeline, sample)
    current_timestep = pipeline.denoising_step_list[int(step_index)]
    b, f = sample.baseline.shape[:2]
    timestep = torch.ones(
        [b, f], device=sample.baseline.device, dtype=torch.int64
    ) * current_timestep
    teacher = sample.need.to(sample.baseline.dtype) * sample.memory + (
        1.0 - sample.need.to(sample.baseline.dtype)
    ) * sample.baseline
    noisy = pipeline.scheduler.add_noise(
        teacher.flatten(0, 1),
        sample.noise_by_step[int(step_index)].flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, teacher.shape[:2])
    context = MemoryAdapterContext(
        target_block=sample.block_id,
        source_chunk=sample.source_chunk,
        memory_latent=sample.memory,
        recent_latent=sample.recent,
        need_mask=sample.need,
    )
    layout = pipeline._runtime_layout(teacher.shape[-2], teacher.shape[-1])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, prediction = pipeline.generator(
            noisy_image_or_video=noisy,
            conditional_dict=sample.conditioning,
            timestep=timestep,
            kv_cache=cache,
            kv_size=(0, layout["kv_size_used_for_nonfirst_block"]),
            render_latent_input=render_block,
            freqs_offset=6,
            memory_adapter_context=context,
        )
    return prediction


def train_adapter(
    *,
    pipeline: CausalInferencePipeline,
    samples_by_case: dict[str, list[AdapterSample]],
    output_dir: Path,
    steps: int,
    learning_rate: float,
    rgb_every: int,
    seed: int,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    adapter = pipeline.generator.model.memory_adapter
    if not isinstance(adapter, MemoryPatchAdapter):
        raise RuntimeError("Training requires an installed MemoryPatchAdapter")
    freeze_audit = freeze_backbone_for_adapter(pipeline.generator.model, adapter)
    pipeline.generator.model.gradient_checkpointing = True
    adapter.train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    case_ids = sorted(samples_by_case)
    curve = []
    initial_loss = None
    best_loss = float("inf")
    started = time.perf_counter()
    for iteration in range(int(steps)):
        case_index, _, _ = balanced_training_position(
            iteration,
            case_count=len(case_ids),
            sample_count=1,
            denoising_step_count=len(pipeline.denoising_step_list),
        )
        case_id = case_ids[case_index]
        choices = samples_by_case[case_id]
        _, sample_index, step_index = balanced_training_position(
            iteration,
            case_count=len(case_ids),
            sample_count=len(choices),
            denoising_step_count=len(pipeline.denoising_step_list),
        )
        sample = choices[sample_index]
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict_sample(
            pipeline=pipeline, sample=sample, step_index=step_index
        )
        rgb_prediction = None
        rgb_memory = None
        rgb_loss_active = (
            rgb_every > 0
            and (iteration // len(case_ids)) % rgb_every == 0
        )
        if rgb_loss_active:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rgb_prediction = pipeline.vae.decode_to_pixel(
                    prediction, use_cache=False
                )
            with torch.no_grad():
                rgb_memory = pipeline.vae.decode_to_pixel(
                    sample.memory, use_cache=False
                )
        losses = compute_adapter_losses(
            prediction.float(),
            memory=sample.memory.float(),
            baseline=sample.baseline.float(),
            need=sample.need.float(),
            source_valid=sample.source_valid.float(),
            rgb_prediction=(None if rgb_prediction is None else rgb_prediction.float()),
            rgb_memory=(None if rgb_memory is None else rgb_memory.float()),
        )
        losses["total"].backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0).item()
        )
        optimizer.step()
        value = float(losses["total"].detach().item())
        if initial_loss is None:
            initial_loss = value
        best_loss = min(best_loss, value)
        record = {
            "iteration": iteration,
            "case_id": case_id,
            "block_id": sample.block_id,
            "step_index": step_index,
            "need_coverage": float(sample.need.mean().item()),
            "rgb_loss_active": bool(rgb_loss_active),
            "gradient_norm": gradient_norm,
            **{
                key: float(loss.detach().item()) for key, loss in losses.items()
            },
        }
        curve.append(record)
        if iteration % 5 == 0 or iteration == steps - 1:
            print(json.dumps(record, ensure_ascii=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_curve.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8"
    )
    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for record in curve:
        grouped.setdefault(
            (record["case_id"], record["block_id"], record["step_index"]), []
        ).append(record)
    matched_reductions = []
    for records in grouped.values():
        if len(records) < 2:
            continue
        first = records[0].get("core_total")
        if first is None:
            first = (
                records[0]["memory_latent"] + records[0]["source"]
                + 0.25 * records[0]["boundary"]
                + 0.25 * records[0]["temporal"]
            )
        last = records[-1].get("core_total")
        if last is None:
            last = (
                records[-1]["memory_latent"] + records[-1]["source"]
                + 0.25 * records[-1]["boundary"]
                + 0.25 * records[-1]["temporal"]
            )
        matched_reductions.append((first - last) / max(abs(first), 1e-8))
    summary = {
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "cases": {
            key: [sample.block_id for sample in value]
            for key, value in samples_by_case.items()
        },
        "initial_total_loss": initial_loss,
        "best_total_loss": best_loss,
        "final_total_loss": curve[-1]["total"],
        "loss_reduction_fraction": (
            None
            if not initial_loss
            else float((initial_loss - curve[-1]["total"]) / initial_loss)
        ),
        "matched_core_loss_reduction_fraction": (
            None
            if not matched_reductions
            else float(sum(matched_reductions) / len(matched_reductions))
        ),
        "seconds": time.perf_counter() - started,
        "freeze_audit": freeze_audit,
        "config": asdict(adapter.config),
    }
    save_adapter_checkpoint(
        output_dir / "checkpoint/adapter.pt", adapter, training=summary
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _parse_case(value: str) -> AdapterCaseSpec:
    parts = value.split("::")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "case must be ID::CASE_DIR::BASELINE_ROOT::SAMPLE_ROOT::NOISE_BUNDLE"
        )
    return AdapterCaseSpec(
        case_id=parts[0],
        case_dir=Path(parts[1]).resolve(),
        baseline_root=Path(parts[2]).resolve(),
        sample_root=Path(parts[3]).resolve(),
        noise_bundle=Path(parts[4]).resolve(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-InSpatio MapKV adapter")
    parser.add_argument("--case", action="append", required=True, type=_parse_case)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--asset_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--inject_middle", action="store_true")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--rgb_every", type=int, default=8)
    parser.add_argument("--minimum_coverage", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    pipeline, config = load_frozen_pipeline(
        repo=repo, asset_root=Path(args.asset_root).resolve(), device=device
    )
    adapter = MemoryPatchAdapter(
        MemoryAdapterConfig(
            hidden_channels=args.hidden_channels,
            model_dim=int(pipeline.generator.model.dim),
            patch_size=tuple(int(value) for value in pipeline.generator.model.patch_size),
            inject_middle=bool(args.inject_middle),
            middle_start=(
                pipeline.num_transformer_blocks // 3
                if args.inject_middle else None
            ),
            middle_stop=(
                2 * pipeline.num_transformer_blocks // 3
                if args.inject_middle else None
            ),
        )
    ).to(device=device)
    pipeline.generator.model.memory_adapter = adapter
    samples = {
        spec.case_id: build_case_samples(
            spec=spec,
            pipeline=pipeline,
            config=config,
            device=device,
            minimum_coverage=args.minimum_coverage,
        )
        for spec in args.case
    }
    # Prompt embeddings and encoded tensors are now resident; release the 11B T5
    # host before keeping activation graphs for the frozen DiT.
    pipeline.text_encoder.to(device="cpu")
    torch.cuda.empty_cache()
    summary = train_adapter(
        pipeline=pipeline,
        samples_by_case=samples,
        output_dir=Path(args.output_dir).resolve(),
        steps=args.steps,
        learning_rate=args.learning_rate,
        rgb_every=args.rgb_every,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
