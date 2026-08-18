from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from mapkv_proto.memory_context import make_memory_context
from wan.modules.model import rope_apply_given_freqs

from .warp_reencode import (
    WarpReencodePlan,
    build_continuous_virtual_recent_plans,
    warp_latent,
)


def _resize_grid(
    grid: torch.Tensor, token_hw: tuple[int, int]
) -> torch.Tensor:
    """Resize normalized target->source coordinates to token centers."""
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"grid must be [F,H,W,2], got {tuple(grid.shape)}")
    if tuple(grid.shape[1:3]) == tuple(token_hw):
        return grid
    resized = F.interpolate(
        grid.permute(0, 3, 1, 2),
        size=token_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).contiguous()


def _warp_token_payload(
    payload: torch.Tensor,
    grid: torch.Tensor,
    *,
    frames: int,
    token_hw: tuple[int, int],
) -> torch.Tensor:
    """Warp [B,F*H*W,...] payload into a target token grid."""
    batch, sequence = payload.shape[:2]
    height, width = token_hw
    if sequence != frames * height * width:
        raise ValueError(
            f"Token payload length {sequence} != {frames}*{height}*{width}"
        )
    tail = payload.shape[2:]
    channels = 1
    for value in tail:
        channels *= int(value)
    video = (
        payload.reshape(batch, frames, height, width, channels)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
    )
    warped = warp_latent(video, _resize_grid(grid, token_hw))
    return (
        warped.permute(0, 1, 3, 4, 2)
        .reshape(batch, sequence, *tail)
        .contiguous()
    )


def _memory_token_gate(
    memory_coverage: torch.Tensor,
    token_hw: tuple[int, int],
) -> torch.Tensor:
    if memory_coverage.ndim != 4:
        raise ValueError(
            "memory coverage must be [B,F,H,W], got "
            f"{tuple(memory_coverage.shape)}"
        )
    batch, frames = memory_coverage.shape[:2]
    pooled = F.adaptive_max_pool2d(
        memory_coverage.float().reshape(
            batch * frames, 1, *memory_coverage.shape[-2:]
        ),
        token_hw,
    )
    return (pooled > 0).reshape(batch, frames * token_hw[0] * token_hw[1])


def _recent_freqs(model, *, frames: int, token_hw: tuple[int, int]) -> torch.Tensor:
    device = model.patch_embedding.weight.device
    if not hasattr(model, "freqs"):
        model.init_freqs(device)
    elif model.freqs.device != device:
        model.freqs = model.freqs.to(device)
    height, width = token_hw
    head_dim = int(model.dim) // int(model.num_heads)
    half = head_dim // 2
    temporal, vertical, horizontal = model.freqs.split(
        [half - 2 * (half // 3), half // 3, half // 3], dim=1
    )
    # Native writer receives [Ref t0-t2, Recent t3-t5].
    return torch.cat(
        [
            temporal[frames : 2 * frames]
            .view(frames, 1, 1, -1)
            .expand(frames, height, width, -1),
            vertical[:height]
            .view(1, height, 1, -1)
            .expand(frames, height, width, -1),
            horizontal[:width]
            .view(1, 1, width, -1)
            .expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, -1)


@torch.no_grad()
def build_canonical_readdress_contexts(
    *,
    pipeline,
    conditional_dict,
    ref_latent: torch.Tensor,
    render_latent: torch.Tensor,
    mask_latent: torch.Tensor,
    source_latents_path: str | Path,
    source_chunk: int,
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
    device: torch.device,
    dtype: torch.dtype,
    min_history_gap_chunks: int = 2,
    memory_dilation_kernel: int = 3,
    query_feather_kernel: int = 3,
) -> tuple[dict, list[dict], dict]:
    """Build geometry-readdressed native Recent K/V contexts.

    The historical clean B1 block is captured once in its original writer
    context as projected pre-normalization K plus V.  Each visible target block
    receives a fixed-size target-layout Recent grid: source features are warped,
    K is normalized and assigned target Recent RoPE, and runtime Recent remains
    the fallback outside ``M_memory_token`` inside attention.
    """
    selected_layers = tuple(dict.fromkeys(int(x) for x in selected_layers))
    geometry_plans, geometry_selections = build_continuous_virtual_recent_plans(
        source_latents_path=source_latents_path,
        source_chunk=source_chunk,
        target_pose_path=target_pose_path,
        intrinsics_path=intrinsics_path,
        surfel_index_path=surfel_index_path,
        surfel_sequence_path=surfel_sequence_path,
        latent_length=latent_length,
        rgb_length=rgb_length,
        frames_per_block=frames_per_block,
        latent_hw=latent_hw,
        image_hw=image_hw,
        selected_layers=selected_layers,
        selected_step_indices=selected_step_indices,
        alpha=alpha,
        feather_kernel=1,
        device=device,
        dtype=dtype,
        min_history_gap_chunks=min_history_gap_chunks,
        warp_short_term_recent=False,
        query_gate_mode="surfel_support_preserving",
        mask_policy="strong_core",
        memory_dilation_kernel=memory_dilation_kernel,
        query_feather_kernel=query_feather_kernel,
    )
    if not geometry_plans:
        raise RuntimeError("Canonical re-addressing has no visible target plans")

    source_payload = torch.load(
        Path(source_latents_path).resolve(), map_location="cpu", weights_only=True
    )
    if isinstance(source_payload, dict):
        source_payload = source_payload["pred_latents"]
    source_start = int(source_chunk) * int(frames_per_block)
    source_stop = source_start + int(frames_per_block)
    clean_source = source_payload[:, source_start:source_stop].to(
        device=device, dtype=dtype
    )
    capture_block = int(source_chunk) + 1
    capture_start = capture_block * int(frames_per_block)
    capture_stop = capture_start + int(frames_per_block)
    ref_block = ref_latent[:, capture_start:capture_stop]
    zeros = torch.zeros_like(ref_block)
    padded_ref = torch.cat(
        [ref_block, zeros[:, :, :4], zeros], dim=2
    )
    capture_render = torch.cat(
        [
            mask_latent[:, capture_start:capture_stop],
            render_latent[:, capture_start:capture_stop],
        ],
        dim=2,
    )
    capture = {layer: {} for layer in selected_layers}
    post_rope, writer_audit = pipeline.encode_clean_latent_as_recent_slot(
        reference_context=padded_ref,
        clean_recent_latent=clean_source,
        conditional_dict=conditional_dict,
        selected_layers=selected_layers,
        render_block=capture_render,
        canonical_capture=capture,
    )
    token_hw = pipeline._runtime_layout(*latent_hw)["token_hw"]
    slot_len = frames_per_block * token_hw[0] * token_hw[1]
    model = pipeline.generator.model
    freqs = _recent_freqs(model, frames=frames_per_block, token_hw=token_hw)
    source_validation = {}
    source_canonical = {}
    for layer in selected_layers:
        item = capture[layer]
        if set(item) != {"k_projected_pre_norm", "v"}:
            raise RuntimeError(f"Incomplete canonical capture for layer {layer}: {item.keys()}")
        k_projected = (
            item["k_projected_pre_norm"][:, slot_len : 2 * slot_len]
            .detach()
            .clone()
        )
        v = item["v"][:, slot_len : 2 * slot_len].detach().clone()
        attention = model.blocks[layer].self_attn
        normalized = attention.norm_k(k_projected).view(
            k_projected.shape[0], slot_len, attention.num_heads, attention.head_dim
        )
        reconstructed = rope_apply_given_freqs(normalized, freqs).type_as(v)
        source_post_k, source_post_v = post_rope[layer]
        k_diff = float((reconstructed.float() - source_post_k.float()).abs().max().item())
        v_diff = float((v.float() - source_post_v.float()).abs().max().item())
        source_validation[str(layer)] = {
            "post_rope_k_max_abs_diff": k_diff,
            "v_max_abs_diff": v_diff,
        }
        source_canonical[layer] = (k_projected, v)
    validation_max = max(
        max(item.values()) for item in source_validation.values()
    )
    if validation_max != 0.0:
        raise RuntimeError(
            "Canonical source reconstruction does not match native writer: "
            f"max_abs_diff={validation_max}"
        )
    del capture, post_rope
    if device.type == "cuda":
        torch.cuda.empty_cache()

    contexts = {}
    per_target = {}
    bytes_total = 0
    for target, plan in sorted(geometry_plans.items()):
        slot_gate = _memory_token_gate(plan.coverage, token_hw).to(device=device)
        layer_payloads = {}
        for layer in selected_layers:
            source_k, source_v = source_canonical[layer]
            warped_k = _warp_token_payload(
                source_k,
                plan.target_to_source_grid,
                frames=frames_per_block,
                token_hw=token_hw,
            )
            warped_v = _warp_token_payload(
                source_v,
                plan.target_to_source_grid,
                frames=frames_per_block,
                token_hw=token_hw,
            )
            attention = model.blocks[layer].self_attn
            normalized = attention.norm_k(warped_k).view(
                warped_k.shape[0], slot_len, attention.num_heads, attention.head_dim
            )
            target_k = rope_apply_given_freqs(normalized, freqs).type_as(warped_v)
            target_v = warped_v
            layer_payloads[layer] = (target_k, target_v)
            bytes_total += target_k.numel() * target_k.element_size()
            bytes_total += target_v.numel() * target_v.element_size()
        context = make_memory_context(
            target_block=target,
            source_chunk=source_chunk,
            layer_payloads=layer_payloads,
            selected_layers=selected_layers,
            selected_step_indices=selected_step_indices,
            alpha=alpha,
            injection_mode="canonical_recent_delta",
            gate_mode="surfel_support_preserving",
            smooth_kernel=query_feather_kernel,
            coverage=plan.coverage,
            memory_slot_gate=slot_gate,
        )
        if context is None:
            raise RuntimeError(f"Canonical context unexpectedly empty for target {target}")
        contexts[target] = context
        per_target[str(target)] = {
            "source_chunk": int(source_chunk),
            "memory_coverage_fraction": float(plan.coverage.float().mean().item()),
            "memory_token_fraction": float(slot_gate.float().mean().item()),
            "target_to_source_grid_shape": list(plan.target_to_source_grid.shape),
            "selected_layers": list(selected_layers),
        }

    audit = {
        "mode": "canonical_kv_spatial_readdressing",
        "capture_type": "clean_recent_writer_projected_k_pre_norm_plus_v",
        "source_chunk": int(source_chunk),
        "source_capture_context_block": capture_block,
        "source_writer": writer_audit,
        "source_reconstruction_validation": source_validation,
        "source_reconstruction_max_abs_diff": validation_max,
        "rope_layout": "target_recent_t3_t5_target_hw",
        "token_hw": list(token_hw),
        "slot_len": slot_len,
        "memory_bytes": int(bytes_total),
        "targets": per_target,
        "runtime_recent_fallback_inside_attention": True,
        "query_gate": "surfel_support_preserving",
    }
    selections = []
    for item in geometry_selections:
        copied = dict(item)
        copied["mode"] = "canonical_kv_spatial_readdressing"
        copied["payload_kind"] = "projected_pre_norm_k_plus_v_target_readdressed"
        copied["source_capture_context_block"] = capture_block
        selections.append(copied)
    return contexts, selections, audit


def save_canonical_audit(audit: dict, output_root: str | Path) -> Path:
    path = Path(output_root).resolve() / "canonical_kv_audit.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return path


__all__ = [
    "build_canonical_readdress_contexts",
    "save_canonical_audit",
]
