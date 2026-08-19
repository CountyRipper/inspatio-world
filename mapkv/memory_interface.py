from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from PIL import Image


MEMORY_INTERFACE_MODES = {
    "masked_hard_x0",
    "dual_branch_recent",
    "native_render",
    "latent_anchor",
}


def _coverage_like(coverage: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Return BF1HW coverage on ``tensor``'s device/spatial grid."""
    value = torch.as_tensor(coverage, dtype=torch.float32, device=tensor.device)
    if value.ndim == 4:
        value = value.unsqueeze(2)
    if value.ndim != 5 or value.shape[2] != 1:
        raise ValueError(
            f"Memory coverage must be [B,F,H,W] or [B,F,1,H,W], got {tuple(value.shape)}"
        )
    if value.shape[:2] != tensor.shape[:2]:
        raise ValueError(
            f"Memory coverage BF={tuple(value.shape[:2])} does not match "
            f"tensor BF={tuple(tensor.shape[:2])}"
        )
    if value.shape[-2:] != tensor.shape[-2:]:
        batch, frames = value.shape[:2]
        value = F.interpolate(
            value.reshape(batch * frames, 1, *value.shape[-2:]),
            size=tensor.shape[-2:],
            mode="nearest",
        ).reshape(batch, frames, 1, *tensor.shape[-2:])
    return value.clamp(0, 1)


@dataclass
class MemoryInterfacePlan:
    """One target block in the frozen-model memory-interface ladder.

    ``memory_latent`` is already camera aligned to the target block. The same
    hard ``need_coverage`` is used by every method so the experiment only
    changes the model interface, not geometry, lifecycle, or accepted pixels.
    """

    target_block: int
    source_chunk: int
    mode: str
    memory_latent: torch.Tensor
    need_coverage: torch.Tensor
    selected_layers: tuple[int, ...] = ()
    anchor_step_indices: tuple[int, ...] = ()
    source_rgb_indices: tuple[int, ...] = ()
    target_rgb_indices: tuple[int, ...] = ()
    dual_recent_payloads: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    source_plan_audit: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    artifacts: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MEMORY_INTERFACE_MODES:
            raise ValueError(f"Unsupported memory interface: {self.mode}")
        if self.memory_latent.ndim != 5:
            raise ValueError("memory_latent must be [B,F,C,H,W]")
        if self.need_coverage.ndim not in {4, 5}:
            raise ValueError("need_coverage must be [B,F,H,W] or [B,F,1,H,W]")
        if int(self.source_chunk) >= int(self.target_block) - 1:
            raise ValueError("Memory source must be older than the runtime Recent chunk")
        self.selected_layers = tuple(int(value) for value in self.selected_layers)
        self.anchor_step_indices = tuple(
            int(value) for value in self.anchor_step_indices
        )
        self.audit.update(
            {
                "mode": self.mode,
                "target_block": int(self.target_block),
                "source_chunk": int(self.source_chunk),
                "need_coverage_fraction": float(
                    self.need_coverage.float().mean().item()
                ),
                "memory_latent_shape": list(self.memory_latent.shape),
                "same_target_aligned_memory": True,
            }
        )

    @property
    def needs_dual_recent_writer(self) -> bool:
        return self.mode == "dual_branch_recent"

    @property
    def owns_render_condition(self) -> bool:
        return self.mode == "native_render"

    def memory_for(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.memory_latent.shape != tensor.shape:
            raise ValueError(
                f"Memory latent {tuple(self.memory_latent.shape)} does not match "
                f"target {tuple(tensor.shape)}"
            )
        return self.memory_latent.to(device=tensor.device, dtype=tensor.dtype)

    def coverage_for(self, tensor: torch.Tensor) -> torch.Tensor:
        return _coverage_like(self.need_coverage, tensor)

    def compose_virtual_recent(self, raw_recent: torch.Tensor) -> torch.Tensor:
        memory = self.memory_for(raw_recent)
        coverage = self.coverage_for(raw_recent).to(dtype=raw_recent.dtype)
        virtual = coverage * memory + (1.0 - coverage) * raw_recent
        self.artifacts.update(
            {
                "memory_latent": memory.detach().cpu(),
                "need_coverage": coverage.detach().cpu(),
                "raw_recent": raw_recent.detach().cpu(),
                "virtual_recent": virtual.detach().cpu(),
            }
        )
        self.audit.update(
            {
                "virtual_recent_vs_raw_l1": float(
                    (virtual.float() - raw_recent.float()).abs().mean().item()
                ),
                "recent_writer": "native_timestep_zero_ref_plus_recent",
            }
        )
        return virtual

    def set_dual_recent_payloads(
        self,
        payloads: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        writer_audit: Mapping,
    ) -> None:
        if not self.needs_dual_recent_writer:
            raise ValueError("Only DualBranchRecent accepts virtual Recent K/V")
        self.dual_recent_payloads = {
            int(layer): (k, v) for layer, (k, v) in payloads.items()
        }
        self.audit["dual_recent_writer"] = dict(writer_audit)

    def fuse_native_render(
        self,
        render_latent: torch.Tensor,
        mask_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse source > memory > new into InSpatio's native render channels."""
        if not self.owns_render_condition:
            return render_latent, mask_latent
        if render_latent.shape != self.memory_latent.shape:
            raise ValueError("MemoryRender requires memory/render latent shape equality")
        if mask_latent.ndim != 5 or mask_latent.shape[:2] != render_latent.shape[:2]:
            raise ValueError("Native render mask must be BFCHW and block aligned")
        memory = self.memory_for(render_latent)
        need = self.coverage_for(render_latent)
        source_mask01 = ((mask_latent.float() + 1.0) * 0.5).clamp(0, 1)
        source_valid = source_mask01.mean(dim=2, keepdim=True)
        memory_accept = (1.0 - source_valid) * need
        fused_render = (
            source_valid.to(render_latent.dtype) * render_latent
            + memory_accept.to(render_latent.dtype) * memory
        )
        fused_valid = torch.maximum(source_valid, memory_accept)
        fused_mask = (fused_valid * 2.0 - 1.0).expand_as(mask_latent).to(
            dtype=mask_latent.dtype
        )
        self.artifacts.update(
            {
                "memory_latent": memory.detach().cpu(),
                "need_coverage": need.detach().cpu(),
                "native_source_valid": source_valid.detach().cpu(),
                "memory_render_accept": memory_accept.detach().cpu(),
                "fused_render_latent": fused_render.detach().cpu(),
                "fused_render_mask": fused_mask.detach().cpu(),
            }
        )
        source_error = float(
            (
                (fused_render.float() - render_latent.float()).abs()
                * source_valid
            ).max().item()
        )
        self.audit.update(
            {
                "render_priority": "source_gt_memory_gt_new",
                "source_region_render_max_abs_diff": source_error,
                "native_render_memory_fraction": float(memory_accept.mean().item()),
                "extra_recent_writer": False,
                "extra_memory_attention": False,
            }
        )
        return fused_render, fused_mask

    def blend_dual_predictions(
        self,
        base_x0: torch.Tensor,
        memory_x0: torch.Tensor,
        *,
        step_index: int,
    ) -> torch.Tensor:
        if self.mode != "dual_branch_recent":
            raise ValueError("Dual prediction blend requested for a non-dual method")
        coverage = self.coverage_for(base_x0).to(dtype=base_x0.dtype)
        guided = base_x0 + coverage * (memory_x0 - base_x0)
        self.audit.setdefault("denoising_steps", []).append(
            {
                "step_index": int(step_index),
                "base_vs_memory_l1": float(
                    (base_x0.float() - memory_x0.float()).abs().mean().item()
                ),
                "guided_vs_base_l1": float(
                    (guided.float() - base_x0.float()).abs().mean().item()
                ),
            }
        )
        return guided

    def apply_x0(
        self,
        predicted_x0: torch.Tensor,
        *,
        step_index: int,
        total_steps: int,
    ) -> torch.Tensor:
        apply = False
        if self.mode == "masked_hard_x0":
            apply = int(step_index) == int(total_steps) - 1
        elif self.mode == "latent_anchor":
            apply = int(step_index) in self.anchor_step_indices
        if not apply:
            return predicted_x0
        memory = self.memory_for(predicted_x0)
        coverage = self.coverage_for(predicted_x0).to(dtype=predicted_x0.dtype)
        anchored = coverage * memory + (1.0 - coverage) * predicted_x0
        self.audit.setdefault("x0_interventions", []).append(
            {
                "step_index": int(step_index),
                "total_steps": int(total_steps),
                "intervention": (
                    "matched_masked_final_x0"
                    if self.mode == "masked_hard_x0"
                    else "noise_consistent_latent_anchor"
                ),
                "output_delta_l1": float(
                    (anchored.float() - predicted_x0.float()).abs().mean().item()
                ),
            }
        )
        return anchored


def build_dual_recent_cache(
    base_cache: list[dict[str, torch.Tensor]],
    payloads: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    recent_slot_len: int,
) -> list[dict[str, torch.Tensor]]:
    """Clone the complete base cache and replace only its native Recent slot."""
    expected = set(range(len(base_cache)))
    provided = {int(layer) for layer in payloads}
    if provided != expected:
        raise ValueError(
            "DualBranchRecent requires every transformer layer: "
            f"expected={sorted(expected)} provided={sorted(provided)}"
        )
    result = []
    start = int(recent_slot_len)
    stop = 2 * int(recent_slot_len)
    for layer, entry in enumerate(base_cache):
        k = entry["k"].detach().clone()
        v = entry["v"].detach().clone()
        memory_k, memory_v = payloads[layer]
        if memory_k.shape != k[:, start:stop].shape:
            raise ValueError(
                f"Dual Recent K shape mismatch at layer {layer}: "
                f"{tuple(memory_k.shape)} vs {tuple(k[:, start:stop].shape)}"
            )
        if memory_v.shape != v[:, start:stop].shape:
            raise ValueError(
                f"Dual Recent V shape mismatch at layer {layer}: "
                f"{tuple(memory_v.shape)} vs {tuple(v[:, start:stop].shape)}"
            )
        k[:, start:stop].copy_(memory_k.to(device=k.device, dtype=k.dtype))
        v[:, start:stop].copy_(memory_v.to(device=v.device, dtype=v.dtype))
        result.append({"k": k, "v": v})
    return result


def from_warp_reencode_plans(
    plans: Mapping[int, object],
    *,
    mode: str,
    anchor_step_indices: Iterable[int] = (),
) -> dict[int, MemoryInterfacePlan]:
    """Freeze WRE geometry/lifecycle and expose only a different model interface."""
    if mode not in MEMORY_INTERFACE_MODES:
        raise ValueError(f"Unsupported memory interface: {mode}")
    result = {}
    for target, source in plans.items():
        need = source.need_coverage
        if need is None:
            need = source.coverage
        if not bool(torch.as_tensor(need).any()):
            continue
        result[int(target)] = MemoryInterfacePlan(
            target_block=int(source.target_block),
            source_chunk=int(source.source_chunk),
            mode=mode,
            memory_latent=source.historical_latent,
            need_coverage=torch.as_tensor(need),
            selected_layers=tuple(source.selected_layers),
            anchor_step_indices=tuple(int(value) for value in anchor_step_indices),
            source_rgb_indices=tuple(source.source_rgb_indices),
            target_rgb_indices=tuple(source.target_rgb_indices),
            source_plan_audit=dict(source.audit),
        )
    return result


def _save_mask(tensor: torch.Tensor, path: Path) -> None:
    value = tensor.detach().float().cpu()
    if value.ndim == 5:
        value = value[0, value.shape[1] // 2, 0]
    elif value.ndim == 4:
        value = value[0, value.shape[1] // 2]
    array = (value.clamp(0, 1).numpy() * 255.0).round().astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_memory_interface_artifacts(
    *,
    plans: Mapping[int, MemoryInterfacePlan],
    vae,
    output_root: str | Path,
) -> dict:
    root = Path(output_root) / "memory_interface"
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for target, plan in sorted(plans.items()):
        block = root / f"block_{int(target):04d}"
        block.mkdir(parents=True, exist_ok=True)
        torch.save(plan.memory_latent.detach().cpu(), block / "L_mem.pt")
        torch.save(plan.need_coverage.detach().cpu(), block / "M_need.pt")
        _save_mask(plan.need_coverage, block / "M_need.png")
        with torch.no_grad():
            rgb = (vae.decode_to_pixel(plan.memory_latent, use_cache=False) * 0.5 + 0.5)
        center = rgb[0, rgb.shape[1] // 2].detach().float().clamp(0, 1)
        array = (
            center.permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().astype("uint8")
        Image.fromarray(array).save(block / "L_mem_decoded.png")
        entries.append(
            {
                "target_block": int(target),
                "source_chunk": int(plan.source_chunk),
                "mode": plan.mode,
                "need_coverage_fraction": float(
                    plan.need_coverage.float().mean().item()
                ),
                "source_rgb_indices": list(plan.source_rgb_indices),
                "target_rgb_indices": list(plan.target_rgb_indices),
                "audit": plan.audit,
                "artifact_dir": str(block.relative_to(Path(output_root))),
            }
        )
    manifest = {
        "mode": next(iter(plans.values())).mode if plans else None,
        "same_memory_and_mask_across_interfaces": True,
        "entries": entries,
    }
    import json

    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


__all__ = [
    "MEMORY_INTERFACE_MODES",
    "MemoryInterfacePlan",
    "build_dual_recent_cache",
    "from_warp_reencode_plans",
    "save_memory_interface_artifacts",
]
