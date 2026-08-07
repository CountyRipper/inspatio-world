from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

from phase1_lsm.adapter import load_adapter
from phase2_memory.rollout import ReturnRecord
from pipeline.causal_inference import CausalInferencePipeline


DEFAULT_CHECKPOINT = Path(
    "/data4/daixiangting/inspatio-world/checkpoints/"
    "InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"
)
DEFAULT_WAN_ROOT = Path(
    "/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B"
)


def init_distributed() -> torch.device:
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    return device


def load_pipeline(
    repo_root: Path,
    checkpoint: Path,
    adapter: Path,
    device: torch.device,
) -> CausalInferencePipeline:
    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/default_config.yaml"),
        OmegaConf.load(repo_root / "configs/inference_1.3b.yaml"),
    )
    config.wan_model_folder = str(DEFAULT_WAN_ROOT)
    config.generator.weight_list[0].path = str(DEFAULT_WAN_ROOT)
    pipeline = CausalInferencePipeline(config, device=device)
    incompatible = pipeline.generator.load_state_dict(
        load_file(str(checkpoint)), strict=False
    )
    if (
        set(incompatible.missing_keys) != {"model.memory_adapter.proj.weight"}
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.generator.to(device=device)
    load_adapter(pipeline.generator.model.memory_adapter, adapter, device=device)
    pipeline.eval().requires_grad_(False)
    return pipeline


def cpu(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = tensor.detach().contiguous().cpu()
    return value if dtype is None else value.to(dtype=dtype)


def record_tensors(
    output: torch.Tensor,
    prompt_embeds: torch.Tensor,
    records: list[ReturnRecord],
) -> dict[str, torch.Tensor]:
    tensors = {
        "output": cpu(output, torch.bfloat16),
        "prompt_embeds": cpu(prompt_embeds, torch.bfloat16),
    }
    fields = (
        "previous_latent",
        "ref_latent",
        "render_condition",
        "step_inputs",
        "transition_noises",
        "projected_memory",
        "memory_mask4",
        "occupancy",
        "anchor_noise",
        "prediction",
        "no_memory_full",
        "no_memory_truncated",
    )
    for index, record in enumerate(records):
        prefix = f"r{index:02d}_"
        for field in fields:
            value = getattr(record, field)
            if value is None:
                raise AssertionError(f"record {index} missing {field}")
            tensors[prefix + field] = cpu(
                value,
                torch.bool if field == "occupancy" else None,
            )
    return tensors


def load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def load_return_record(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, object],
    index: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], ReturnRecord]:
    prefix = f"r{index:02d}_"

    def get(name: str) -> torch.Tensor:
        return tensors[prefix + name].to(device)

    info = metadata["records"][index]
    record = ReturnRecord(
        block=int(info["block"]),
        memory_id=str(info["memory_id"]),
        memory_version=int(info["memory_version"]),
        retrieved_ids=list(info["retrieved_ids"]),
        retrieved_scores=[float(value) for value in info["retrieved_scores"]],
        previous_latent=get("previous_latent"),
        ref_latent=get("ref_latent"),
        render_condition=get("render_condition"),
        step_inputs=get("step_inputs"),
        transition_noises=get("transition_noises"),
        projected_memory=get("projected_memory"),
        memory_mask4=get("memory_mask4"),
        occupancy=get("occupancy"),
        anchor_noise=get("anchor_noise"),
        prediction=get("prediction"),
        no_memory_full=get("no_memory_full"),
        no_memory_truncated=get("no_memory_truncated"),
    )
    conditional = {"prompt_embeds": tensors["prompt_embeds"].to(device)}
    return conditional, record


def json_dump(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def finish_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
