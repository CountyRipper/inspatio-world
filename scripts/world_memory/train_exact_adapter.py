#!/usr/bin/env python3
"""Overfit the 122,880-parameter adapter on one exact-pose A/B pair."""

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.causal_inference import denoise_block
from scripts.world_memory.common import (
    exact_memory_inputs,
    init_single_gpu_distributed,
    initialize_kv_cache,
    load_configs,
    load_frozen_generator,
    resolve_repo_path,
)
from world_memory import (
    attach_latent_memory_adapter,
    save_latent_memory_adapter,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/world_memory/exact_identity.yaml",
    )
    parser.add_argument("--stage-a-steps", type=int, default=None)
    parser.add_argument("--stage-b-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_denoise_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    experiment_config, base_config = load_configs(args.config)
    artifact_dir = resolve_repo_path(experiment_config.experiment.artifact_dir)
    query_path = artifact_dir / "captured_latents/query_state.pt"
    if not query_path.exists():
        raise FileNotFoundError("capture exact pairs before training")
    query = torch.load(query_path, map_location="cpu", weights_only=True)

    device = init_single_gpu_distributed()
    generator = load_frozen_generator(
        base_config,
        experiment_config.experiment.checkpoint,
        device,
    )
    set_denoise_seed(int(experiment_config.training.adapter_seed))
    adapter = attach_latent_memory_adapter(
        generator.model,
        device=device,
        dtype=torch.float32,
    )
    if bool(experiment_config.training.gradient_checkpointing):
        generator.enable_gradient_checkpointing()

    trainable = sum(parameter.numel() for parameter in adapter.parameters())
    if trainable != 122_880:
        raise AssertionError(f"expected 122,880 trainable parameters, got {trainable}")
    if not all(parameter.requires_grad for parameter in adapter.parameters()):
        raise AssertionError("adapter parameters must require gradients")
    if any(
        parameter.requires_grad
        for name, parameter in generator.named_parameters()
        if not name.startswith("model.memory_adapter.")
    ):
        raise AssertionError("the base InSpatio model must remain frozen")

    learning_rate = float(experiment_config.training.learning_rate)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=learning_rate,
        weight_decay=float(experiment_config.training.weight_decay),
    )
    training_state_path = artifact_dir / "training_state.pt"
    history = []
    start_stage = "A"
    start_step = 0
    if args.resume and training_state_path.exists():
        state = torch.load(training_state_path, map_location=device, weights_only=True)
        adapter.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        history = state.get("history", [])
        start_stage = state.get("stage", "A")
        start_step = int(state.get("step", 0))
        print(f"Resuming stage {start_stage} at step {start_step}")

    dtype = torch.bfloat16
    context_frames = query["context_frames"].to(device=device, dtype=dtype)
    render_block = query["render_block"].to(device=device, dtype=dtype)
    query_noise = query["query_noise"].to(device=device, dtype=dtype)
    conditional_dict = {
        "prompt_embeds": query["prompt_embeds"].to(device=device, dtype=dtype)
    }
    targets = {
        "A": query["target_A"].to(device=device, dtype=dtype),
        "B": query["target_B"].to(device=device, dtype=dtype),
    }
    memories = {
        identity: exact_memory_inputs(target)
        for identity, target in targets.items()
    }
    denoising_steps = query["denoising_steps"].to(device=device)
    denoise_seed = int(query["denoise_seed"])
    kv_cache = initialize_kv_cache(
        generator,
        batch_size=1,
        dtype=dtype,
        device=device,
    )
    scheduler = generator.get_scheduler()

    def predict(identity: str, grad: bool) -> torch.Tensor:
        memory_condition, memory_occupancy = memories[identity]
        set_denoise_seed(denoise_seed)
        context = torch.enable_grad() if grad else torch.no_grad()
        # The first three steps run under no_grad. Disabling the autocast weight
        # cache prevents their detached BF16 Conv3d cast from being reused by
        # the differentiable final step.
        with context, torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            prediction, _ = denoise_block(
                generator,
                scheduler,
                query_noise.clone(),
                conditional_dict,
                kv_cache,
                context_frames=context_frames,
                context_no_grad=True,
                context_freqs_offset=0,
                render_block=render_block,
                denoising_kv_size=1560 * 6,
                denoising_steps=denoising_steps,
                memory_condition=memory_condition,
                memory_occupancy=memory_occupancy,
            )
        return prediction

    @torch.no_grad()
    def measure(identity: str) -> float:
        prediction = predict(identity, grad=False)
        return float((prediction.float() - targets[identity].float()).abs().mean())

    initial_stage_a_loss = measure("A")
    print(
        f"trainable={trainable:,} initial_A_L1={initial_stage_a_loss:.6f} "
        f"adapter_dtype={next(adapter.parameters()).dtype}"
    )

    def save_state(stage: str, step: int) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        save_latent_memory_adapter(adapter, artifact_dir / "adapter.safetensors")
        torch.save(
            {
                "adapter": {
                    key: value.detach().cpu()
                    for key, value in adapter.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "stage": stage,
                "step": step,
                "history": history,
                "trainable_parameters": trainable,
            },
            training_state_path,
        )

    save_every = int(experiment_config.training.save_every)

    def train_stage(stage: str, steps: int, offset: int = 0) -> None:
        adapter.train()
        for local_step in range(offset, steps):
            identity = "A" if stage == "A" or local_step % 2 == 0 else "B"
            optimizer.zero_grad(set_to_none=True)
            activation_grad_states = []
            hook = None
            if stage == "A" and local_step == offset:
                hook = adapter.register_forward_hook(
                    lambda module, inputs, output: activation_grad_states.append(
                        (torch.is_grad_enabled(), output.requires_grad)
                    )
                )
            prediction = predict(identity, grad=True)
            if hook is not None:
                hook.remove()
                print(
                    "adapter_grad_debug "
                    f"parameter={adapter.proj.weight.requires_grad} "
                    f"activations={activation_grad_states} "
                    f"prediction={prediction.requires_grad}"
                )
            loss = (prediction.float() - targets[identity].float()).abs().mean()
            loss.backward()
            gradient_norm = float(
                torch.linalg.vector_norm(adapter.proj.weight.grad.detach().float())
            )
            optimizer.step()

            record = {
                "stage": stage,
                "step": local_step + 1,
                "identity": identity,
                "loss": float(loss.detach()),
                "gradient_norm": gradient_norm,
            }
            history.append(record)
            if local_step == 0 or (local_step + 1) % 10 == 0:
                print(
                    f"stage={stage} step={local_step + 1}/{steps} "
                    f"identity={identity} loss={record['loss']:.6f} "
                    f"grad={gradient_norm:.6f}"
                )
            if (local_step + 1) % save_every == 0:
                save_state(stage, local_step + 1)
        save_state(stage, steps)

    stage_a_steps = (
        int(args.stage_a_steps)
        if args.stage_a_steps is not None
        else int(experiment_config.training.stage_a_steps)
    )
    stage_b_steps = (
        int(args.stage_b_steps)
        if args.stage_b_steps is not None
        else int(experiment_config.training.stage_b_steps)
    )

    if start_stage == "A":
        train_stage("A", stage_a_steps, start_step)
        stage_a_loss = measure("A")
        save_latent_memory_adapter(adapter, artifact_dir / "adapter_stage_A.safetensors")
        print(f"stage_A_final_L1={stage_a_loss:.6f}")
        if not stage_a_loss < initial_stage_a_loss:
            raise RuntimeError("Stage A did not reduce exact-pose target L1")
        start_step = 0

    train_stage("B", stage_b_steps, start_step if start_stage == "B" else 0)
    final_metrics = {
        "stage_A_initial_L1": initial_stage_a_loss,
        "final_A_L1": measure("A"),
        "final_B_L1": measure("B"),
        "steps_A": stage_a_steps,
        "steps_B": stage_b_steps,
        "trainable_parameters": trainable,
    }
    save_state("complete", stage_b_steps)
    with (artifact_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
