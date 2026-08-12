# Exact-pose implicit latent memory readout

This baseline asks one narrow question: can frozen InSpatio-World 1.3B read a
previously generated clean latent identity through a 122,880-parameter sidecar?
It is not a world-state system.

## Adapter and injection

`LatentMemoryAdapter` is one bias-free `Conv3d(20, model_dim, (1,2,2),
stride=(1,2,2))`. For the 1.3B model, `model_dim=1536`, so it has exactly
`20 * 1536 * 1 * 2 * 2 = 122,880` parameters. The external condition layout is
`[B,F,C,H,W]`; the wrapper converts it to Conv3d layout `[B,C,F,H,W]`.

The 20 input channels are four all-valid mask channels followed by 16 clean
memory-latent channels. Binary occupancy is separate. The residual is computed
after the base 36-channel patch embedding and before token flattening, then
multiplied by a max-pooled hard occupancy gate. It never changes the noisy
latent. When `memory_condition is None`, no adapter or zero residual is
computed, preserving the original path.

The adapter is attached only after loading the complete InSpatio checkpoint:

```python
from world_memory import attach_latent_memory_adapter

adapter = attach_latent_memory_adapter(pipeline.generator.model)
```

`save_latent_memory_adapter` and `load_latent_memory_adapter` operate on a
standalone safetensors sidecar; adapter weights are never mixed into the base
checkpoint. Attachment follows the base patch embedding dtype by default;
training passes `dtype=torch.float32` explicitly and uses BF16 autocast.

## Pipeline hooks

`CausalInferencePipeline.inference` accepts two optional hooks:

- `memory_provider(block_index, latent_start, block_size)` returns `None` or
  `(memory_condition, memory_occupancy)` in external layout.
- `block_output_callback(block_index, latent_start, denoised_latent)` receives
  a detached clone of each block's final clean x0.

The context/KV-fill pass is always memory-off. A provider result is reused for
all four denoise steps of only that query block, so memory cannot enter the
STAR/KV prefix early.

## Reproduction

The experiment config is `configs/world_memory/exact_identity.yaml`. The three
scripts prepare/capture the exact trajectory, train only the adapter, and run
the fixed no-memory / memory-A / content-swap-B comparison:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  scripts/world_memory/capture_exact_pairs.py --prepare-render
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  scripts/world_memory/train_exact_adapter.py
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  scripts/world_memory/eval_exact_revisit.py
```

The selected cup-and-tray scene follows `0° -> +40° -> 0° -> +40°` with
rotation only. The first and final `+40°` read/write blocks use identical
per-frame pose and intrinsics. All generated latents, videos, optimizer state,
montage, metrics, and the Chinese result note live under
`artifacts/exact_identity/`, which is Git-ignored.

This baseline does not implement anchors, noisy-state blending, MASH/MARB,
near-view or 6DoF projection, retrieval, submaps, RGB compositing, natural
handoff, or long-term persistence.
