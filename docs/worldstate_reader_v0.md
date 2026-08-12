# Read-only WorldState Teacher v0

This branch keeps the existing 122,880-parameter direct latent adapter as a
frozen one-shot baseline. The formal Teacher does **not** add that adapter's
output to the DiT patch embedding. It loads the same sidecar only as a frozen
`Conv3d(20, 1536, (1,2,2))` content encoder.

## Protocol

The block-aligned trajectory is:

```text
0°
→ blocks 2–4: first 0→40 traversal
→ blocks 5–6: 40° hold, write immutable M40
→ blocks 7–9: 40→0
→ blocks 10–12: 0° hold
→ snapshot STAR/KV/last-pred/noise
→ blocks 13–15: identical second 0→40 traversal
→ blocks 16–18: 40° continuous-read hold
```

Pixel-frame poses from both rising traversals are identical. Each latent pose
uses the actual saved `K` and `c2w_W0`, where
`c2w_W0[t] = inverse(target_c2w[0]) @ target_c2w[t]`. No block yaw label is
used for projection or activation.

The World State is immutable. Each paired bank contains conservative static
source observations and exactly one generated observation, `M40_A` or
`M40_B`. First-traversal targets at other poses are supervision only and are
never inserted into the bank.

Exact M40 validity remains all-true except real padding. A temporal-stability
confidence estimated from the three-frame stationary M40 block is metadata for
attention and separates `generated_memory_owned` supervision from dynamic
`unknown`; low-confidence people/liquid motion is not trained as long-term
identity.

## Visibility and candidates

`RotationProjector` applies the pure-rotation homography
`K_query R_query<-memory inverse(K_memory)` per latent frame. Exact 40° uses an
identity fast path. Candidate validity is per pixel/patch and depends only on
in-bounds geometry, stored validity, and higher-authority source conflicts.
Coverage is logged but is never an on/off switch.

Source and generated observations remain separate. `WorldTokenEncoder`
patchifies each 20-channel candidate independently, expands each observation
to a 3×3 local patch neighborhood, and appends a learned null candidate. With
top-2 observations this yields `2×9+1=19` candidates per query patch. Invalid
attention logits are `-inf`; the null value is forced to zero.

## DiT integration

Blocks 8, 14, and 20 run a width-512, 8-head local cross-attention reader
immediately before native self-attention:

```text
current hidden + denoise timestep → Q
independent projected candidates → frozen adapter → encoder → K/V
QK + confidence/pose/authority/provenance bias → local attention or null
hidden += learned_layer_scale × output
```

World K/V is computed once per block/layer and reused for all four denoise
steps. Q is recomputed from the current noisy hidden at every step. Optional
rank-8 Q/O LoRA is active only when a non-`None` world context exists. Context
KV-fill passes and memory-off calls structurally bypass both Reader and LoRA.
World tokens are never written into native ST-Cache; only the resulting clean
`x0` is committed by the existing STAR path.

Formal training backpropagates through all four denoise steps. The 1.3B
backbone and frozen direct content convolution remain frozen; Teacher encoder,
Reader, and enabled LoRA parameters stay FP32 under BF16 autocast.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python training/world_teacher/build_paired_records.py \
  --scene S0 --prepare-render

CUDA_VISIBLE_DEVICES=1 python training/world_teacher/build_paired_records.py \
  --scene S1 --prepare-render

CUDA_VISIBLE_DEVICES=0 python training/world_teacher/train.py \
  --stage exact-reader --scenes S0

CUDA_VISIBLE_DEVICES=0 python training/world_teacher/train.py \
  --stage exact-lora --scenes S0 S1 \
  --init-checkpoint artifacts/world_teacher_v0/checkpoints/exact-reader.safetensors

CUDA_VISIBLE_DEVICES=0 python training/world_teacher/train.py \
  --stage continuous --scenes S0 S1 \
  --init-checkpoint artifacts/world_teacher_v0/checkpoints/exact-lora.safetensors

CUDA_VISIBLE_DEVICES=0 python training/world_teacher/evaluate.py \
  --scene S0 \
  --checkpoint artifacts/world_teacher_v0/checkpoints/continuous_final.safetensors

CUDA_VISIBLE_DEVICES=0 python training/world_teacher/train.py \
  --stage two-block --scenes S0 S1 \
  --init-checkpoint artifacts/world_teacher_v0/checkpoints/continuous.safetensors

CUDA_VISIBLE_DEVICES=0 python training/world_teacher/evaluate.py \
  --scene S0 \
  --checkpoint artifacts/world_teacher_v0/checkpoints/teacher_final.safetensors

CUDA_VISIBLE_DEVICES=1 python training/world_teacher/evaluate.py \
  --scene S1 \
  --checkpoint artifacts/world_teacher_v0/checkpoints/teacher_final.safetensors
```

## Teacher v0 result

The measured result is **PARTIAL**. With direct residual injection disabled,
both scenes recover content-specific A/B identity at exact 40° and preserve its
direction through three HOLD blocks. S0 recovers the sparse-vs-dense bowl and
cookie layout; S1 recovers different bottle and metal-cup arrangements. The
first visible block does not suffer a global brightness collapse, and repeated
HOLD reads do not progressively degrade.

The result is not marked WORKS because near/exact outputs remain visibly softer
than the first-traversal references, especially for S1. The experiment therefore
supports immutable, visibility-driven continuous read and query-conditioned
identity selection, but not yet sharp, conflict-free natural fusion.

At exact one-shot, S0 Teacher-A-to-A latent L1 is 0.2487 versus 0.6075 for
no-memory; Teacher-B is 0.2682 to B versus 0.5635 to A. For S1 the corresponding
values are 0.2775 versus 0.4438, and 0.2688 to B versus 0.3701 to A. A single-GPU
inclusive S0 run measured roughly 1974 ms/block for no-memory and 2020 ms/block
for continuous Teacher-A, with about 6.0--6.5 GiB peak allocated VRAM. Timing
includes retrieval, projection, token encoding, per-layer K/V precomputation,
and four denoise steps.

`data/world_teacher_v0/` and `artifacts/world_teacher_v0/` are ignored by Git.
The experiment does not implement online write, merge, eviction, translation,
6DoF projection, generated-view geometry reconstruction, output RGB
compositing, distillation, or real-time inference.
