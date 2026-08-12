# Three-Domain WorldStateReader v1

## Scope

Reader v1 keeps the immutable M40 bank, camera-only retrieval, pure-rotation
projection, selected DiT layers `[8, 14, 20]`, and the v0 continuous revisit
trajectory. It changes only the query-time domain contract and the Reader.

The formal path never attaches the direct adapter residual. The existing
122,880-parameter adapter is frozen and used only to encode projected generated
memory into its full 1536-dimensional patch feature.

## Three mutually exclusive domains

For each query block:

```text
S = strict_source_valid
M = (~S) & trusted_projected_generated_M40
U = (~S) & (~trusted_projected_generated_M40)
```

`S` is recovered conservatively from the four-channel render mask: all four
encoded values must be positive. A one-latent-pixel erosion produces `S_core`.
At every denoise step, `S_core` receives the correctly noised render clean
latent before DiT, and the predicted clean x0 is clamped after DiT. The clamp is
also reapplied after scheduler transitions. Thus STAR `last_pred` and commit use
the clamped x0. The collar `S-S_core` remains a native render-condition boundary;
there is no decoded RGB composite.

Generated memory is hard-excluded from every source-owned pixel and patch. U has
no Reader residual or memory LoRA update. Native self-attention may still carry
the consequence of a memory correction across patches; this is not direct
memory injection into U.

## Identity-preserving Reader

The encoder extracts only the center-aligned generated candidate. It has no 3x3
neighborhood, no source candidate in the selector, and no 1536-to-512 content
bottleneck.

```text
projected M40 + hard M mask
  -> frozen adapter Conv3d
  -> content_1536                         (value path)

norm(content_1536) + pose/conf/subpixel
  -> selector_key_256                     (selector path)

current hidden + timestep + selector_key
  -> alpha(memory vs null)

innovation = value_projection(content_1536)
           - current_projection(norm(hidden))
delta = fusion(innovation, current hidden, timestep)
hidden += M_patch * alpha * delta
```

Metadata never enters the value. Confidence is only a trust threshold and a
selector feature; it never scales content amplitude. The hard M patch gate is
applied after the learned selector/fusion, so residuals outside M are exactly
zero. World content is precomputed once per block and reused for all four
denoise steps; the query and selector are recomputed each step.

Rank-8 Q/O LoRA modules remain available, but their updates are multiplied by
the same per-patch `M_patch * alpha` gate. The selected final experiment did not
enable them: Reader-only restored exact identity and stayed stable on HOLD,
while later continuous training began to darken S1.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python training/world_teacher/train_v1.py \
  --stage exact-reader --scenes S0 S1

CUDA_VISIBLE_DEVICES=0 python training/world_teacher/train_v1.py \
  --stage continuous-reader --scenes S0 S1 \
  --init-checkpoint artifacts/world_teacher_v1/checkpoints/exact-reader.safetensors

CUDA_VISIBLE_DEVICES=0 python training/world_teacher/evaluate_v1.py \
  --scene S0 \
  --checkpoint artifacts/world_teacher_v1/checkpoints/reader_v1_final.safetensors
```

The selected final sidecar is the continuous Reader-only milestone at step 120.
Step 240 is retained only as an ignored diagnostic because it introduced
systematic S1 darkening without a meaningful identity gain.

## Result boundary

The two paired scenes show exact and HOLD A/B identity recovery with sharper
objects than Teacher v0 and no decoded RGB compositing. `S_core` is preserved
exactly. The result remains `PARTIAL`: first-visible is smooth, but near blocks
still contain local projection/fusion artifacts and some content propagation
into U. The remaining failure is continuous innovation, not the exact value
carrier. Writer, merge, eviction, translation/6DoF, distillation, and larger data
remain outside this result.
