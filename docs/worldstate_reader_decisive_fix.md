# WorldStateReader v1 decisive exact-pose fix

## Question

Can the existing innovation Reader recover clear, content-specific A/B memory
identity from one exact-pose read after fixing the three-domain contract?

This experiment intentionally excludes near views, continuous read, LoRA,
TBPTT, writer, DA3, data expansion, and distillation.

## Corrected contract

For one exact query, S0/S1 each build one spatial domain shared by A and B:

```text
S = eroded current source mask (S_core)
V = generated M40 exact-projection in-bounds validity
M = (~S) & V
U = ~(S | M)
```

There is no confidence threshold in `M`. The generated observation is projected
without a source candidate, so the v0 projector's source-authority conflict rule
cannot punch extra holes in `V`. A provides shared validity, confidence, pose,
subpixel metadata, occupancy, causal state, and noise; B replaces only the
projected latent16 content. Exact Reader contexts force `alpha=1` in hard M
patches and structurally bypass all inactive Reader layers. LoRA and direct
residual are disabled in Reader conditions.

Targets use the same source clamp as inference:

```text
target_A = where(S, source_clean, first_G_A)
target_B = where(S, source_clean, first_G_B)
```

Blocks 13--15 are memory-off. Block 16 reads memory during all four denoise
steps. Block 17 is memory-off and observes STAR carryover.

## Reproduction

Initial layer ablation:

```bash
CUDA_VISIBLE_DEVICES=0 python training/world_teacher/evaluate_decisive_fix.py \
  --scene S0
CUDA_VISIBLE_DEVICES=2 python training/world_teacher/evaluate_decisive_fix.py \
  --scene S1
```

The unmodified checkpoint had no sufficiently clear single layer. Layer 8 had
the least visual fragmentation, so the one permitted repair trained only that
Reader for 80 balanced S0/S1 x A/B exact steps:

```bash
CUDA_VISIBLE_DEVICES=0 python training/world_teacher/train_decisive_fix.py
```

Final evaluation loads the ignored sidecar
`artifacts/worldstate_reader_decisive_fix/checkpoints/reader_layer8_exact_fix_ablation.safetensors`.
The final deployment sidecar `reader_layer8_exact_fix.safetensors` contains only
the shared encoder and layer 8 Reader; it has no layer 14/20 or LoRA tensors.

```bash
CUDA_VISIBLE_DEVICES=0 python training/world_teacher/evaluate_decisive_fix.py \
  --scene S0 \
  --reader-checkpoint \
    artifacts/worldstate_reader_decisive_fix/checkpoints/reader_layer8_exact_fix_ablation.safetensors \
  --output-name final_layer8
CUDA_VISIBLE_DEVICES=2 python training/world_teacher/evaluate_decisive_fix.py \
  --scene S1 \
  --reader-checkpoint \
    artifacts/worldstate_reader_decisive_fix/checkpoints/reader_layer8_exact_fix_ablation.safetensors \
  --output-name final_layer8
```

## Decision

`WORKS` for the narrowly scoped exact-pose question. The repaired layer 8
restores distinct A/B cup/bowl/cookie and bottle-layout identities without the
fragmentation seen in layers 14/20. Block 17 has no visual collapse; S1 remains
stable, while S0 has a modest latent-L1 regression that leaves STAR carryover as
follow-up work. Applying the repaired layer 8 together with the unchanged
layers 14/20 is worse, so repeated multi-layer injection is the dominant exact
failure once one layer has sufficient read strength. The selected design is a
single layer-8 Reader; the three-layer stack is rejected.

This result does not establish near-view projection, continuous read, long-term
HOLD, retrieval, writing, or generalization beyond S0/S1.
