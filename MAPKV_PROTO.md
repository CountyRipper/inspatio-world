# CUT3R Surfel-Indexed KV quick check

This branch is a deliberately narrow, training-free prototype. It preserves the
upstream reference/recent cache and stores selected-layer historical KV in a
separate CPU bank. Memory is injected only as a same-length auxiliary
recent-slot counterfactual, before the single output projection.

## Fixed revisions and environments

- InSpatio-World: `2d15b7c742fbc90bfd7e67052a260ff87d97abc3`
- VMem: `39291e4f272f6b4f270691d930926ab5930f942e`
- CUT3R weight: `cut3r_512_dpt_4_64.pth`
- InSpatio: its original environment, bf16, one GPU, native VAE, four scheduler
  steps, `num_frame_per_block=3`, no `torch.compile`.
- CUT3R: a separate Python process/environment. Do not install its Torch stack
  into the InSpatio environment.

`scripts/setup_mapkv_cut3r.sh` pins VMem, builds CUT3R's CUDA RoPE extension,
and downloads the official 512 checkpoint. Review VMem's MIT license and the
CUT3R CC BY-NC-SA 4.0 terms in `mapkv_proto/cut3r/ATTRIBUTION.md` before reuse.

## Environment variables

Activate the InSpatio environment, or set `INSPATIO_PYTHON` to its Python.
When this repository is a linked worktree, the scripts automatically locate
checkpoints and `test/example/new_vggt` in the main checkout. Override that with
`INSPATIO_ASSET_ROOT` if needed.

```bash
export INSPATIO_PYTHON=/path/to/inspatio_world/bin/python
export MAPKV_GPU=0
export MAPKV_ARTIFACT_ROOT="$PWD/artifacts"
```

For geometry, additionally set:

```bash
export CUT3R_PYTHON=/path/to/mapkv_cut3r/bin/python
export VMEM_ROOT="$PWD/third_party/vmem"
export CUT3R_CHECKPOINT="$VMEM_ROOT/extern/CUT3R/src/cut3r_512_dpt_4_64.pth"
```

## Phase 0 — deterministic baseline and KV bank

```bash
bash scripts/run_mapkv_baseline.sh
```

This creates the bundle, runs memory-off twice in the same process, records the
latent `max_abs_diff`, captures only the configured four layers, saves lossless
keyframes/masks, and writes the five best revisit candidates. Inspect:

```text
artifacts/baseline/run_metadata.json
artifacts/baseline/revisit_candidates.json
artifacts/baseline/revisit_candidates.png
artifacts/baseline/kv_bank/metadata.json
```

Choose a visible generated-region revisit with gap at least three, then choose a
causally valid unrelated wrong chunk. Do not proceed based only on pose distance;
inspect the contact sheet and masks.

## Phase I — Oracle payload/injection

For `B=SOURCE`, `R=TARGET`, and `W=WRONG`:

```bash
bash scripts/run_mapkv_oracle.sh --mode baseline \
  --run_name baseline

bash scripts/run_mapkv_oracle.sh --mode alpha_zero \
  --source_chunk B --target_chunk R --run_name alpha_zero

bash scripts/run_mapkv_oracle.sh --mode wrong \
  --source_chunk W --target_chunk R --alpha 0.10 --run_name wrong_kv

for alpha in 0.05 0.10 0.20; do
  bash scripts/run_mapkv_oracle.sh --mode oracle \
    --source_chunk B --target_chunk R --alpha "${alpha}" \
    --run_name "oracle_a${alpha/./}"
done
```

The AlphaZero metadata must report zero saved-baseline latent difference. Each
active run also contains an exact block/step/layer activation audit. Stop here if
correct OracleKV has no attributable benefit after checking capture, pair choice,
and one diagnostic `alpha=1.0` run.

## Phase II — causal two-pass CUT3R retrieval

Only after Phase I is GO:

```bash
bash scripts/build_mapkv_geometry.sh \
  --target_chunk R --oracle_source B
```

The builder exports baseline-only PNG/pose/K views, reconstructs incremental
prefixes with fixed poses and frozen previous depths, writes only each new view's
surfels, and excludes current/future/immediate-previous chunks during voting.
It also attaches a selected-chunk surfel gate to Pose and Oracle plans, so the
three stage-II methods differ in address rather than memory budget or gate rule.

```bash
bash scripts/run_mapkv_geometry.sh --retrieval pose \
  --retrieval_plan artifacts/geometry/pose_plan.json \
  --alpha 0.10 --run_name posekv_a010

bash scripts/run_mapkv_geometry.sh --retrieval geometry \
  --retrieval_plan artifacts/geometry/surfel_plan.json \
  --alpha 0.10 --run_name geometrykv_a010

bash scripts/run_mapkv_geometry.sh --retrieval oracle \
  --retrieval_plan artifacts/geometry/oracle_plan.json \
  --source_chunk B --alpha 0.10 --run_name oraclekv_a010
```

An empty selected-surface coverage is recorded and falls back to memory-off. It
never enables memory globally.

## Final evidence bundle

After visually assigning both GO/NO-GO decisions and exactly one failure class:

```bash
python scripts/make_mapkv_comparison.py \
  --baseline_run artifacts/baseline \
  --pose_run artifacts/geometry/runs/posekv_a010 \
  --geometry_run artifacts/geometry/runs/geometrykv_a010 \
  --oracle_run artifacts/geometry/runs/oraclekv_a010 \
  --pose_plan artifacts/geometry/pose_plan.json \
  --geometry_plan artifacts/geometry/surfel_plan.json \
  --source_chunk B --target_chunk R --wrong_chunk W \
  --phase1_conclusion GO --phase2_conclusion GO \
  --primary_failure 5 \
  --why_revisit "..." --oracle_effect "..." --wrong_effect "..." \
  --activation_discontinuity "..." --next_action "..."
```

The output is `artifacts/final/{REPORT.md,metrics.json,contact_sheet.png,
baseline_vs_pose_vs_geometry_vs_oracle.mp4}`. LPIPS is recorded when the optional
`lpips` package is available; visual judgment remains explicit in the report.
