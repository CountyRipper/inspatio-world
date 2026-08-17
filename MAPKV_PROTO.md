# CUT3R Surfel-Indexed KV quick check

This branch is a deliberately narrow, training-free prototype. It preserves the
upstream reference/recent cache and stores selected-layer historical KV in a
separate CPU bank. Memory is injected only as a same-length auxiliary
recent-slot counterfactual, before the single output projection.

The official `test/example + x_y_circle_cycle.txt` run is an upstream smoke test
only. It is not decision-eligible: any historical result from that trajectory is
labelled `INCONCLUSIVE / INVALID_BENCHMARK`. Phase-I GO/NO-GO decisions must use
the exact-pose repeated-static-frame controls below.

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

## Upstream compatibility smoke

```bash
bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --disable_adaptive_frame \
  --output_folder ./output/mapkv_proto/upstream_smoke
```

Do not use this video to accept or reject the historical-KV payload.

## Phase 0 — exact-pose controlled baseline and KV bank

First calibrate the native VAE temporal mapping from an upstream run, then build
a repeated-static-frame pure-yaw case. `target_poses.npy` is absolute `c2w` and
is the single source of truth for rendering, inference, block mapping, and
evaluation. The exact path bypasses trajectory text and all spline logic.

```bash
python scripts/build_mapkv_control_case.py \
  --case_id yaw15_scene01 \
  --source_json /path/to/one-scene-source.json \
  --source_frame_index 240 \
  --theta 15 \
  --vae_calibration_metadata /path/to/upstream/run_metadata.json \
  --render

MAPKV_REQUIRE_EXACT=1 bash scripts/run_mapkv_baseline.sh \
  --case_dir artifacts/control/yaw15_scene01 --seed 0

bash scripts/run_mapkv_oracle.sh \
  --case_dir artifacts/control/yaw15_scene01 --seed 0 \
  --mode alpha_zero --source_chunk 6 --target_chunk 17 \
  --run_name alpha_zero

python -m mapkv_proto.revisit_pair validate \
  --case_dir artifacts/control/yaw15_scene01 \
  --baseline_root artifacts/control/yaw15_scene01/baseline/seed_0 \
  --alpha_zero_root artifacts/control/yaw15_scene01/oracle/seed_0/runs/alpha_zero \
  --b1_quality_pass --headroom_pass
```

The two visual flags are explicit human checks; do not pass them before
inspecting `pair_contact_sheet.png`. Validation records V1–V10, KV checksums and
shapes, exact input checksums, same-view render/mask equality, and AlphaZero
equality. A failed check is `INVALID_CASE`, never payload `NO-GO`.

Inspect:

```text
artifacts/control/yaw15_scene01/pose_validation.json
artifacts/control/yaw15_scene01/render_revisit_diff.json
artifacts/control/yaw15_scene01/pair_validation.json
artifacts/control/yaw15_scene01/baseline/seed_0/run_metadata.json
artifacts/control/yaw15_scene01/baseline/seed_0/kv_bank/metadata.json
```

## Phase I — Oracle payload/injection

The manifest fixes B1/source, B2/target, and wrong chunks before results are
viewed. After the pair gate passes, run the fixed diagnostic order:

```bash
bash scripts/run_mapkv_control_matrix.sh \
  --case_dir artifacts/control/yaw15_scene01 \
  --seeds 0 --alphas 0.05,0.10,0.20
```

The matrix is resumable. Seed 0 runs AlphaZero, diagnostic Oracle `alpha=1.0`,
the stable alpha sweep, WrongKV, and RandomKV. Wrong/Random use `alpha=0.10`, so
only Oracle `alpha=0.10` is used for the primary matched-strength discrimination;
`alpha=0.20` is a strength sweep.

If yaw15 shows attributable correct-vs-control sensitivity, build two yaw30
static scenes and run seeds 0–2:

```bash
bash scripts/run_mapkv_control_matrix.sh \
  --case_dir artifacts/control/yaw30_scene01 \
  --seeds 0,1,2 --alphas 0.05,0.10,0.20
```

Every active run records the exact target block/step/layer audit and asserts the
runtime KV cache is unchanged. If `alpha=1.0` does not change the target latent,
the implementation is invalid. Slight metric movement without recognizable
first-visit identity recovery is not a GO.

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

## Final controlled evidence bundle

Generate each per-seed report, then aggregate the two-scene primary matrix:

```bash
python scripts/make_mapkv_control_report.py \
  --case_dir artifacts/control/yaw30_scene01 --seed 0 \
  --conclusion NO-GO --visual_summary "..." --quiet

python scripts/make_mapkv_final_report.py \
  --case_dirs artifacts/control/yaw30_scene01 artifacts/control/yaw30_scene02 \
  --seeds 0,1,2 --output_dir artifacts/final \
  --conclusion NO-GO --visual_summary "..."
```

When Phase I does not clear, the final bundle deliberately contains Phase-I
comparison videos and no fabricated PoseKV/GeometryKV comparison. CUT3R remains
gated. If Phase I is GO, use the Phase-II commands above and then create the
four-way geometry bundle:

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
