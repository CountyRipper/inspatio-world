# MapKV rapid prototype

This package closes a deliberately coarse causal loop:

```text
generated chunk PNGs -> fixed-pose official CUT3R prefix
-> radius-normal surfels -> eligible-first chunk vote
-> CPU native post-RoPE KV bank -> replace-recent attention delta at B2
```

The original InSpatio reference and recent caches remain intact. With memory
disabled, the original attention path is used and no second attention call is
made. The main injector counterfactually substitutes historical KV into the
recent slot in an auxiliary branch and blends before the single output
projection.

Run the cached yaw30 loop:

```bash
INSPATIO_PYTHON=/path/to/inspatio/bin/python \
bash scripts/run_mapkv_fast.sh \
  --case yaw30_scene01 \
  --stage full \
  --seed 0 \
  --output results/mapkv_fast/yaw30_scene01_seed0 \
  --methods baseline,wrongkv,posekv,surfelkv,manualcorrect \
  --memory-alpha 0.10 \
  --memory-layers uniform8 \
  --gate global \
  --top-k 1 \
  --gpu 0
```

Stages `kv_sanity`, `cut3r`, `surfel`, `retrieval`, `generation`, and `report`
may be rerun independently. `--reuse-baseline`, `--reuse-cut3r`, and
`--reuse-surfel` preserve expensive intermediates.

## Replication and partial-overlap locality

Build the exact 0→30→0→20 case once:

~~~bash
PYTHONPATH=. /mnt/16T2/daixiangting/conda_envs/inspatio/bin/python \
  scripts/build_mapkv_control_case.py \
  --case_id yaw30to20_scene01 \
  --source_json test/example/new.json \
  --data_path_root /mnt/16T2/daixiangting/inspatio-world \
  --source_frame_index 240 \
  --theta 30 --revisit_theta 20 \
  --vae_calibration_metadata \
    artifacts/control/yaw30_scene01/baseline/seed_0/run_metadata.json \
  --vae_time_map artifacts/control/vae_time_map.json \
  --render --render_device 0
~~~

Then run cached stages independently or together:

~~~bash
bash scripts/run_mapkv_next_stage.sh --stage full --seed 0 --gpu 0
~~~

The stage performs:

- scene02 same-pose replication with all-layer Manual/Wrong/SurfelKV;
- actual-target-pose 0→30→0→20 retrieval;
- global versus projected-surfel query gating;
- conditional surfel-to-historical-token selection;
- All/Early10/Middle10/Late10 layer budget;
- a compact synchronized HTML report.

## Source-protected generated-history revisit

Build the stronger exact-yaw benchmark from the same static scene:

~~~bash
PYTHONPATH=. /mnt/16T2/daixiangting/conda_envs/inspatio/bin/python \
  scripts/build_mapkv_control_case.py \
  --case_id yaw45m20to35_scene01 \
  --source_json test/example/new.json \
  --data_path_root /mnt/16T2/daixiangting/inspatio-world \
  --source_frame_index 240 \
  --theta 45 --leave_theta -20 --revisit_theta 35 \
  --vae_calibration_metadata \
    artifacts/control/yaw30_scene01/baseline/seed_0/run_metadata.json \
  --vae_time_map artifacts/control/vae_time_map.json \
  --render --render_device 0
~~~

Then run the cached baseline → causal CUT3R → tagged surfel → RGB-Warp WRE
pipeline:

~~~bash
bash scripts/run_mapkv_source_protected.sh \
  --stage full --gpu 0 \
  --output_root \
    results/mapkv_fast/yaw45m20to35_scene01_seed0_source_protected
~~~

The source-protected path records `reference_blind_at_write` per surfel
observation and uses
`M_need = M_history × (1 - M_ref_protected)` for both Virtual Recent
composition and a source-clamped query gate. The report separates source
stability from true generated-history revisit recovery and always includes the
complete B1 → leave → return → B2 videos.

For the current partial-overlap control, unconstrained retrieval and the
declared B1-locality ablation are stored separately. The latter restricts
candidates to the manifest B1 plateau only to isolate *where* memory acts; it
does not change CUT3R, surfel fusion, visibility, or score parameters and is
never reported as unconstrained retrieval.

## Camera-aligned Warp-and-Reencode Recent

The changed-view diagnostic freezes the historical source to B1 chunk 8 and
replaces direct post-RoPE replay with:

    B1 clean latent + exact B1/B2 c2w
    -> target-to-source rotation warp on the VAE latent grid
    -> blend with runtime Recent outside valid coverage
    -> isolated native [Ref, Virtual Recent] timestep-zero writer
    -> target-layout recent K/V
    -> replace_recent_delta

Run or reuse the deterministic experiment and build its synchronized report:

    bash scripts/run_mapkv_warp_reencode.sh --stage full --seed 0 --gpu 0

The default output is
results/mapkv_fast/yaw30to20_scene01_seed0_warp_reencode. The runner reuses
the established same-GPU Baseline and fixed-chunk-8 HardKV controls, asserts
their configuration, and verifies that the Warp-Reencode prefix through chunk
20 is exactly equal to Baseline.

## Masked Continuous Warp-Reencode Recent

Run the repaired visibility-driven architecture with:

    bash scripts/run_mapkv_masked_continuous_wre.sh --stage full --seed 0 --gpu 0

It reuses the same fixed B1 chunk and frozen known-pose surfel index, and
queries source-chunk visibility for every causally eligible block. Only B1 is
camera-warped. Runtime `last_pred` remains in the native short-term Recent
distribution:

    VirtualRecent = M_history * warp(B1 -> camera_t)
                  + (1 - M_history) * raw_last_pred

The native timestep-zero writer produces counterfactual recent K/V. The
masked method then applies exactly the same feathered geometry mask at query
resolution:

    A_out = A_base + tokenize(M_history) * (A_virtual - A_base)

The runner compares Baseline, the successful fixed-B2 Block-on WRE,
Continuous RawRecent with a global delta, and Masked Continuous WRE. Its
default output is
`results/mapkv_fast/yaw30to20_scene01_seed0_masked_continuous_wre`.

The previous warped-short-term/global-delta failure remains reproducible by
passing `--continuous_recent_fallback warped --continuous_query_gate global`
to `inference_mapkv_proto.py`; its existing result directory is not replaced.

## Report architecture framework

All new MapKV reports use `mapkv.report_framework`. A report must describe the
complete path from controlled inputs through generation, geometry/addressing,
memory payload, context/attention, output, and evaluation. The current focus is
marked in Chinese; unchanged modules remain visible; added/modified/removed
modules are color-coded and paired with a before/after change table.

Every report emits:

    architecture_state.json
    architecture_changes.json
    architecture.md
    assets/architecture_graph.svg

The validator rejects incomplete role coverage, unknown graph edges, changed
nodes without annotations, and change records without affected files or a
rationale. Project-wide defaults live in `AGENTS.md` and
`mapkv/report_preferences.yaml`.

## Identity recovery stage

Run the three controlled representation ablations with:

    bash scripts/run_mapkv_identity_recovery.sh --stage full --gpu 0

The runner freezes B1 chunk 8, known-pose CUT3R geometry, continuous
visibility, all transformer layers, all four denoising steps, and `alpha=1`.
It changes one variable at a time:

1. `strong_core_latent_wre` separates a binary/dilated `M_memory` from the
   support-preserving, soft-boundary `M_query`;
2. `rgb_warp_vae_wre` applies the exact camera warp to lossless generated RGB
   before the native WanVAE encode;
3. `canonical_kv` captures writer-only projected pre-normalization K plus V,
   warps them into the target token grid, applies `norm_k` and target Recent
   T/H/W RoPE, then retains runtime Recent outside the memory-slot mask.

Canonical capture is forbidden during denoising and must reconstruct source
post-RoPE K/V exactly before generation. The report contains complete B1-to-B2
videos, Chinese method labels, real-RGB surfel views, object identity crops,
and the full architecture/change graph.
