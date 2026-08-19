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

## Re-entry memory refinement

Run the lifecycle / observation / edge-safety matrix while reusing the
validated baseline and causal CUT3R prefix:

    bash scripts/run_mapkv_reentry_refinement.sh --stage full --gpu 0

The stage compares:

1. the original Source-Protected RGB-Warp WRE that reads whenever history is
   visible;
2. a group-level re-entry-only policy with fixed chunk 11;
3. a stable view-adaptive first-episode observation;
4. the view-adaptive method with `reference_blind_at_write >= 0.75`, an
   eroded warp-valid interior, RGB border padding, and inward-only query
   feathering;
5. the same edge-safe path on denoising steps `[0,1,2]` only.

The lifecycle is causal: the first visibility episode only writes memory,
two consecutive absent blocks arm re-entry, the first source-blind supported
block reads once, and the group is then handed back to native Recent. Source
selection uses the first-episode generated-only observations and scores
`coverage × camera-view alignment × observation quality × center margin`;
the selected chunk is locked for the episode.

The default output is
`results/mapkv_fast/yaw45m20to35_scene01_seed0_reentry_refinement`. It includes
complete 453-frame comparison videos, separate first-departure and re-entry
clips, a lifecycle timeline, real-RGB surfels, a B2 right-edge review, metrics,
and the architecture graph. The current controlled result is intentionally
reported as `REENTRY_HANDOFF_INSUFFICIENT`: group-level one-shot serving fixes
the first-departure transition but retains too little long-term memory and
marks later re-entering surfaces served too early. The next scoped correction
is per-surfel one-shot lifecycle, not another alpha/mask/CUT3R sweep.

## Re-entry continuous refresh

Run the gated lifecycle convergence stage with:

    bash scripts/run_mapkv_reentry_refresh.sh --stage full --gpu 0

The full command reuses the same controlled trajectory, canonical chunk-11
identity target, known-pose CUT3R map, noise, checkpoint, and RGB-Warp WRE
quality path. It advances conditionally:

1. first-visit write-only, then continuous read throughout the true re-entry
   episode;
2. per-surface TTL=2 only after the episode policy succeeds;
3. view-adaptive observation selection restricted to actual shared chunk-11
   anchor surfel IDs;
4. edge-safe support and memory steps [0,1,2] on the best successful
   lifecycle/source.

All methods are evaluated against canonical chunk 11 even when another
observation is selected. The default output is
`results/mapkv_fast/yaw45m20to35_scene01_seed0_reentry_refresh`, with complete
453-frame synchronized revisit videos, departure/re-entry clips, real-RGB
surfel views, a lifecycle timeline, architecture/change graph, metrics, and a
Chinese HTML report.

The controlled result is `REENTRY_EPISODE_CONTINUOUS_WORKS`: it preserves
102.8% of the old continuously-reading memory gain while restoring the first
departure peak to baseline. TTL=2, same-surface view adaptation, edge-safe
support, and steps012 do not improve the retained chunk-11 identity and remain
rejected ablations.

## Fixed 3D surfel address repair

The former pts3d_in_self_view × known c2w path is retained only as the
explicit legacy backend. The repaired geometry path adds:

1. CUT3R cross-view global alignment with fixed known poses and intrinsics;
2. an audited previous-depth incremental backend plus a full-prefix joint
   quality backend;
3. tentative/stable surfels with confidence calibration and reprojection
   consistency;
4. stable surface-neighborhood, unique-cell chunk voting with cluster-max
   aggregation and retrieval margin/entropy;
5. separate pure-yaw angular and 0.08-translation depth Gates.

Run:

    bash scripts/run_mapkv_geometry_repair.sh --stage full
    bash scripts/run_mapkv_translation_geometry.sh --stage full
    bash scripts/run_mapkv_geometry_repair.sh --stage report

The combined report is written to
results/mapkv_fast/yaw45m20to35_scene01_seed0_geometry_repair/report.html.
No KV generation runs inside these stages. The observed result is
GEOMETRY_ADDRESS_REPAIR_WORKS: the pure-yaw angular Gate and translation
depth Gate both pass. Strictly freezing every previous depth was implemented
and audited but did not converge as a quality path; the current quality oracle
uses a causal full-prefix joint fixed-pose alignment.

## Frozen memory-interface convergence

Run the fixed-memory interface ladder with:

    bash scripts/run_mapkv_memory_interface.sh --stage full --gpu 0

This stage reuses the validated geometry, fixed canonical chunk 11,
episode-continuous re-entry lifecycle, exact RGB camera warp, Wan VAE, source
protection, deterministic noise bundle, and the same hard `M_need`. It changes
only the frozen model interface:

1. matched masked final-x0 replacement (`MaskedHardX0` quality upper bound);
2. two coherent full denoising branches with independent raw/Virtual Recent
   caches (`DualBranchRecent`);
3. source-priority fusion through InSpatio's native render+validity channels
   (`MemoryRender`);
4. noise-consistent x0 anchoring before the same scheduler re-noise on steps
   `[0,1,2]`, plus the conditional all-four-step diagnostic.

The default output is
`results/mapkv_fast/yaw45m20to35_scene01_seed0_memory_interface`. The report
contains Chinese method/focus labels, a complete pipeline/change graph,
complete 453-frame synchronized revisit videos, re-entry clips, and automatic
generated-history structure crops. Average L1 is explicitly reported as
historical *appearance* proximity; `STRONG/PARTIAL/NONE` identity ratings come
from the synchronized videos and crops.

The current controlled decision is `LATENT_ANCHOR_REQUIRED`: coherent Recent
guidance is partial, native Render memory does not recover the canonical
instances, and a free final denoising step washes out most steps-012 anchoring.
All-four-step latent anchoring matches the hard upper bound while keeping the
measured first-departure and re-entry peaks close to baseline.

## Geometry-addressing HTML presentation

Generate the lightweight 16:9 presentation from the fixed-pose geometry
artifacts with:

    python scripts/make_mapkv_geometry_presentation.py \
      --root results/mapkv_fast/yaw45m20to35_scene01_seed0_geometry_repair

The output `presentation.html` is a nine-slide Chinese walkthrough of generated
history → CUT3R depth/confidence → fixed world alignment → stable RGB surfels →
target-view visibility → chunk voting → selected KV address. It uses only
relative artifact paths, includes the full baseline revisit video, supports
keyboard/fullscreen navigation and `?slide=N`, and ends with the controlled
coarse-addressing result plus its unrestricted-retrieval limitation.

## Lightweight changed-view memory adapter

The adapter stage freezes the complete InSpatio/CUT3R memory pipeline and only
trains a small parallel patch-token residual:

```text
known-pose generated history
  -> RGB camera warp -> Wan VAE L_mem
  -> M_need = generated-only history x current source-blind
  -> concat[L_mem, raw last_pred, M_need]
  -> 2 x Conv3D(32) + zero-init 1x1 projection
  -> max-pooled token support x patch-token residual
  -> frozen InSpatio transformer
```

The native patch embedding is untouched. With no adapter context, no additional
Conv3D call is made. With a freshly initialized adapter, the residual is exactly
zero; both controlled cases require a full latent `max_abs_diff == 0` before
training. Backbone, text encoder, Wan VAE, scheduler, render/source pipeline,
CUT3R, surfel map, camera warp, and re-entry lifecycle remain frozen.

Run the cached two-scene workflow with:

```bash
bash scripts/run_mapkv_memory_adapter.sh \
  --stage full --gpu 0 \
  --output_root results/mapkv_fast/memory_adapter_two_scene
```

Individual stages are `prepare`, `zero_init`, `baselines`, `overfit`, `joint`,
`evaluate`, and `report`. Training first overfits
`yaw45m20to35_scene01`, then uses balanced sampling over the independent
`yaw45m20to35_scene01/02` cases. It consumes only re-entry blocks with at least
5% accepted `M_need` coverage. Checkpoints and videos are experiment artifacts
under the result root and are not committed.

If patch-only fails the two-scene identity gate, `full` performs exactly one
allowed refinement: a second zero-init projection of the same compact memory
feature into transformer blocks `[10,20)`. No geometry, mask, loss, width,
step count, or backbone parameter changes. It is also runnable explicitly as
`--stage refine`.

Scene01 uses `0 -> +45 -> -20 -> +35`. The independent indoor scene02 keeps the
same pure-yaw changed-view revisit class but uses `0 -> +45 -> -45 -> +35`:
the stronger leave segment is required because its wide-FOV anchor surfaces
never become fully absent at -20 degrees. The absence threshold and lifecycle
are not relaxed.

The final `report.html` includes the complete B1 -> leave -> re-entry -> B2
videos for both scenes, Chinese labels for the current focus, architecture and
architecture-change diagrams, target-aligned memory/mask previews,
automatically selected generated-history identity crops, training curves, and
separate identity/source/boundary/temporal measurements.
