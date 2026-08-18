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
