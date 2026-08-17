# MapKV rapid prototype

This package closes a deliberately coarse causal loop:

```text
generated chunk PNGs -> official CUT3R prefix -> voxel surfels -> chunk vote
-> CPU native post-RoPE KV bank -> selected-layer residual attention at B2
```

The original InSpatio reference and recent caches remain intact. With memory
disabled, the original attention path is used and no second attention call is
made. The first injector supports one retrieved whole chunk; retrieval itself
can report a larger top-K for diagnostics.

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

