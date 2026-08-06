# Phase 1 LSM-style adapter reproduction

This directory contains the source-only reproduction path for the exact-pose
Phase 1 experiment. All generated files are written below
`artifacts/phase1_lsm/`, which is intentionally ignored by Git.

Run from the repository root with the existing InSpatio environment:

```bash
export PHASE1_PY=/data4/daixiangting/conda_envs/inspatio/bin/python
export PHASE1_TORCHRUN=/data4/daixiangting/conda_envs/inspatio/bin/torchrun
export PHASE1_CKPT=/data4/daixiangting/inspatio-world/checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
```

Confirm that physical GPU 0 is idle before every GPU stage. Do not proceed if
the trajectory/pose assertions or any earlier gate fails.

## 1. CPU checks

The bundled runner executes all six tests without requiring pytest:

```bash
"$PHASE1_PY" scripts/phase1_lsm/run_cpu_tests.py
"$PHASE1_PY" -m py_compile \
  phase1_lsm/*.py scripts/phase1_lsm/*.py tests/phase1_lsm/*.py \
  pipeline/causal_inference.py utils/wan_wrapper.py wan/modules/causal_model.py
```

## 2. Generate the four fixed conditions

```bash
"$PHASE1_PY" scripts/phase1_lsm/prepare_conditions.py --source S0 --trajectory P
"$PHASE1_PY" scripts/phase1_lsm/prepare_conditions.py --source S0 --trajectory N
"$PHASE1_PY" scripts/phase1_lsm/prepare_conditions.py --source S1 --trajectory P
"$PHASE1_PY" scripts/phase1_lsm/prepare_conditions.py --source S1 --trajectory N
```

This creates both trajectory files, the fixed-8 manifest, rendered conditions,
raw target depth, intrinsics, and validated target poses.

## 3. Capture and gate the only smoke sample

```bash
"$PHASE1_TORCHRUN" --standalone --nproc_per_node=1 \
  scripts/phase1_lsm/capture_sample.py \
  --condition-dir artifacts/phase1_lsm/conditions/S0/P \
  --checkpoint "$PHASE1_CKPT" \
  --output-dir artifacts/phase1_lsm/samples/S0_P_seed0 --seed 0

"$PHASE1_TORCHRUN" --standalone --nproc_per_node=1 \
  scripts/phase1_lsm/train_adapter.py \
  --sample artifacts/phase1_lsm/samples/S0_P_seed0/sample.safetensors \
  --checkpoint "$PHASE1_CKPT" \
  --output-dir artifacts/phase1_lsm/train/smoke_direct \
  --memory-kind direct --max-steps 1000 --lr 1e-3 --weight-decay 0 \
  --early-stop-ratio 0.2

"$PHASE1_PY" scripts/phase1_lsm/make_montage.py \
  --training-outputs artifacts/phase1_lsm/train/smoke_direct/training_outputs.safetensors \
  --output-dir artifacts/phase1_lsm/train/smoke_direct --gate-key direct

"$PHASE1_TORCHRUN" --standalone --nproc_per_node=1 \
  scripts/phase1_lsm/train_adapter.py \
  --sample artifacts/phase1_lsm/samples/S0_P_seed0/sample.safetensors \
  --checkpoint "$PHASE1_CKPT" \
  --output-dir artifacts/phase1_lsm/train/smoke_projected \
  --memory-kind projected --max-steps 1000 --lr 1e-3 --weight-decay 0 \
  --early-stop-ratio 0.2

"$PHASE1_PY" scripts/phase1_lsm/make_montage.py \
  --training-outputs artifacts/phase1_lsm/train/smoke_projected/training_outputs.safetensors \
  --output-dir artifacts/phase1_lsm/train/smoke_projected --gate-key projected
```

Do not create the remaining samples unless both smoke stages pass.

## 4. Capture the remaining fixed samples

Use this helper only after the smoke gate:

```bash
capture_phase1_sample() {
  phase1_source="$1"
  phase1_trajectory="$2"
  phase1_seed="$3"
  "$PHASE1_TORCHRUN" --standalone --nproc_per_node=1 \
    scripts/phase1_lsm/capture_sample.py \
    --condition-dir "artifacts/phase1_lsm/conditions/${phase1_source}/${phase1_trajectory}" \
    --checkpoint "$PHASE1_CKPT" \
    --output-dir "artifacts/phase1_lsm/samples/${phase1_source}_${phase1_trajectory}_seed${phase1_seed}" \
    --seed "$phase1_seed"
}

capture_phase1_sample S0 P 1
capture_phase1_sample S0 N 0
capture_phase1_sample S0 N 1
capture_phase1_sample S1 P 0
capture_phase1_sample S1 P 1
capture_phase1_sample S1 N 0
capture_phase1_sample S1 N 1

"$PHASE1_PY" scripts/phase1_lsm/audit_dataset.py \
  --artifacts-root artifacts/phase1_lsm
```

## 5. Train the fixed eight samples

```bash
"$PHASE1_TORCHRUN" --standalone --nproc_per_node=1 \
  scripts/phase1_lsm/train_adapter_8.py \
  --samples-root artifacts/phase1_lsm/samples \
  --checkpoint "$PHASE1_CKPT" \
  --output-dir artifacts/phase1_lsm/train/fixed8_direct \
  --memory-kind direct --max-steps 400 --lr 1e-3 --weight-decay 0 \
  --eval-every 32 --early-stop-ratio 0.6

"$PHASE1_TORCHRUN" --standalone --nproc_per_node=1 \
  scripts/phase1_lsm/train_adapter_8.py \
  --samples-root artifacts/phase1_lsm/samples \
  --checkpoint "$PHASE1_CKPT" \
  --output-dir artifacts/phase1_lsm/train/fixed8_projected \
  --memory-kind projected --max-steps 400 --lr 1e-3 --weight-decay 0 \
  --eval-every 32 --early-stop-ratio 0.6
```

## 6. Generate final plots

```bash
"$PHASE1_PY" scripts/phase1_lsm/compose_final_outputs.py \
  --direct-outputs artifacts/phase1_lsm/train/smoke_direct/training_outputs.safetensors \
  --projected-outputs artifacts/phase1_lsm/train/smoke_projected/training_outputs.safetensors \
  --output artifacts/phase1_lsm/final/training_outputs.safetensors

MPLBACKEND=Agg "$PHASE1_PY" scripts/phase1_lsm/plot_loss_curves.py \
  --train-root artifacts/phase1_lsm/train \
  --output artifacts/phase1_lsm/final/loss_curves.png

"$PHASE1_PY" scripts/phase1_lsm/make_montage.py \
  --training-outputs artifacts/phase1_lsm/final/training_outputs.safetensors \
  --output-dir artifacts/phase1_lsm/final --gate-key projected
```

The adapter-only checkpoints and configs are written inside each training
directory as `memory_adapter.safetensors` and `memory_adapter_config.json`.
