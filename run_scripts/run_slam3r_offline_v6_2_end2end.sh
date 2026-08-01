#!/usr/bin/env bash
set -euo pipefail

# Public v6_2 entrypoint. It prepares the PNG-only official SLAM3R dataset,
# then delegates the fixed-map build and InSpatio Pass 2 to the stage runner.
PROJECT_ROOT="${PROJECT_ROOT:-/data4/daixiangting/inspatio-world}"
SLAM3R_ROOT="${SLAM3R_ROOT:-/data4/daixiangting/SLAM3R}"
CONDA_ROOT="${CONDA_ROOT:-/data3/daixiangting/miniconda3}"
SLAM3R_ENV="${SLAM3R_ENV:-/data4/daixiangting/conda_envs/slam3r}"
GPU="${GPU:-0}"
PRED_VIDEO="${PRED_VIDEO:?Set PRED_VIDEO}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
I2P_MODEL="${I2P_MODEL:-/data4/daixiangting/models/slam3r_i2p}"
L2W_MODEL="${L2W_MODEL:-/data4/daixiangting/models/slam3r_l2w}"

KEYFRAME_DIR="${OUTPUT_DIR}/slam3r_input"
OFFICIAL_DIR="${OUTPUT_DIR}/offline_slam3r/official"
PREDS_DIR="${OFFICIAL_DIR}/preds"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
export PYTHONPATH="${PROJECT_ROOT}:${SLAM3R_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

if [[ ! -s "${KEYFRAME_DIR}/manifest.json" ]]; then
  conda activate "${SLAM3R_ENV}"
  python scripts/export_slam3r_keyframes_v6_2.py \
    --pred-video "${PRED_VIDEO}" \
    --output-dir "${KEYFRAME_DIR}"
fi

PREDS_COMPLETE=1
for name in local_pcds.npy registered_pcds.npy local_confs.npy registered_confs.npy input_imgs.npy metadata.json; do
  if [[ ! -s "${PREDS_DIR}/${name}" ]]; then
    PREDS_COMPLETE=0
  fi
done
if [[ "${PREDS_COMPLETE}" != "1" ]]; then
  conda activate "${SLAM3R_ENV}"
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/run_slam3r_offline_v6_2_official.py \
    --slam3r-root "${SLAM3R_ROOT}" \
    --input-dir "${KEYFRAME_DIR}" \
    --output-dir "${OFFICIAL_DIR}" \
    --i2p-model "${I2P_MODEL}" \
    --l2w-model "${L2W_MODEL}" \
    --device cuda:0 \
    --buffer-size 100
fi

exec "${PROJECT_ROOT}/run_scripts/run_slam3r_offline_v6_2.sh"
