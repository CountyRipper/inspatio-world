#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data4/daixiangting/inspatio-world}"
SLAM3R_ROOT="${SLAM3R_ROOT:-/data4/daixiangting/SLAM3R}"
CONDA_ROOT="${CONDA_ROOT:-/data3/daixiangting/miniconda3}"
CONDA_ENV="${CONDA_ENV:-/data4/daixiangting/conda_envs/slam3r}"
GPU="${GPU:-0}"

PRED_VIDEO="${PRED_VIDEO:-/data4/daixiangting/inspatio_v4_results/example0_yaw_0_45_0_45/checkpoints/InSpatio-World-1.3B/version_0/0-pred_video_rank0.mp4}"
REFERENCE_DIR="${REFERENCE_DIR:-${PROJECT_ROOT}/test/example/new_vggt/cropped_source/render}"
OUTPUT_DIR="${OUTPUT_DIR:-/data4/daixiangting/inspatio_v6_results/example0_slam3r_offline}"
I2P_MODEL="${I2P_MODEL:-siyan824/slam3r_i2p}"
L2W_MODEL="${L2W_MODEL:-siyan824/slam3r_l2w}"

INITIAL_WINSIZE="${INITIAL_WINSIZE:-5}"
WIN_R="${WIN_R:-3}"
NUM_SCENE_FRAME="${NUM_SCENE_FRAME:-10}"
BUFFER_SIZE="${BUFFER_SIZE:-30}"
CONF_THRES_I2P="${CONF_THRES_I2P:-1.5}"
CONF_THRES_L2W="${CONF_THRES_L2W:-12.0}"
FRAME_MEAN_CONF_THRES="${FRAME_MEAN_CONF_THRES:-10.0}"
SIM3_MAX_CANDIDATES="${SIM3_MAX_CANDIDATES:-11}"
SIM3_MAX_NORMALIZED_RMSE="${SIM3_MAX_NORMALIZED_RMSE:-0.08}"
SIM3_MAX_VALIDATION_P90="${SIM3_MAX_VALIDATION_P90:-0.20}"
MAX_KEYFRAMES="${MAX_KEYFRAMES:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -f "${PRED_VIDEO}" ]]; then
  echo "Missing generated video: ${PRED_VIDEO}" >&2
  exit 2
fi
if [[ ! -d "${REFERENCE_DIR}" ]]; then
  echo "Missing reference directory: ${REFERENCE_DIR}" >&2
  exit 2
fi
if [[ ! -f "${SLAM3R_ROOT}/slam3r/models.py" ]]; then
  echo "Missing SLAM3R checkout: ${SLAM3R_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda initialization: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 2
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_ROOT}:${SLAM3R_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
if [[ "${MAX_KEYFRAMES}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max-keyframes "${MAX_KEYFRAMES}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python scripts/run_slam3r_offline_v6.py \
  --pred-video "${PRED_VIDEO}" \
  --reference-dir "${REFERENCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --slam3r-root "${SLAM3R_ROOT}" \
  --i2p-model "${I2P_MODEL}" \
  --l2w-model "${L2W_MODEL}" \
  --device cuda:0 \
  --initial-winsize "${INITIAL_WINSIZE}" \
  --win-r "${WIN_R}" \
  --num-scene-frame "${NUM_SCENE_FRAME}" \
  --buffer-size "${BUFFER_SIZE}" \
  --conf-thres-i2p "${CONF_THRES_I2P}" \
  --conf-thres-l2w "${CONF_THRES_L2W}" \
  --frame-mean-conf-thres "${FRAME_MEAN_CONF_THRES}" \
  --sim3-max-candidates "${SIM3_MAX_CANDIDATES}" \
  --sim3-max-normalized-rmse "${SIM3_MAX_NORMALIZED_RMSE}" \
  --sim3-max-validation-p90 "${SIM3_MAX_VALIDATION_P90}" \
  "${EXTRA_ARGS[@]}"
