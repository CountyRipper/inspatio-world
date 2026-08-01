#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data4/daixiangting/inspatio-world}"
SLAM3R_ROOT="${SLAM3R_ROOT:-/data4/daixiangting/SLAM3R}"
CONDA_ROOT="${CONDA_ROOT:-/data3/daixiangting/miniconda3}"
SLAM3R_ENV="${SLAM3R_ENV:-/data4/daixiangting/conda_envs/slam3r}"
INSPATIO_ENV="${INSPATIO_ENV:-/data4/daixiangting/conda_envs/inspatio}"
GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-29562}"

PRED_VIDEO="${PRED_VIDEO:?Set PRED_VIDEO to the completed v4 Pass-1 video}"
REFERENCE_DIR="${REFERENCE_DIR:?Set REFERENCE_DIR to the original render directory}"
ORIGINAL_JSON="${ORIGINAL_JSON:?Set ORIGINAL_JSON to the original one-sample new.json}"
TRAJ_TXT_PATH="${TRAJ_TXT_PATH:?Set TRAJ_TXT_PATH to the unchanged planned trajectory}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the v6_2 run}"

I2P_MODEL="${I2P_MODEL:-/data4/daixiangting/models/slam3r_i2p}"
L2W_MODEL="${L2W_MODEL:-/data4/daixiangting/models/slam3r_l2w}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/inference_1.3b.yaml}"
MAP_CONFIDENCE="${MAP_CONFIDENCE:-12.0}"
TARGET_PIXEL_SPACING="${TARGET_PIXEL_SPACING:-6.0}"
MAX_SIM3_VALIDATION_P90="${MAX_SIM3_VALIDATION_P90:-0.20}"

for path in \
  "${PRED_VIDEO}" \
  "${ORIGINAL_JSON}" \
  "${TRAJ_TXT_PATH}" \
  "${CHECKPOINT_PATH}" \
  "${CONFIG_PATH}" \
  "${I2P_MODEL}/model.safetensors" \
  "${L2W_MODEL}/model.safetensors"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
done
for path in "${REFERENCE_DIR}" "${SLAM3R_ROOT}" "${SLAM3R_ENV}" "${INSPATIO_ENV}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda initialization under ${CONDA_ROOT}" >&2
  exit 2
fi

KEYFRAME_DIR="${OUTPUT_DIR}/slam3r_input"
OFFLINE_DIR="${OUTPUT_DIR}/offline_slam3r"
OFFICIAL_DIR="${OFFLINE_DIR}/official"
PREDS_DIR="${OFFICIAL_DIR}/preds"
PASS2_DIR="${OUTPUT_DIR}/pass2"
PASS2_JSON="${OFFLINE_DIR}/pass2_input/new_v6_2.json"
PASS2_VIDEO="${PASS2_DIR}/checkpoints/InSpatio-World-1.3B/version_0/0-pred_video_rank0.mp4"
mkdir -p "${OUTPUT_DIR}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
export PYTHONPATH="${PROJECT_ROOT}:${SLAM3R_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT_ROOT}"

echo "[v6_2 1/4] Export one generated RGB keyframe per latent"
if [[ -s "${KEYFRAME_DIR}/manifest.json" ]]; then
  echo "  resume: ${KEYFRAME_DIR}/manifest.json already exists"
else
  conda activate "${SLAM3R_ENV}"
  python scripts/export_slam3r_keyframes_v6_2.py \
    --pred-video "${PRED_VIDEO}" \
    --output-dir "${KEYFRAME_DIR}"
fi

echo "[v6_2 2/4] Official SLAM3R offline reconstruction"
SLAM_PREDS_COMPLETE=1
for name in local_pcds.npy registered_pcds.npy local_confs.npy registered_confs.npy input_imgs.npy metadata.json; do
  if [[ ! -s "${PREDS_DIR}/${name}" ]]; then
    SLAM_PREDS_COMPLETE=0
  fi
done
if [[ "${SLAM_PREDS_COMPLETE}" == "1" ]]; then
  echo "  resume: official per-frame predictions already exist"
else
  conda activate "${SLAM3R_ENV}"
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/run_slam3r_offline_v6_2_backend.py \
    --slam3r-root "${SLAM3R_ROOT}" \
    --input-dir "${KEYFRAME_DIR}" \
    --output-dir "${OFFICIAL_DIR}" \
    --i2p-model "${I2P_MODEL}" \
    --l2w-model "${L2W_MODEL}" \
    --device cuda:0 \
    --buffer-size 100
fi

echo "[v6_2 3/4] Frozen Sim(3), best-confidence voxel map, and full render"
if [[ -s "${OFFLINE_DIR}/summary.json" && -s "${PASS2_JSON}" \
      && -s "${OFFLINE_DIR}/pass2_input/vggt_depth/render/render_offline.mp4" \
      && -s "${OFFLINE_DIR}/pass2_input/vggt_depth/render/mask_offline.mp4" ]]; then
  echo "  resume: fixed canonical map and condition videos already exist"
else
  conda activate "${SLAM3R_ENV}"
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/build_slam3r_offline_v6_2.py \
    --preds-dir "${PREDS_DIR}" \
    --manifest "${KEYFRAME_DIR}/manifest.json" \
    --reference-dir "${REFERENCE_DIR}" \
    --original-json "${ORIGINAL_JSON}" \
    --output-dir "${OFFLINE_DIR}" \
    --device cuda:0 \
    --map-confidence "${MAP_CONFIDENCE}" \
    --target-pixel-spacing "${TARGET_PIXEL_SPACING}" \
    --max-sim3-validation-p90 "${MAX_SIM3_VALIDATION_P90}"
fi

echo "[v6_2 4/4] InSpatio Pass 2 with the fixed fused condition"
if [[ -s "${PASS2_VIDEO}" ]]; then
  echo "  resume: ${PASS2_VIDEO} already exists"
else
  conda activate "${INSPATIO_ENV}"
  PASS2_CONFIG="${OUTPUT_DIR}/pass2_inference_config.yaml"
  cp "${CONFIG_PATH}" "${PASS2_CONFIG}"
  sed -i "/^[[:space:]]*#/!s|traj_txt_path:.*|traj_txt_path: ${TRAJ_TXT_PATH}|g" "${PASS2_CONFIG}"
  sed -i "/^[[:space:]]*#/!s|relative_to_source:.*|relative_to_source: false|g" "${PASS2_CONFIG}"
  sed -i "/^[[:space:]]*#/!s|rotation_only:.*|rotation_only: true|g" "${PASS2_CONFIG}"
  sed -i "/^[[:space:]]*#/!s|adaptive_frame:.*|adaptive_frame: false|g" "${PASS2_CONFIG}"
  CUDA_VISIBLE_DEVICES="${GPU}" torchrun \
    --nproc_per_node=1 \
    --master_port "${MASTER_PORT}" \
    inference_causal_test.py \
    --config_path "${PASS2_CONFIG}" \
    --json_path "${PASS2_JSON}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --output_folder "${PASS2_DIR}" \
    --seed 0
fi

if [[ ! -s "${PASS2_VIDEO}" ]]; then
  echo "v6_2 failed to produce ${PASS2_VIDEO}" >&2
  exit 1
fi
echo "v6_2 complete: ${PASS2_VIDEO}"
