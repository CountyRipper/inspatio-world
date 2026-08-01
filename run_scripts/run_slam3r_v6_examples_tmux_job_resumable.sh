#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data4/daixiangting/inspatio-world
CONDA_ROOT=/data3/daixiangting/miniconda3
CONDA_ENV=/data4/daixiangting/conda_envs/slam3r
SLAM3R_ROOT=/data4/daixiangting/SLAM3R
RESULT_ROOT=/data4/daixiangting/inspatio_v6_results
JOB_ROOT="${RESULT_ROOT}/_slam3r_v6_examples_job"
LOG_PATH="${JOB_ROOT}/job.log"
STATUS_PATH="${JOB_ROOT}/status.txt"
I2P_MODEL_DIR=/data4/daixiangting/models/slam3r_i2p
L2W_MODEL_DIR=/data4/daixiangting/models/slam3r_l2w
RANGE_DOWNLOADER=/data4/daixiangting/parallel_range_download_linux.sh

mkdir -p "${JOB_ROOT}"
exec >>"${LOG_PATH}" 2>&1

status() {
  local message="$1"
  local line
  line="$(date '+%Y-%m-%d %H:%M:%S') ${message}"
  printf '%s\n' "${line}"
  printf '%s\n' "${line}" >"${STATUS_PATH}"
}

on_exit() {
  local code=$?
  if [[ ${code} -eq 0 ]]; then
    status "COMPLETE all examples"
  else
    status "FAILED exit_code=${code}"
  fi
}
trap on_exit EXIT

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${SLAM3R_ROOT}:${PYTHONPATH:-}"
export HF_HOME=/data3/daixiangting/cache
export TOKENIZERS_PARALLELISM=false

download_model() {
  local model_id="$1"
  local model_dir="$2"
  local size="$3"
  local sha256="$4"
  local weight_path="${model_dir}/model.safetensors"
  local actual_sha256=""

  mkdir -p "${model_dir}"
  if [[ -s "${weight_path}" ]]; then
    actual_sha256="$(sha256sum "${weight_path}" | awk '{print $1}')"
  fi
  if [[ "${actual_sha256}" != "${sha256}" ]]; then
    status "DOWNLOAD ${model_id} resumable_ranges"
    "${RANGE_DOWNLOADER}" \
      "https://huggingface.co/${model_id}/resolve/main/model.safetensors" \
      "${size}" \
      "${sha256}" \
      "${weight_path}" \
      16 \
      32
  fi

  curl --location --fail --silent --show-error \
    --connect-timeout 30 --max-time 120 --retry 10 --retry-all-errors \
    --output "${model_dir}/config.json.tmp" \
    "https://huggingface.co/${model_id}/resolve/main/config.json"
  mv "${model_dir}/config.json.tmp" "${model_dir}/config.json"
  MODEL_DIR="${model_dir}" EXPECTED_SIZE="${size}" EXPECTED_SHA256="${sha256}" \
    python -c 'import hashlib, json, os; from pathlib import Path; p=Path(os.environ["MODEL_DIR"]); w=p/"model.safetensors"; assert w.stat().st_size==int(os.environ["EXPECTED_SIZE"]); assert hashlib.sha256(w.read_bytes()).hexdigest()==os.environ["EXPECTED_SHA256"]; json.loads((p/"config.json").read_text()); print("validated model", p)'
  status "DOWNLOAD_COMPLETE ${model_id}"
}

validate_case() {
  local label="$1"
  local output_dir="$2"
  LABEL="${label}" OUTPUT_DIR="${output_dir}" python -c 'import json, os; from pathlib import Path; root=Path(os.environ["OUTPUT_DIR"]); summary=json.loads((root/"summary.json").read_text()); assert summary["status"]=="complete", summary; assert summary["final_map_points"]>0, summary; required=[root/"canonical_map.ply",root/"canonical_map.npz",root/"slam3r_to_canonical_sim3.npz",root/"fused_causal.mp4"]; missing=[str(p) for p in required if not p.is_file() or p.stat().st_size==0]; assert not missing, missing; print("validated",os.environ["LABEL"],"points",summary["final_map_points"],"accepted",summary["accepted_frame_count"],"rejected",summary["rejected_frame_count"])'
}

run_case() {
  local label="$1"
  local pred_video="$2"
  local reference_dir="$3"
  local output_dir="$4"
  status "RUN_START ${label}"
  GPU=0 \
  OVERWRITE=1 \
  I2P_MODEL="${I2P_MODEL_DIR}" \
  L2W_MODEL="${L2W_MODEL_DIR}" \
  PRED_VIDEO="${pred_video}" \
  REFERENCE_DIR="${reference_dir}" \
  OUTPUT_DIR="${output_dir}" \
  bash "${PROJECT_ROOT}/run_scripts/run_slam3r_offline_v6.sh"
  validate_case "${label}" "${output_dir}"
  status "RUN_COMPLETE ${label}"
}

status "START pid=$$"
download_model siyan824/slam3r_i2p "${I2P_MODEL_DIR}" 2131831108 9f58b4a39f9e3641ed2fe19d16a82be956975ec1ecb568899c8852370aa78b8c
download_model siyan824/slam3r_l2w "${L2W_MODEL_DIR}" 922557296 637c3bd3d440d84e2976d898569716ef64cadd446fce61ad5dd9236fa2c58359
export HF_HUB_OFFLINE=1

run_case \
  example0 \
  /data4/daixiangting/inspatio_v4_results/example0_yaw_0_45_0_45/checkpoints/InSpatio-World-1.3B/version_0/0-pred_video_rank0.mp4 \
  /data4/daixiangting/inspatio-world/test/example/new_vggt/cropped_source/render \
  /data4/daixiangting/inspatio_v6_results/example0_slam3r_offline

run_case \
  example1 \
  /data4/daixiangting/inspatio_v4_results/example1_yaw_0_45_0_45/checkpoints/InSpatio-World-1.3B/version_0/0-pred_video_rank0.mp4 \
  /data4/daixiangting/inspatio-world/test/example2/new_vggt/coffee_martini/render \
  /data4/daixiangting/inspatio_v6_results/example1_slam3r_offline
