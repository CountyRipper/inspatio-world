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
unset HF_HUB_OFFLINE

download_model() {
  local model_id="$1"
  local attempt
  for attempt in 1 2 3 4; do
    status "DOWNLOAD ${model_id} attempt=${attempt}/4"
    if MODEL_ID="${model_id}" python -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; p=Path(snapshot_download(os.environ["MODEL_ID"])); weights=list(p.glob("*.safetensors")); assert weights, f"no safetensors in {p}"; print("downloaded", p, [(x.name, x.stat().st_size) for x in weights])'; then
      status "DOWNLOAD_COMPLETE ${model_id}"
      return 0
    fi
    status "DOWNLOAD_RETRY ${model_id} attempt=${attempt}/4"
    sleep 30
  done
  status "DOWNLOAD_FAILED ${model_id}"
  return 1
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
  I2P_MODEL=siyan824/slam3r_i2p \
  L2W_MODEL=siyan824/slam3r_l2w \
  PRED_VIDEO="${pred_video}" \
  REFERENCE_DIR="${reference_dir}" \
  OUTPUT_DIR="${output_dir}" \
  bash "${PROJECT_ROOT}/run_scripts/run_slam3r_offline_v6.sh"
  validate_case "${label}" "${output_dir}"
  status "RUN_COMPLETE ${label}"
}

status "START pid=$$"
download_model siyan824/slam3r_i2p
download_model siyan824/slam3r_l2w
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
