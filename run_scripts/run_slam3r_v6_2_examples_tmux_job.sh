#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data4/daixiangting/inspatio-world}"
RESULT_ROOT="${RESULT_ROOT:-/data4/daixiangting/inspatio_v6_2_results}"
GPU="${GPU:-0}"
STATUS_FILE="${RESULT_ROOT}/tmux_status.txt"
mkdir -p "${RESULT_ROOT}"

on_exit() {
  code=$?
  if [[ ${code} -eq 0 ]]; then
    printf 'complete\n' > "${STATUS_FILE}"
  else
    printf 'failed exit_code=%s\n' "${code}" > "${STATUS_FILE}"
  fi
}
trap on_exit EXIT
printf 'running example0\n' > "${STATUS_FILE}"

cd "${PROJECT_ROOT}"
GPU="${GPU}" \
MASTER_PORT=29562 \
PRED_VIDEO="/data4/daixiangting/inspatio_v4_results/example0_yaw_0_45_0_45/checkpoints/InSpatio-World-1.3B/version_0/0-pred_video_rank0.mp4" \
REFERENCE_DIR="${PROJECT_ROOT}/test/example/new_vggt/cropped_source/render" \
ORIGINAL_JSON="${PROJECT_ROOT}/test/example/new.json" \
TRAJ_TXT_PATH="/data4/daixiangting/inspatio_v4_results/trajectories/yaw_0_45_0_45_247.txt" \
OUTPUT_DIR="${RESULT_ROOT}/example0_slam3r_offline_v6_2" \
./run_scripts/run_slam3r_offline_v6_2.sh

printf 'running example1\n' > "${STATUS_FILE}"
GPU="${GPU}" \
MASTER_PORT=29563 \
PRED_VIDEO="/data4/daixiangting/inspatio_v4_results/example1_yaw_0_45_0_45/checkpoints/InSpatio-World-1.3B/version_0/0-pred_video_rank0.mp4" \
REFERENCE_DIR="${PROJECT_ROOT}/test/example2/new_vggt/coffee_martini/render" \
ORIGINAL_JSON="${PROJECT_ROOT}/test/example2/new.json" \
TRAJ_TXT_PATH="/data4/daixiangting/inspatio_v4_results/trajectories/yaw_0_45_0_45_300.txt" \
OUTPUT_DIR="${RESULT_ROOT}/example1_slam3r_offline_v6_2" \
./run_scripts/run_slam3r_offline_v6_2.sh
