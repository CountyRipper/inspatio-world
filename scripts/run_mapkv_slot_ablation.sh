#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
inspatio_python="${INSPATIO_PYTHON:-/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python}"
base_run="${MAPKV_BASE_RUN:-${repo_root}/results/mapkv_fast/yaw30_scene01_seed0_core_repair}"
case_dir="${MAPKV_CASE_DIR:-${repo_root}/artifacts/control/yaw30_scene01}"
output="${MAPKV_SLOT_OUTPUT:-${repo_root}/results/mapkv_fast/yaw30_scene01_seed0_slot_ablation}"

exec "${inspatio_python}" -m mapkv.slot_ablation \
  --base-run-root "${base_run}" \
  --case-dir "${case_dir}" \
  --output "${output}" \
  --inspatio-python "${inspatio_python}" \
  "$@"
