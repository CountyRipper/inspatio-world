#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_DIR=""
SEEDS="0,1,2"
ALPHAS="0.05,0.10,0.20"
RUN_ACTIVATION=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case_dir) CASE_DIR="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    --skip_activation) RUN_ACTIVATION=false; shift ;;
    *) echo "Unsupported argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${CASE_DIR}" ]] || { echo "--case_dir is required" >&2; exit 2; }
CASE_DIR="$(cd "${CASE_DIR}" && pwd)"
SOURCE_CHUNK="$(jq -r '.source_chunk' "${CASE_DIR}/trajectory_manifest.json")"
TARGET_CHUNK="$(jq -r '.target_chunk' "${CASE_DIR}/trajectory_manifest.json")"
WRONG_CHUNK="$(jq -r '.wrong_chunk' "${CASE_DIR}/trajectory_manifest.json")"

run_if_missing() {
  local marker="$1"
  shift
  if [[ -f "${marker}" ]]; then
    echo "[MapKV matrix] skip existing ${marker}"
  else
    "$@"
  fi
}

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
for SEED in "${SEED_ARRAY[@]}"; do
  BASELINE_ROOT="${CASE_DIR}/baseline/seed_${SEED}"
  RUN_ROOT="${CASE_DIR}/oracle/seed_${SEED}/runs"
  run_if_missing "${BASELINE_ROOT}/run_metadata.json" \
    env MAPKV_REQUIRE_EXACT=1 bash "${REPO_ROOT}/scripts/run_mapkv_baseline.sh" \
      --case_dir "${CASE_DIR}" --seed "${SEED}"
  run_if_missing "${RUN_ROOT}/alpha_zero/run_metadata.json" \
    bash "${REPO_ROOT}/scripts/run_mapkv_oracle.sh" \
      --case_dir "${CASE_DIR}" --seed "${SEED}" --mode alpha_zero \
      --source_chunk "${SOURCE_CHUNK}" --target_chunk "${TARGET_CHUNK}" \
      --run_name alpha_zero
  if [[ "${RUN_ACTIVATION}" == true && "${SEED}" == "0" ]]; then
    run_if_missing "${RUN_ROOT}/oracle_activation_a100/run_metadata.json" \
      bash "${REPO_ROOT}/scripts/run_mapkv_oracle.sh" \
        --case_dir "${CASE_DIR}" --seed "${SEED}" --mode oracle \
        --source_chunk "${SOURCE_CHUNK}" --target_chunk "${TARGET_CHUNK}" \
        --alpha 1.0 --run_name oracle_activation_a100
  fi
  IFS=',' read -r -a ALPHA_ARRAY <<< "${ALPHAS}"
  for ALPHA in "${ALPHA_ARRAY[@]}"; do
    RUN_NAME="oracle_a${ALPHA/./}"
    run_if_missing "${RUN_ROOT}/${RUN_NAME}/run_metadata.json" \
      bash "${REPO_ROOT}/scripts/run_mapkv_oracle.sh" \
        --case_dir "${CASE_DIR}" --seed "${SEED}" --mode oracle \
        --source_chunk "${SOURCE_CHUNK}" --target_chunk "${TARGET_CHUNK}" \
        --alpha "${ALPHA}" --run_name "${RUN_NAME}"
  done
  run_if_missing "${RUN_ROOT}/wrong_a010/run_metadata.json" \
    bash "${REPO_ROOT}/scripts/run_mapkv_oracle.sh" \
      --case_dir "${CASE_DIR}" --seed "${SEED}" --mode wrong \
      --source_chunk "${WRONG_CHUNK}" --target_chunk "${TARGET_CHUNK}" \
      --alpha 0.10 --run_name wrong_a010
  run_if_missing "${RUN_ROOT}/random_a010/run_metadata.json" \
    bash "${REPO_ROOT}/scripts/run_mapkv_oracle.sh" \
      --case_dir "${CASE_DIR}" --seed "${SEED}" --mode random \
      --source_chunk "${SOURCE_CHUNK}" --target_chunk "${TARGET_CHUNK}" \
      --random_seed 0 --alpha 0.10 --run_name random_a010
done
