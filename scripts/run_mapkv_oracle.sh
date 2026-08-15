#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir)"
ASSET_ROOT="${INSPATIO_ASSET_ROOT:-$(dirname "${GIT_COMMON_DIR}")}"
PYTHON_BIN="${INSPATIO_PYTHON:-python}"
ARTIFACT_ROOT="${MAPKV_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
GPU="${MAPKV_GPU:-0}"

MODE="oracle"
SOURCE_CHUNK=""
TARGET_CHUNK=""
ALPHA="0.10"
RUN_NAME=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --source_chunk) SOURCE_CHUNK="$2"; shift 2 ;;
    --target_chunk) TARGET_CHUNK="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

RUNNER_MODE="${MODE}"
SOURCE_ARGS=()
TARGET_ARGS=()
COMPARE_ARGS=()
case "${MODE}" in
  baseline|off)
    RUNNER_MODE="baseline"
    COMPARE_ARGS=(--compare_latents_to "${ARTIFACT_ROOT}/baseline/pred_latents.pt")
    ;;
  alpha_zero|oracle)
    if [[ "${MODE}" == "alpha_zero" ]]; then
      RUNNER_MODE="oracle"
      ALPHA="0.0"
    fi
    [[ -n "${SOURCE_CHUNK}" && -n "${TARGET_CHUNK}" ]] || {
      echo "oracle/alpha_zero requires --source_chunk and --target_chunk" >&2
      exit 2
    }
    SOURCE_ARGS=(--source_chunk "${SOURCE_CHUNK}")
    TARGET_ARGS=(--target_chunks "${TARGET_CHUNK}")
    if [[ "${ALPHA}" == "0" || "${ALPHA}" == "0.0" || "${ALPHA}" == "0.00" ]]; then
      COMPARE_ARGS=(--compare_latents_to "${ARTIFACT_ROOT}/baseline/pred_latents.pt")
    fi
    ;;
  wrong)
    [[ -n "${SOURCE_CHUNK}" && -n "${TARGET_CHUNK}" ]] || {
      echo "wrong requires --source_chunk WRONG_CHUNK and --target_chunk" >&2
      exit 2
    }
    SOURCE_ARGS=(--wrong_chunk "${SOURCE_CHUNK}")
    TARGET_ARGS=(--target_chunks "${TARGET_CHUNK}")
    ;;
  *) echo "Unsupported mode: ${MODE}" >&2; exit 2 ;;
esac
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="${MODE}_a${ALPHA/./}"
fi

RUN_ROOT="${ARTIFACT_ROOT}/oracle/runs/${RUN_NAME}"
VIDEO_OUTPUT="${ARTIFACT_ROOT}/oracle/${RUN_NAME}.mp4"
cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" inference_mapkv_proto.py \
  --config_path configs/inference_1.3b.yaml \
  --mapkv_config configs/mapkv_proto.yaml \
  --checkpoint_path "${ASSET_ROOT}/checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors" \
  --wan_model_folder "${ASSET_ROOT}/checkpoints/Wan2.1-T2V-1.3B" \
  --json_path "${ASSET_ROOT}/test/example/new.json" \
  --data_path_root "${ASSET_ROOT}" \
  --traj_txt_path "${REPO_ROOT}/traj/x_y_circle_cycle.txt" \
  --output_dir "${RUN_ROOT}" \
  --video_output "${VIDEO_OUTPUT}" \
  --run_name "${RUN_NAME}" \
  --noise_bundle "${ARTIFACT_ROOT}/baseline/noise_bundle.pt" \
  --bank_root "${ARTIFACT_ROOT}/baseline/kv_bank" \
  --mode "${RUNNER_MODE}" \
  --alpha "${ALPHA}" \
  --gate_mode ref_blind \
  "${SOURCE_ARGS[@]}" \
  "${TARGET_ARGS[@]}" \
  "${COMPARE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
