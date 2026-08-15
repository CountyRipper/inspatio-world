#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir)"
ASSET_ROOT="${INSPATIO_ASSET_ROOT:-$(dirname "${GIT_COMMON_DIR}")}"
PYTHON_BIN="${INSPATIO_PYTHON:-python}"
ARTIFACT_ROOT="${MAPKV_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
GPU="${MAPKV_GPU:-0}"

RETRIEVAL="geometry"
PLAN=""
ALPHA="0.10"
RUN_NAME=""
SOURCE_CHUNK=""
TARGET_CHUNK=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --retrieval) RETRIEVAL="$2"; shift 2 ;;
    --retrieval_plan) PLAN="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --source_chunk) SOURCE_CHUNK="$2"; shift 2 ;;
    --target_chunk) TARGET_CHUNK="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="${RETRIEVAL}kv_a${ALPHA/./}"
fi

MODE_ARGS=()
PLAN_ARGS=()
GATE="surfel_ref_blind"
case "${RETRIEVAL}" in
  baseline)
    MODE_ARGS=(--mode baseline)
    GATE="ref_blind"
    ;;
  pose|geometry)
    [[ -n "${PLAN}" ]] || PLAN="${ARTIFACT_ROOT}/geometry/${RETRIEVAL/geometry/surfel}_plan.json"
    [[ -f "${PLAN}" ]] || { echo "Retrieval plan not found: ${PLAN}" >&2; exit 2; }
    MODE_ARGS=(--mode "${RETRIEVAL}")
    PLAN_ARGS=(--retrieval_plan "${PLAN}")
    ;;
  oracle)
    [[ -n "${PLAN}" ]] || { echo "oracle geometry replay requires --retrieval_plan with Oracle coverage" >&2; exit 2; }
    MODE_ARGS=(--mode oracle)
    PLAN_ARGS=(--retrieval_plan "${PLAN}")
    [[ -z "${SOURCE_CHUNK}" ]] || MODE_ARGS+=(--source_chunk "${SOURCE_CHUNK}")
    ;;
  *) echo "Unsupported retrieval: ${RETRIEVAL}" >&2; exit 2 ;;
esac
if [[ -n "${TARGET_CHUNK}" ]]; then
  MODE_ARGS+=(--target_chunks "${TARGET_CHUNK}")
fi

RUN_ROOT="${ARTIFACT_ROOT}/geometry/runs/${RUN_NAME}"
VIDEO_OUTPUT="${ARTIFACT_ROOT}/geometry/${RUN_NAME}.mp4"
cd "${REPO_ROOT}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
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
  --alpha "${ALPHA}" \
  --gate_mode "${GATE}" \
  "${MODE_ARGS[@]}" \
  "${PLAN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
