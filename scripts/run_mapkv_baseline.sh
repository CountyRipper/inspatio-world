#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir)"
ASSET_ROOT="${INSPATIO_ASSET_ROOT:-$(dirname "${GIT_COMMON_DIR}")}"
PYTHON_BIN="${INSPATIO_PYTHON:-python}"
ARTIFACT_ROOT="${MAPKV_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
GPU="${MAPKV_GPU:-0}"

BASELINE_ROOT="${ARTIFACT_ROOT}/baseline"
NOISE_BUNDLE="${BASELINE_ROOT}/noise_bundle.pt"
CREATE_NOISE=()
if [[ ! -f "${NOISE_BUNDLE}" ]]; then
  CREATE_NOISE=(--create_noise_bundle)
fi
REPLAY_ARGS=(--verify_memory_off_replay)
if [[ "${MAPKV_SKIP_REPLAY_CHECK:-0}" == "1" ]]; then
  REPLAY_ARGS=()
fi
STRICT_ARGS=()
if [[ "${MAPKV_REQUIRE_EXACT:-0}" == "1" ]]; then
  STRICT_ARGS=(--require_replay_tolerance)
fi

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
  --output_dir "${BASELINE_ROOT}" \
  --run_name baseline \
  --noise_bundle "${NOISE_BUNDLE}" \
  --bank_root "${BASELINE_ROOT}/kv_bank" \
  --mode baseline \
  --capture_kv \
  "${CREATE_NOISE[@]}" \
  "${REPLAY_ARGS[@]}" \
  "${STRICT_ARGS[@]}" \
  "$@"

"${PYTHON_BIN}" -m mapkv_proto.revisit_pair \
  --block_mapping "${BASELINE_ROOT}/block_mapping.json" \
  --output_json "${BASELINE_ROOT}/revisit_candidates.json" \
  --contact_sheet "${BASELINE_ROOT}/revisit_candidates.png"
