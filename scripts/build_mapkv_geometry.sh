#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir)"
ASSET_ROOT="${INSPATIO_ASSET_ROOT:-$(dirname "${GIT_COMMON_DIR}")}"
INSPATIO_PYTHON="${INSPATIO_PYTHON:-python}"
ARTIFACT_ROOT="${MAPKV_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
VMEM_ROOT="${VMEM_ROOT:-${REPO_ROOT}/third_party/vmem}"
CUT3R_CHECKPOINT="${CUT3R_CHECKPOINT:-${VMEM_ROOT}/extern/CUT3R/src/cut3r_512_dpt_4_64.pth}"
CUT3R_DEVICE="${CUT3R_DEVICE:-cuda:0}"
CUT3R_GPU="${CUT3R_GPU:-${MAPKV_GPU:-0}}"
TARGET_CHUNK=""
ORACLE_SOURCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target_chunk) TARGET_CHUNK="$2"; shift 2 ;;
    --oracle_source) ORACLE_SOURCE="$2"; shift 2 ;;
    --vmem_root) VMEM_ROOT="$2"; shift 2 ;;
    --cut3r_checkpoint) CUT3R_CHECKPOINT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${TARGET_CHUNK}" && -n "${ORACLE_SOURCE}" ]] || {
  echo "Usage: $0 --target_chunk R --oracle_source B" >&2
  exit 2
}
[[ -n "${CUT3R_PYTHON:-}" ]] || {
  echo "Set CUT3R_PYTHON to the Python executable in the separate CUT3R environment." >&2
  exit 2
}

GEOMETRY_ROOT="${ARTIFACT_ROOT}/geometry"
VIEWS_ROOT="${GEOMETRY_ROOT}/views"
cd "${REPO_ROOT}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUT3R_GPU}" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${INSPATIO_PYTHON}" -m mapkv_proto.cut3r.export_views \
  --block_mapping "${ARTIFACT_ROOT}/baseline/block_mapping.json" \
  --anchor_image "${ARTIFACT_ROOT}/baseline/anchor.png" \
  --intrinsic_path "${ASSET_ROOT}/test/example/new_vggt/cropped_source/intrinsics.txt" \
  --intrinsic_source_height 480 \
  --intrinsic_source_width 832 \
  --output_root "${VIEWS_ROOT}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUT3R_GPU}" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${CUT3R_PYTHON}" -m mapkv_proto.cut3r.build_surfel_index \
  --views_json "${VIEWS_ROOT}/views.json" \
  --vmem_root "${VMEM_ROOT}" \
  --checkpoint "${CUT3R_CHECKPOINT}" \
  --output_root "${GEOMETRY_ROOT}" \
  --device "${CUT3R_DEVICE}" \
  --niter 100 \
  --lr 0.01 \
  --grid_height 30 \
  --grid_width 52

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${INSPATIO_PYTHON}" -m mapkv_proto.retrieval \
  --block_mapping "${ARTIFACT_ROOT}/baseline/block_mapping.json" \
  --target_chunks "${TARGET_CHUNK}" \
  --output "${GEOMETRY_ROOT}/pose_address_plan.json"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${INSPATIO_PYTHON}" -m mapkv_proto.cut3r.build_retrieval_plan \
  --surfel_index "${GEOMETRY_ROOT}/surfel_index.npz" \
  --views_json "${VIEWS_ROOT}/views.json" \
  --target_chunks "${TARGET_CHUNK}" \
  --oracle_source "${ORACLE_SOURCE}" \
  --output "${GEOMETRY_ROOT}/surfel_plan.json"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${INSPATIO_PYTHON}" -m mapkv_proto.cut3r.build_retrieval_plan \
  --surfel_index "${GEOMETRY_ROOT}/surfel_index.npz" \
  --views_json "${VIEWS_ROOT}/views.json" \
  --target_chunks "${TARGET_CHUNK}" \
  --oracle_source "${ORACLE_SOURCE}" \
  --address_plan "${GEOMETRY_ROOT}/pose_address_plan.json" \
  --output "${GEOMETRY_ROOT}/pose_plan.json"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${INSPATIO_PYTHON}" -m mapkv_proto.cut3r.build_retrieval_plan \
  --surfel_index "${GEOMETRY_ROOT}/surfel_index.npz" \
  --views_json "${VIEWS_ROOT}/views.json" \
  --target_chunks "${TARGET_CHUNK}" \
  --oracle_source "${ORACLE_SOURCE}" \
  --fixed_selected_chunk "${ORACLE_SOURCE}" \
  --output "${GEOMETRY_ROOT}/oracle_plan.json"
