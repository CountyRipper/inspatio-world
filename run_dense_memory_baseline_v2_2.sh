#!/bin/bash
set -euo pipefail

# Online dense-memory v2.2: reuse the verified v2.1 reference geometry, but
# replace DA3 generated-block depth with a persistent Align3R worker. Every
# decoded RGB frame is written to M_gen and fused into subsequent STAR blocks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="${SERVER_ROOT:-/mnt/16T2/daixiangting}"
V21_ROOT="${V21_ROOT:-${SERVER_ROOT}/tmp/dense_memory_baseline_v2_1_full_gpu0_20260723}"
WORK_ROOT="${WORK_ROOT:-${SERVER_ROOT}/tmp/dense_memory_baseline_v2_2_align3r_fullblock_20260724}"
INPUT_DIR="${INPUT_DIR:-${V21_ROOT}/official_example0_1_24fps_237f}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/full_block_align3r}"
ALIGN3R_WORK_DIR="${ALIGN3R_WORK_DIR:-${WORK_ROOT}/align3r_memory_blocks}"

ALIGN3R_ROOT="${ALIGN3R_ROOT:-${SERVER_ROOT}/Align3R}"
ALIGN3R_ENV="${ALIGN3R_ENV:-${SERVER_ROOT}/conda_envs/align3r}"
INSPATIO_ENV="${INSPATIO_ENV:-${SERVER_ROOT}/conda_envs/inspatio}"
ALIGN3R_WEIGHTS="${ALIGN3R_WEIGHTS:-${ALIGN3R_ROOT}/checkpoints/align3r_depthpro}"
TORCH_HOME="${TORCH_HOME:-${SERVER_ROOT}/torch_cache}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${SERVER_ROOT}/xdg_config}"

INFERENCE_GPU="${INFERENCE_GPU:-1}"
ALIGN3R_GPU="${ALIGN3R_GPU:-0}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-false}"
MASTER_PORT="${MASTER_PORT:-29652}"
TRAJECTORY="${TRAJECTORY:-${SCRIPT_DIR}/traj/yaw_0_45_0_45_237.txt}"

INSPATIO_PYTHON="${INSPATIO_ENV}/bin/python"
ALIGN3R_PYTHON="${ALIGN3R_ENV}/bin/python"

if [ "$INFERENCE_GPU" = "$ALIGN3R_GPU" ] && [ "$ALLOW_SHARED_GPU" != true ]; then
    echo "Error: use separate physical GPUs for resident InSpatio and Align3R worker"
    echo "Set ALLOW_SHARED_GPU=true only after verifying that the selected GPU has enough VRAM"
    exit 1
fi

for required in \
    "$INSPATIO_PYTHON" \
    "$ALIGN3R_PYTHON" \
    "$INPUT_DIR/new.json" \
    "$INPUT_DIR/new_vggt/example0_da3_tmp/frames_pcd" \
    "$INPUT_DIR/new_vggt/example1_da3_tmp/frames_pcd" \
    "$TRAJECTORY" \
    "$ALIGN3R_WEIGHTS/model.safetensors" \
    "$ALIGN3R_ROOT/third_party/ml-depth-pro/checkpoints/depth_pro.pt" \
    "$ALIGN3R_ROOT/third_party/RAFT/models/Tartan-C-T432x960-M.pth"; do
    if [ ! -e "$required" ]; then
        echo "Missing v2.2 dependency: $required"
        exit 1
    fi
done

ENTRY_COUNT="$($INSPATIO_PYTHON -c \
    'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
    "$INPUT_DIR/new.json")"
if [ "$ENTRY_COUNT" != "2" ]; then
    echo "Error: expected exactly two official examples, found $ENTRY_COUNT"
    exit 1
fi

if find "$OUTPUT_DIR" -type f -name '*-memory_timing_rank0.json' -print -quit \
    2>/dev/null | grep -q .; then
    echo "Error: completed timing artifacts already exist under $OUTPUT_DIR"
    exit 1
fi

ALIGN3R_GPU_NAME="$(nvidia-smi --id="$ALIGN3R_GPU" --query-gpu=name --format=csv,noheader)"
ALIGN3R_DISABLE_CUROPE=false
if [[ "$ALIGN3R_GPU_NAME" == *A100* ]]; then
    ALIGN3R_DISABLE_CUROPE=true
fi

mkdir -p "$WORK_ROOT" "$OUTPUT_DIR" "$ALIGN3R_WORK_DIR"

echo "============================================================"
echo "Dense memory baseline v2.2: online Align3R full-block writes"
echo "  samples:              $ENTRY_COUNT (example0, example1)"
echo "  reference input:      $INPUT_DIR"
echo "  output:               $OUTPUT_DIR"
echo "  trajectory:           $TRAJECTORY"
echo "  memory map:           dense_two_layer"
echo "  memory update:        full_block (all 237 generated RGB frames)"
echo "  depth backend:        align3r"
echo "  InSpatio physical GPU:$INFERENCE_GPU"
echo "  Align3R physical GPU: $ALIGN3R_GPU ($ALIGN3R_GPU_NAME)"
echo "  Shared GPU allowed:   $ALLOW_SHARED_GPU"
echo "  Align3R work dir:     $ALIGN3R_WORK_DIR"
echo "  cuRoPE fallback:      $ALIGN3R_DISABLE_CUROPE"
echo "============================================================"

ALIGN3R_FALLBACK_ARG=()
if [ "$ALIGN3R_DISABLE_CUROPE" = true ]; then
    ALIGN3R_FALLBACK_ARG+=(--memory_align3r_disable_curope)
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID PATH="${INSPATIO_ENV}/bin:${PATH}" \
    bash "$SCRIPT_DIR/run_test_pipeline.sh" \
    --input_dir "$INPUT_DIR" \
    --traj_txt_path "$TRAJECTORY" \
    --skip_step1 \
    --skip_step2 \
    --rotation_only \
    --disable_adaptive_frame \
    --step2_gpus "$INFERENCE_GPU" \
    --step3_gpus "$INFERENCE_GPU" \
    --step3_nproc 1 \
    --master_port "$MASTER_PORT" \
    --output_folder "$OUTPUT_DIR" \
    --historical_memory \
    --memory_map_mode dense_two_layer \
    --memory_update_mode full_block \
    --memory_point_size 1 \
    --memory_depth_backend align3r \
    --memory_align3r_python "$ALIGN3R_PYTHON" \
    --memory_align3r_root "$ALIGN3R_ROOT" \
    --memory_align3r_weights "$ALIGN3R_WEIGHTS" \
    --memory_align3r_work_dir "$ALIGN3R_WORK_DIR" \
    --memory_align3r_gpu "$ALIGN3R_GPU" \
    --memory_align3r_torch_home "$TORCH_HOME" \
    --memory_align3r_xdg_config_home "$XDG_CONFIG_HOME" \
    "${ALIGN3R_FALLBACK_ARG[@]}"

RESULT_DIR="$OUTPUT_DIR/checkpoints/InSpatio-World-1.3B/version_0"
"$INSPATIO_PYTHON" "$SCRIPT_DIR/scripts/audit_dense_two_layer_output.py" \
    --output_dir "$RESULT_DIR" \
    --reference_render_dirs \
        "$INPUT_DIR/new_vggt/example0/render" \
        "$INPUT_DIR/new_vggt/example1/render" \
    --expected_output_frames 237 \
    --expected_reference_frames 237 \
    --expected_point_count 94648320 \
    --expected_depth_backend align3r \
    --yaw_indices 0 79 158 236 \
    --yaw_values 0 45 0 45 | tee "$WORK_ROOT/audit.json"

echo "Completed online Align3R dense-memory v2.2: $WORK_ROOT"
