#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="${SERVER_ROOT:-/mnt/16T2/daixiangting}"
V21_ROOT="${V21_ROOT:-${SERVER_ROOT}/tmp/dense_memory_baseline_v2_1_full_gpu0_20260723}"
WORK_ROOT="${WORK_ROOT:-${SERVER_ROOT}/tmp/align3r_full_frames_example0_1_237f_20260724}"
ALIGN3R_ROOT="${ALIGN3R_ROOT:-${SERVER_ROOT}/Align3R}"
ALIGN3R_ENV="${ALIGN3R_ENV:-${SERVER_ROOT}/conda_envs/align3r}"
INSPATIO_ENV="${INSPATIO_ENV:-${SERVER_ROOT}/conda_envs/inspatio}"
TORCH_HOME="${TORCH_HOME:-${SERVER_ROOT}/torch_cache}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${SERVER_ROOT}/xdg_config}"
ALIGN3R_WEIGHTS="${ALIGN3R_WEIGHTS:-${ALIGN3R_ROOT}/checkpoints/align3r_depthpro}"
GPU="${GPU:-0}"
FRAME_COUNT=237

V21_RESULTS="${V21_ROOT}/latent_keyframe/checkpoints/InSpatio-World-1.3B/version_0"
V21_GEOMETRY="${V21_ROOT}/official_example0_1_24fps_237f/new_vggt"
INSPATIO_PYTHON="${INSPATIO_ENV}/bin/python"
ALIGN3R_PYTHON="${ALIGN3R_ENV}/bin/python"

for required in \
    "$INSPATIO_PYTHON" \
    "$ALIGN3R_PYTHON" \
    "$ALIGN3R_WEIGHTS/model.safetensors" \
    "$ALIGN3R_ROOT/third_party/ml-depth-pro/checkpoints/depth_pro.pt" \
    "$ALIGN3R_ROOT/third_party/RAFT/models/Tartan-C-T432x960-M.pth"; do
    if [ ! -e "$required" ]; then
        echo "Missing full-frame Align3R dependency: $required"
        exit 1
    fi
done

mkdir -p "$WORK_ROOT"
for pair in "0 example0" "1 example1"; do
    read -r video_index example_name <<< "$pair"
    example_root="$WORK_ROOT/$example_name"
    prepared="$example_root/prepared"
    align3r_output="$example_root/align3r_raw"
    result="$example_root/result"
    mkdir -p "$align3r_output/$example_name" "$result"

    "$INSPATIO_PYTHON" "$SCRIPT_DIR/scripts/v2_2_depth_pointcloud.py" prepare \
        --video "$V21_RESULTS/${video_index}-pred_video_rank0.mp4" \
        --target-c2w "$V21_GEOMETRY/$example_name/render/target_c2w.npy" \
        --intrinsic "$V21_GEOMETRY/$example_name/render/intrinsic.npy" \
        --reference-depth "$V21_GEOMETRY/$example_name/render/depth_offline.npy" \
        --output-dir "$prepared" \
        --expected-frames "$FRAME_COUNT" \
        --frame-step 1

    align3r_depth_count="$(find "$align3r_output/$example_name" -maxdepth 1 \
        -type f -name 'frame_[0-9][0-9][0-9][0-9].npy' 2>/dev/null \
        | wc -l | tr -d ' ')"
    if [ "$align3r_depth_count" -ne "$FRAME_COUNT" ]; then
        (
            cd "$ALIGN3R_ROOT"
            CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" TORCH_HOME="$TORCH_HOME" \
                XDG_CONFIG_HOME="$XDG_CONFIG_HOME" \
                PYTHONPATH="$ALIGN3R_ROOT" \
                "$ALIGN3R_PYTHON" tool/demo.py \
                --input_dir "$prepared/frames" \
                --output_dir "$align3r_output" \
                --seq_name "$example_name" \
                --interval "$FRAME_COUNT" \
                --mode eval_pose_h \
                --weights "$ALIGN3R_WEIGHTS" \
                --device cuda \
                --silent
        )
    fi

    "$INSPATIO_PYTHON" "$SCRIPT_DIR/scripts/v2_2_depth_pointcloud.py" \
        build-align3r-full-frames \
        --prepared-dir "$prepared" \
        --align3r-depth-dir "$align3r_output/$example_name" \
        --output-dir "$result"
done

echo "Align3R full-frame example0/example1 completed: $WORK_ROOT"
