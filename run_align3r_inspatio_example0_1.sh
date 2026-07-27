#!/bin/bash
set -euo pipefail

# End-to-end InSpatio-World baseline with Align3R replacing DA3.
# The official example0/example1 videos are normalized to 24 fps / 237 frames,
# reconstructed with every consecutive frame, adapted to InSpatio's geometry
# interface, rendered along 0->45->0->45 yaw, and passed through STAR/JDMD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="${SERVER_ROOT:-/mnt/16T2/daixiangting}"
WORK_ROOT="${WORK_ROOT:-${SERVER_ROOT}/tmp/align3r_inspatio_example0_1_237f_20260724}"
INPUT_DIR="${INPUT_DIR:-${WORK_ROOT}/official_example0_1_24fps_237f}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/prediction}"
ALIGN3R_INPUT_ROOT="${ALIGN3R_INPUT_ROOT:-${WORK_ROOT}/align3r_input}"
ALIGN3R_OUTPUT_ROOT="${ALIGN3R_OUTPUT_ROOT:-${WORK_ROOT}/align3r_raw}"

ALIGN3R_ROOT="${ALIGN3R_ROOT:-${SERVER_ROOT}/Align3R}"
ALIGN3R_ENV="${ALIGN3R_ENV:-${SERVER_ROOT}/conda_envs/align3r}"
INSPATIO_ENV="${INSPATIO_ENV:-${SERVER_ROOT}/conda_envs/inspatio}"
ALIGN3R_WEIGHTS="${ALIGN3R_WEIGHTS:-${ALIGN3R_ROOT}/checkpoints/align3r_depthpro}"
TORCH_HOME="${TORCH_HOME:-${SERVER_ROOT}/torch_cache}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${SERVER_ROOT}/xdg_config}"

GPU="${GPU:-1}"
MASTER_PORT="${MASTER_PORT:-29645}"
FRAME_COUNT=237
TRAJECTORY="${TRAJECTORY:-${SCRIPT_DIR}/traj/yaw_0_45_0_45_237.txt}"
EXAMPLE0_SOURCE="${EXAMPLE0_SOURCE:-${SCRIPT_DIR}/test/example/cropped_source.mp4}"
EXAMPLE1_SOURCE="${EXAMPLE1_SOURCE:-${SCRIPT_DIR}/test/example2/coffee_martini.mp4}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"

INSPATIO_PYTHON="${INSPATIO_ENV}/bin/python"
ALIGN3R_PYTHON="${ALIGN3R_ENV}/bin/python"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n "$((GPU + 1))p")"
if [[ "$GPU_NAME" == *A100* ]]; then
    ALIGN3R_DISABLE_CUROPE="${ALIGN3R_DISABLE_CUROPE:-1}"
else
    ALIGN3R_DISABLE_CUROPE="${ALIGN3R_DISABLE_CUROPE:-0}"
fi

for required in \
    "$INSPATIO_PYTHON" \
    "$ALIGN3R_PYTHON" \
    "$TRAJECTORY" \
    "$EXAMPLE0_SOURCE" \
    "$EXAMPLE1_SOURCE" \
    "$ALIGN3R_WEIGHTS/model.safetensors" \
    "$ALIGN3R_ROOT/third_party/ml-depth-pro/checkpoints/depth_pro.pt" \
    "$ALIGN3R_ROOT/third_party/RAFT/models/Tartan-C-T432x960-M.pth"; do
    if [ ! -e "$required" ]; then
        echo "Missing required dependency: $required"
        exit 1
    fi
done

normalize_video() {
    local source_path="$1"
    local output_path="$2"
    "$FFMPEG_BIN" -y -hide_banner -loglevel error \
        -i "$source_path" \
        -vf "fps=24,scale=832:480:flags=lanczos" \
        -frames:v "$FRAME_COUNT" -an -c:v libx264 -crf 18 \
        -pix_fmt yuv420p -r 24 "$output_path"

    local frame_count
    local frame_rate
    frame_count="$($FFPROBE_BIN -v error -select_streams v:0 \
        -show_entries stream=nb_frames -of csv=p=0 "$output_path")"
    frame_rate="$($FFPROBE_BIN -v error -select_streams v:0 \
        -show_entries stream=avg_frame_rate -of csv=p=0 "$output_path")"
    if [ "$frame_count" != "$FRAME_COUNT" ] || [ "$frame_rate" != "24/1" ]; then
        echo "Video contract failed: $output_path frames=$frame_count fps=$frame_rate"
        exit 1
    fi
}

run_align3r() {
    local example_name="$1"
    local video_path="$2"
    local frame_dir="${ALIGN3R_INPUT_ROOT}/${example_name}"
    local align_dir="${ALIGN3R_OUTPUT_ROOT}/${example_name}"
    local depth_count
    local rgb_count

    depth_count=0
    rgb_count=0
    if [ -d "$align_dir" ]; then
        depth_count="$(find "$align_dir" -maxdepth 1 -type f \
            -name 'frame_[0-9][0-9][0-9][0-9].npy' | wc -l | tr -d ' ')"
        rgb_count="$(find "$align_dir" -maxdepth 1 -type f \
            -name 'frame_[0-9][0-9][0-9][0-9]_rgb.png' | wc -l | tr -d ' ')"
    fi
    if [ "$depth_count" -eq "$FRAME_COUNT" ] \
        && [ "$rgb_count" -eq "$FRAME_COUNT" ] \
        && [ -f "$align_dir/pred_traj.txt" ] \
        && [ -f "$align_dir/pred_intrinsics.txt" ]; then
        echo "Reusing complete Align3R reconstruction: $align_dir"
    else
        rm -rf "$frame_dir" "$align_dir"
        mkdir -p "$frame_dir" "$ALIGN3R_OUTPUT_ROOT"
        "$FFMPEG_BIN" -y -hide_banner -loglevel error \
            -i "$video_path" -frames:v "$FRAME_COUNT" -start_number 0 \
            "$frame_dir/frame_%04d.png"

        (
            cd "$ALIGN3R_ROOT"
            CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
                TORCH_HOME="$TORCH_HOME" XDG_CONFIG_HOME="$XDG_CONFIG_HOME" \
                ALIGN3R_DISABLE_CUROPE="$ALIGN3R_DISABLE_CUROPE" \
                PYTHONPATH="$ALIGN3R_ROOT" \
                "$ALIGN3R_PYTHON" tool/demo.py \
                --input_dir "$frame_dir" \
                --output_dir "$ALIGN3R_OUTPUT_ROOT" \
                --seq_name "$example_name" \
                --interval "$FRAME_COUNT" \
                --mode eval_pose_h \
                --weights "$ALIGN3R_WEIGHTS" \
                --device cuda \
                --silent
        )
    fi

    "$INSPATIO_PYTHON" "$SCRIPT_DIR/scripts/convert_align3r_to_inspatio.py" \
        --align3r_dir "$align_dir" \
        --output_dir "$INPUT_DIR/new_vggt/${example_name}_da3_tmp" \
        --expected_frames "$FRAME_COUNT"

    "$INSPATIO_PYTHON" "$SCRIPT_DIR/scripts/convert_da3_to_pi3.py" \
        --da3_dir "$INPUT_DIR/new_vggt/${example_name}_da3_tmp" \
        --output_dir "$INPUT_DIR/new_vggt/${example_name}" \
        --video_path "$video_path"
}

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$ALIGN3R_INPUT_ROOT" "$ALIGN3R_OUTPUT_ROOT"
normalize_video "$EXAMPLE0_SOURCE" "$INPUT_DIR/example0.mp4"
normalize_video "$EXAMPLE1_SOURCE" "$INPUT_DIR/example1.mp4"

echo "============================================================"
echo "Align3R -> InSpatio-World full inference"
echo "  input:       $INPUT_DIR"
echo "  output:      $OUTPUT_DIR"
echo "  trajectory:  $TRAJECTORY"
echo "  frames:      $FRAME_COUNT continuous frames per video"
echo "  GPU:         physical device $GPU ($GPU_NAME, PCI_BUS_ID ordering)"
echo "  PyTorch RoPE fallback: $ALIGN3R_DISABLE_CUROPE"
echo "============================================================"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    "$INSPATIO_PYTHON" "$SCRIPT_DIR/scripts/gen_json.py" \
    --root_dir "$INPUT_DIR" \
    --model_path "$SCRIPT_DIR/checkpoints/Florence-2-large"

run_align3r example0 "$INPUT_DIR/example0.mp4"
run_align3r example1 "$INPUT_DIR/example1.mp4"

PATH="${INSPATIO_ENV}/bin:${PATH}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    bash "$SCRIPT_DIR/run_test_pipeline.sh" \
    --input_dir "$INPUT_DIR" \
    --traj_txt_path "$TRAJECTORY" \
    --skip_step1 \
    --skip_step2 \
    --rotation_only \
    --disable_adaptive_frame \
    --render_backend warper \
    --step2_gpus "$GPU" \
    --step3_gpus "$GPU" \
    --step3_nproc 1 \
    --master_port "$MASTER_PORT" \
    --output_folder "$OUTPUT_DIR"

echo "Completed Align3R-backed InSpatio inference: $WORK_ROOT"
