#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MEMORY_WRITE_MODE="${MEMORY_WRITE_MODE:-latent_keyframe}"
GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-29631}"
WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/output/dense_memory_baseline_v2_1}"
INPUT_DIR="${INPUT_DIR:-${WORK_ROOT}/official_example0_1_24fps_237f}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/${MEMORY_WRITE_MODE}}"
TRAJECTORY="${TRAJECTORY:-${SCRIPT_DIR}/traj/yaw_0_45_0_45_237.txt}"
EXAMPLE0_SOURCE="${EXAMPLE0_SOURCE:-${SCRIPT_DIR}/test/example/cropped_source.mp4}"
EXAMPLE1_SOURCE="${EXAMPLE1_SOURCE:-${SCRIPT_DIR}/test/example2/coffee_martini.mp4}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"

case "$MEMORY_WRITE_MODE" in
    latent_keyframe)
        MEMORY_UPDATE_MODE="latent_keyframe"
        EXPECTED_MAP_FRAMES=60
        ;;
    all_frames)
        MEMORY_UPDATE_MODE="full_block"
        EXPECTED_MAP_FRAMES=237
        ;;
    *)
        echo "Error: MEMORY_WRITE_MODE must be latent_keyframe or all_frames"
        exit 1
        ;;
esac

normalize_official_video() {
    local source_path="$1"
    local output_path="$2"
    "$FFMPEG_BIN" -y -hide_banner -loglevel error \
        -i "$source_path" \
        -vf "fps=24,scale=832:480:flags=lanczos" \
        -frames:v 237 -an -c:v libx264 -crf 18 -pix_fmt yuv420p -r 24 \
        "$output_path"

    local frame_count
    local frame_rate
    frame_count="$($FFPROBE_BIN -v error -select_streams v:0 \
        -show_entries stream=nb_frames -of csv=p=0 "$output_path")"
    frame_rate="$($FFPROBE_BIN -v error -select_streams v:0 \
        -show_entries stream=avg_frame_rate -of csv=p=0 "$output_path")"
    if [ "$frame_count" != "237" ] || [ "$frame_rate" != "24/1" ]; then
        echo "Error: normalized video contract failed for $output_path: frames=$frame_count fps=$frame_rate"
        exit 1
    fi
}

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
normalize_official_video "$EXAMPLE0_SOURCE" "$INPUT_DIR/example0.mp4"
normalize_official_video "$EXAMPLE1_SOURCE" "$INPUT_DIR/example1.mp4"

echo "============================================"
echo "Dense memory baseline v2.1"
echo "  Inputs:            $INPUT_DIR"
echo "  Output:            $OUTPUT_DIR"
echo "  Trajectory:        $TRAJECTORY"
echo "  Memory write mode: $MEMORY_WRITE_MODE"
echo "  CLI update mode:   $MEMORY_UPDATE_MODE"
echo "  Map RGB frames:    $EXPECTED_MAP_FRAMES per video"
echo "  GPU:               $GPU"
echo "============================================"

bash "$SCRIPT_DIR/run_test_pipeline.sh" \
    --input_dir "$INPUT_DIR" \
    --traj_txt_path "$TRAJECTORY" \
    --rotation_only \
    --disable_adaptive_frame \
    --step1_gpus "$GPU" \
    --step2_gpus "$GPU" \
    --step3_gpus "$GPU" \
    --step3_nproc 1 \
    --master_port "$MASTER_PORT" \
    --output_folder "$OUTPUT_DIR" \
    --historical_memory \
    --memory_map_mode dense_two_layer \
    --memory_update_mode "$MEMORY_UPDATE_MODE" \
    --memory_point_size 1
