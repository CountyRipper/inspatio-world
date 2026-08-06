#!/bin/bash
set -e

##############################################################################
# One-stop inference pipeline:
#   Step 1: Generate JSON (Florence-2 caption + depth path)
#   Step 2: Generate depth with DA3 + convert to Pi3 format + render point clouds
#   Step 3: Run v2v model inference
#
# Usage:
#   bash run_scripts/run_test_pipeline.sh \
#     --input_dir ./test_input \
#     --traj_txt_path ./traj/y_left_30.txt \
#     --checkpoint_path ./checkpoints/InSpatio-World/InSpatio-World.safetensors
#
# Arguments:
#   --input_dir           (required) Input video folder containing .mp4 files
#   --traj_txt_path       (required) Trajectory file path, e.g. ./traj/y_left_30.txt
#   --checkpoint_path     (optional) v2v model checkpoint path (.safetensors)
#                         Default: ./checkpoints/InSpatio-World/InSpatio-World.safetensors
#   --config_path         (optional) Config file path, default configs/inference.yaml
#   --da3_model_path      (optional) DA3 model path, default ./checkpoints/DA3
#   --step1_gpus          (optional) GPUs for Step 1, comma-separated for parallel (e.g. 0,1,2,3), default 0
#   --step2_gpus          (optional) GPUs for Step 2, comma-separated for parallel (e.g. 0,1,2,3), default 0
#   --step3_gpus          (optional) GPUs for Step 3, default 0
#   --step3_nproc         (optional) Number of GPUs for Step 3, default 1
#   --florence_model_path (optional) Florence-2 model path (HuggingFace ID or local)
#   --output_folder       (optional) Output folder (default: ./output/<input_dir_name>/<traj>)
#   --master_port         (optional) Master port for torchrun, default 29513
#   --skip_step1          (optional) Skip Step 1
#   --skip_step2          (optional) Skip Step 2
#   --skip_step3          (optional) Skip Step 3
#   --relative_to_source  (optional) Compose trajectory poses relative to initial view
#   --rotation_only       (optional) Only apply rotation, ignore translation (tripod pan/tilt)
#   --disable_adaptive_frame (optional) Disable adaptive frame expansion/subsampling
#   --use_tae             (optional) Use Tiny Auto Encoder (TAE) instead of WanVAE
#   --tae_checkpoint_path (optional) Path to TAE checkpoint file (required when --use_tae is set)
#   --compile_dit         (optional) Apply torch.compile to the DiT model
#   --historical_memory   (optional) Enable training-free historical RGB point memory
#   --memory_depth_backend (optional) da3, align3r, or mapanything (default: da3)
#   --memory_map_mode     (optional) also supports overlap_voxel_v3_2, v4, and v5
#   --memory_depth_device (optional) Logical CUDA device for memory DA3
#   --memory_mapanything_model_path (required for overlap_voxel_v5)
#   --memory_update_mode  (optional) keyframe, latent_keyframe, or full_block (default: keyframe)
#   --memory_voxel_size   (optional) Historical point voxel size (default: 0.02)
#   --memory_voxel_target_pixels (optional) V3.1 median-depth projected spacing (default: 3.0)
#   --memory_max_points   (optional) Maximum historical point count (V4: 3000000; otherwise: 500000)
#   --memory_point_size   (optional) Historical point splat size (V4: 3; otherwise: 1)
#   --memory_geometry_voxel_factor (optional) V4 voxel tolerance multiplier (default: 2.0)
#   --memory_geometry_depth_ratio (optional) V4 relative-depth tolerance (default: 0.03)
#   --memory_anchor_count (optional) V3 DA3 anchor count; only 1 is implemented
#   --memory_single_keyframe_index (required for V3.2) sole RGB keyframe index
#   --disable_memory_diagnostics (optional) Disable historical/fused diagnostic videos
#   --profile_blocks      (optional) Save per-block DiT timing
#   --save_denoised_latents (optional) Save final denoised latent tensor
#   --freeze_repeat       (optional) Repeat a frame N times for time-freeze effect (default: 0, disabled)
#   --freeze_frame        (optional) Frame index to freeze (default: middle frame)
#   --render_backend      (optional) Rendering backend: warper (default, no point-cloud save) or ply
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default arguments
STEP1_GPUS="0"
STEP2_GPUS="0"
STEP3_GPUS="0"
STEP3_NPROC=1
CHECKPOINT_PATH="${SCRIPT_DIR}/checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"
CONFIG_PATH="${SCRIPT_DIR}/configs/inference_1.3b.yaml"
FLORENCE_MODEL_PATH="${SCRIPT_DIR}/checkpoints/Florence-2-large"
DA3_MODEL_PATH="${SCRIPT_DIR}/checkpoints/DA3"
OUTPUT_FOLDER=""
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false
RELATIVE_TO_SOURCE=false
ROTATION_ONLY=false
ADAPTIVE_FRAME=true
MASTER_PORT=29513
FREEZE_REPEAT=0
FREEZE_FRAME=""
USE_TAE=false
TAE_CHECKPOINT_PATH="${SCRIPT_DIR}/checkpoints/taehv/taew2_1.pth"
COMPILE_DIT=false
RENDER_BACKEND="warper"
HISTORICAL_MEMORY=false
MEMORY_MAP_MODE="bounded_voxel"
MEMORY_DEPTH_BACKEND="da3"
MEMORY_DEPTH_DEVICE=""
MEMORY_ALIGN3R_PYTHON=""
MEMORY_MAPANYTHING_MODEL_PATH=""
MEMORY_MAPANYTHING_CONFIDENCE_PERCENTILE="10.0"
MEMORY_MAPANYTHING_MIN_CONSISTENT_RATIO="0.01"
MEMORY_ALIGN3R_ROOT=""
MEMORY_ALIGN3R_WEIGHTS=""
MEMORY_ALIGN3R_WORK_DIR=""
MEMORY_ALIGN3R_GPU=""
MEMORY_ALIGN3R_TORCH_HOME=""
MEMORY_ALIGN3R_XDG_CONFIG_HOME=""
MEMORY_ALIGN3R_DISABLE_CUROPE=false
MEMORY_UPDATE_MODE="keyframe"
MEMORY_VOXEL_SIZE="0.02"
MEMORY_VOXEL_TARGET_PIXELS="3.0"
MEMORY_MAX_POINTS=""
MEMORY_POINT_SIZE=""
MEMORY_ANCHOR_COUNT="1"
MEMORY_SINGLE_KEYFRAME_INDEX=""
MEMORY_GEOMETRY_VOXEL_FACTOR="2.0"
MEMORY_GEOMETRY_DEPTH_RATIO="0.03"
MEMORY_DIAGNOSTICS=true
PROFILE_BLOCKS=false
SAVE_DENOISED_LATENTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --traj_txt_path)
            TRAJ_TXT_PATH="$2"
            shift 2
            ;;
        --step1_gpus)
            STEP1_GPUS="$2"
            shift 2
            ;;
        --step2_gpus)
            STEP2_GPUS="$2"
            shift 2
            ;;
        --step3_gpus)
            STEP3_GPUS="$2"
            shift 2
            ;;
        --step3_nproc)
            STEP3_NPROC="$2"
            shift 2
            ;;
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --florence_model_path)
            FLORENCE_MODEL_PATH="$2"
            shift 2
            ;;
        --da3_model_path)
            DA3_MODEL_PATH="$2"
            shift 2
            ;;
        --output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --skip_step1)
            SKIP_STEP1=true
            shift
            ;;
        --skip_step2)
            SKIP_STEP2=true
            shift
            ;;
        --skip_step3)
            SKIP_STEP3=true
            shift
            ;;
        --relative_to_source)
            RELATIVE_TO_SOURCE=true
            shift
            ;;
        --rotation_only)
            ROTATION_ONLY=true
            shift
            ;;
        --disable_adaptive_frame)
            ADAPTIVE_FRAME=false
            shift
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --freeze_repeat)
            FREEZE_REPEAT="$2"
            shift 2
            ;;
        --freeze_frame)
            FREEZE_FRAME="$2"
            shift 2
            ;;
        --use_tae)
            USE_TAE=true
            shift
            ;;
        --tae_checkpoint_path)
            TAE_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --compile_dit)
            COMPILE_DIT=true
            shift
            ;;
        --historical_memory)
            HISTORICAL_MEMORY=true
            shift
            ;;
        --memory_map_mode)
            MEMORY_MAP_MODE="$2"
            shift 2
            ;;
        --memory_depth_backend)
            MEMORY_DEPTH_BACKEND="$2"
            shift 2
            ;;
        --memory_depth_device)
            MEMORY_DEPTH_DEVICE="$2"
            shift 2
            ;;
        --memory_align3r_python)
            MEMORY_ALIGN3R_PYTHON="$2"
            shift 2
            ;;
        --memory_mapanything_model_path)
            MEMORY_MAPANYTHING_MODEL_PATH="$2"
            shift 2
            ;;
        --memory_mapanything_confidence_percentile)
            MEMORY_MAPANYTHING_CONFIDENCE_PERCENTILE="$2"
            shift 2
            ;;
        --memory_mapanything_min_consistent_ratio)
            MEMORY_MAPANYTHING_MIN_CONSISTENT_RATIO="$2"
            shift 2
            ;;
        --memory_align3r_root)
            MEMORY_ALIGN3R_ROOT="$2"
            shift 2
            ;;
        --memory_align3r_weights)
            MEMORY_ALIGN3R_WEIGHTS="$2"
            shift 2
            ;;
        --memory_align3r_work_dir)
            MEMORY_ALIGN3R_WORK_DIR="$2"
            shift 2
            ;;
        --memory_align3r_gpu)
            MEMORY_ALIGN3R_GPU="$2"
            shift 2
            ;;
        --memory_align3r_torch_home)
            MEMORY_ALIGN3R_TORCH_HOME="$2"
            shift 2
            ;;
        --memory_align3r_xdg_config_home)
            MEMORY_ALIGN3R_XDG_CONFIG_HOME="$2"
            shift 2
            ;;
        --memory_align3r_disable_curope)
            MEMORY_ALIGN3R_DISABLE_CUROPE=true
            shift
            ;;
        --memory_update_mode)
            MEMORY_UPDATE_MODE="$2"
            shift 2
            ;;
        --memory_voxel_size)
            MEMORY_VOXEL_SIZE="$2"
            shift 2
            ;;
        --memory_voxel_target_pixels)
            MEMORY_VOXEL_TARGET_PIXELS="$2"
            shift 2
            ;;
        --memory_max_points)
            MEMORY_MAX_POINTS="$2"
            shift 2
            ;;
        --memory_point_size)
            MEMORY_POINT_SIZE="$2"
            shift 2
            ;;
        --memory_geometry_voxel_factor)
            MEMORY_GEOMETRY_VOXEL_FACTOR="$2"
            shift 2
            ;;
        --memory_geometry_depth_ratio)
            MEMORY_GEOMETRY_DEPTH_RATIO="$2"
            shift 2
            ;;
        --memory_anchor_count)
            MEMORY_ANCHOR_COUNT="$2"
            shift 2
            ;;
        --memory_single_keyframe_index)
            MEMORY_SINGLE_KEYFRAME_INDEX="$2"
            shift 2
            ;;
        --disable_memory_diagnostics)
            MEMORY_DIAGNOSTICS=false
            shift
            ;;
        --profile_blocks)
            PROFILE_BLOCKS=true
            shift
            ;;
        --save_denoised_latents)
            SAVE_DENOISED_LATENTS=true
            shift
            ;;
        --render_backend)
            RENDER_BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input_dir is required"
    exit 1
fi
if [ -z "$TRAJ_TXT_PATH" ]; then
    echo "Error: --traj_txt_path is required"
    exit 1
fi
if [ -z "$MEMORY_MAX_POINTS" ]; then
    if [ "$MEMORY_MAP_MODE" = "overlap_voxel_v4" ] || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v5" ]; then
        MEMORY_MAX_POINTS="3000000"
    else
        MEMORY_MAX_POINTS="500000"
    fi
fi
if [ -z "$MEMORY_POINT_SIZE" ]; then
    if [ "$MEMORY_MAP_MODE" = "overlap_voxel_v4" ] || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v5" ]; then
        MEMORY_POINT_SIZE="3"
    else
        MEMORY_POINT_SIZE="1"
    fi
fi
if [ "$MEMORY_MAP_MODE" != "bounded_voxel" ] \
        && [ "$MEMORY_MAP_MODE" != "dense_two_layer" ] \
        && [ "$MEMORY_MAP_MODE" != "overlap_voxel_v3" ] \
        && [ "$MEMORY_MAP_MODE" != "overlap_voxel_v3_1" ] \
        && [ "$MEMORY_MAP_MODE" != "overlap_voxel_v3_2" ] \
        && [ "$MEMORY_MAP_MODE" != "overlap_voxel_v4" ] \
        && [ "$MEMORY_MAP_MODE" != "overlap_voxel_v5" ]; then
    echo "Error: invalid --memory_map_mode"
    exit 1
fi
if [ "$MEMORY_DEPTH_BACKEND" != "da3" ] && [ "$MEMORY_DEPTH_BACKEND" != "align3r" ] && [ "$MEMORY_DEPTH_BACKEND" != "mapanything" ]; then
    echo "Error: invalid --memory_depth_backend"
    exit 1
fi
if [ "$HISTORICAL_MEMORY" = true ] && [ "$MEMORY_DEPTH_BACKEND" = "align3r" ]; then
    for required_value in \
        "$MEMORY_ALIGN3R_PYTHON" \
        "$MEMORY_ALIGN3R_ROOT" \
        "$MEMORY_ALIGN3R_WEIGHTS" \
        "$MEMORY_ALIGN3R_WORK_DIR" \
        "$MEMORY_ALIGN3R_GPU"; do
        if [ -z "$required_value" ]; then
            echo "Error: Align3R memory backend requires python/root/weights/work_dir/gpu"
            exit 1
        fi
    done
fi
if [ "$MEMORY_MAP_MODE" = "dense_two_layer" ]; then
    if [ "$MEMORY_UPDATE_MODE" != "latent_keyframe" ] && [ "$MEMORY_UPDATE_MODE" != "full_block" ]; then
        echo "Error: dense_two_layer requires --memory_update_mode latent_keyframe or full_block"
        exit 1
    fi
    # Dense reference depth is the visible PLY z-buffer. Legacy runs retain
    # the latest upstream fast-warper default.
    RENDER_BACKEND="ply"
fi
if [ "$MEMORY_MAP_MODE" = "overlap_voxel_v3" ] \
        || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v3_1" ] \
        || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v3_2" ] \
        || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v4" ] \
        || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v5" ]; then
    if [ "$MEMORY_UPDATE_MODE" != "latent_keyframe" ]; then
        echo "Error: overlap-voxel modes require --memory_update_mode latent_keyframe"
        exit 1
    fi
    if [ "$MEMORY_MAP_MODE" = "overlap_voxel_v3_2" ] \
            && { [ -z "$MEMORY_SINGLE_KEYFRAME_INDEX" ] \
            || ! [[ "$MEMORY_SINGLE_KEYFRAME_INDEX" =~ ^[0-9]+$ ]]; }; then
        echo "Error: overlap_voxel_v3_2 requires a non-negative --memory_single_keyframe_index"
        exit 1
    fi
    if [ "$MEMORY_MAP_MODE" = "overlap_voxel_v5" ]; then
        if [ "$MEMORY_DEPTH_BACKEND" != "mapanything" ] || [ -z "$MEMORY_MAPANYTHING_MODEL_PATH" ]; then
            echo "Error: overlap_voxel_v5 requires mapanything and its local model path"
            exit 1
        fi
    elif [ "$MEMORY_DEPTH_BACKEND" != "da3" ]; then
        echo "Error: overlap_voxel_v3/v4 require DA3"
        exit 1
    fi
    if [ "$MEMORY_MAP_MODE" != "overlap_voxel_v4" ] \
            && [ "$MEMORY_MAP_MODE" != "overlap_voxel_v5" ] \
            && [ "$MEMORY_ANCHOR_COUNT" != "1" ]; then
        echo "Error: multi-anchor V3 is reserved but not implemented"
        exit 1
    fi
    if { [ "$MEMORY_MAP_MODE" = "overlap_voxel_v4" ] \
            || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v5" ]; } \
            && [ "$MEMORY_POINT_SIZE" != "3" ]; then
        echo "Error: overlap_voxel_v4/v5 requires --memory_point_size 3"
        exit 1
    fi
    RENDER_BACKEND="ply"
fi
if [ "$RENDER_BACKEND" != "warper" ] && [ "$RENDER_BACKEND" != "ply" ]; then
    echo "Error: --render_backend must be 'warper' or 'ply'"
    exit 1
fi
if [ "$RENDER_BACKEND" = "ply" ]; then
    SAVE_POINT_CLOUD=true
else
    SAVE_POINT_CLOUD=false
fi

INPUT_DIR_NAME=$(basename "$INPUT_DIR")
TRAJ_NAME=$(basename "$TRAJ_TXT_PATH" .txt)
JSON_PATH="${INPUT_DIR}/new.json"
if [ -z "$OUTPUT_FOLDER" ]; then
    OUTPUT_FOLDER="./output/${INPUT_DIR_NAME}/${TRAJ_NAME}"
fi

echo "============================================"
echo "Pipeline Configuration:"
echo "  Input dir:       $INPUT_DIR"
echo "  Traj txt path:   $TRAJ_TXT_PATH"
echo "  JSON path:       $JSON_PATH"
echo "  Output folder:   $OUTPUT_FOLDER"
echo "  Step1 GPUs:      $STEP1_GPUS"
echo "  Step2 GPUs:      $STEP2_GPUS"
echo "  Step3 GPUs:      $STEP3_GPUS (nproc=$STEP3_NPROC)"
echo "  Checkpoint:      $CHECKPOINT_PATH"
echo "  Config:          $CONFIG_PATH"
echo "  DA3 model:       $DA3_MODEL_PATH"
echo "  Florence model:  $FLORENCE_MODEL_PATH"
echo "  Relative to source: $RELATIVE_TO_SOURCE"
echo "  Rotation only:   $ROTATION_ONLY"
echo "  Adaptive frame:  $ADAPTIVE_FRAME"
echo "  Freeze repeat:   $FREEZE_REPEAT"
echo "  Freeze frame:    ${FREEZE_FRAME:-auto (middle)}"
echo "  Render backend:  $RENDER_BACKEND"
echo "  Save point cloud:$SAVE_POINT_CLOUD"
echo "  Use TAE:         $USE_TAE"
echo "  TAE checkpoint:  ${TAE_CHECKPOINT_PATH:-N/A}"
echo "  Compile DiT:     $COMPILE_DIT"
echo "  Historical memory: $HISTORICAL_MEMORY"
echo "  Memory depth backend: $MEMORY_DEPTH_BACKEND"
echo "  Memory map mode: $MEMORY_MAP_MODE"
echo "  Memory depth device: ${MEMORY_DEPTH_DEVICE:-same as DiT}"
echo "  Align3R memory GPU: ${MEMORY_ALIGN3R_GPU:-N/A}"
echo "  Align3R memory work dir: ${MEMORY_ALIGN3R_WORK_DIR:-N/A}"
echo "  Memory update mode: $MEMORY_UPDATE_MODE"
echo "  Memory anchor count: $MEMORY_ANCHOR_COUNT (multi-anchor reserved)"
echo "  Memory single keyframe: ${MEMORY_SINGLE_KEYFRAME_INDEX:-N/A}"
if [ "$MEMORY_MAP_MODE" = "overlap_voxel_v3_1" ] \
        || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v3_2" ] \
        || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v4" ] \
        || [ "$MEMORY_MAP_MODE" = "overlap_voxel_v5" ]; then
    echo "  Memory voxel/max points/splat: adaptive / $MEMORY_MAX_POINTS / $MEMORY_POINT_SIZE"
else
    echo "  Memory voxel/max points/splat: $MEMORY_VOXEL_SIZE / $MEMORY_MAX_POINTS / $MEMORY_POINT_SIZE"
fi
echo "  Memory voxel target pixels: $MEMORY_VOXEL_TARGET_PIXELS"
echo "  V4 geometry voxel/depth tolerance: $MEMORY_GEOMETRY_VOXEL_FACTOR / $MEMORY_GEOMETRY_DEPTH_RATIO"
echo "  Memory diagnostics: $MEMORY_DIAGNOSTICS"
echo "  Profile blocks: $PROFILE_BLOCKS"
echo "  Save denoised latents: $SAVE_DENOISED_LATENTS"
echo "============================================"

##############################################################################
# Step 1: Generate JSON (Florence-2 caption)
##############################################################################
if [ "$SKIP_STEP1" = false ]; then
    echo ""
    echo "========== Step 1: Generating JSON with Florence-2 =========="

    # Parse GPU list
    IFS=',' read -ra STEP1_GPU_ARRAY <<< "$STEP1_GPUS"
    STEP1_NUM_GPUS=${#STEP1_GPU_ARRAY[@]}
    echo "  Using ${STEP1_NUM_GPUS} GPU(s) for Step 1: ${STEP1_GPUS}"

    if [ "$STEP1_NUM_GPUS" -eq 1 ]; then
        # Single GPU: run directly
        CUDA_VISIBLE_DEVICES=${STEP1_GPU_ARRAY[0]} python "$SCRIPT_DIR/scripts/gen_json.py" \
            --root_dir "$INPUT_DIR" \
            --model_path "$FLORENCE_MODEL_PATH"
    else
        # Multi-GPU: launch one worker per GPU, each writes a partial JSON,
        # then merge all partial JSONs into the final new.json
        STEP1_PIDS=()
        STEP1_PARTIAL_JSONS=()
        for (( i=0; i<STEP1_NUM_GPUS; i++ )); do
            PARTIAL_JSON="${INPUT_DIR}/new_partial_${i}.json"
            STEP1_PARTIAL_JSONS+=("$PARTIAL_JSON")
            CUDA_VISIBLE_DEVICES=${STEP1_GPU_ARRAY[$i]} python "$SCRIPT_DIR/scripts/gen_json.py" \
                --root_dir "$INPUT_DIR" \
                --model_path "$FLORENCE_MODEL_PATH" \
                --worker_id "$i" \
                --num_workers "$STEP1_NUM_GPUS" \
                --output_json "$PARTIAL_JSON" &
            STEP1_PIDS+=($!)
        done

        # Wait for all workers and check exit codes
        STEP1_FAIL=false
        for pid in "${STEP1_PIDS[@]}"; do
            if ! wait "$pid"; then
                STEP1_FAIL=true
            fi
        done
        if [ "$STEP1_FAIL" = true ]; then
            echo "Error: Step 1 failed on one or more GPUs"
            exit 1
        fi

        # Merge partial JSONs into final new.json
        python "$SCRIPT_DIR/scripts/merge_partial_jsons.py" \
            --input_dir "$INPUT_DIR" \
            --output_json "$JSON_PATH"
    fi

    echo "Step 1 completed. JSON saved to: $JSON_PATH"
else
    echo ""
    echo "========== Step 1: SKIPPED =========="
fi

##############################################################################
# Step 2: Generate depth with DA3 + convert + render point clouds
##############################################################################
DA3_CLI="${SCRIPT_DIR}/depth/depth_predict_da3_cli.py"
DA3_CONFIG="{\"model_path\":\"${DA3_MODEL_PATH}\",\"fix_resize\":true,\"fix_resize_height\":480,\"fix_resize_width\":832,\"num_frames\":1000,\"save_point_cloud\":${SAVE_POINT_CLOUD}}"
CONVERT_SCRIPT="${SCRIPT_DIR}/scripts/convert_da3_to_pi3.py"
RENDER_SCRIPT="${SCRIPT_DIR}/scripts/render_point_cloud.py"

# Parse GPU list (e.g. "0,1,2,3" -> array)
IFS=',' read -ra GPU_ARRAY <<< "$STEP2_GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

##############################################################################
# Step 2a: DA3 depth estimation + format conversion (skippable)
##############################################################################
if [ "$SKIP_STEP2" = false ]; then
    echo ""
    echo "========== Step 2a: DA3 depth + convert (multi-GPU parallel) =========="
    echo "  Using ${NUM_GPUS} GPU(s): ${STEP2_GPUS}"

    python "$SCRIPT_DIR/scripts/run_da3_parallel.py" \
        --json_path "$JSON_PATH" \
        --gpu_list "$STEP2_GPUS" \
        --da3_cli "$DA3_CLI" \
        --da3_config "$DA3_CONFIG" \
        --convert_script "$CONVERT_SCRIPT"

    echo "Step 2a completed. Depth maps generated."
else
    echo ""
    echo "========== Step 2a: DA3 depth + convert SKIPPED =========="
fi

##############################################################################
# Step 2b: Render point clouds (always runs — depends on trajectory)
# Render uses the trajectory file, so it must re-run when switching
# trajectories even if depth is already computed.
##############################################################################
echo ""
echo "========== Step 2b: Rendering point clouds (multi-GPU parallel) =========="
echo "  Using ${NUM_GPUS} GPU(s): ${STEP2_GPUS}"

python "$SCRIPT_DIR/scripts/run_render_parallel.py" \
    --json_path "$JSON_PATH" \
    --gpu_list "$STEP2_GPUS" \
    --render_script "$RENDER_SCRIPT" \
    --traj_txt_path "$TRAJ_TXT_PATH" \
    --width 832 --height 480 \
    --render_backend "$RENDER_BACKEND" \
    $([ "$RELATIVE_TO_SOURCE" = true ] && echo "--relative_to_source") \
    $([ "$ROTATION_ONLY" = true ] && echo "--rotation_only") \
    $([ "$FREEZE_REPEAT" -gt 0 ] 2>/dev/null && echo "--freeze_repeat $FREEZE_REPEAT") \
    $([ -n "$FREEZE_FRAME" ] && echo "--freeze_frame $FREEZE_FRAME")

echo "Step 2b completed. Point clouds rendered."

##############################################################################
# Step 3: v2v model inference
##############################################################################
if [ "$SKIP_STEP3" = false ]; then
    echo ""
    echo "========== Step 3: Running v2v inference =========="

    # Convert T5 encoder .pth -> .safetensors if needed
    T5_PTH="${SCRIPT_DIR}/checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
    T5_ST="${SCRIPT_DIR}/checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.safetensors"
    if [ ! -f "$T5_ST" ]; then
        if [ ! -f "$T5_PTH" ]; then
            echo "Error: Wan T5 encoder weight not found at: $T5_PTH"
            echo "Please download Wan2.1-T2V-1.3B weights to: ${SCRIPT_DIR}/checkpoints/Wan2.1-T2V-1.3B/"
            exit 1
        fi
        echo "  Converting T5 encoder: .pth -> .safetensors ..."
        python "$SCRIPT_DIR/utils/convert_pth_to_safetensors.py" \
            --input "$T5_PTH" \
            --output "$T5_ST"
        echo "  Conversion done: $T5_ST"
    fi

    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "Error: --checkpoint_path is required for Step 3"
        exit 1
    fi

    cd "$SCRIPT_DIR"

    # Generate a temporary config overriding traj_txt_path
    TMP_CONFIG=$(mktemp /tmp/pipeline_config_XXXXXX.yaml)
    cp "$CONFIG_PATH" "$TMP_CONFIG"
    sed -i "/^[[:space:]]*#/!s|traj_txt_path:.*|traj_txt_path: ${TRAJ_TXT_PATH}|g" "$TMP_CONFIG"
    sed -i "/^[[:space:]]*#/!s|relative_to_source:.*|relative_to_source: ${RELATIVE_TO_SOURCE}|g" "$TMP_CONFIG"
    sed -i "/^[[:space:]]*#/!s|rotation_only:.*|rotation_only: ${ROTATION_ONLY}|g" "$TMP_CONFIG"
    sed -i "/^[[:space:]]*#/!s|adaptive_frame:.*|adaptive_frame: ${ADAPTIVE_FRAME}|g" "$TMP_CONFIG"
    sed -i "/^[[:space:]]*#/!s|freeze_repeat:.*|freeze_repeat: ${FREEZE_REPEAT}|g" "$TMP_CONFIG"
    if [ -n "$FREEZE_FRAME" ]; then
        sed -i "/^[[:space:]]*#/!s|freeze_frame:.*|freeze_frame: ${FREEZE_FRAME}|g" "$TMP_CONFIG"
    fi

    CUDA_VISIBLE_DEVICES=$STEP3_GPUS torchrun \
        --nproc_per_node=$STEP3_NPROC \
        --master_port $MASTER_PORT \
        inference_causal_test.py \
        --config_path "$TMP_CONFIG" \
        --json_path "$JSON_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --output_folder "$OUTPUT_FOLDER" \
        $([ "$USE_TAE" = true ] && echo "--use_tae") \
        $([ -n "$TAE_CHECKPOINT_PATH" ] && echo "--tae_checkpoint_path $TAE_CHECKPOINT_PATH") \
        $([ "$COMPILE_DIT" = true ] && echo "--compile_dit") \
        $([ "$HISTORICAL_MEMORY" = true ] && echo "--historical_memory") \
        --memory_depth_backend "$MEMORY_DEPTH_BACKEND" \
        --memory_map_mode "$MEMORY_MAP_MODE" \
        $([ -n "$MEMORY_DEPTH_DEVICE" ] && echo "--memory_depth_device $MEMORY_DEPTH_DEVICE") \
        --memory_update_mode "$MEMORY_UPDATE_MODE" \
        --memory_da3_model_path "$DA3_MODEL_PATH" \
        $([ -n "$MEMORY_MAPANYTHING_MODEL_PATH" ] && echo "--memory_mapanything_model_path $MEMORY_MAPANYTHING_MODEL_PATH") \
        --memory_mapanything_confidence_percentile "$MEMORY_MAPANYTHING_CONFIDENCE_PERCENTILE" \
        --memory_mapanything_min_consistent_ratio "$MEMORY_MAPANYTHING_MIN_CONSISTENT_RATIO" \
        $([ -n "$MEMORY_ALIGN3R_PYTHON" ] && echo "--memory_align3r_python $MEMORY_ALIGN3R_PYTHON") \
        $([ -n "$MEMORY_ALIGN3R_ROOT" ] && echo "--memory_align3r_root $MEMORY_ALIGN3R_ROOT") \
        $([ -n "$MEMORY_ALIGN3R_WEIGHTS" ] && echo "--memory_align3r_weights $MEMORY_ALIGN3R_WEIGHTS") \
        $([ -n "$MEMORY_ALIGN3R_WORK_DIR" ] && echo "--memory_align3r_work_dir $MEMORY_ALIGN3R_WORK_DIR") \
        $([ -n "$MEMORY_ALIGN3R_GPU" ] && echo "--memory_align3r_gpu $MEMORY_ALIGN3R_GPU") \
        $([ -n "$MEMORY_ALIGN3R_TORCH_HOME" ] && echo "--memory_align3r_torch_home $MEMORY_ALIGN3R_TORCH_HOME") \
        $([ -n "$MEMORY_ALIGN3R_XDG_CONFIG_HOME" ] && echo "--memory_align3r_xdg_config_home $MEMORY_ALIGN3R_XDG_CONFIG_HOME") \
        $([ "$MEMORY_ALIGN3R_DISABLE_CUROPE" = true ] && echo "--memory_align3r_disable_curope") \
        --memory_voxel_size "$MEMORY_VOXEL_SIZE" \
        --memory_voxel_target_pixels "$MEMORY_VOXEL_TARGET_PIXELS" \
        --memory_max_points "$MEMORY_MAX_POINTS" \
        --memory_point_size "$MEMORY_POINT_SIZE" \
        --memory_anchor_count "$MEMORY_ANCHOR_COUNT" \
        $([ -n "$MEMORY_SINGLE_KEYFRAME_INDEX" ] && echo "--memory_single_keyframe_index $MEMORY_SINGLE_KEYFRAME_INDEX") \
        --memory_geometry_voxel_factor "$MEMORY_GEOMETRY_VOXEL_FACTOR" \
        --memory_geometry_depth_ratio "$MEMORY_GEOMETRY_DEPTH_RATIO" \
        $([ "$MEMORY_DIAGNOSTICS" = false ] && echo "--disable_memory_diagnostics") \
        $([ "$PROFILE_BLOCKS" = true ] && echo "--profile_blocks") \
        $([ "$SAVE_DENOISED_LATENTS" = true ] && echo "--save_denoised_latents")

    rm -f "$TMP_CONFIG"

    echo "Step 3 completed. Results saved to: $OUTPUT_FOLDER"
else
    echo ""
    echo "========== Step 3: SKIPPED =========="
fi

echo ""
echo "============================================"
echo "Pipeline finished!"
echo "  JSON:    $JSON_PATH"
echo "  Output:  $OUTPUT_FOLDER"
echo "============================================"
