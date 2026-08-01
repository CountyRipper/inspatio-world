#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data4/daixiangting/inspatio-world}"
CONDA_ROOT="${CONDA_ROOT:-/data3/daixiangting/miniconda3}"
CONDA_ENV="${CONDA_ENV:-/data4/daixiangting/conda_envs/inspatio}"
GPU="${GPU:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data4/daixiangting/inspatio_v4_results}"
TARGET_PIXEL_SPACING="${TARGET_PIXEL_SPACING:-3.0}"
MAX_POINTS="${MAX_POINTS:-3000000}"
GEOMETRY_VOXEL_FACTOR="${GEOMETRY_VOXEL_FACTOR:-2.0}"
GEOMETRY_DEPTH_RATIO="${GEOMETRY_DEPTH_RATIO:-0.03}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

TRAJECTORY_DIR="${OUTPUT_ROOT}/trajectories"
mkdir -p "${TRAJECTORY_DIR}"
python scripts/create_dense_yaw_trajectory.py \
    --frames 247 --output "${TRAJECTORY_DIR}/yaw_0_45_0_45_247.txt"
python scripts/create_dense_yaw_trajectory.py \
    --frames 300 --output "${TRAJECTORY_DIR}/yaw_0_45_0_45_300.txt"

run_case() {
    local case_name="$1"
    local input_dir="$2"
    local trajectory="$3"
    local port="$4"
    local output_dir="${OUTPUT_ROOT}/${case_name}"
    mkdir -p "${output_dir}"

    /usr/bin/time -v -o "${output_dir}/pipeline_resource_time.txt" \
        bash "${PROJECT_ROOT}/run_scripts/run_test_pipeline.sh" \
        --input_dir "${input_dir}" \
        --traj_txt_path "${trajectory}" \
        --output_folder "${output_dir}" \
        --historical_memory \
        --memory_depth_backend da3 \
        --memory_map_mode overlap_voxel_v4 \
        --memory_update_mode latent_keyframe \
        --memory_voxel_target_pixels "${TARGET_PIXEL_SPACING}" \
        --memory_max_points "${MAX_POINTS}" \
        --memory_point_size 3 \
        --memory_geometry_voxel_factor "${GEOMETRY_VOXEL_FACTOR}" \
        --memory_geometry_depth_ratio "${GEOMETRY_DEPTH_RATIO}" \
        --rotation_only \
        --disable_adaptive_frame \
        --render_backend ply \
        --step1_gpus "${GPU}" \
        --step2_gpus "${GPU}" \
        --step3_gpus "${GPU}" \
        --step3_nproc 1 \
        --master_port "${port}" \
        2>&1 | tee "${output_dir}/pipeline.log"

    local result_dir="${output_dir}/checkpoints/InSpatio-World-1.3B/version_0"
    python scripts/summarize_overlap_voxel_v4.py \
        "${result_dir}/0-overlap_voxel_v4-rank0" \
        --csv "${OUTPUT_ROOT}/${case_name}_v4_block_metrics.csv" \
        --summary-json "${OUTPUT_ROOT}/${case_name}_v4_summary.json"
}

run_case \
    "example0_yaw_0_45_0_45" \
    "./test/example" \
    "${TRAJECTORY_DIR}/yaw_0_45_0_45_247.txt" \
    29751
run_case \
    "example1_yaw_0_45_0_45" \
    "./test/example2" \
    "${TRAJECTORY_DIR}/yaw_0_45_0_45_300.txt" \
    29752
