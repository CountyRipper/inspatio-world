#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data4/daixiangting
PROJECT="$ROOT/inspatio-world"
ENV_PATH="$ROOT/conda_envs/inspatio"
GPU_INDEX=${GPU_INDEX:-2}

export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="$PROJECT/server15/cache/tmp"
export HF_HOME="$PROJECT/server15/cache/huggingface"
export XDG_CACHE_HOME="$PROJECT/server15/cache"

cd "$PROJECT"
MASTER_PORT=$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)

exec bash "$PROJECT/run_scripts/run_test_pipeline.sh" \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --disable_adaptive_frame \
  --step1_gpus "$GPU_INDEX" \
  --step2_gpus "$GPU_INDEX" \
  --step3_gpus "$GPU_INDEX" \
  --step3_nproc 1 \
  --master_port "$MASTER_PORT"
