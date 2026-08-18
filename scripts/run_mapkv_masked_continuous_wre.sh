#!/usr/bin/env bash
set -euo pipefail

STAGE="full"
GPU="0"
SEED="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
exec /mnt/16T2/daixiangting/conda_envs/inspatio/bin/python \
  -m mapkv.continuous_cavr_stage \
  --stage "${STAGE}" \
  --gpu "${GPU}" \
  --seed "${SEED}"
