#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${INSPATIO_PYTHON:-/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python}"
exec "${PYTHON_BIN}" -m mapkv.memory_interface_stage "$@"
