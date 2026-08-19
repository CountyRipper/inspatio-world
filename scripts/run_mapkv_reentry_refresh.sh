#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MAPKV_INSPATIO_PYTHON:-/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m mapkv.reentry_refresh_stage "$@"
