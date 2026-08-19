#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${INSPATIO_PYTHON:-/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" -m mapkv.source_protected_stage "$@"
