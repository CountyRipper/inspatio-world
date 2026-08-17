#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSPATIO_PYTHON="${INSPATIO_PYTHON:-/mnt/16T2/daixiangting/conda_envs/inspatio/bin/python}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${INSPATIO_PYTHON}" -m mapkv.fast_pipeline \
  --inspatio-python "${INSPATIO_PYTHON}" "$@"
