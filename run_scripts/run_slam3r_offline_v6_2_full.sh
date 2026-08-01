#!/usr/bin/env bash
set -euo pipefail

# Stable public entrypoint. The gate is diagnostic only; the robust Sim(3)
# estimator, its inlier threshold, and all downstream geometry are unchanged.
MAX_SIM3_VALIDATION_P90="${MAX_SIM3_VALIDATION_P90:-0.50}"
export MAX_SIM3_VALIDATION_P90
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_slam3r_offline_v6_2_end2end.sh"
