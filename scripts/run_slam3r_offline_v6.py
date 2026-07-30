#!/usr/bin/env python3
"""CUDA-initializing entrypoint for the SLAM3R offline v6 replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_slam3r_offline_v6_impl as _impl  # noqa: E402


# Keep the small public surface used by tests and direct Python callers.
FrozenSimilarity = _impl.FrozenSimilarity
SequentialVideoReader = _impl.SequentialVideoReader


def run(args) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        # PyTorch 2.7.1+cu128 rejects reset_peak_memory_stats(device) before
        # an explicit CUDA context has selected the logical device.
        torch.cuda.set_device(device)
    return _impl.run(args)


def main() -> None:
    args = _impl._build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
