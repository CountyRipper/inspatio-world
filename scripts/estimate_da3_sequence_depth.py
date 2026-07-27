#!/usr/bin/env python3
"""Estimate one jointly processed DA3 depth sequence for V2.2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.depth_only_da3 import DA3DepthOnlyEstimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.png"))
    if len(frame_paths) != 60:
        raise RuntimeError(f"Expected 60 V2.2 keyframes, found {len(frame_paths)}")
    frames = []
    for path in frame_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    rgb = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    height, width = frames[0].shape[:2]

    estimator = DA3DepthOnlyEstimator(args.model_path, torch.device(args.device))
    _, depth, _, _ = estimator.estimate_block(
        rgb,
        output_size=(height, width),
        output_device=torch.device("cpu"),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, depth.numpy().astype(np.float32, copy=False))
    manifest = {
        "frame_count": len(frame_paths),
        "output_shape": list(depth.shape),
        "native_shape": list(estimator.last_native_shape),
        "processed_shape": list(estimator.last_processed_shape),
        "intrinsics_shape": list(estimator.last_intrinsics_shape),
        "extrinsics_shape": list(estimator.last_extrinsics_shape),
        "cuda_peak_allocated_gb": estimator.last_peak_memory_gb,
        "model_path": str(Path(args.model_path).resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
