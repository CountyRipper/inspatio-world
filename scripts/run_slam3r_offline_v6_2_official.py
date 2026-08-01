#!/usr/bin/env python3
"""Run official SLAM3R offline reconstruction on the v6_2 PNG keyframes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slam3r-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--i2p-model", required=True)
    parser.add_argument("--l2w-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=100)
    args = parser.parse_args()

    slam3r_root = args.slam3r_root.resolve()
    sys.path.insert(0, str(slam3r_root))
    from slam3r.datasets.wild_seq import Seq_Data
    from slam3r.models import Image2PointsModel, Local2WorldModel
    from slam3r.pipeline.recon_offline_pipeline import scene_recon_pipeline_offline

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official SLAM3R offline v6_2 requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = json.loads((input_dir / "manifest.json").read_text())
    png_count = len(list(input_dir.glob("*.png")))
    if png_count != manifest["keyframe_count"]:
        raise RuntimeError(
            f"Keyframe PNG count {png_count} != manifest {manifest['keyframe_count']}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading I2P from {args.i2p_model}", flush=True)
    i2p_model = Image2PointsModel.from_pretrained(args.i2p_model).to(device).eval()
    print(f"Loading L2W from {args.l2w_model}", flush=True)
    l2w_model = Local2WorldModel.from_pretrained(args.l2w_model).to(device).eval()
    dataset = Seq_Data(
        img_dir=str(input_dir),
        img_size=224,
        silent=False,
        sample_freq=1,
        start_idx=0,
        num_views=-1,
        start_freq=1,
        postfix=".png",
        to_tensor=True,
    )
    pipeline_args = SimpleNamespace(
        device=str(device),
        keyframe_stride=1,
        initial_winsize=5,
        win_r=3,
        conf_thres_i2p=1.5,
        num_scene_frame=10,
        max_num_register=10,
        conf_thres_l2w=12.0,
        num_points_save=2_000_000,
        norm_input=False,
        update_buffer_intv=1,
        buffer_size=args.buffer_size,
        buffer_strategy="reservoir",
        save_all_views=False,
        save_preds=True,
        save_for_eval=False,
        keyframe_adapt_min=1,
        keyframe_adapt_max=20,
        keyframe_adapt_stride=1,
    )
    scene_recon_pipeline_offline(
        i2p_model,
        l2w_model,
        dataset,
        pipeline_args,
        str(output_dir),
    )
    required = (
        "local_pcds.npy",
        "registered_pcds.npy",
        "local_confs.npy",
        "registered_confs.npy",
        "input_imgs.npy",
        "metadata.json",
    )
    missing = [name for name in required if not (output_dir / "preds" / name).is_file()]
    if missing:
        raise RuntimeError(f"Official SLAM3R did not save predictions: {missing}")
    print(json.dumps({
        "status": "complete",
        "keyframes": manifest["keyframe_count"],
        "preds_dir": str(output_dir / "preds"),
    }, indent=2))


if __name__ == "__main__":
    main()
