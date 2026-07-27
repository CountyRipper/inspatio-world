#!/usr/bin/env python3
"""Long-lived Align3R worker used by online dense historical memory."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


READY_PREFIX = "ALIGN3R_WORKER_READY\t"
RESULT_PREFIX = "ALIGN3R_WORKER_RESULT\t"
ERROR_PREFIX = "ALIGN3R_WORKER_ERROR\t"


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def is_retryable_pose_failure(error: BaseException) -> bool:
    """Return whether Align3R failed because pose optimization was non-finite."""
    message = str(error)
    return (
        ("linalg.inv" in message and "singular" in message)
        or "SVD did not converge" in message
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--align3r-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    align3r_root = args.align3r_root.resolve()
    os.chdir(align3r_root)
    sys.path.insert(0, str(align3r_root))
    import depth_pro
    import tool.demo as demo
    from dust3r.model import AsymmetricCroCo3DStereo

    model = AsymmetricCroCo3DStereo.from_pretrained(str(args.weights)).to(args.device)
    model.eval()
    depth_model, depth_transform = depth_pro.create_model_and_transforms(
        device=args.device
    )
    depth_model.eval()
    demo.generate_monocular_depth_maps = lambda *_args, **_kwargs: None
    demo_args = SimpleNamespace(weights=str(args.weights), output_dir=None)
    emit(READY_PREFIX, {"device": args.device, "weights": str(args.weights)})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            if request.get("command") != "estimate":
                raise ValueError(f"Unknown worker command: {request}")
            input_dir = Path(request["input_dir"])
            output_dir = Path(request["output_dir"])
            sequence_name = request["sequence_name"]
            frame_count = int(request["frame_count"])
            input_files = sorted(input_dir.glob("*.png"))
            if len(input_files) != frame_count:
                raise ValueError(
                    f"Expected {frame_count} input frames, found {len(input_files)}"
                )

            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for image_path in input_files:
                prior_path = image_path.with_name(
                    image_path.stem + "_pred_depth_depthpro.npz"
                )
                if prior_path.is_file():
                    continue
                image, _, focal_px = depth_pro.load_rgb(str(image_path))
                image = depth_transform(image)
                with torch.inference_mode():
                    prediction = depth_model.infer(image, f_px=focal_px)
                np.savez_compressed(
                    prior_path,
                    depth=prediction["depth"].detach().cpu().numpy(),
                    focallength_px=prediction["focallength_px"].detach().cpu().numpy(),
                )
            retry_reasons = []
            attempt_configs = (
                (0.01, 0.01, "default"),
                (0.0, 0.01, "no_temporal"),
                (0.0, 0.0, "geometry_only"),
            )
            for attempt, (
                temporal_smoothing_weight,
                flow_loss_weight,
                retry_mode,
            ) in enumerate(attempt_configs):
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)
                demo_args.output_dir = str(output_dir)
                print(
                    f"Align3R optimizing {frame_count} consecutive frames "
                    f"(attempt {attempt + 1}, mode={retry_mode}, "
                    f"temporal_smoothing_weight={temporal_smoothing_weight}, "
                    f"flow_loss_weight={flow_loss_weight})",
                    flush=True,
                )
                try:
                    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                        scene = demo.get_reconstructed_scene_hierachical(
                            demo_args,
                            str(output_dir),
                            model,
                            args.device,
                            True,
                            512,
                            filelist=[str(path) for path in input_files],
                            schedule="linear",
                            niter=300,
                            min_conf_thr=1.1,
                            as_pointcloud=True,
                            mask_sky=False,
                            clean_depth=True,
                            transparent_cams=False,
                            cam_size=0.05,
                            show_cam=True,
                            scenegraph_type="swinstride",
                            winsize=5,
                            refid=0,
                            seq_name=sequence_name,
                            new_model_weights=str(args.weights),
                            temporal_smoothing_weight=temporal_smoothing_weight,
                            translation_weight="1.0",
                            shared_focal=True,
                            flow_loss_weight=flow_loss_weight,
                            flow_loss_start_iter=0.1,
                            flow_loss_threshold=25,
                            use_gt_mask=False,
                            fps=0,
                            interval=frame_count,
                            depth_prior_name="depthpro",
                        )
                    break
                except BaseException as error:
                    if (
                        attempt == len(attempt_configs) - 1
                        or not is_retryable_pose_failure(error)
                    ):
                        raise
                    retry_reasons.append(repr(error))
                    torch.cuda.empty_cache()
                    print(
                        "Align3R pose optimization became non-finite; retrying "
                        f"this block with mode={attempt_configs[attempt + 1][2]}",
                        flush=True,
                    )
            del scene
            elapsed = time.perf_counter() - start
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            emit(RESULT_PREFIX, {
                "elapsed_seconds": elapsed,
                "frame_count": frame_count,
                "peak_memory_gb": peak_memory_gb,
                "reconstruction_dir": str(output_dir / sequence_name),
                "retry_count": len(retry_reasons),
                "retry_reason": retry_reasons[-1] if retry_reasons else None,
                "retry_reasons": retry_reasons,
                "retry_mode": retry_mode,
                "temporal_smoothing_weight": temporal_smoothing_weight,
                "flow_loss_weight": flow_loss_weight,
            })
        except BaseException as error:
            emit(ERROR_PREFIX, {
                "error": repr(error),
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    main()
