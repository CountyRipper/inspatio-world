#!/usr/bin/env python3
"""Export one ordered RGB keyframe per Wan latent for SLAM3R v6_2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _crop_metadata(height: int, width: int, size: int = 224) -> dict:
    if width >= height:
        resized_height = size
        resized_width = int(round(size * width / height))
    else:
        resized_height = int(round(size * height / width))
        resized_width = size
    return {
        "source_height": height,
        "source_width": width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "crop_top": (resized_height - size) // 2,
        "crop_left": (resized_width - size) // 2,
        "crop_size": size,
    }


def export_keyframes(
    pred_video: Path,
    output_dir: Path,
    *,
    latents_per_block: int = 3,
) -> dict:
    if latents_per_block <= 0:
        raise ValueError("latents_per_block must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(pred_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {pred_video}")

    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    crop = _crop_metadata(height, width)
    keyframes = []
    frame_index = 0
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            if frame_index % 4 == 0:
                keyframe_index = len(keyframes)
                filename = f"{keyframe_index:06d}.png"
                if not cv2.imwrite(str(output_dir / filename), frame_bgr):
                    raise RuntimeError(f"Failed to write {output_dir / filename}")
                keyframes.append({
                    "keyframe_index": keyframe_index,
                    "block_index": keyframe_index // latents_per_block,
                    "latent_index": keyframe_index,
                    "source_rgb_frame": frame_index,
                    "planned_c2w_index": frame_index,
                    "planned_keyframe_index": keyframe_index,
                    "filename": filename,
                    "crop": crop,
                })
            frame_index += 1
    finally:
        capture.release()

    if frame_index != frame_count:
        frame_count = frame_index
    expected = (frame_count - 1) // 4 + 1
    if len(keyframes) != expected:
        raise AssertionError(f"Exported {len(keyframes)} keyframes, expected {expected}")
    manifest = {
        "schema_version": 1,
        "pred_video": str(pred_video.resolve()),
        "source_frame_count": frame_count,
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "keyframe_rule": "frame_0_then_every_4th_rgb_frame",
        "internal_keyframe_stride": 1,
        "latents_per_block": latents_per_block,
        "keyframe_count": len(keyframes),
        "keyframes": keyframes,
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latents-per-block", type=int, default=3)
    args = parser.parse_args()
    manifest = export_keyframes(
        args.pred_video.resolve(),
        args.output_dir.resolve(),
        latents_per_block=args.latents_per_block,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "keyframe_count": manifest["keyframe_count"],
        "source_frame_count": manifest["source_frame_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
