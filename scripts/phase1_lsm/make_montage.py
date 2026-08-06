#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from utils.wan_wrapper import WanVAEWrapper


COLUMNS = (
    ("A", "A"),
    ("no_memory_Aprime", "no-memory A'"),
    ("direct_memory_Aprime", "direct-memory A'"),
    ("projected_memory_Aprime", "LSM-projected A'"),
    ("wrong_memory_Aprime", "wrong-memory A'"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-outputs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-key", choices=("direct", "projected"), default="direct")
    parser.add_argument(
        "--wan-root",
        default="/data4/daixiangting/inspatio-world/checkpoints/Wan2.1-T2V-1.3B",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = load_file(args.training_outputs, device="cpu")
    device = torch.device("cuda")
    vae = WanVAEWrapper(args.wan_root).to(device=device, dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)
    decoded = {}
    with torch.inference_mode():
        for key, _ in COLUMNS:
            video = vae.decode_to_pixel(tensors[key].to(device), use_cache=False)
            decoded[key] = (video[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
            vae.model.clear_cache()

    frame_count = decoded["A"].shape[0]
    frame_indices = np.linspace(0, frame_count - 1, 3).round().astype(int).tolist()
    cell_width, cell_height = 416, 240
    header_height = 28
    canvas = Image.new(
        "RGB",
        (cell_width * len(COLUMNS), header_height + cell_height * 3),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, (key, label) in enumerate(COLUMNS):
        draw.text((column * cell_width + 6, 7), label, fill="black")
        for row, frame_index in enumerate(frame_indices):
            frame = decoded[key][frame_index].permute(1, 2, 0).numpy()
            image = Image.fromarray((frame * 255).round().astype(np.uint8))
            image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            canvas.paste(image, (column * cell_width, header_height + row * cell_height))
    montage_path = output_dir / "montage.png"
    canvas.save(montage_path)

    target = decoded["A"]
    metrics = {
        "decoded_frames_per_block": frame_count,
        "montage_frame_indices": frame_indices,
        "pixel_l1_to_A": {
            key: float((decoded[key] - target).abs().mean())
            for key, _ in COLUMNS[1:]
        },
    }
    gate_key = f"{args.gate_key}_memory_Aprime"
    metrics["visual_gate_key"] = gate_key
    metrics["memory_closer_than_no_memory"] = (
        metrics["pixel_l1_to_A"][gate_key]
        < metrics["pixel_l1_to_A"]["no_memory_Aprime"]
    )
    (output_dir / "montage_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    if not metrics["memory_closer_than_no_memory"]:
        raise RuntimeError(f"{gate_key} is not visually closer to A than no-memory A'")


if __name__ == "__main__":
    main()
