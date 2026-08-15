from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_gate_overlay(
    image_path: str | Path,
    gate: torch.Tensor | np.ndarray,
    output_path: str | Path,
    *,
    opacity: float = 0.55,
) -> None:
    image = Image.open(image_path).convert("RGB")
    gate = np.asarray(torch.as_tensor(gate).float().cpu())
    while gate.ndim > 2:
        gate = gate.mean(axis=0)
    gate_image = Image.fromarray((np.clip(gate, 0, 1) * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.BILINEAR
    )
    heat = Image.new("RGB", image.size, (255, 255, 255))
    heat.putalpha(gate_image)
    base = image.convert("RGBA")
    heat.putalpha(gate_image.point(lambda value: int(value * opacity)))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, heat).convert("RGB").save(output_path)


def make_contact_sheet(
    images: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    cell_size: tuple[int, int] = (416, 240),
) -> None:
    cells = []
    for label, path in images.items():
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_size[0], cell_size[1] - 24))
        cell = Image.new("RGB", cell_size, "black")
        cell.paste(image, ((cell.width - image.width) // 2, 0))
        ImageDraw.Draw(cell).text((6, cell.height - 20), label, fill="white")
        cells.append(cell)
    sheet = Image.new("RGB", (cell_size[0] * len(cells), cell_size[1]), "black")
    for index, cell in enumerate(cells):
        sheet.paste(cell, (index * cell_size[0], 0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def make_comparison_video(
    videos: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    fps: float | None = None,
    retrieved_chunk_ids: Mapping[str, int] | None = None,
) -> None:
    from torchvision.io import read_video, write_video

    decoded = []
    source_fps = []
    for label, path in videos.items():
        frames, _, info = read_video(str(path), pts_unit="sec")
        decoded.append((label, frames))
        source_fps.append(float(info.get("video_fps", 24.0)))
    length = min(frames.shape[0] for _, frames in decoded)
    height = min(frames.shape[1] for _, frames in decoded)
    width = min(frames.shape[2] for _, frames in decoded)
    output_frames = []
    for frame_index in range(length):
        cells = []
        for label, frames in decoded:
            frame = Image.fromarray(frames[frame_index].numpy()).resize((width, height))
            draw = ImageDraw.Draw(frame)
            suffix = ""
            if retrieved_chunk_ids and label in retrieved_chunk_ids:
                suffix = f" | retrieved chunk {retrieved_chunk_ids[label]}"
            draw.rectangle((0, 0, width, 22), fill="black")
            draw.text((6, 4), f"{label}{suffix}", fill="white")
            cells.append(np.asarray(frame))
        output_frames.append(np.concatenate(cells, axis=1))
    tensor = torch.from_numpy(np.stack(output_frames))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(output_path),
        tensor,
        fps=int(round(fps or min(source_fps))),
        video_codec="h264",
        options={"crf": "18"},
    )
