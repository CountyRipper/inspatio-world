"""Small native-decode visualization helpers for Teacher v0."""

from pathlib import Path
import subprocess
from typing import Iterable, Tuple

import torch
from PIL import Image, ImageDraw


def tensor_image(frame: torch.Tensor) -> Image.Image:
    value = (
        frame.detach().float().clamp(0, 1).mul(255).round().byte()
        .permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(value)


def block_frame(video: torch.Tensor, block_index: int, block_size: int = 3) -> Image.Image:
    if video.ndim == 5:
        video = video[0]
    latent_middle = block_index * block_size + block_size // 2
    pixel_index = min(4 * latent_middle, video.shape[0] - 1)
    return tensor_image(video[pixel_index])


def labeled(image: Image.Image, label: str, bar_height: int = 34) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + bar_height), "white")
    output.paste(image.convert("RGB"), (0, bar_height))
    ImageDraw.Draw(output).text((8, 9), label, fill="black")
    return output


def write_grid(
    path,
    items: Iterable[Tuple[str, Image.Image]],
    *,
    columns: int = 5,
) -> None:
    tiles = [labeled(image, label) for label, image in items]
    if not tiles:
        raise ValueError("montage requires at least one image")
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def valid_overlay(candidate: Image.Image, valid: torch.Tensor) -> Image.Image:
    mask = valid.detach().float()
    if mask.ndim == 5:
        mask = mask[0]
    if mask.ndim == 4:
        mask = mask[mask.shape[0] // 2, 0]
    mask_image = Image.fromarray(
        mask.mul(255).byte().cpu().numpy(), mode="L"
    ).resize(candidate.size, Image.Resampling.NEAREST)
    red = Image.new("RGB", candidate.size, (255, 40, 40))
    return Image.composite(candidate.convert("RGB"), red, mask_image)


def write_sync_comparison(path, videos) -> None:
    """Write a labeled 3x2 diagnostic grid without changing native branches."""
    if len(videos) != 6:
        raise ValueError("the synchronized comparison expects six branches")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for _, video_path in videos:
        command.extend(("-i", str(video_path)))
    filters = []
    for index, (label, _) in enumerate(videos):
        safe_label = label.replace("'", "")
        filters.append(
            f"[{index}:v]scale=416:240,"
            f"drawtext=text='{safe_label}':x=8:y=8:fontsize=18:"
            "fontcolor=white:box=1:boxcolor=black@0.6"
            f"[v{index}]"
        )
    filters.extend(
        (
            "[v0][v1][v2]hstack=inputs=3[row0]",
            "[v3][v4][v5]hstack=inputs=3[row1]",
            "[row0][row1]vstack=inputs=2[out]",
        )
    )
    command.extend(
        (
            "-filter_complex", ";".join(filters),
            "-map", "[out]", "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p", str(path),
        )
    )
    subprocess.run(command, check=True)
