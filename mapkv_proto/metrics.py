from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image


def image_tensor(path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def lpips_distance(
    image_a: str | Path,
    image_b: str | Path,
    *,
    model=None,
    device: torch.device | str = "cpu",
) -> float:
    if model is None:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("Install the 'lpips' package to compute prototype metrics") from error
        model = lpips.LPIPS(net="alex").to(device).eval()
    a = image_tensor(image_a, device=device) * 2.0 - 1.0
    b = image_tensor(image_b, device=device) * 2.0 - 1.0
    with torch.no_grad():
        return float(model(a, b).item())


def compute_revisit_metrics(
    *,
    source_keyframe: str | Path,
    revisit_keyframes: Mapping[str, str | Path],
    boundary_pairs: Mapping[str, tuple[str | Path, str | Path]] | None = None,
    block_latencies: Mapping[str, Mapping[int, float]] | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    try:
        import lpips
    except ImportError:
        lpips = None
    result = {
        "lpips_available": lpips is not None,
        "revisit_lpips": {},
        "boundary_lpips": {},
        "per_target_block_latency_seconds": block_latencies or {},
    }
    if lpips is None:
        return result
    model = lpips.LPIPS(net="alex").to(device).eval()
    for name, path in revisit_keyframes.items():
        result["revisit_lpips"][name] = lpips_distance(
            source_keyframe, path, model=model, device=device
        )
    for name, pair in (boundary_pairs or {}).items():
        result["boundary_lpips"][name] = lpips_distance(
            pair[0], pair[1], model=model, device=device
        )
    return result


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
