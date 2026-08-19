from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .continuous_cavr_evaluation import _video_transition
from .evaluation import _image, _masked_l1
from .locality_evaluation import _block, _intrinsics, _rotation_warp, _save_mask, _save_rgb


METHOD_ROOTS = {
    "baseline": "baseline",
    "episode_wre": "generation/episode_wre",
    "masked_hard_x0": "generation/masked_hard_x0",
    "dual_branch_recent": "generation/dual_branch_recent",
    "memory_render": "generation/memory_render",
    "latent_anchor012": "generation/latent_anchor012",
    "latent_anchor_all4": "generation/latent_anchor_all4",
}

METHOD_LABELS_ZH = {
    "baseline": "M0 原始 Baseline",
    "episode_wre": "M1 当前 Episode Continuous WRE",
    "masked_hard_x0": "M2 匹配掩码 Hard X0 上界",
    "dual_branch_recent": "M3 双分支一致 Recent Guidance",
    "memory_render": "M4 原生 Render Memory",
    "latent_anchor012": "M5 Noise-consistent Latent Anchor 012",
    "latent_anchor_all4": "M6 Noise-consistent Latent Anchor All4",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_map(case: Path) -> dict[str, dict]:
    return {
        item["name"]: item
        for item in _json(case / "phase_labels.json")["phases"]
    }


def _method_image(root: Path, method: str, chunk: int) -> np.ndarray:
    return _image(
        root
        / METHOD_ROOTS[method]
        / "keyframes"
        / f"chunk_{int(chunk):04d}.png"
    )


def _resize_mask(value: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        value.astype(np.float32),
        (int(hw[1]), int(hw[0])),
        interpolation=cv2.INTER_NEAREST,
    )


def _identity_regions(
    mask: np.ndarray,
    warped_anchor: np.ndarray,
    *,
    maximum: int = 2,
) -> list[dict]:
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    gray = cv2.cvtColor(
        (warped_anchor * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    edges = (cv2.Canny(gray, 60, 140) > 0).astype(np.float32)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area < 64 or width < 8 or height < 8:
            continue
        component = labels == label
        edge_density = float(edges[component].mean()) if component.any() else 0.0
        score = float(area) * (0.25 + edge_density)
        padding = max(6, int(round(0.04 * max(width, height))))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(mask.shape[1], x + width + padding)
        y1 = min(mask.shape[0], y + height + padding)
        candidates.append(
            {
                "component": int(label),
                "bbox_xyxy": [x0, y0, x1, y1],
                "area_pixels": area,
                "edge_density": edge_density,
                "selection_score": score,
            }
        )
    candidates.sort(key=lambda item: (-item["selection_score"], item["component"]))
    if not candidates:
        ys, xs = np.nonzero(binary)
        if not len(xs):
            raise RuntimeError("M_need has no identity evaluation pixels")
        candidates = [
            {
                "component": 0,
                "bbox_xyxy": [
                    int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
                ],
                "area_pixels": int(len(xs)),
                "edge_density": float(edges[binary > 0].mean()),
                "selection_score": float(len(xs)),
            }
        ]
    return candidates[:maximum]


def _crop_sheet(
    path: Path,
    *,
    region: dict,
    images: list[tuple[str, np.ndarray]],
) -> None:
    x0, y0, x1, y1 = region["bbox_xyxy"]
    crops = []
    label_height = 40
    for label, image in images:
        crop = (image[y0:y1, x0:x1].clip(0, 1) * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(crop)
        canvas = Image.new("RGB", (pil.width, pil.height + label_height), "white")
        canvas.paste(pil, (0, label_height))
        ImageDraw.Draw(canvas).text((8, 10), label, fill=(20, 28, 42))
        crops.append(canvas)
    width = max(item.width for item in crops)
    height = max(item.height for item in crops)
    sheet = Image.new("RGB", (width * len(crops), height), (240, 243, 248))
    for index, item in enumerate(crops):
        sheet.paste(item, (index * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=94)


def evaluate_memory_interfaces(
    *,
    run_root: str | Path,
    case_dir: str | Path,
) -> dict:
    root = Path(run_root).resolve()
    case = Path(case_dir).resolve()
    manifest = _json(case / "trajectory_manifest.json")
    phases = _phase_map(case)
    anchor_chunk = int(manifest["source_chunk"])
    target_chunk = int(manifest["target_chunk"])
    available = {
        method: relative
        for method, relative in METHOD_ROOTS.items()
        if (root / relative / "run_metadata.json").exists()
    }
    required = {"baseline", "episode_wre", "masked_hard_x0"}
    if not required.issubset(available):
        raise FileNotFoundError(
            f"Memory-interface methods missing: {sorted(required - set(available))}"
        )

    baseline_mapping = root / "baseline/block_mapping.json"
    anchor = _method_image(root, "baseline", anchor_chunk)
    baseline_target = _method_image(root, "baseline", target_chunk)
    image_hw = anchor.shape[:2]
    anchor_pose = np.asarray(_block(baseline_mapping, anchor_chunk)["c2w"])
    target_pose = np.asarray(_block(baseline_mapping, target_chunk)["c2w"])
    intrinsics = _intrinsics(case / "intrinsics.txt", image_hw)
    warped_anchor, warp_valid, homography = _rotation_warp(
        anchor, anchor_pose, target_pose, intrinsics
    )

    hard_root = root / METHOD_ROOTS["masked_hard_x0"] / "memory_interface"
    need_path = hard_root / f"block_{target_chunk:04d}" / "M_need.pt"
    if not need_path.exists():
        candidates = sorted(hard_root.glob("block_*/M_need.pt"))
        if not candidates:
            raise FileNotFoundError("MaskedHardX0 did not save M_need")
        need_path = candidates[len(candidates) // 2]
    need = torch.load(need_path, map_location="cpu", weights_only=True).float()
    if need.ndim == 5:
        need = need[0, need.shape[1] // 2, 0]
    else:
        need = need[0, need.shape[1] // 2]
    revisit = _resize_mask(need.numpy(), image_hw) * (warp_valid > 0).astype(np.float32)
    reference_valid = np.asarray(
        Image.open(
            root
            / "baseline/masks"
            / f"chunk_{target_chunk:04d}_reference_valid.png"
        ).convert("L"),
        dtype=np.float32,
    ) / 255.0
    source_region = (reference_valid > 0.5).astype(np.float32)
    right = np.zeros(image_hw, dtype=np.float32)
    right[:, int(round(image_hw[1] * 0.68)) :] = 1.0
    right_revisit = right * revisit
    if float(revisit.mean()) <= 0:
        raise RuntimeError("M_need has no target-image support")

    assets = root / "assets/memory_interface"
    assets.mkdir(parents=True, exist_ok=True)
    _save_rgb(assets / "canonical_b1_anchor.png", anchor)
    _save_rgb(assets / "canonical_b1_warped_to_b2.png", warped_anchor)
    _save_rgb(assets / "baseline_b2.png", baseline_target)
    _save_mask(assets / "M_need_b2.png", revisit)
    _save_mask(assets / "M_source_b2.png", source_region)

    regions = _identity_regions(revisit, warped_anchor)
    (assets / "identity_regions.json").write_text(
        json.dumps(regions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    method_images = {
        method: _method_image(root, method, target_chunk)
        for method in available
    }
    ordered = [
        method
        for method in (
            "baseline",
            "episode_wre",
            "masked_hard_x0",
            "dual_branch_recent",
            "memory_render",
            "latent_anchor012",
            "latent_anchor_all4",
        )
        if method in method_images
    ]
    for index, region in enumerate(regions, start=1):
        _crop_sheet(
            assets / f"identity_region_{index}.jpg",
            region=region,
            images=[("B1 warp", warped_anchor)]
            + [(METHOD_LABELS_ZH[item], method_images[item]) for item in ordered],
        )

    departure_start = int(phases["B1_hold"]["rgb_start"])
    departure_stop = int(phases["B1_to_Leave"]["rgb_stop_exclusive"])
    reentry_start = int(phases["Leave_to_B2"]["rgb_start"])
    reentry_stop = int(phases["B2_hold"]["rgb_stop_exclusive"])
    methods = {}
    metadata = {}
    for method in ordered:
        image = method_images[method]
        video_path = root / METHOD_ROOTS[method] / "pred.mp4"
        departure = _video_transition(
            video_path,
            departure_start,
            window_start=departure_start,
            window_stop=departure_stop,
        )
        reentry = _video_transition(
            video_path,
            reentry_start,
            window_start=reentry_start,
            window_stop=reentry_stop,
        )
        methods[method] = {
            "warped_historical_appearance_l1": _masked_l1(
                warped_anchor, image, revisit
            ),
            "source_region_delta_vs_baseline": (
                0.0
                if method == "baseline"
                else _masked_l1(image, baseline_target, source_region)
            ),
            "right_edge_l1": _masked_l1(warped_anchor, image, right_revisit),
            "first_departure_peak": float(
                departure["reentry_window_peak_l1"]
            ),
            "reentry_peak": float(reentry["reentry_window_peak_l1"]),
            "reentry_mean": float(reentry["reentry_window_mean_l1"]),
            "identity": "UNRATED",
        }
        metadata[method] = _json(
            root / METHOD_ROOTS[method] / "run_metadata.json"
        )

    review_path = root / "identity_review.json"
    review = _json(review_path) if review_path.exists() else {}
    for method, rating in review.get("methods", {}).items():
        if method in methods and rating in {"STRONG", "PARTIAL", "NONE"}:
            methods[method]["identity"] = rating

    hard = methods["masked_hard_x0"]
    baseline = methods["baseline"]
    hard_upper_bound_works = (
        hard["warped_historical_appearance_l1"]
        < baseline["warped_historical_appearance_l1"] * 0.4
    )
    latent012_needs_all4 = False
    if "latent_anchor012" in methods:
        anchor012 = methods["latent_anchor012"]
        latent012_needs_all4 = (
            anchor012["warped_historical_appearance_l1"]
            > hard["warped_historical_appearance_l1"] * 2.5
            and anchor012["identity"] != "STRONG"
        )

    status = "INTERFACE_LADDER_INCOMPLETE"
    strongest = None
    rated = {
        method: value["identity"] for method, value in methods.items()
    }
    if all(
        item in methods
        for item in (
            "dual_branch_recent",
            "memory_render",
            "latent_anchor012",
        )
    ):
        if rated.get("dual_branch_recent") == "STRONG":
            status, strongest = "DUAL_BRANCH_RECENT_VIABLE", "dual_branch_recent"
        elif rated.get("memory_render") == "STRONG":
            status, strongest = "NATIVE_RENDER_MEMORY_VIABLE", "memory_render"
        elif rated.get("latent_anchor012") == "STRONG" or rated.get(
            "latent_anchor_all4"
        ) == "STRONG":
            status = "LATENT_ANCHOR_REQUIRED"
            strongest = (
                "latent_anchor012"
                if rated.get("latent_anchor012") == "STRONG"
                else "latent_anchor_all4"
            )
        elif review:
            status = "TRAINING_FREE_IDENTITY_LIMITED"

    result = {
        "status": status,
        "strongest_viable_interface": strongest,
        "trajectory": {
            "type": manifest.get("trajectory_type"),
            "anchor_chunk": anchor_chunk,
            "target_chunk": target_chunk,
            "reentry_active_chunks": sorted(
                {
                    int(item["target_chunk"])
                    for item in metadata["masked_hard_x0"]["mapkv"]["selections"]
                    if item["status"] == "scheduled_reentry_read"
                }
            ),
        },
        "controls": {
            "same_memory": "canonical chunk-11 RGB-Warp→Wan-VAE latent",
            "same_mask": "M_need generated-only history × current reference-blind",
            "same_noise": True,
            "same_geometry_lifecycle": True,
            "homography_source_to_target": homography.tolist(),
            "revisit_fraction": float(revisit.mean()),
        },
        "identity_regions": regions,
        "methods": methods,
        "decisions": {
            "hard_upper_bound_works": bool(hard_upper_bound_works),
            "latent_anchor012_needs_all4": bool(latent012_needs_all4),
            "identity_review_present": bool(review),
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "status.json").write_text(
        json.dumps(
            {"status": status, "strongest_viable_interface": strongest},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result


__all__ = [
    "METHOD_LABELS_ZH",
    "METHOD_ROOTS",
    "evaluate_memory_interfaces",
]
