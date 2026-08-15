from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mapkv_proto.pose_utils import pose_distance, scale_intrinsics
from mapkv_proto.cut3r.surfel_index import SurfelIndex, pointmap_to_surfels


PINNED_VMEM_COMMIT = "39291e4f272f6b4f270691d930926ab5930f942e"


def verify_vmem_revision(vmem_root: str | Path) -> str:
    vmem_root = Path(vmem_root).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(vmem_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_VMEM_COMMIT:
        raise RuntimeError(
            f"VMem revision {revision} != required {PINNED_VMEM_COMMIT}"
        )
    return revision


def load_cut3r(vmem_root: Path, checkpoint: Path, device: str):
    sys.path.insert(0, str(vmem_root))
    sys.path.insert(0, str(vmem_root / "extern" / "CUT3R"))
    from extern.CUT3R.add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(str(checkpoint))
    from extern.CUT3R.src.dust3r.model import ARCroco3DStereo
    from extern.CUT3R.surfel_inference import run_inference_from_pil

    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    model.eval()
    return model, run_inference_from_pil


def resize_field(array: np.ndarray, grid_hw: tuple[int, int], *, channels_last: bool) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array)).float()
    if channels_last:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=grid_hw, mode="bilinear", align_corners=False)
    if channels_last:
        return tensor[0].permute(1, 2, 0).numpy()
    return tensor[0, 0].numpy()


def camera_matrices(camera_info: dict) -> np.ndarray:
    rotations = np.asarray(camera_info["R"])
    translations = np.asarray(camera_info["t"])
    matrices = np.repeat(np.eye(4)[None], len(rotations), axis=0)
    matrices[:, :3, :3] = rotations
    matrices[:, :3, 3] = translations
    return matrices


def build_index(
    *,
    views_json: str | Path,
    vmem_root: str | Path,
    checkpoint: str | Path,
    output_root: str | Path,
    grid_hw: tuple[int, int] = (30, 52),
    confidence_threshold: float = 1.0,
    radius_scale: float = 0.5,
    merge_normal_cosine: float = 0.6,
    niter: int = 100,
    lr: float = 0.01,
    image_size: int = 512,
    device: str = "cuda",
    maximum_views: int | None = None,
) -> dict:
    vmem_root = Path(vmem_root).resolve()
    revision = verify_vmem_revision(vmem_root)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CUT3R checkpoint not found: {checkpoint}")
    payload_path = Path(views_json).resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    views = payload["views"]
    if maximum_views is not None:
        views = views[:maximum_views]
    if len(views) < 2:
        raise ValueError("CUT3R incremental reconstruction needs anchor plus at least one view")
    model, run_inference_from_pil = load_cut3r(vmem_root, checkpoint, device)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    index = SurfelIndex()
    previous_depths: list[np.ndarray] = []
    processed_view_ids: set[int] = set()
    records = []
    for prefix_length in range(2, len(views) + 1):
        prefix = views[:prefix_length]
        images = [
            Image.open(payload_path.parent / view["image_path"]).convert("RGB")
            for view in prefix
        ]
        fixed_poses = np.asarray([view["c2w_cut3r"] for view in prefix], dtype=np.float32)
        preset_depths = (
            torch.from_numpy(np.stack(previous_depths))
            if previous_depths
            else None
        )
        started = time.perf_counter()
        scene = run_inference_from_pil(
            images,
            model,
            poses=fixed_poses,
            depths=preset_depths,
            lr=lr,
            niter=niter,
            device=device,
            size=image_size,
            visualize=False,
            save_flag=False,
        )
        runtime = time.perf_counter() - started
        pointmaps = torch.cat(scene["point_clouds"], dim=0).float().cpu().numpy()
        depths = torch.cat(scene["depths"], dim=0).float().cpu().numpy()
        confidences = torch.cat(scene["confidences"], dim=0).float().cpu().numpy()
        output_poses = camera_matrices(scene["camera_info"])
        if len(pointmaps) != prefix_length:
            raise RuntimeError(
                f"CUT3R returned {len(pointmaps)} views for prefix length {prefix_length}"
            )
        pose_errors = [
            pose_distance(fixed_poses[index_], output_poses[index_])
            for index_ in range(prefix_length)
        ]
        maximum_pose_error = max(error[0] for error in pose_errors)
        previous_depth_change = 0.0
        if previous_depths:
            previous_depth_change = max(
                float(np.max(np.abs(depths[index_] - previous_depths[index_])))
                for index_ in range(len(previous_depths))
            )

        new_indices = [
            index_ for index_ in range(prefix_length)
            if int(prefix[index_]["view_id"]) not in processed_view_ids
        ]
        merge_records = []
        for view_index in new_indices:
            view = prefix[view_index]
            source_h, source_w = pointmaps[view_index].shape[:2]
            scale = 0.5 * (grid_hw[0] / source_h + grid_hw[1] / source_w)
            focal_value = float(np.asarray(scene["camera_info"]["focal"][view_index]).mean())
            surfels = pointmap_to_surfels(
                pointmap=resize_field(
                    pointmaps[view_index], grid_hw, channels_last=True
                ),
                depth=resize_field(depths[view_index], grid_hw, channels_last=False),
                confidence=resize_field(
                    confidences[view_index], grid_hw, channels_last=False
                ),
                c2w=fixed_poses[view_index],
                focal=focal_value * scale,
                chunk_id=int(view["chunk_id"]),
                confidence_threshold=confidence_threshold,
                radius_scale=radius_scale,
            )
            merge_record = index.merge(
                surfels, normal_cosine=merge_normal_cosine
            )
            merge_record["view_id"] = int(view["view_id"])
            merge_record["chunk_id"] = int(view["chunk_id"])
            merge_records.append(merge_record)
            processed_view_ids.add(int(view["view_id"]))
        finite_confidence = confidences[np.isfinite(confidences)]
        if finite_confidence.size:
            confidence_quantiles = np.quantile(
                finite_confidence, [0.0, 0.25, 0.5, 0.75, 1.0]
            ).tolist()
        else:
            confidence_quantiles = []
        records.append(
            {
                "prefix_length": prefix_length,
                "new_chunk": int(prefix[-1]["chunk_id"]),
                "runtime_seconds": runtime,
                "maximum_input_output_pose_error": maximum_pose_error,
                "maximum_previous_depth_change": previous_depth_change,
                "confidence_quantiles": confidence_quantiles,
                "merge": merge_records,
                "surfel_count": len(index.surfels),
            }
        )
        previous_depths = [depth.copy() for depth in depths]

    anchor = views[0]
    anchor_source_hw = tuple(anchor["image_hw"])
    anchor_intrinsic = scale_intrinsics(
        np.asarray(anchor["intrinsics"]),
        source_hw=anchor_source_hw,
        target_hw=grid_hw,
    )
    anchor_render = index.render(
        c2w=np.asarray(anchor["c2w_cut3r"]),
        intrinsic=anchor_intrinsic,
        image_hw=grid_hw,
        maximum_created_chunk=-1,
    )
    anchor_coverage = float(np.mean(anchor_render["surfel_id"] >= 0))
    positive_depth_fraction = float(np.mean(anchor_render["depth"] > 0))
    if anchor_coverage <= 0 or positive_depth_fraction <= 0:
        raise RuntimeError(
            "Anchor self-reprojection is empty; check Tcw/c2w and CUT3R Y/Z convention"
        )
    index.save(
        output_root / "surfel_index.npz",
        output_root / "surfel_index.ply",
    )
    metadata = {
        "version": 1,
        "vmem_commit": revision,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cut3r_checkpoint": str(checkpoint),
        "views_json": str(payload_path),
        "grid_hw": list(grid_hw),
        "niter": niter,
        "lr": lr,
        "confidence_threshold": confidence_threshold,
        "radius_scale": radius_scale,
        "merge_normal_cosine": merge_normal_cosine,
        "surfel_count": len(index.surfels),
        "anchor_self_reprojection": {
            "coverage_fraction": anchor_coverage,
            "positive_depth_fraction": positive_depth_fraction,
        },
        "increments": records,
    }
    (output_root / "surfel_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a causal CUT3R surfel-to-chunk index")
    parser.add_argument("--views_json", required=True)
    parser.add_argument("--vmem_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--grid_height", type=int, default=30)
    parser.add_argument("--grid_width", type=int, default=52)
    parser.add_argument("--confidence_threshold", type=float, default=1.0)
    parser.add_argument("--radius_scale", type=float, default=0.5)
    parser.add_argument("--merge_normal_cosine", type=float, default=0.6)
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum_views", type=int)
    args = parser.parse_args()
    build_index(
        views_json=args.views_json,
        vmem_root=args.vmem_root,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        grid_hw=(args.grid_height, args.grid_width),
        confidence_threshold=args.confidence_threshold,
        radius_scale=args.radius_scale,
        merge_normal_cosine=args.merge_normal_cosine,
        niter=args.niter,
        lr=args.lr,
        image_size=args.image_size,
        device=args.device,
        maximum_views=args.maximum_views,
    )


if __name__ == "__main__":
    main()
