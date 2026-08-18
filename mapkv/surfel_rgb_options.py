from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from mapkv_proto.pose_utils import to_cut3r_c2w

from .surfel_index import SurfelIndex


def _prepare_cut3r_rgb(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Reproduce CUT3R's long-edge resize and centered multiple-of-16 crop."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    long_edge = max(shape)
    scale = long_edge / max(width, height)
    resized = image.resize(
        (int(round(width * scale)), int(round(height * scale))),
        Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC,
    )
    target_h, target_w = shape
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    if cropped.size != (target_w, target_h):
        raise ValueError(
            f"CUT3R RGB crop {cropped.size} does not match {(target_w, target_h)}"
        )
    return np.asarray(cropped, dtype=np.uint8)


def _project_point(
    xyz: np.ndarray, c2w: np.ndarray, intrinsics: np.ndarray
) -> tuple[float, float, float] | None:
    homogeneous = np.append(np.asarray(xyz, dtype=np.float64), 1.0)
    camera = np.linalg.inv(np.asarray(c2w, dtype=np.float64)) @ homogeneous
    if not np.isfinite(camera).all() or camera[2] <= 1e-6:
        return None
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    u = intrinsic[0, 0] * camera[0] / camera[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[1] / camera[2] + intrinsic[1, 2]
    return float(u), float(v), float(camera[2])


def sample_historical_rgb(
    index: SurfelIndex, sequence_path: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Color established surfels from their real first-seen generated view."""
    sequence_path = Path(sequence_path).resolve()
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    frames = {int(item["chunk_id"]): item for item in sequence["frames"]}
    image_cache: dict[int, np.ndarray] = {}
    colors = np.zeros((len(index.cells), 3), dtype=np.uint8)
    valid = np.zeros(len(index.cells), dtype=bool)
    color_chunks = np.full(len(index.cells), -1, dtype=np.int32)
    for cell_index, cell in enumerate(index.cells):
        candidates = [int(cell.first_seen_chunk)] + [
            int(chunk)
            for chunk in cell.observing_chunks
            if int(chunk) != int(cell.first_seen_chunk)
        ]
        for chunk in candidates:
            frame = frames.get(chunk)
            if frame is None:
                continue
            shape = tuple(int(value) for value in frame["shape"])
            if chunk not in image_cache:
                image_cache[chunk] = _prepare_cut3r_rgb(
                    frame["image_path"], shape
                )
            projected = _project_point(
                cell.xyz,
                np.asarray(frame["camera_pose"], dtype=np.float64),
                np.asarray(frame["intrinsics"], dtype=np.float64),
            )
            if projected is None:
                continue
            u, v, _ = projected
            x = int(round(u))
            y = int(round(v))
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                colors[cell_index] = image_cache[chunk][y, x]
                valid[cell_index] = True
                color_chunks[cell_index] = chunk
                break
    stats = {
        "num_cells": len(index.cells),
        "rgb_colored_cells": int(valid.sum()),
        "rgb_colored_fraction": float(valid.mean()) if len(valid) else 0.0,
        "color_source": "real generated first-seen observation",
        "cut3r_resize_crop_reproduced": True,
        "invented_colors": False,
    }
    return colors, valid, color_chunks, stats


def _set_world_bounds(axis, positions: np.ndarray, voxel_size: float) -> None:
    lower = np.quantile(positions, 0.01, axis=0)
    upper = np.quantile(positions, 0.99, axis=0)
    margin = np.maximum((upper - lower) * 0.04, voxel_size)
    axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axis.set_ylim(lower[2] - margin[2], upper[2] + margin[2])
    axis.set_zlim(lower[1] - margin[1], upper[1] + margin[1])
    axis.set_box_aspect(np.maximum((upper - lower)[[0, 2, 1]], voxel_size))


def render_rgb_world_splats(
    index: SurfelIndex,
    colors: np.ndarray,
    valid: np.ndarray,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.asarray([cell.xyz for cell in index.cells], dtype=np.float32)
    keep = valid & np.isfinite(positions).all(axis=1)
    figure = plt.figure(figsize=(10, 7), facecolor="#101418")
    axis = figure.add_subplot(111, projection="3d", facecolor="#101418")
    axis.scatter(
        positions[keep, 0],
        positions[keep, 2],
        positions[keep, 1],
        c=colors[keep].astype(np.float32) / 255.0,
        s=7,
        alpha=0.94,
        depthshade=False,
    )
    _set_world_bounds(axis, positions[keep], index.voxel_size)
    axis.view_init(elev=30, azim=-68)
    axis.set(title="A — RGB world-space surfel splats", xlabel="x", ylabel="z", zlabel="y")
    axis.tick_params(colors="white")
    axis.title.set_color("white")
    axis.xaxis.label.set_color("white")
    axis.yaxis.label.set_color("white")
    axis.zaxis.label.set_color("white")
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def render_rgb_oriented_disks(
    index: SurfelIndex,
    colors: np.ndarray,
    valid: np.ndarray,
    path: str | Path,
    *,
    max_disks: int = 3500,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    candidates = [
        item
        for item, cell in enumerate(index.cells)
        if valid[item]
        and cell.normal is not None
        and np.isfinite(cell.xyz).all()
        and np.isfinite(cell.normal).all()
        and np.isfinite(cell.radius)
        and cell.radius > 0
    ]
    if len(candidates) > max_disks:
        sampled = np.rint(np.linspace(0, len(candidates) - 1, max_disks)).astype(int)
        candidates = [candidates[item] for item in sampled]
    radii = np.asarray([index.cells[item].radius for item in candidates])
    radius_cap = float(np.quantile(radii, 0.95))
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    polygons = []
    facecolors = []
    positions = []
    for item in candidates:
        cell = index.cells[item]
        normal = np.asarray(cell.normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(normal, seed))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(normal, seed)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        radius = min(float(cell.radius), radius_cap)
        disk = (
            cell.xyz[None]
            + radius * np.cos(theta)[:, None] * tangent[None]
            + radius * np.sin(theta)[:, None] * bitangent[None]
        )
        polygons.append(disk[:, [0, 2, 1]])
        facecolors.append(colors[item].astype(np.float32) / 255.0)
        positions.append(cell.xyz)
    figure = plt.figure(figsize=(10, 7), facecolor="#101418")
    axis = figure.add_subplot(111, projection="3d", facecolor="#101418")
    collection = Poly3DCollection(
        polygons, facecolors=facecolors, edgecolors="none", alpha=0.92
    )
    axis.add_collection3d(collection)
    _set_world_bounds(axis, np.asarray(positions), index.voxel_size)
    axis.view_init(elev=30, azim=-68)
    axis.set(title="B — RGB oriented surfel disks", xlabel="x", ylabel="z", zlabel="y")
    axis.tick_params(colors="white")
    axis.title.set_color("white")
    axis.xaxis.label.set_color("white")
    axis.yaxis.label.set_color("white")
    axis.zaxis.label.set_color("white")
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def render_target_rgb(
    index: SurfelIndex,
    colors: np.ndarray,
    valid_colors: np.ndarray,
    query_pose: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
    *,
    eligible_chunks: set[int] | None = None,
    eligible_max_chunk: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    visible = index.visible_cells(
        query_pose,
        intrinsics,
        image_hw,
        source_image_size=image_hw,
        eligible_max_chunk=eligible_max_chunk,
        eligible_chunks=eligible_chunks,
        use_occlusion=True,
        front_facing=False,
        maximum_radius_pixels=12.0,
    )
    rendered = np.zeros((*image_hw, 3), dtype=np.uint8)
    mask = np.zeros(image_hw, dtype=bool)
    pixels = np.asarray(visible["pixels"], dtype=np.int32)
    indices = np.asarray(visible["indices"], dtype=np.int32)
    if len(pixels):
        keep = valid_colors[indices]
        pixels = pixels[keep]
        indices = indices[keep]
        rendered[pixels[:, 0], pixels[:, 1]] = colors[indices]
        mask[pixels[:, 0], pixels[:, 1]] = True
    stats = {
        "eligible_chunks": None if eligible_chunks is None else sorted(eligible_chunks),
        "visible_cells": int(visible["num_visible_cells"]),
        "rgb_pixels": int(mask.sum()),
        "rgb_coverage": float(mask.mean()),
    }
    return rendered, mask, stats


def _option_page(output_dir: Path, manifest: dict) -> None:
    cards = [
        (
            "A — RGB 世界坐标 splat",
            "A_rgb_world_splats.png",
            "看整体几何布局和颜色分布；点状、结构最诚实。",
        ),
        (
            "B — RGB 定向 surfel disk",
            "B_rgb_oriented_disks.png",
            "同时展示 position / normal / radius；适合审计 fusion，但可能显得杂乱。",
        ),
        (
            "C — B2 相机视角 RGB z-buffer（推荐主视图）",
            "C_rgb_target_zbuffer.png",
            "直接看历史 surfel 从目标相机能渲染出什么，最容易判断几何是否有意义。",
        ),
        (
            "D — B1-only RGB z-buffer（推荐 memory 主视图）",
            "D_rgb_b1_target_zbuffer.png",
            "纯 B1 chunk 8 surfel，不混入 B2 图像；直接看长期 memory 覆盖与外观。",
        ),
        (
            "E — B1 RGB 与 B2 overlay（推荐对齐审计）",
            "E_rgb_b1_target_overlay.png",
            "mask 外是 B2 baseline，mask 内 78% surfel + 22% B2；只用于检查落点。",
        ),
    ]
    figures = "".join(
        f"<figure><figcaption><b>{html.escape(title)}</b><br>"
        f"{html.escape(note)}</figcaption><img src='{path}'></figure>"
        for title, path, note in cards
    )
    payload = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>MapKV RGB Surfel 可视化选项</title><style>
body{{font:15px/1.5 system-ui;margin:0;background:#f3f5f8;color:#172033}}
main{{max-width:1280px;margin:auto;padding:24px}}section{{background:white;padding:20px;
border-radius:12px;margin:14px 0}}.grid{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px}}
figure{{margin:0}}figcaption{{min-height:70px}}img{{width:100%;background:#111;border-radius:8px}}
code{{background:#eef1f5;padding:2px 5px}}</style></head><body><main>
<section><h1>MapKV RGB Surfel 可视化候选</h1>
<p><b>本次关注：</b>选择未来 HTML report 的默认 surfel 主视图。</p>
<p>所有颜色均来自真实 generated historical frame 的 first-seen observation；
没有使用 chunk 伪彩色，也没有改变 CUT3R geometry / merge / retrieval。</p>
	<p>请直接回复 A / B / C / D / E，或组合，例如
	<code>D 主视图 + E 对齐审计 + A 补充</code>。</p></section>
<section class='grid'>{figures}</section>
<section><h2>数据审计</h2><pre>{html.escape(json.dumps(manifest, indent=2, ensure_ascii=False))}</pre></section>
</main></body></html>"""
    (output_dir / "report.html").write_text(payload, encoding="utf-8")


def generate_options(
    *,
    index_path: str | Path,
    sequence_path: str | Path,
    block_mapping_path: str | Path,
    target_chunk: int,
    source_chunk: int,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = SurfelIndex.load(index_path)
    sequence = json.loads(Path(sequence_path).read_text(encoding="utf-8"))
    colors, valid, color_chunks, color_stats = sample_historical_rgb(
        index, sequence_path
    )
    render_rgb_world_splats(
        index, colors, valid, output_dir / "A_rgb_world_splats.png"
    )
    render_rgb_oriented_disks(
        index, colors, valid, output_dir / "B_rgb_oriented_disks.png"
    )
    mapping_path = Path(block_mapping_path).resolve()
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    blocks = mapping_payload["blocks"]
    target = next(
        item for item in blocks if int(item["chunk_id"]) == int(target_chunk)
    )
    query_pose = to_cut3r_c2w(
        np.asarray(target["c2w"], dtype=np.float64)
    )
    image_hw = tuple(int(value) for value in sequence["frames"][0]["shape"])
    intrinsics = np.asarray(sequence["query_intrinsics"], dtype=np.float64)
    all_rgb, all_mask, all_stats = render_target_rgb(
        index,
        colors,
        valid,
        query_pose,
        intrinsics,
        image_hw,
        eligible_max_chunk=int(sequence["prefix_last_chunk"]),
    )
    Image.fromarray(all_rgb).save(output_dir / "C_rgb_target_zbuffer.png")
    b1_rgb, b1_mask, b1_stats = render_target_rgb(
        index,
        colors,
        valid,
        query_pose,
        intrinsics,
        image_hw,
        eligible_chunks={int(source_chunk)},
        eligible_max_chunk=int(target_chunk) - 2,
    )
    target_rgb = _prepare_cut3r_rgb(
        mapping_path.parent / target["png_path"], image_hw
    )
    Image.fromarray(b1_rgb).save(
        output_dir / "D_rgb_b1_target_zbuffer.png"
    )
    overlay = target_rgb.copy()
    overlay[b1_mask] = np.clip(
        0.78 * b1_rgb[b1_mask].astype(np.float32)
        + 0.22 * target_rgb[b1_mask].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    Image.fromarray(overlay).save(output_dir / "E_rgb_b1_target_overlay.png")
    Image.fromarray(target_rgb).save(output_dir / "target_b2.png")
    Image.fromarray((b1_mask.astype(np.uint8) * 255)).save(
        output_dir / "b1_support_mask.png"
    )
    manifest = {
        **color_stats,
        "index": str(Path(index_path).resolve()),
        "sequence": str(Path(sequence_path).resolve()),
        "target_chunk": int(target_chunk),
        "source_chunk": int(source_chunk),
        "target_camera_all_history": all_stats,
        "target_camera_b1_only": b1_stats,
        "color_chunk_min": int(color_chunks[valid].min()) if valid.any() else None,
        "color_chunk_max": int(color_chunks[valid].max()) if valid.any() else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _option_page(output_dir, manifest)
    thumbnails = []
    for name in (
        "A_rgb_world_splats.png",
        "B_rgb_oriented_disks.png",
        "C_rgb_target_zbuffer.png",
        "D_rgb_b1_target_zbuffer.png",
        "E_rgb_b1_target_overlay.png",
        "target_b2.png",
    ):
        thumbnails.append(
            ImageOps.fit(
                Image.open(output_dir / name).convert("RGB"),
                (480, 270),
                method=Image.Resampling.LANCZOS,
            )
        )
    sheet = Image.new("RGB", (1440, 540), "white")
    for item, image in enumerate(thumbnails):
        sheet.paste(image, ((item % 3) * 480, (item // 3) * 270))
    sheet.save(output_dir / "options_contact_sheet.jpg", quality=88)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate real-observation RGB surfel visualization options"
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--block_mapping", required=True)
    parser.add_argument("--target_chunk", type=int, required=True)
    parser.add_argument("--source_chunk", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_options(
                index_path=args.index,
                sequence_path=args.sequence,
                block_mapping_path=args.block_mapping,
                target_chunk=args.target_chunk,
                source_chunk=args.source_chunk,
                output_dir=args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
