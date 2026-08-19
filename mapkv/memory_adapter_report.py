from __future__ import annotations

import html
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .memory_adapter_evaluation import (
    METHODS,
    OPTIONAL_METHODS,
    evaluate_memory_adapter,
)


LABELS = {
    "baseline": "Baseline（原始 InSpatio）",
    "episode_wre": "Episode Continuous WRE",
    "latent_anchor_all4": "LatentAnchorAll4（硬控制）",
    "adapter_patch_only": "本次最新方法：MemoryPatchAdapter",
    "adapter_overfit": "Example A Overfit Adapter",
    "adapter_patch_middle": "最小 Refinement：Patch + Middle Adapter",
}


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _transcode(
    source: Path, destination: Path, *, start: float | None = None,
    duration: float | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        command += ["-ss", str(start)]
    command += ["-i", str(source)]
    if duration is not None:
        command += ["-t", str(duration)]
    command += [
        "-vf", "scale=-2:480", "-c:v", "libx264", "-crf", "28",
        "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(destination),
    ]
    subprocess.run(command, check=True)


def _videos(root: Path, case_id: str) -> None:
    case = root / case_id
    phases = {
        value["name"]: value
        for value in json.loads(
            (case / "trajectory/phase_labels.json").read_text()
        )["phases"]
    }
    fps = 24.0
    reentry_start = int(phases["Leave_to_B2"]["rgb_start"]) / fps
    reentry_stop = int(phases["B2_hold"]["rgb_stop_exclusive"]) / fps
    b2_start = int(phases["B2_hold"]["rgb_start"]) / fps
    b2_stop = int(phases["B2_hold"]["rgb_stop_exclusive"]) / fps
    for method, relative in {**METHODS, **OPTIONAL_METHODS}.items():
        source = case / relative / "pred.mp4"
        if not source.exists():
            continue
        outputs = {
            f"full_revisit_{method}.mp4": (None, None),
            f"reentry_{method}.mp4": (reentry_start, reentry_stop - reentry_start),
            f"b2_{method}.mp4": (b2_start, b2_stop - b2_start),
        }
        for name, (start, duration) in outputs.items():
            destination = case / "videos/report" / name
            if not destination.exists():
                _transcode(source, destination, start=start, duration=duration)


def _architecture_graph(root: Path) -> Path:
    destination = root / "architecture_graph.svg"
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="360" viewBox="0 0 1200 360">
<style>.n{fill:#17243a;stroke:#56c7ff;stroke-width:2;rx:16}.m{fill:#17372e;stroke:#62e6a7;stroke-width:3;rx:16}.t{fill:#eef7ff;font:18px sans-serif;text-anchor:middle}.s{fill:#a9bdd2;font:14px sans-serif;text-anchor:middle}.a{stroke:#7d91aa;stroke-width:3;marker-end:url(#e)}</style>
<defs><marker id="e" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#7d91aa"/></marker></defs>
<rect class="n" x="25" y="105" width="190" height="100"/><text class="t" x="120" y="140">RGB 历史 + known camera</text><text class="s" x="120" y="169">RGB Warp → Wan VAE</text>
<rect class="n" x="265" y="105" width="180" height="100"/><text class="t" x="355" y="140">M_need</text><text class="s" x="355" y="169">generated-only × source-blind</text>
<rect class="m" x="495" y="70" width="240" height="170"/><text class="t" x="615" y="115">本次架构修改</text><text class="t" x="615" y="147">MemoryPatchAdapter</text><text class="s" x="615" y="178">2×Conv3D + zero-init 1×1</text><text class="s" x="615" y="204">masked patch-token residual</text>
<rect class="n" x="785" y="105" width="180" height="100"/><text class="t" x="875" y="140">Frozen InSpatio</text><text class="s" x="875" y="169">native patch + 30 blocks</text>
<rect class="n" x="1015" y="105" width="160" height="100"/><text class="t" x="1095" y="140">Revisit output</text><text class="s" x="1095" y="169">identity / boundary</text>
<line class="a" x1="215" y1="155" x2="265" y2="155"/><line class="a" x1="445" y1="155" x2="495" y2="155"/><line class="a" x1="735" y1="155" x2="785" y2="155"/><line class="a" x1="965" y1="155" x2="1015" y2="155"/>
<text class="s" x="600" y="300">冻结：backbone / VAE / text / scheduler / CUT3R / surfel / lifecycle　　仅训练绿色模块</text></svg>"""
    destination.write_text(svg, encoding="utf-8")
    return destination


def _training_plot(root: Path) -> Path:
    destination = root / "assets/training_curve.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        ("overfit_scene01", "Example A overfit"),
        ("joint_scene01_scene02", "A+B patch-only"),
    ]
    if (root / "training/joint_patch_middle/training_curve.json").exists():
        entries.append(("joint_patch_middle", "A+B patch+middle"))
    fig, axes = plt.subplots(1, len(entries), figsize=(5 * len(entries), 3.2))
    axes = np.atleast_1d(axes)
    for axis, (name, title) in zip(axes, entries):
        curve = json.loads(
            (root / f"training/{name}/training_curve.json").read_text()
        )
        axis.plot([value["iteration"] for value in curve], [value["total"] for value in curve])
        axis.set_title(title)
        axis.set_xlabel("iteration")
        axis.set_ylabel("total loss")
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def _largest_boxes(mask_path: Path, count: int = 2) -> list[tuple[int, int, int, int]]:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    try:
        import cv2
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        components = sorted(stats[1:], key=lambda value: int(value[4]), reverse=True)
        boxes = [
            (int(x), int(y), int(x + w), int(y + h))
            for x, y, w, h, area in components if int(area) >= 64
        ][:count]
    except ImportError:
        ys, xs = np.nonzero(mask)
        boxes = [] if not len(xs) else [(int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))]
    padded = []
    for x0, y0, x1, y1 in boxes:
        pad = 20
        padded.append((max(0, x0-pad), max(0, y0-pad), min(mask.shape[1], x1+pad), min(mask.shape[0], y1+pad)))
    return padded


def _identity_crops(root: Path, case_id: str) -> list[Path]:
    case = root / case_id
    trajectory = json.loads(
        (case / "trajectory/trajectory_manifest.json").read_text()
    )
    block = int(trajectory["target_chunk"])
    source_chunk = int(trajectory["source_chunk"])
    adapter_root = case / "methods/adapter_patch_only"
    mask_path = adapter_root / f"memory_adapter/block_{block:04d}/M_need.png"
    memory_path = adapter_root / f"memory_adapter/block_{block:04d}/L_mem_decoded.png"
    crop_labels = {
        "baseline": "Baseline",
        "episode_wre": "Episode WRE",
        "latent_anchor_all4": "LatentAnchorAll4",
        "adapter_patch_only": "MemoryPatchAdapter",
        "adapter_patch_middle": "Patch+Middle",
    }
    sources = [
        (
            "B1 canonical",
            case / f"methods/baseline/keyframes/chunk_{source_chunk:04d}.png",
        ),
        ("B1 warped to B2", memory_path),
        *[
            (crop_labels[name], case / relative / f"keyframes/chunk_{block:04d}.png")
            for name, relative in {**METHODS, **OPTIONAL_METHODS}.items()
            if (case / relative / f"keyframes/chunk_{block:04d}.png").exists()
        ],
    ]
    outputs = []
    for index, box in enumerate(_largest_boxes(mask_path), start=1):
        tiles = []
        for label, source in sources:
            image = Image.open(source).convert("RGB").crop(box).resize((280, 180))
            tile = Image.new("RGB", (280, 212), "#101826")
            tile.paste(image, (0, 32))
            ImageDraw.Draw(tile).text((8, 8), label, fill="white")
            tiles.append(tile)
        sheet = Image.new("RGB", (280 * len(tiles), 212), "#101826")
        for offset, tile in enumerate(tiles):
            sheet.paste(tile, (offset * 280, 0))
        output = case / f"identity_crops/region_{index}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, quality=90)
        outputs.append(output)
    return outputs


def _video_grid(case_id: str, kind: str, methods: list[str]) -> str:
    videos = "".join(
        f"<figure><figcaption>{html.escape(LABELS[name])}</figcaption>"
        f"<video class='sync-{case_id}-{kind}' controls preload='metadata' "
        f"src='{case_id}/videos/report/{kind}_{name}.mp4'></video></figure>"
        for name in methods
    )
    key = f"sync-{case_id}-{kind}"
    return (
        f"<div class='buttons'><button onclick=playAll('{key}')>全部播放</button>"
        f"<button onclick=pauseAll('{key}')>全部暂停</button>"
        f"<button onclick=resetAll('{key}')>全部归零</button></div>"
        f"<div class='video-grid'>{videos}</div>"
    )


def build_memory_adapter_report(root: str | Path) -> Path:
    root = Path(root).resolve()
    metrics = evaluate_memory_adapter(root)
    for case_id in metrics["cases"]:
        _videos(root, case_id)
    architecture = _architecture_graph(root)
    training_plot = _training_plot(root)
    crops = {case_id: _identity_crops(root, case_id) for case_id in metrics["cases"]}
    tables = []
    review_tables = []
    case_sections = []
    for case_id, case in metrics["cases"].items():
        video_methods = list(METHODS)
        if (root / case_id / OPTIONAL_METHODS["adapter_patch_middle"] / "pred.mp4").exists():
            video_methods.append("adapter_patch_middle")
        rows = "".join(
            "<tr><td>{}</td><td>{:.5f}</td><td>{:.4f}</td><td>{:.5f}</td>"
            "<td>{:.5f}</td><td>{:.5f}</td></tr>".format(
                LABELS[name], value["historical_appearance_l1"],
                value["generated_history_feature_similarity"],
                value["source_region_delta_vs_baseline"],
                value["boundary_band_error"], value["reentry_temporal_peak"],
            )
            for name, value in case["methods"].items()
        )
        tables.append(
            f"<h3>{case_id}</h3><p>轨迹 0°→{case['trajectory']['b1_yaw']:.0f}°"
            f"→{case['trajectory']['leave_yaw']:.0f}°→{case['trajectory']['b2_yaw']:.0f}°；"
            f"M_need 平均覆盖 {case['mean_need_coverage']:.1%}；"
            f"adapter-off max diff = {case['adapter_off_max_abs_diff']:.1e}</p>"
            "<table><tr><th>方法</th><th>历史外观 L1↓</th><th>结构相似↑</th>"
            f"<th>Source Δ↓</th><th>Boundary↓</th><th>Re-entry peak↓</th></tr>{rows}</table>"
        )
        reviews = metrics["human_review"].get(case_id, {})
        review_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                LABELS[name], value["identity"], value["transition"],
                value["boundary"], value["source_protection"],
            )
            for name, value in reviews.items()
            if name in LABELS
        )
        review_tables.append(
            f"<h3>{case_id}</h3><table><tr><th>方法</th><th>Identity</th>"
            f"<th>Transition</th><th>Boundary</th><th>Source protection</th>"
            f"</tr>{review_rows}</table>"
        )
        crop_html = "".join(
            f"<img class='wide' src='{_rel(path, root)}'>" for path in crops[case_id]
        )
        adapter = (
            root / case_id / "methods/adapter_patch_only/memory_adapter"
            / f"block_{case['target_chunk']:04d}"
        )
        case_sections.append(
            f"<section><h2>{case_id}：mask 与 identity 区域</h2>"
            "<div class='image-grid'>"
            f"<figure><figcaption>B2 的 M_need</figcaption><img src='{_rel(adapter/'M_need.png', root)}'></figure>"
            f"<figure><figcaption>camera-aligned L_mem</figcaption><img src='{_rel(adapter/'L_mem_decoded.png', root)}'></figure>"
            "</div>" + crop_html + "</section>"
            f"<section><h2>{case_id}：完整回访视频（B1→离开→回访→B2）</h2>{_video_grid(case_id, 'full_revisit', video_methods)}</section>"
            f"<section><h2>{case_id}：Re-entry window</h2>{_video_grid(case_id, 'reentry', video_methods)}</section>"
            f"<section><h2>{case_id}：B2 hold identity window</h2>{_video_grid(case_id, 'b2', video_methods)}</section>"
        )
    joint = metrics["training"]["joint_scene01_scene02"]
    best_training = (
        metrics["training"]["joint_patch_middle"]
        if metrics["best_adapter"] == "adapter_patch_middle"
        else joint
    )
    architecture_state = {
        "backbone": "InSpatio-World-1.3B frozen",
        "memory": "known-pose surfel → RGB warp → Wan VAE L_mem",
        "adapter": best_training["config"],
        "best_adapter": metrics["best_adapter"],
        "trainable_parameters": best_training["freeze_audit"]["adapter_trainable_parameters"],
        "frozen_parameters": "backbone/text/VAE/scheduler/CUT3R",
        "injection": "parallel masked patch-token residual",
        "cases": list(metrics["cases"]),
    }
    (root / "architecture_state.json").write_text(
        json.dumps(architecture_state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    html_text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>MapKV Lightweight Memory Adapter</title>
<style>body{{margin:auto;max-width:1500px;background:#09111f;color:#e8f1fb;font:16px/1.55 system-ui;padding:26px}}section{{background:#101b2d;border:1px solid #263b55;border-radius:14px;padding:20px;margin:18px 0}}h1,h2{{color:#73d7ff}}.focus{{background:#17372e;border-left:5px solid #62e6a7;padding:14px}}img,video{{max-width:100%;border-radius:8px;background:#050910}}.video-grid,.image-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0}}figcaption{{font-weight:650;margin:6px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #334861;padding:7px;text-align:left}}th{{background:#172842}}button{{padding:8px 14px;margin:4px;background:#2c6f99;color:white;border:0;border-radius:7px}}.wide{{width:100%;margin:10px 0}}code{{color:#8ee8bc}}@media(max-width:800px){{.video-grid,.image-grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>MapKV Lightweight Memory Adapter</h1><p class='focus'><b>本次最新方法 / Focus：</b>冻结 InSpatio，只训练零初始化 Conv3D <code>MemoryPatchAdapter</code>，测试 changed-view identity 是否能强于 Episode WRE，同时比 LatentAnchorAll4 更少 seam / popping。</p>
<section><h2>结论</h2><p><b>Status: {metrics['status']}</b></p><p>Best adapter: <b>{metrics['best_adapter']}</b>；两个案例；adapter-off exact={metrics['decisions']['adapter_off_exact']}；identity metric better={metrics['decisions']['joint_adapter_identity_metric_better_than_episode_wre']}；boundary better than hard={metrics['decisions']['joint_adapter_boundary_better_than_latent_anchor_all4']}。</p></section>
<section><h2>完整 Pipeline / Framework</h2><img class='wide' src='{_rel(architecture, root)}'><h3>Architecture Changes</h3><table><tr><th>模块</th><th>Before</th><th>Now</th><th>训练</th></tr><tr><td>Memory interface</td><td>WRE attention delta / hard x0</td><td>parallel masked patch-token residual</td><td>仅 Adapter</td></tr><tr><td>Native patch</td><td>frozen input embed</td><td>保持冻结并加 sidecar residual</td><td>不训练</td></tr><tr><td>Geometry / lifecycle</td><td>validated source-protected episode</td><td>完全不变</td><td>不训练</td></tr></table></section>
<section><h2>训练</h2><p>Trainable params: {best_training['freeze_audit']['adapter_trainable_parameters']:,}；backbone trainable: {best_training['freeze_audit']['backbone_trainable_parameters']}；A+B balanced；matched core-loss reduction: {best_training['matched_core_loss_reduction_fraction']:.1%}。</p><img class='wide' src='{_rel(training_plot, root)}'></section>
<section><h2>核心指标</h2>{''.join(tables)}<p>注意：L1 是 historical appearance，不单独等价于 instance identity；最终判断依赖下方同步视频与结构 crop。</p></section>
<section><h2>人眼评级</h2>{''.join(review_tables)}<p>Patch+Middle 在两例均只有 PARTIAL identity，且仍有 clear seam；因此不满足 GO。</p></section>
{''.join(case_sections)}
<section><h2>Findings 与下一步</h2><p><b>Worked：</b>zero-init 严格退化、appearance L1 与 re-entry peak 改善、backbone 完全冻结。</p><p><b>Failed：</b>两例 changed-view instance identity 均未稳定超过 Episode WRE；scene01 暗斑、scene02 白雾/瓶体重复仍明显。</p><p><b>Uncertain：</b>未进入 held-out 泛化，因为两例 joint gate 已失败。</p><p><b>下一步唯一动作：</b>保持当前 geometry/adapter 接口与 frozen backbone，改用更高保真的 target-aligned intermediate-feature teacher 训练同一轻量 adapter；不要继续扩大当前 RGB/VAE patch residual。</p></section>
<script>function all(c){{return [...document.querySelectorAll('.'+c)]}}function playAll(c){{let v=all(c),t=Math.min(...v.map(x=>x.currentTime));v.forEach(x=>{{x.currentTime=t;x.play()}})}}function pauseAll(c){{all(c).forEach(x=>x.pause())}}function resetAll(c){{all(c).forEach(x=>{{x.pause();x.currentTime=0}})}}</script></body></html>"""
    report = root / "report.html"
    report.write_text(html_text, encoding="utf-8")
    markdown = f"""# MapKV Lightweight Memory Adapter

- Status: **{metrics['status']}**
- Focus: 冻结 InSpatio，仅训练 source-protected patch-token memory residual
- Examples: scene01 与独立 scene02（均为 pure-yaw leave/re-entry changed-view revisit）
- Best adapter: {metrics['best_adapter']}
- Trainable parameters: {best_training['freeze_audit']['adapter_trainable_parameters']:,}
- Adapter-off max diff: 0 for both cases
- HTML: `report.html`

## Architecture

Known-pose surfel → RGB Warp → Wan VAE L_mem；`[L_mem, raw last_pred, M_need]` → tiny zero-init Conv3D → masked patch-token residual → frozen InSpatio。

## Decision

- Identity metric better than Episode WRE: {metrics['decisions']['joint_adapter_identity_metric_better_than_episode_wre']}
- Boundary metric better than LatentAnchorAll4: {metrics['decisions']['joint_adapter_boundary_better_than_latent_anchor_all4']}
- Human review GO: {metrics['decisions']['human_review_go']}

## Next action

Keep geometry and the frozen backbone fixed; train the same lightweight adapter
from a higher-fidelity target-aligned intermediate-feature teacher instead of
expanding the current RGB/VAE patch residual.
"""
    (root / "report.md").write_text(markdown, encoding="utf-8")
    return report


__all__ = ["build_memory_adapter_report"]
