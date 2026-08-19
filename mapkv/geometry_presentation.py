from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from string import Template

import numpy as np


DEFAULT_ROOT = (
    "results/mapkv_fast/"
    "yaw45m20to35_scene01_seed0_geometry_repair"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cut3r_depth_confidence(root: Path, output: Path, frame: int = 8) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = np.load(root / "cut3r/frames" / f"frame_{frame:04d}.npz")
    depth = np.asarray(payload["depth"], dtype=np.float32)
    confidence = np.asarray(payload["confidence"], dtype=np.float32)
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    valid_confidence = confidence[np.isfinite(confidence) & (confidence > 0)]
    depth_limits = tuple(np.percentile(valid_depth, [2, 98]))
    confidence_limits = (
        0.0,
        float(np.percentile(valid_confidence, 98)) if valid_confidence.size else 1.0,
    )
    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#07111f")
    depth_image = axes[0].imshow(
        depth,
        cmap="turbo",
        vmin=depth_limits[0],
        vmax=depth_limits[1],
    )
    axes[0].set_title("CUT3R depth / fixed known pose")
    confidence_image = axes[1].imshow(
        confidence,
        cmap="magma",
        vmin=confidence_limits[0],
        vmax=confidence_limits[1],
    )
    axes[1].set_title("CUT3R confidence")
    for axis in axes:
        axis.set_axis_off()
    figure.colorbar(depth_image, ax=axes[0], fraction=0.045, pad=0.02)
    figure.colorbar(confidence_image, ax=axes[1], fraction=0.045, pad=0.02)
    figure.tight_layout(pad=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


def _score_rows(scores: dict[str, float], positive: set[int], selected: int) -> str:
    maximum = max(float(value) for value in scores.values())
    rows = []
    for chunk, raw_score in sorted(
        ((int(key), float(value)) for key, value in scores.items()),
        key=lambda item: item[0],
    ):
        width = 100.0 * raw_score / max(maximum, 1e-12)
        classes = ["score-row"]
        if chunk in positive:
            classes.append("positive")
        if chunk == selected:
            classes.append("selected")
        rows.append(
            f"<div class='{' '.join(classes)}'>"
            f"<span class='chunk-label'>chunk {chunk}</span>"
            f"<span class='bar-track'><span class='bar' style='width:{width:.2f}%'></span></span>"
            f"<span class='score-value'>{raw_score:.3f}</span></div>"
        )
    return "".join(rows)


def _keyframe_cards(chunks: list[tuple[int, str, str]]) -> str:
    return "".join(
        "<figure class='frame-card'>"
        f"<img src='baseline/keyframes/chunk_{chunk:04d}.png'>"
        f"<figcaption><b>{html.escape(title)}</b><span>{html.escape(note)}</span></figcaption>"
        "</figure>"
        for chunk, title, note in chunks
    )


def build_presentation(root: str | Path) -> Path:
    root = Path(root).resolve()
    cut3r = _json(root / "cut3r/stats.json")
    surfel = _json(root / "surfel/stats.json")
    retrieval_payload = _json(root / "retrieval/retrieval.json")
    retrieval = retrieval_payload["targets"][0]
    target = int(retrieval["target_chunk"])
    selected = int(retrieval["selected_chunks"][0])
    positive = {int(value) for value in retrieval["positive_cluster"]}
    top3 = retrieval["retrieved"][:3]
    assets = root / "presentation_assets"
    _cut3r_depth_confidence(
        root, assets / "cut3r_depth_confidence.png", frame=selected
    )

    keyframes = _keyframe_cards(
        [
            (0, "初始视角", "history begins"),
            (selected, f"chunk {selected}", "retrieval 最终选中"),
            (11, "B1 plateau", "correct positive cluster"),
            (target, f"target {target}", "当前回访视角"),
        ]
    )
    top3_cards = "".join(
        "<article class='rank-card'>"
        f"<div class='rank'>#{rank}</div>"
        f"<img src='baseline/keyframes/chunk_{int(item['chunk_id']):04d}.png'>"
        f"<h3>chunk {int(item['chunk_id'])}</h3>"
        f"<p>score <b>{float(item['score']):.3f}</b></p>"
        f"<small>{int(item['visible_support'])} visible stable surfels</small>"
        "</article>"
        for rank, item in enumerate(top3, start=1)
    )
    score_rows = _score_rows(retrieval["scores"], positive, selected)
    positive_text = ", ".join(str(value) for value in sorted(positive))

    template = Template(
        r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MapKV — 3D Surfel Addressing Presentation</title>
<style>
:root{--bg:#07111f;--panel:#101e31;--panel2:#16263b;--text:#f6f8fc;--muted:#9fb0c8;--blue:#5ca7ff;--cyan:#49d5d0;--green:#73d17c;--orange:#ffb55f;--red:#ff7067;--line:#29405e}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:Inter,"Noto Sans SC","Microsoft YaHei",Arial,sans-serif}.deck{position:relative;width:100%;height:100%}.slide{position:absolute;inset:0;display:none;padding:4.4vh 5.2vw 7.5vh;background:radial-gradient(circle at 84% 12%,#183653 0,transparent 31%),linear-gradient(135deg,#07111f,#0a1728 58%,#07111f);overflow:auto}.slide.active{display:block}.eyebrow{color:var(--cyan);font-weight:800;letter-spacing:.12em;text-transform:uppercase;font-size:clamp(12px,1vw,18px)}h1{font-size:clamp(42px,5.3vw,88px);line-height:1.02;margin:.2em 0}.slide h2{font-size:clamp(32px,3.5vw,60px);margin:.15em 0 .55em}.slide h3{margin:.35em 0}.lead{font-size:clamp(19px,1.65vw,31px);line-height:1.5;color:#dce6f4;max-width:1250px}.muted{color:var(--muted)}.accent{color:var(--green)}.warn{color:var(--orange)}.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:2.3vw;align-items:center}.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5vw}.media{width:100%;max-height:64vh;object-fit:contain;border-radius:14px;background:#02070e;border:1px solid var(--line);box-shadow:0 18px 45px #0008}.panel{background:linear-gradient(145deg,#13243aee,#0c192aee);border:1px solid var(--line);border-radius:18px;padding:1.4vw;box-shadow:0 18px 45px #0004}.flow{display:flex;align-items:stretch;gap:.7vw;margin:4vh 0;flex-wrap:wrap}.flow-node{flex:1;min-width:160px;background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:1.15vw}.flow-node b{display:block;color:var(--cyan);font-size:clamp(16px,1.3vw,24px);margin-bottom:.45em}.arrow{display:flex;align-items:center;color:var(--green);font-size:32px;font-weight:900}.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:1vw;margin:1.2vw 0}.metric{background:#0e1d30;border:1px solid var(--line);border-radius:14px;padding:1vw}.metric strong{display:block;font-size:clamp(23px,2.25vw,42px);color:var(--green)}.metric span{color:var(--muted)}.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:1vw}.frame-card{margin:0;background:#0c1828;border:1px solid var(--line);border-radius:14px;overflow:hidden}.frame-card img{width:100%;aspect-ratio:16/9;object-fit:cover}.frame-card figcaption{padding:.7vw}.frame-card figcaption b,.frame-card figcaption span{display:block}.frame-card figcaption span{color:var(--muted);margin-top:.25em}.two-images{display:grid;grid-template-columns:1fr 1fr;gap:1.2vw}.two-images figure,.three-images figure{margin:0}.two-images figcaption,.three-images figcaption{font-size:clamp(14px,1.1vw,20px);color:#d7e4f5;margin:.5em 0}.three-images{display:grid;grid-template-columns:repeat(3,1fr);gap:1vw}.score-list{display:grid;grid-template-columns:1fr 1fr;column-gap:1.8vw}.score-row{display:grid;grid-template-columns:88px 1fr 56px;align-items:center;gap:8px;margin:5px 0;padding:3px 6px;border-radius:7px}.bar-track{height:12px;background:#21344d;border-radius:99px;overflow:hidden}.bar{display:block;height:100%;background:#587fae;border-radius:99px}.score-row.positive{background:#173b2b}.score-row.positive .bar{background:var(--green)}.score-row.selected{outline:2px solid var(--orange);background:#563e1a}.score-row.selected .bar{background:var(--orange)}.chunk-label,.score-value{font-variant-numeric:tabular-nums}.rank-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.3vw}.rank-card{position:relative;background:#101f33;border:1px solid var(--line);border-radius:16px;padding:1vw}.rank-card:first-child{border:2px solid var(--orange);box-shadow:0 0 34px #ffb55f33}.rank-card img{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:10px}.rank{position:absolute;top:-12px;left:-10px;background:var(--orange);color:#101010;font-weight:900;font-size:24px;width:48px;height:48px;border-radius:50%;display:grid;place-items:center}.rank-card small{color:var(--muted)}.callout{border-left:5px solid var(--green);background:#102c29;padding:1vw 1.3vw;border-radius:0 12px 12px 0;font-size:clamp(17px,1.3vw,24px);line-height:1.45}.caveat{border-left-co
}
.caveat{border-left-color:var(--orange);background:#332718}
.legend{display:flex;flex-wrap:wrap;gap:.65vw;color:#dce6f4;font-size:14px;margin-bottom:.55vw}
.legend span{display:inline-flex;align-items:center;background:#0a1728;border:1px solid var(--line);padding:.35em .65em;border-radius:99px}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}
.controls{position:fixed;z-index:20;left:0;right:0;bottom:0;height:6vh;background:#07111fe8;border-top:1px solid #21354e;display:flex;align-items:center;padding:0 2vw;gap:.7vw;backdrop-filter:blur(8px)}
button{background:#15263b;color:white;border:1px solid #345170;border-radius:9px;padding:.55em .9em;cursor:pointer}button:hover{background:#23405e}
.progress{height:5px;flex:1;background:#20334a;border-radius:20px;overflow:hidden}.progress span{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .25s}
.counter{font-variant-numeric:tabular-nums;color:var(--muted);min-width:72px;text-align:center}.slide-nav{display:flex;gap:5px}.slide-nav button{width:10px;height:10px;border-radius:50%;padding:0;border:0;background:#405671}.slide-nav button.active{background:var(--orange);transform:scale(1.3)}
.pipeline-mini{position:absolute;right:5vw;top:4.4vh;color:#7189a6;font-size:13px}.pipeline-mini b{color:var(--cyan)}
ul.clean{font-size:clamp(17px,1.35vw,25px);line-height:1.55;padding-left:1.2em}code{color:#8be4dc;background:#12283b;padding:.15em .35em;border-radius:5px}
@media(max-width:900px){.grid-2,.grid-3,.frames,.two-images,.three-images,.rank-grid{grid-template-columns:1fr}.slide{padding-bottom:10vh}.metric-row{grid-template-columns:1fr 1fr}.score-list{grid-template-columns:1fr}.pipeline-mini{display:none}}
@media print{html,body{overflow:visible;background:white}.controls{display:none}.slide{position:relative;display:block!important;width:13.333in;height:7.5in;page-break-after:always;overflow:hidden;color:white}}
</style></head><body><main class="deck">

<section class="slide active" data-title="结论">
  <div class="eyebrow">MapKV Geometry Addressing · Controlled Demo</div>
  <h1>3D Surfel 如何<br><span class="accent">找到历史 KV-Cache？</span></h1>
  <p class="lead">显式几何不保存 appearance。它只回答一个地址问题：<br><b>“当前相机看见的表面，曾经由哪个历史 chunk 观察过？”</b></p>
  <div class="flow">
    <div class="flow-node"><b>1 · Generated History</b>历史 RGB / chunk IDs</div><div class="arrow">→</div>
    <div class="flow-node"><b>2 · CUT3R</b>depth + confidence</div><div class="arrow">→</div>
    <div class="flow-node"><b>3 · RGB Surfels</b>position / normal / radius</div><div class="arrow">→</div>
    <div class="flow-node"><b>4 · Visibility Vote</b>surfel → observing chunks</div><div class="arrow">→</div>
    <div class="flow-node"><b>5 · KV Address</b>选择 chunk $selected</div>
  </div>
  <div class="callout"><b>本例：</b>target chunk $target → top-1 <b>chunk $selected</b>；命中 B1 positive cluster [$positive_text]。</div>
</section>

<section class="slide" data-title="输入历史">
  <div class="pipeline-mini">Generated history <b>●</b> → CUT3R → Surfels → Retrieval</div>
  <div class="eyebrow">Step 1 · Raw generated history</div><h2>输入不是 GT，而是模型自己生成的历史</h2>
  <div class="grid-2">
    <div><video class="media" controls preload="metadata" src="baseline/pred.mp4"></video><p class="muted">完整 causal baseline：B1 first visit → leave → B2 revisit。CUT3R 只看到 target 之前的 prefix。</p></div>
    <div><div class="frames">$keyframes</div><div class="callout" style="margin-top:1.2vw">我们为每个历史 chunk 同时保存：<b>RGB keyframe + known camera pose + native KV payload</b>。</div></div>
  </div>
</section>

<section class="slide" data-title="CUT3R">
  <div class="pipeline-mini">Generated history → <b>CUT3R ●</b> → Surfels → Retrieval</div>
  <div class="eyebrow">Step 2 · CUT3R fixed-pose processing</div><h2>CUT3R 提供 depth；相机 pose / K 使用已知真值</h2>
  <div class="grid-2">
    <div><img class="media" src="baseline/keyframes/chunk_$selected4.png"><p class="muted">示例输入：historical chunk $selected</p></div>
    <div><img class="media" src="presentation_assets/cut3r_depth_confidence.png"></div>
  </div>
  <div class="metric-row">
    <div class="metric"><strong>$cut3r_frames</strong><span>causal prefix frames</span></div>
    <div class="metric"><strong>$accepted_ratio</strong><span>accepted point ratio</span></div>
    <div class="metric"><strong>fixed</strong><span>known c2w + intrinsics</span></div>
    <div class="metric"><strong>0</strong><span>future frames leaked</span></div>
  </div>
  <p class="callout caveat">CUT3R predicted pose 只做诊断；即使它有 drift，也<b>不会</b>进入 world map。</p>
</section>

<section class="slide" data-title="点云对齐">
  <div class="pipeline-mini">Generated history → <b>CUT3R ●</b> → Surfels → Retrieval</div>
  <div class="eyebrow">Step 2.5 · Fixed global alignment</div><h2>每帧 depth 按 known pose 放进同一 world frame</h2>
  <div class="grid-2">
    <figure><img class="media" src="cut3r/pointcloud_preview.png"><figcaption class="muted">Causal prefix aligned point cloud</figcaption></figure>
    <figure><img class="media" src="cut3r/camera_trajectory.png"><figcaption class="muted">Known camera trajectory</figcaption></figure>
  </div>
  <div class="callout">这里得到的是粗几何 state，不追求 photoreal 3D reconstruction；目标只是让同一真实表面在多视图中落到相近的 3D 邻域。</div>
</section>

<section class="slide" data-title="Surfel 构建">
  <div class="pipeline-mini">Generated history → CUT3R → <b>RGB Surfels ●</b> → Retrieval</div>
  <div class="eyebrow">Step 3 · Stable RGB surfel index</div><h2>Pointmap → radius/normal surfels → stable address</h2>
  <div class="two-images">
    <figure><figcaption>RGB surfel centers（真实历史 RGB）</figcaption><img class="media" src="surfel_rgb_options/A_rgb_world_splats.png"></figure>
    <figure><figcaption>Oriented disks（position + normal + radius）</figcaption><img class="media" src="surfel_rgb_options/B_rgb_oriented_disks.png"></figure>
  </div>
  <div class="metric-row">
    <div class="metric"><strong>$num_cells</strong><span>all surfel cells</span></div>
    <div class="metric"><strong>$stable_cells</strong><span>stable, allowed to vote</span></div>
    <div class="metric"><strong>$stable_fraction</strong><span>stable fraction</span></div>
    <div class="metric"><strong>$reprojection_p95 px</strong><span>reprojection residual p95</span></div>
  </div>
  <p class="muted">Merge = spatial radius + normal consistency + reprojection residual；单视图 tentative surfel 不参与长期 voting / occlusion。</p>
</section>

<section class="slide" data-title="Target 可见性">
  <div class="pipeline-mini">Generated history → CUT3R → RGB Surfels → <b>Visibility ●</b></div>
  <div class="eyebrow">Step 4 · Target-view visibility</div><h2>先过滤 eligible history，再做 target-view z-buffer</h2>
  <div class="three-images">
    <figure><figcaption>Target B2 / chunk $target</figcaption><img class="media" src="surfel_rgb_options/target_b2.png"></figure>
    <figure><figcaption>B1 history 投影到 target</figcaption><img class="media" src="surfel_rgb_options/D_rgb_b1_target_zbuffer.png"></figure>
    <figure><figcaption>投影 overlay：对齐误差会直接显现</figcaption><img class="media" src="surfel_rgb_options/E_rgb_b1_target_overlay.png"></figure>
  </div>
  <div class="grid-2" style="margin-top:1vw"><img class="media" src="retrieval/visible_support_target_0037.png"><div><div class="legend"><span><i class="dot" style="background:#4678d2"></i>全部 visible stable surfels</span><span><i class="dot" style="background:#eb5546"></i>selected chunk support</span><span><i class="dot" style="background:#000"></i>无几何 support</span></div><ul class="clean"><li>future / current / recent chunks 先排除</li><li>只让 stable surfels 参与遮挡与投票</li><li>每个 visible surfel 对 observing chunks 投票</li><li>同一 pose plateau 用 cluster-max，避免帧数多就占便宜</li></ul></div></div>
</section>

<section class="slide" data-title="检索分数">
  <div class="pipeline-mini">Generated history → CUT3R → RGB Surfels → <b>Chunk Vote ●</b></div>
  <div class="eyebrow">Step 4.5 · Geometry → chunk score</div><h2>Target-visible surfels 给历史 chunk 投票</h2>
  <div class="grid-2">
    <div class="panel"><div class="legend"><span><i class="dot" style="background:var(--green)"></i>B1 positive cluster</span><span><i class="dot" style="background:var(--orange)"></i>selected</span></div><div class="score-list">$score_rows</div></div>
    <div><img class="media" src="retrieval/retrieval_timeline.png"><p class="callout"><code>score += confidence × view_alignment ÷ depth</code><br>单位是 <b>unique visible stable surfel</b>，不是 raw point 数。</p></div>
  </div>
</section>

<section class="slide" data-title="最终选取">
  <div class="pipeline-mini">Generated history → CUT3R → Surfels → Vote → <b>KV Address ●</b></div>
  <div class="eyebrow">Step 5 · Select KV payload address</div><h2>Top-1 选择 chunk $selected</h2>
  <div class="rank-grid">$top3_cards</div>
  <div class="metric-row">
    <div class="metric"><strong>$coverage</strong><span>selected target coverage</span></div>
    <div class="metric"><strong>$margin</strong><span>top-1 score margin</span></div>
    <div class="metric"><strong>$entropy</strong><span>normalized entropy</span></div>
    <div class="metric"><strong>$gap</strong><span>temporal gap / chunks</span></div>
  </div>
  <div class="flow" style="margin:1.2vh 0"><div class="flow-node"><b>Selected address</b>chunk $selected</div><div class="arrow">→</div><div class="flow-node"><b>KVChunkBank</b>native K/V of chunk $selected</div><div class="arrow">→</div><div class="flow-node"><b>Memory interface</b>Recent / Render / Latent</div></div>
</section>

<section class="slide" data-title="Takeaway">
  <div class="eyebrow">Takeaway · What is proven?</div><h2>当前 surfel 系统已能做<br><span class="accent">受控的 coarse KV-cluster addressing</span></h2>
  <div class="grid-2">
    <div class="panel"><h3 class="accent">✓ 已成立</h3><ul class="clean"><li>known-pose CUT3R 构建可用粗 3D state</li><li>stable surfel target-view visibility</li><li>first-episode candidates 中命中 B1 cluster</li><li>本例 target $target → chunk $selected</li></ul></div>
    <div class="panel"><h3 class="warn">△ 尚未完全成立</h3><ul class="clean"><li>unrestricted all-history exact-chunk retrieval</li><li>cluster 内唯一 chunk 的高置信区分</li><li>open-domain / dynamic scene generalization</li><li>最新 identity ladder 仍固定 chunk 11</li></ul></div>
  </div>
  <p class="callout" style="margin-top:2vw"><b>一句话：</b>Geometry 已经能回答“去哪个历史区域找 memory”；下一步只需验证自动选择的 cluster representative 能否复现 Manual KV 的生成趋势。</p>
</section>

</main><nav class="controls"><button id="prev">←</button><button id="next">→</button><button id="fullscreen">全屏</button><div class="slide-nav" id="dots"></div><div class="progress"><span id="bar"></span></div><div class="counter" id="counter"></div></nav>
<script>
const slides=[...document.querySelectorAll('.slide')];let current=0;const dots=document.getElementById('dots');
slides.forEach((slide,index)=>{const button=document.createElement('button');button.title=(index+1)+' · '+slide.dataset.title;button.onclick=()=>show(index);dots.appendChild(button)});
function show(index){slides[current].classList.remove('active');current=(index+slides.length)%slides.length;slides[current].classList.add('active');[...dots.children].forEach((x,i)=>x.classList.toggle('active',i===current));document.getElementById('bar').style.width=((current+1)/slides.length*100)+'%';document.getElementById('counter').textContent=(current+1)+' / '+slides.length;slides.forEach((slide,i)=>{if(i!==current)slide.querySelectorAll('video').forEach(v=>v.pause())})}
document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.getElementById('fullscreen').onclick=()=>document.documentElement.requestFullscreen();
document.addEventListener('keydown',event=>{if(['ArrowRight','PageDown',' '].includes(event.key)){event.preventDefault();show(current+1)}if(['ArrowLeft','PageUp'].includes(event.key)){event.preventDefault();show(current-1)}if(event.key==='Home')show(0);if(event.key==='End')show(slides.length-1)});
const requested=Number.parseInt(new URLSearchParams(location.search).get('slide')||'1',10)-1;show(Number.isFinite(requested)?Math.max(0,Math.min(slides.length-1,requested)):0);
</script></body></html>"""
    )
    values = {
        "selected": selected,
        "selected4": f"{selected:04d}",
        "target": target,
        "positive_text": positive_text,
        "keyframes": keyframes,
        "cut3r_frames": int(cut3r["frames"]),
        "accepted_ratio": f"{100.0 * float(cut3r['accepted_point_ratio']):.1f}%",
        "num_cells": f"{int(surfel['num_cells']):,}",
        "stable_cells": f"{int(surfel['stable_cells']):,}",
        "stable_fraction": f"{100.0 * float(surfel['stable_cell_fraction']):.1f}%",
        "reprojection_p95": f"{float(surfel['reprojection_residual_p95_pixels']):.1f}",
        "score_rows": score_rows,
        "top3_cards": top3_cards,
        "coverage": f"{100.0 * float(retrieval['coverage_fraction']):.1f}%",
        "margin": f"{float(retrieval['top1_margin']):.3f}",
        "entropy": f"{float(retrieval['normalized_entropy']):.3f}",
        "gap": int(top3[0]["temporal_gap_chunks"]),
    }
    presentation = template.substitute(values)
    output = root / "presentation.html"
    output.write_text(presentation, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the MapKV CUT3R/surfel addressing HTML presentation"
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(build_presentation(args.root))


if __name__ == "__main__":
    main()


__all__ = ["build_presentation"]
