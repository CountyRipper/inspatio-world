from __future__ import annotations

import html
import json
from pathlib import Path

from .report_framework import (
    ArchitectureChange,
    ArchitectureEdge,
    ArchitectureSnapshot,
    node,
    render_changes_html,
    render_pipeline_table_html,
    write_architecture_bundle,
)


LABELS = {
    "baseline": "原始 InSpatio（Baseline）",
    "current_rgb_wre": "当前未保护 RGB-Warp WRE（S1）",
    "source_protected": "本次主方法：Source-Protected RGB-Warp WRE（S2）",
    "middle10": "本次轻量候选：Source-Protected Middle10（S3）",
}


def _snapshot(metrics: dict) -> ArchitectureSnapshot:
    return ArchitectureSnapshot(
        name="MapKV Source-Protected Generated-History Memory",
        focus_zh="source/reference 保护与真正 generated-history 回访身份",
        focus_en=(
            "Generated-only B1 observations × current reference blindness "
            "control RGB-Warp→VAE Recent correction"
        ),
        nodes=(
            node(
                id="input",
                label_zh="强受控静态轨迹",
                label_en="0→45→-20→35 exact yaw",
                role="input",
                column=0,
                row=0,
                summary="静态 source、known c2w、同 seed/noise/render，B1=45°、B2=35°。",
                change_type="modified",
                files=("mapkv_proto/trajectory_builder.py",),
            ),
            node(
                id="generation",
                label_zh="原始 InSpatio 生成",
                label_en="Frozen InSpatio-World 1.3B",
                role="generation",
                column=1,
                row=0,
                summary="原始 VAE、4-step、bf16、三帧 block；模型参数全部冻结。",
                files=("pipeline/causal_inference.py",),
            ),
            node(
                id="geometry",
                label_zh="固定 pose CUT3R",
                label_en="Known-pose causal CUT3R prefix",
                role="geometry",
                column=2,
                row=0,
                summary="仅 target 前历史帧；CUT3R 供 depth/pointmap/confidence，map pose 来自控制轨迹。",
                files=("mapkv/cut3r_adapter.py",),
            ),
            node(
                id="address",
                label_zh="generated-only Surfel 地址",
                label_en="reference_blind_at_write observations",
                role="address",
                column=3,
                row=0,
                summary="B1 observation 写入时记录 source blind；查询先过滤 generated-only 再 z-buffer。",
                change_type="modified",
                files=("mapkv/surfel_index.py", "mapkv/warp_reencode.py"),
            ),
            node(
                id="payload",
                label_zh="RGB-Warp 质量路径",
                label_en="RGB warp → Wan VAE",
                role="payload",
                column=4,
                row=0,
                summary="B1 generated RGB 按 known camera warp 到当前 view，再由原生 Wan VAE 编码。",
                files=("mapkv/warp_reencode.py",),
            ),
            node(
                id="context",
                label_zh="Source-Protected Virtual Recent",
                label_en="M_need-composed Virtual Recent",
                role="context",
                column=5,
                row=0,
                summary="M_need=M_history×(1-M_ref_protected)；其内用历史，外部保留 raw last_pred。",
                change_type="modified",
                focus=True,
                files=("mapkv/warp_reencode.py",),
            ),
            node(
                id="attention",
                label_zh="受保护 Recent 修正",
                label_en="M_query-gated replace_recent_delta",
                role="attention",
                column=6,
                row=0,
                summary="source-valid query 的 gate 严格为零；memory interior 强修正，边界轻 feather。",
                change_type="modified",
                focus=True,
                files=(
                    "mapkv_proto/memory_context.py",
                    "inference_mapkv_proto.py",
                ),
            ),
            node(
                id="output",
                label_zh="完整回访视频",
                label_en="B1 → leave → return → B2",
                role="output",
                column=7,
                row=0,
                summary="Baseline / Current / Source-Protected / Middle10 完整同步视频。",
                files=("mapkv/source_protected_stage.py",),
            ),
            node(
                id="evaluation",
                label_zh="分区回访评估",
                label_en="source region vs true revisit region",
                role="evaluation",
                column=8,
                row=0,
                summary="source 稳定性与 generated-history revisit 分开度量，自动选择最大 memory regions。",
                change_type="added",
                files=("mapkv/source_protected_evaluation.py",),
            ),
        ),
        edges=tuple(
            ArchitectureEdge(left, right)
            for left, right in zip(
                (
                    "input",
                    "generation",
                    "geometry",
                    "address",
                    "payload",
                    "context",
                    "attention",
                    "output",
                ),
                (
                    "generation",
                    "geometry",
                    "address",
                    "payload",
                    "context",
                    "attention",
                    "output",
                    "evaluation",
                ),
            )
        ),
        changes=(
            ArchitectureChange(
                component_id="input",
                change_type="modified",
                before="0→30→0→20 partial-overlap trajectory",
                after="0→45→-20→35 memory-required exact-yaw trajectory",
                affected_files=("mapkv_proto/trajectory_builder.py",),
                rationale="扩大 source 视场外生成区域，并在 changed-view B2 重新观察。",
            ),
            ArchitectureChange(
                component_id="address",
                change_type="modified",
                before="B1 所有 surfel observations 均可成为长期 memory",
                after="observation 写入时记录 reference_blind_at_write；仅 generated-only B1 surfels 可用",
                affected_files=("mapkv/surfel_index.py", "mapkv/warp_reencode.py"),
                rationale="长期记忆只表示模型自己生成、而非 source 直接提供的世界内容。",
            ),
            ArchitectureChange(
                component_id="context",
                change_type="modified",
                before="M_history 直接构造 RGB-Warp Virtual Recent",
                after="M_need=M_history×(1-M_ref_protected) 后才构造 Virtual Recent",
                affected_files=("mapkv/warp_reencode.py",),
                rationale="禁止历史 context 与可靠 source/reference 竞争。",
            ),
            ArchitectureChange(
                component_id="attention",
                change_type="modified",
                before="query gate 可 feather 回 source-valid token",
                after="support-preserving query gate 最后被 conservative source-protection token mask 截断",
                affected_files=(
                    "mapkv_proto/memory_context.py",
                    "inference_mapkv_proto.py",
                ),
                rationale="protected source 区域保持 exact baseline attention path。",
            ),
            ArchitectureChange(
                component_id="evaluation",
                change_type="added",
                before="固定中央物体或一般 overlap 指标",
                after="source region 与 true generated-history revisit region 分区，并自动生成 connected-region crops",
                affected_files=("mapkv/source_protected_evaluation.py",),
                rationale="避免 source-supported 内容掩盖真实回访记忆结论。",
            ),
        ),
        metadata={
            "status": metrics["status"],
            "payload": "RGB-Warp→Wan VAE→native Recent writer",
            "canonical_k": "paused",
            "source_chunk": metrics["trajectory"]["source_chunk"],
            "target_chunk": metrics["trajectory"]["target_chunk"],
        },
    )


def _metric_table(metrics: dict) -> str:
    rows = []
    for method in LABELS:
        value = metrics["methods"][method]
        rows.append(
            "<tr>"
            f"<td>{html.escape(LABELS[method])}</td>"
            f"<td>{value['revisit_region_b1_to_b2_l1']:.5f}</td>"
            f"<td>{value['source_region_delta_vs_baseline_l1']:.5f}</td>"
            f"<td>{value['reentry_peak_l1']:.5f}</td>"
            "</tr>"
        )
    return (
        "<table><tr><th>方法</th><th>True revisit B1→B2 ↓</th>"
        "<th>Source-region Δ vs Baseline ↓</th><th>Re-entry peak ↓</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _videos(prefix: str) -> str:
    cards = []
    for method in LABELS:
        cards.append(
            "<figure><figcaption>"
            + html.escape(LABELS[method])
            + "</figcaption><video controls preload='metadata' src='videos/report/"
            + prefix
            + method
            + ".mp4'></video></figure>"
        )
    return "<div class='videos'>" + "".join(cards) + "</div>"


def _crops(metrics: dict) -> str:
    boxes = metrics["automatic_revisit_regions"]["boxes_xyxy"]
    if not boxes:
        return "<p>没有足够大的 connected revisit region。</p>"
    method_order = (
        "b1_warped",
        "baseline",
        "current_rgb_wre",
        "source_protected",
        "middle10",
    )
    labels = {
        "b1_warped": "B1 warp 到 B2（目标）",
        **LABELS,
    }
    blocks = []
    for region in range(1, len(boxes) + 1):
        figures = "".join(
            "<figure><figcaption>"
            + html.escape(labels[method])
            + "</figcaption><img src='assets/source_protected/revisit_regions/"
            + f"region_{region}_{method}.png'></figure>"
            for method in method_order
        )
        blocks.append(
            f"<h3>自动历史回访区域 {region}</h3><div class='images'>{figures}</div>"
        )
    return "".join(blocks)


def build_report(run_root: str | Path) -> Path:
    root = Path(run_root).resolve()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    snapshot = _snapshot(metrics)
    write_architecture_bundle(root, snapshot)
    conclusion = (
        "Source protection 与 true revisit recovery 同时达到判据。"
        if metrics["status"] == "SOURCE_PROTECTED_REVISIT_WORKS"
        else (
            "Source interference 已被隔离，但 frozen training-free path 尚未稳定保留"
            "真正 generated-history instance identity。"
        )
    )
    source_residual = metrics["methods"]["source_protected"][
        "source_region_delta_vs_baseline_l1"
    ]
    source_reduction = 100.0 * metrics["source_protection"][
        "source_region_gain_vs_current"
    ] / max(
        metrics["methods"]["current_rgb_wre"][
            "source_region_delta_vs_baseline_l1"
        ],
        1e-12,
    )
    styles = """
body{font-family:system-ui,'Noto Sans SC',sans-serif;margin:0;background:#f5f7fb;color:#172033}
main{max-width:1480px;margin:auto;padding:28px}section{background:white;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 3px 16px #18203312}
h1,h2,h3{margin-top:.3em}.focus{border-left:6px solid #6b46c1;background:#f5f0ff;padding:12px 16px}
.status{display:inline-block;padding:8px 13px;border-radius:99px;background:#e9f7ee;color:#17633a;font-weight:750}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d9e0ea;padding:9px;text-align:left}th{background:#eef2f7}
.images,.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
figure{margin:0;background:#f7f9fc;padding:10px;border-radius:10px}figcaption{font-weight:650;margin-bottom:8px}
img,video{width:100%;max-height:520px;object-fit:contain;background:#0c111b;border-radius:8px}
.architecture{overflow-x:auto}.architecture img{min-width:1450px;max-height:none}
button{padding:8px 14px;margin:4px;border:0;border-radius:7px;background:#2d5bd1;color:white;cursor:pointer}
code{background:#eef2f7;padding:2px 5px;border-radius:4px}
"""
    script = """
function allVideos(){return Array.from(document.querySelectorAll('video'))}
function playAll(){allVideos().forEach(v=>{v.currentTime=allVideos()[0].currentTime;v.play()})}
function pauseAll(){allVideos().forEach(v=>v.pause())}
function resetAll(){allVideos().forEach(v=>{v.pause();v.currentTime=0})}
"""
    masks = (
        "B1_reference_blind",
        "B1_generated_only_surfels",
        "M_ref_valid",
        "M_history",
        "M_need",
        "M_query",
    )
    mask_labels = {
        "B1_reference_blind": "B1 reference-blind",
        "B1_generated_only_surfels": "B1 generated-only surfels",
        "M_ref_valid": "B2 current reference-valid",
        "M_history": "B2 generated-history visibility",
        "M_need": "本次核心 M_need",
        "M_query": "source-protected M_query",
    }
    mask_html = "".join(
        f"<figure><figcaption>{mask_labels[name]}</figcaption>"
        f"<img src='assets/source_protected/{name}.png'></figure>"
        for name in masks
    )
    document = f"""<!doctype html><html lang='zh-CN'><head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>MapKV Source-Protected Revisit</title><style>{styles}</style></head>
<body><main><h1>MapKV Source-Protected Revisit Memory</h1>
<p class='status'>{html.escape(metrics['status'])}</p>
<p class='focus'><b>本次最新方法 / Focus：</b>Source-Protected RGB-Warp WRE。
长期 memory 只接受 <code>B1 generated-only × current reference-blind</code>；
source-valid query 保持 exact baseline attention。</p>
<p>{html.escape(conclusion)}</p>

<section><h2>A. 完整 Pipeline 与本次架构变化</h2>
<div class='architecture'><img src='assets/architecture_graph.svg'></div>
{render_pipeline_table_html(snapshot)}
<h3>Architecture Changes</h3>{render_changes_html(snapshot)}</section>

<section><h2>B. Benchmark 与 Masks</h2>
<p>精确 pure-yaw：0°→+45° (B1)→−20° (Leave)→+35° (B2)；
source chunk {metrics['trajectory']['source_chunk']}，target chunk
{metrics['trajectory']['target_chunk']}，间隔
{metrics['trajectory']['history_gap_chunks']} chunks。</p>
<div class='images'>
<figure><figcaption>B1 第一次访问</figcaption><img src='assets/source_protected/b1_first_visit.png'></figure>
<figure><figcaption>B1 camera-warp 到 B2</figcaption><img src='assets/source_protected/b1_warped_to_b2.png'></figure>
<figure><figcaption>B2 Baseline</figcaption><img src='assets/source_protected/b2_baseline.png'></figure>
</div><div class='images'>{mask_html}</div></section>

<section><h2>C. Real-RGB Surfel（主视图）</h2><div class='images'>
<figure><figcaption>真实历史 RGB world splats</figcaption><img src='surfel_rgb_options/A_rgb_world_splats.png'></figure>
<figure><figcaption>真实历史 RGB oriented disks</figcaption><img src='surfel_rgb_options/B_rgb_oriented_disks.png'></figure>
<figure><figcaption>B1 generated-only 在 B2 的 RGB z-buffer</figcaption><img src='surfel_rgb_options/D_rgb_b1_target_zbuffer.png'></figure>
<figure><figcaption>B1/B2 support overlay</figcaption><img src='surfel_rgb_options/E_rgb_b1_target_overlay.png'></figure>
</div></section>

<section><h2>D. 自动 True-Revisit Identity Regions</h2>{_crops(metrics)}</section>

<section><h2>E. 完整回访同步视频（B1→离开→返回→B2）</h2>
<p><button onclick='playAll()'>全部播放</button><button onclick='pauseAll()'>全部暂停</button>
<button onclick='resetAll()'>全部重置</button></p>{_videos('full_revisit_')}</section>

<section><h2>F. Re-entry Window</h2>
<figure><figcaption>同帧回程 strip：Baseline / 未保护 / Source-Protected / Middle10</figcaption>
<img src='assets/source_protected/reentry_review_small.jpg'></figure>
{_videos('reentry_')}</section>

<section><h2>G. Core Metrics</h2>{_metric_table(metrics)}
<p>Source 区域占比 {metrics['masks']['source_fraction']:.3f}；true revisit 区域占比
{metrics['masks']['revisit_fraction']:.3f}；protected source 上 query gate 最大值
{metrics['masks']['protected_query_gate_max']:.1f}。</p>
<p><b>限制：</b>当前 block 的 protected source query 确实走 exact baseline
attention，但更早 block 的 memory 会通过后续 raw <code>last_pred</code> 因果传播，
所以最终 source-region Δ 不是逐像素零：{source_residual:.5f}。相对未保护 WRE
仍降低 {source_reduction:.1f}% 。</p></section>

<section><h2>H. 最终解释</h2><p><b>{html.escape(metrics['status'])}</b></p>
<p>{html.escape(conclusion)}</p>
<p><b>下一步唯一优先实验：</b>
{('在另一静态室内场景复现 generated-history identity。' if metrics['status']=='SOURCE_PROTECTED_REVISIT_WORKS' else '固定当前 source protection，评估最小 memory-aware adapter 是否能稳定 generated-history instance identity。')}</p>
</section></main><script>{script}</script></body></html>"""
    report = root / "report.html"
    report.write_text(document, encoding="utf-8")
    lines = [
        "# MapKV Source-Protected Revisit Report",
        "",
        f"- Status: **{metrics['status']}**",
        "- 本次 Focus: generated-only history × current reference-blind",
        "- Payload: RGB-Warp → Wan VAE → native Recent writer",
        "- Injection: source-protected query-gated replace_recent_delta",
        "- Canonical-K: paused",
        "",
        "## Core metrics",
        "",
        "| Method | Revisit ↓ | Source Δ ↓ | Re-entry peak ↓ |",
        "|---|---:|---:|---:|",
    ]
    for method in LABELS:
        value = metrics["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {value['revisit_region_b1_to_b2_l1']:.5f} | "
            f"{value['source_region_delta_vs_baseline_l1']:.5f} | "
            f"{value['reentry_peak_l1']:.5f} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Videos",
        "",
    ]
    lines += [
        f"- {LABELS[method]}: videos/report/full_revisit_{method}.mp4"
        for method in LABELS
    ]
    (root / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["build_report"]
