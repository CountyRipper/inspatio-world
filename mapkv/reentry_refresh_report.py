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
    "baseline": "R0 原始 InSpatio",
    "current_continuous": "R1 旧方案：首次/回访都持续读",
    "one_shot": "R2 旧失败对照：整组 One-shot",
    "episode_continuous": "R3 本次成功：Re-entry Episode 持续刷新",
    "per_surface_ttl": "P1 本次：Per-surface TTL=2",
    "same_surface_adaptive": "V1 本次：Same-surface 自适应 observation",
    "edge_safe": "E1 本次：固定 chunk-11 Edge-safe",
    "final_step": "E2 本次：E1 + memory steps012",
}


def _snapshot(metrics: dict) -> ArchitectureSnapshot:
    return ArchitectureSnapshot(
        name="MapKV Re-entry Continuous Refresh",
        focus_zh=(
            "首次只写；真缺席后整段持续刷新；再测试 per-surface TTL "
            "和 same-surface observation"
        ),
        focus_en=(
            "Episode-level continuous refresh, per-surface TTL, "
            "same-surface observation selection"
        ),
        nodes=(
            node(
                id="input",
                label_zh="受控完整回访",
                label_en="0→45→-20→35 known c2w",
                role="input",
                column=0,
                row=0,
                summary="静态 source、同 seed/noise/checkpoint、canonical identity=chunk 11。",
            ),
            node(
                id="geometry",
                label_zh="冻结 RGB Surfel Memory",
                label_en="known-pose CUT3R / chunk-11 anchor",
                role="geometry",
                column=1,
                row=0,
                summary="CUT3R、surfel fusion、retrieval 参数均未改变。",
            ),
            node(
                id="lifecycle",
                label_zh="本次核心：Re-entry Episode",
                label_en="FIRST_VISIBILITY→ABSENT→REENTRY_ACTIVE",
                role="address",
                column=2,
                row=0,
                summary="首次 write-only；缺席 2 blocks 后，在整个回访可见 episode 持续读。",
                change_type="modified",
                focus=True,
                files=("mapkv/reentry_memory.py", "mapkv/reentry_wre.py"),
            ),
            node(
                id="surface",
                label_zh="Per-surface TTL 对照",
                label_en="independent TTL=2 refresh windows",
                role="address",
                column=3,
                row=0,
                summary="后进入 surfel 独立激活；仅作为机制 ablation。",
                change_type="added",
                files=("mapkv/reentry_memory.py",),
            ),
            node(
                id="source",
                label_zh="Same-surface Observation",
                label_en="candidate must share anchor surfel IDs",
                role="retrieval",
                column=4,
                row=0,
                summary="评分只看 chunk-11 同一组 surface，episode 内 source 锁定。",
                change_type="modified",
                files=("mapkv/reentry_wre.py",),
            ),
            node(
                id="warp",
                label_zh="固定 Quality Path",
                label_en="RGB warp→Wan VAE→Virtual Recent",
                role="payload",
                column=5,
                row=0,
                summary="source protection、raw last_pred fallback 与 alpha=1 保持不变。",
            ),
            node(
                id="writer",
                label_zh="原生 Recent Writer",
                label_en="native timestep-0 clean writer",
                role="context",
                column=6,
                row=0,
                summary="生成合法 target-aligned Recent K/V；runtime cache 不被 auxiliary writer 改写。",
            ),
            node(
                id="attention",
                label_zh="Masked Recent Correction",
                label_en="M_query × (A_virtual − A_base)",
                role="attention",
                column=7,
                row=0,
                summary="source-valid 区域保持 exact base attention。",
            ),
            node(
                id="generation",
                label_zh="冻结 InSpatio 4-step",
                label_en="all 30 layers / normal noise",
                role="generation",
                column=8,
                row=0,
                summary="不训练、不改 Ref cache、不改 CUT3R。",
            ),
            node(
                id="evaluation",
                label_zh="统一 Chunk-11 评价",
                label_en="identity / source / leave / re-entry / edge",
                role="evaluation",
                column=10,
                row=0,
                summary="即使自适应选了其它 chunk，ground truth 仍固定 chunk 11。",
                change_type="modified",
                files=("mapkv/reentry_refresh_evaluation.py",),
            ),
            node(
                id="output",
                label_zh="完整回访视频",
                label_en="decoded B1→leave→return→B2",
                role="output",
                column=9,
                row=0,
                summary="memory 仅作为 context，不直接覆盖当前输出。",
            ),
        ),
        edges=tuple(
            ArchitectureEdge(left, right)
            for left, right in zip(
                (
                    "input",
                    "geometry",
                    "lifecycle",
                    "surface",
                    "source",
                    "warp",
                    "writer",
                    "attention",
                    "generation",
                    "output",
                ),
                (
                    "geometry",
                    "lifecycle",
                    "surface",
                    "source",
                    "warp",
                    "writer",
                    "attention",
                    "generation",
                    "output",
                    "evaluation",
                ),
            )
        ),
        changes=(
            ArchitectureChange(
                component_id="lifecycle",
                change_type="modified",
                before="整组首次 re-entry 只读一个 block，随后立即 SERVED",
                after="首次 episode write-only；真缺席后整个 REENTRY_ACTIVE episode 持续刷新",
                affected_files=("mapkv/reentry_memory.py", "mapkv/reentry_wre.py"),
                rationale="one-shot 只保留 2.4% memory gain，无法把 identity 交给 native Recent。",
            ),
            ArchitectureChange(
                component_id="surface",
                change_type="added",
                before="整组统一 read lifecycle",
                after="每个 surfel 缺席/重新进入后获得独立 TTL=2 refresh window",
                affected_files=("mapkv/reentry_memory.py", "mapkv/reentry_wre.py"),
                rationale="检验后进入画面的 surface 是否可更局部地恢复。",
            ),
            ArchitectureChange(
                component_id="source",
                change_type="modified",
                before="自适应候选可来自不相关 generated-only surfels",
                after="候选和评分只允许实际共享 chunk-11 anchor surfel IDs",
                affected_files=("mapkv/reentry_wre.py",),
                rationale="避免 pose/coverage 相近但不是同一 identity surface 的 observation。",
            ),
            ArchitectureChange(
                component_id="evaluation",
                change_type="modified",
                before="adaptive method 随 selected chunk 更换 identity reference",
                after="所有方法统一使用 canonical chunk 11 warp/mask",
                affected_files=("mapkv/reentry_refresh_evaluation.py",),
                rationale="防止自适应选源通过改变 ground truth 获得虚假改善。",
            ),
        ),
    )


def _controls(group: str) -> str:
    return (
        f"<button onclick=\"playGroup('{group}')\">同步播放</button>"
        f"<button onclick=\"pauseGroup('{group}')\">全部暂停</button>"
        f"<button onclick=\"resetGroup('{group}')\">全部复位</button>"
    )


def _videos(prefix: str, group: str, methods: list[str]) -> str:
    return "<div class='videos'>" + "".join(
        "<figure><figcaption>"
        + html.escape(LABELS[method])
        + "</figcaption><video controls preload='metadata' data-group='"
        + group
        + "' src='videos/report/"
        + prefix
        + method
        + ".mp4'></video></figure>"
        for method in methods
    ) + "</div>"


def _metric_table(metrics: dict) -> str:
    rows = []
    for method in LABELS:
        if method not in metrics["methods"]:
            continue
        value = metrics["methods"][method]
        rows.append(
            "<tr><td>"
            + html.escape(LABELS[method])
            + "</td>"
            + f"<td>{value['canonical_revisit_region_l1']:.5f}</td>"
            + f"<td>{value['source_region_delta_vs_baseline_l1']:.5f}</td>"
            + f"<td>{value['leave_window_peak_l1']:.5f}</td>"
            + f"<td>{value['reentry_window_peak_l1']:.5f}</td>"
            + f"<td>{value['right_edge_revisit_l1']:.5f}</td>"
            + f"<td>{value['active_read_chunks']}</td>"
            + f"<td>{100*value['mean_active_read_coverage']:.1f}%</td></tr>"
        )
    return (
        "<table><tr><th>Method</th><th>Chunk11→B2 revisit ↓</th>"
        "<th>Source Δ ↓</th><th>首次离开 Peak ↓</th>"
        "<th>Re-entry Peak ↓</th><th>右侧 revisit ↓</th>"
        "<th>Active chunks</th><th>平均 read coverage</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _timeline_table(metrics: dict) -> str:
    rows = []
    for item in metrics["lifecycle_timeline"]:
        rows.append(
            "<tr>"
            f"<td>{item['chunk']}</td><td>{item['yaw_degrees']:.1f}°</td>"
            f"<td>{100*item['historical_visibility_fraction']:.1f}%</td>"
            f"<td>{item['state_before']} → {item['state_after']}</td>"
            f"<td>{item['active_refresh_surface_count']}</td>"
            f"<td>{item['selected_source_chunk'] or '—'}</td>"
            f"<td>{100*item['read_coverage_fraction']:.1f}%</td></tr>"
        )
    return (
        "<details><summary>展开逐 chunk lifecycle</summary><table>"
        "<tr><th>chunk</th><th>yaw</th><th>anchor visible</th>"
        "<th>state</th><th>active surfels</th><th>source</th>"
        "<th>read coverage</th></tr>"
        + "".join(rows)
        + "</table></details>"
    )


def build_report(run_root: str | Path) -> Path:
    root = Path(run_root).resolve()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    snapshot = _snapshot(metrics)
    write_architecture_bundle(root, snapshot)
    available = [method for method in LABELS if method in metrics["methods"]]
    decisions = metrics["decisions"]
    styles = """
body{font-family:system-ui,'Noto Sans SC',sans-serif;margin:0;background:#f4f6fa;color:#172033}
main{max-width:1550px;margin:auto;padding:28px}section{background:white;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 3px 16px #17203312}
h1,h2,h3{margin-top:.3em}.focus{border-left:6px solid #6b46c1;background:#f5f0ff;padding:12px 16px}.status{display:inline-block;padding:8px 13px;border-radius:99px;background:#e9f7ee;color:#17633a;font-weight:750}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d9e0ea;padding:8px;text-align:left}th{background:#eef2f7}
.images,.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(265px,1fr));gap:13px}figure{margin:0;background:#f7f9fc;padding:10px;border-radius:10px}figcaption{font-weight:650;margin-bottom:8px}
img,video{width:100%;max-height:530px;object-fit:contain;background:#0c111b;border-radius:8px}.architecture{overflow-x:auto}.architecture img{min-width:1500px;max-height:none}
button{padding:8px 14px;margin:4px;border:0;border-radius:7px;background:#2d5bd1;color:white;cursor:pointer}
"""
    script = """
function vids(g){return Array.from(document.querySelectorAll('video[data-group="'+g+'"]'))}
function playGroup(g){let v=vids(g);if(!v.length)return;let t=v[0].currentTime;v.forEach(x=>{x.currentTime=t;x.play()})}
function pauseGroup(g){vids(g).forEach(x=>x.pause())}
function resetGroup(g){vids(g).forEach(x=>{x.pause();x.currentTime=0})}
"""
    right_edge = "".join(
        "<figure><figcaption>"
        + html.escape(
            "Canonical B1 warp"
            if method == "canonical_b1_warped"
            else LABELS[method]
        )
        + "</figcaption><img src='assets/reentry_refresh/right_edge/"
        + method
        + ".png'></figure>"
        for method in ["canonical_b1_warped", *available]
    )
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>MapKV Re-entry Refresh</title>
<style>{styles}</style></head><body><main>
<h1>MapKV Re-entry Refresh</h1><span class='status'>{html.escape(metrics['status'])}</span>
<p class='focus'><b>本次最新 Method / Focus：</b>
R3 Re-entry Episode Continuous Refresh。首次 B1 episode 只写不读；anchor surfaces
缺席 ≥2 blocks 后，在完整 re-entry episode 内用固定 chunk 11 持续刷新。
同时测试但不自动采用：per-surface TTL=2、same-surface view-adaptive、edge-safe、steps012。</p>

<section><h2>A. 完整 Pipeline / Framework 与架构变更</h2>
<div class='architecture'><img src='assets/architecture_graph.svg'></div>
{render_pipeline_table_html(snapshot)}<h3>Architecture Changes</h3>
{render_changes_html(snapshot)}</section>

<section><h2>B. Geometry、Canonical Identity 与 Lifecycle</h2><div class='images'>
<figure><figcaption>真实 RGB surfel splats</figcaption><img src='surfel_rgb_options/A_rgb_world_splats.png'></figure>
<figure><figcaption>真实 RGB oriented disks</figcaption><img src='surfel_rgb_options/B_rgb_oriented_disks.png'></figure>
<figure><figcaption>Canonical B1 chunk 11</figcaption><img src='assets/reentry_refresh/canonical_b1_chunk11.png'></figure>
<figure><figcaption>Chunk 11 warp 到 B2</figcaption><img src='assets/reentry_refresh/canonical_b1_warped_to_b2.png'></figure>
<figure><figcaption>Canonical revisit mask</figcaption><img src='assets/reentry_refresh/M_canonical_revisit_eval.png'></figure>
<figure><figcaption>Lifecycle timeline（{metrics['lifecycle_timeline_method']}）</figcaption><img src='assets/reentry_refresh/lifecycle_timeline.png'></figure>
</div>{_timeline_table(metrics)}</section>

<section><h2>C. 完整回访同步视频（B1→首次离开→Leave→Return→B2）</h2>
<p>{_controls('full')}</p>{_videos('full_revisit_', 'full', available)}</section>

<section><h2>D. 首次离开窗口</h2>
<p>{_controls('leave')}</p>{_videos('departure_', 'leave', available)}</section>

<section><h2>E. 真正 Re-entry 窗口</h2>
<p>{_controls('reentry')}</p>{_videos('reentry_', 'reentry', available)}</section>

<section><h2>F. B2 Canonical Identity / 右侧重叠</h2>
<div class='images'>{right_edge}</div></section>

<section><h2>G. Core Metrics（所有方法统一 chunk 11 ground truth）</h2>
{_metric_table(metrics)}
<p>Priority 1 episode-continuous：<b>{decisions['priority1_episode_continuous_works']}</b>，
memory gain retention={100*decisions['priority1_memory_gain_retention_ratio']:.1f}%。</p>
<p>Priority 2 per-surface TTL=2：<b>{decisions['priority2_per_surface_ttl_works']}</b>；
Priority 3 same-surface adaptive：<b>{decisions['priority3_same_surface_adaptive_works']}</b>；
Edge-safe：<b>{decisions['edge_safe_support_works']}</b>；
steps012：<b>{decisions['final_step_stabilization_useful']}</b>。</p></section>

<section><h2>H. Final Interpretation</h2>
<p><b>保留的 lifecycle：</b>固定 chunk 11 的 Re-entry Episode Continuous Refresh。</p>
<p><b>Per-surface TTL：</b>若指标为 false，则 2-block handoff 仍不足以维持 frozen student identity。</p>
<p><b>Same-surface observation：</b>只有同时保持 canonical chunk-11 identity
并降低右侧重叠时才采用；不会因改换 ground truth 获得虚假改善。</p>
<p><b>下一步唯一任务：</b>基于本报告的最佳 method 决定是保留 episode 持续刷新，
还是采用 edge-safe / steps012；不进入 Canonical-K、CUT3R tuning 或训练。</p></section>
</main><script>{script}</script></body></html>"""
    report = root / "report.html"
    report.write_text(document, encoding="utf-8")

    lines = [
        "# MapKV Re-entry Refresh",
        "",
        f"- Status: {metrics['status']}",
        "- 本次 Focus: first-visit write-only → true re-entry episode continuous refresh",
        f"- Canonical identity reference: chunk {metrics['trajectory']['anchor_chunk']}",
        "",
        "## Metrics",
        "",
        "| Method | Chunk11→B2 ↓ | Source Δ ↓ | Leave peak ↓ | Re-entry peak ↓ | Right edge ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in available:
        value = metrics["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {value['canonical_revisit_region_l1']:.5f} | "
            f"{value['source_region_delta_vs_baseline_l1']:.5f} | "
            f"{value['leave_window_peak_l1']:.5f} | "
            f"{value['reentry_window_peak_l1']:.5f} | "
            f"{value['right_edge_revisit_l1']:.5f} |"
        )
    lines += [
        "",
        "## Decisions",
        "",
        f"- Episode continuous: {decisions['priority1_episode_continuous_works']}",
        f"- Per-surface TTL=2: {decisions['priority2_per_surface_ttl_works']}",
        f"- Same-surface adaptive: {decisions['priority3_same_surface_adaptive_works']}",
        f"- Edge-safe: {decisions['edge_safe_support_works']}",
        f"- Steps012: {decisions['final_step_stabilization_useful']}",
        "",
        "## Complete videos",
        "",
    ]
    lines += [
        f"- {LABELS[method]}: videos/report/full_revisit_{method}.mp4"
        for method in available
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


__all__ = ["LABELS", "build_report"]
