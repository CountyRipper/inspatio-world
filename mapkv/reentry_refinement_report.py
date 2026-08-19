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
    "baseline": "E0 原始 InSpatio",
    "current_source_protected": "E1 当前持续读取 Source-Protected WRE",
    "reentry_only": "E2 本次：仅 Re-entry 读取（固定 +45°）",
    "view_adaptive": "E3 本次：视角自适应 Re-entry",
    "edge_safe": "E4 本次主方法：Edge-Safe View-Adaptive",
    "final_step": "E5 本次：E4 + 最后一步回归 Baseline",
}


def _snapshot(metrics: dict) -> ArchitectureSnapshot:
    return ArchitectureSnapshot(
        name="MapKV Re-entry Memory Refinement",
        focus_zh="何时读、读哪个 observation、在哪些 FOV 区域安全读",
        focus_en=(
            "One-shot re-entry lifecycle + stable view-adaptive observation "
            "+ edge-safe RGB-Warp WRE"
        ),
        nodes=(
            node(
                id="input",
                label_zh="强受控回访轨迹",
                label_en="0→45→-20→35 exact yaw",
                role="input",
                column=0,
                row=0,
                summary="静态 source、known c2w、相同 noise/seed/checkpoint。",
            ),
            node(
                id="generation",
                label_zh="冻结 InSpatio 生成",
                label_en="Frozen 1.3B / native 4-step",
                role="generation",
                column=1,
                row=0,
                summary="原始 Ref+Recent+Current 推理与 runtime cache 不变。",
            ),
            node(
                id="geometry",
                label_zh="冻结 Known-pose CUT3R",
                label_en="causal prefix depth/pointmap",
                role="geometry",
                column=2,
                row=0,
                summary="不改 CUT3R、surfel merge、confidence 或 geometry 参数。",
            ),
            node(
                id="address",
                label_zh="Surface Episode Lifecycle",
                label_en="VISIBLE_RECENT→ABSENT→REENTERED→SERVED",
                role="address",
                column=3,
                row=0,
                summary="首次只写；缺席 2 blocks；re-entry 只读一次后 handoff。",
                change_type="added",
                focus=True,
                files=("mapkv/reentry_memory.py", "mapkv/reentry_wre.py"),
            ),
            node(
                id="payload",
                label_zh="稳定历史 Observation",
                label_en="coverage×view×quality×center margin",
                role="payload",
                column=4,
                row=0,
                summary="首次 episode 多视角评分，选中后整个 re-entry episode 锁定。",
                change_type="modified",
                focus=True,
                files=("mapkv/surfel_index.py", "mapkv/reentry_wre.py"),
            ),
            node(
                id="context",
                label_zh="Edge-Safe RGB-Warp WRE",
                label_en="border padding + eroded valid interior",
                role="context",
                column=5,
                row=0,
                summary="只 warp 历史 RGB；raw last_pred fallback；不向 FOV 外 dilation。",
                change_type="modified",
                focus=True,
                files=("mapkv/reentry_wre.py", "mapkv/warp_reencode.py"),
            ),
            node(
                id="attention",
                label_zh="向内 Feather 的 Recent 修正",
                label_en="inward M_query × replace_recent_delta",
                role="attention",
                column=6,
                row=0,
                summary="outside 与 source-protected query 严格为 baseline；边界只向内软化。",
                change_type="modified",
                focus=True,
                files=("mapkv_proto/memory_context.py",),
            ),
            node(
                id="output",
                label_zh="完整回访与双窗口视频",
                label_en="full / departure / re-entry",
                role="output",
                column=7,
                row=0,
                summary="中文方法名、完整 B1→Leave→Return→B2 与同步播放。",
                change_type="added",
                files=("mapkv/reentry_refinement_stage.py",),
            ),
            node(
                id="evaluation",
                label_zh="身份/稳定性分区评估",
                label_en="revisit/source/leave/re-entry/right-edge",
                role="evaluation",
                column=8,
                row=0,
                summary="分别量化 true revisit、source、首次离开、re-entry 与右侧边缘。",
                change_type="added",
                files=("mapkv/reentry_refinement_evaluation.py",),
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
                component_id="address",
                change_type="added",
                before="surfels 只要可见就持续读取长期 memory",
                after="首次只写；缺席 2 blocks 后 re-entry 只读一次，随后 SERVED/handoff",
                affected_files=("mapkv/reentry_memory.py", "mapkv/reentry_wre.py"),
                rationale="消除首次离开时 Recent 与同 episode 长期 memory 的重复 conditioning。",
            ),
            ArchitectureChange(
                component_id="payload",
                change_type="modified",
                before="固定 +45° chunk 11 作为所有回访的 RGB source",
                after="首次 episode 多 observation 评分，并在 re-entry episode 内锁定最佳 source",
                affected_files=("mapkv/surfel_index.py", "mapkv/reentry_wre.py"),
                rationale="降低 +45°→+35° warp 幅度与右侧边缘 view conflict。",
            ),
            ArchitectureChange(
                component_id="context",
                change_type="modified",
                before="zero padding，support 可向 warp-valid 边界外 dilation",
                after="RGB border padding；warp-valid 内缩 3；M_memory 不向安全边界外扩张",
                affected_files=("mapkv/reentry_wre.py", "mapkv/warp_reencode.py"),
                rationale="阻止 VAE 黑边 halo 与 FOV 边缘 residual identity 扩张。",
            ),
            ArchitectureChange(
                component_id="attention",
                change_type="modified",
                before="support-preserving gate 在边界向外 feather",
                after="hard interior 保持 1，只在安全 support 内向内 feather",
                affected_files=("mapkv_proto/memory_context.py",),
                rationale="memory influence 在 safe support 外严格为零。",
            ),
            ArchitectureChange(
                component_id="output",
                change_type="added",
                before="只有完整回访与 re-entry clip",
                after="完整回访 + 首次离开 + re-entry 三组同步视频",
                affected_files=("mapkv/reentry_refinement_stage.py",),
                rationale="直接审查两类 transition，而不是只看 B2。",
            ),
            ArchitectureChange(
                component_id="evaluation",
                change_type="added",
                before="未单独量化 first-departure 和右侧 identity overlap",
                after="leave/re-entry mean+peak 与固定 right-edge identity crop/metric",
                affected_files=("mapkv/reentry_refinement_evaluation.py",),
                rationale="将本轮三个已知 failure mode 分开定位。",
            ),
        ),
        metadata={
            "statuses": metrics["statuses"],
            "anchor_chunk": metrics["trajectory"]["anchor_chunk"],
            "selected_source_chunk": metrics["trajectory"][
                "selected_source_chunk"
            ],
            "read_chunk": metrics["trajectory"]["read_chunk"],
        },
    )


def _metric_table(metrics: dict) -> str:
    rows = []
    for method, label in LABELS.items():
        value = metrics["methods"][method]
        right = value["right_edge_revisit_l1"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>chunk {value['revisit_reference_chunk']}</td>"
            f"<td>{value['revisit_region_b1_to_b2_l1']:.5f}</td>"
            f"<td>{value['source_region_delta_vs_baseline_l1']:.5f}</td>"
            f"<td>{value['leave_window_mean_l1']:.5f}</td>"
            f"<td>{value['leave_window_peak_l1']:.5f}</td>"
            f"<td>{value['reentry_window_mean_l1']:.5f}</td>"
            f"<td>{value['reentry_window_peak_l1']:.5f}</td>"
            f"<td>{'—' if right is None else f'{right:.5f}'}</td>"
            "</tr>"
        )
    return (
        "<table><tr><th>方法</th><th>历史 reference</th><th>True revisit ↓</th>"
        "<th>Source Δ ↓</th><th>Leave mean ↓</th><th>Leave peak ↓</th>"
        "<th>Re-entry mean ↓</th><th>Re-entry peak ↓</th>"
        "<th>Right-edge revisit ↓</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _videos(prefix: str, group: str) -> str:
    cards = []
    for method, label in LABELS.items():
        cards.append(
            "<figure><figcaption>"
            + html.escape(label)
            + "</figcaption><video controls preload='metadata' data-group='"
            + group
            + "' src='videos/report/"
            + prefix
            + method
            + ".mp4'></video></figure>"
        )
    return "<div class='videos'>" + "".join(cards) + "</div>"


def _controls(group: str) -> str:
    return (
        f"<button onclick=\"playGroup('{group}')\">全部播放</button>"
        f"<button onclick=\"pauseGroup('{group}')\">全部暂停</button>"
        f"<button onclick=\"resetGroup('{group}')\">全部复位</button>"
    )


def _selection_table(metrics: dict) -> str:
    rows = []
    for item in metrics["observation_selection"]["edge_safe_top3"]:
        rows.append(
            "<tr>"
            f"<td>{item['chunk_id']}</td><td>{item['score']:.6f}</td>"
            f"<td>{item['visible_coverage']:.4f}</td>"
            f"<td>{item['view_alignment']:.4f}</td>"
            f"<td>{item['observation_quality']:.4f}</td>"
            f"<td>{item['image_center_margin']:.4f}</td>"
            f"<td>{item['rotation_distance_degrees']:.2f}°</td></tr>"
        )
    return (
        "<table><tr><th>chunk</th><th>score</th><th>coverage</th>"
        "<th>view alignment</th><th>quality</th><th>center margin</th>"
        "<th>warp rotation</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _lifecycle_table(metrics: dict) -> str:
    rows = []
    for item in metrics["lifecycle_timeline"]:
        rows.append(
            "<tr>"
            f"<td>{item['chunk']}</td><td>{item['yaw_degrees']:.1f}°</td>"
            f"<td>{100*item['historical_visibility_fraction']:.1f}%</td>"
            f"<td>{item['absence_count']}</td>"
            f"<td>{item['state_before']} → {item['state_after']}</td>"
            f"<td>{item['selected_source_chunk'] or '—'}</td>"
            f"<td>{100*item['read_coverage_fraction']:.1f}%</td></tr>"
        )
    return (
        "<details><summary>展开逐 chunk lifecycle 数据</summary>"
        "<table><tr><th>chunk</th><th>yaw</th><th>history visible</th>"
        "<th>absence</th><th>state</th><th>source</th><th>read</th></tr>"
        + "".join(rows)
        + "</table></details>"
    )


def build_report(run_root: str | Path) -> Path:
    root = Path(run_root).resolve()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    snapshot = _snapshot(metrics)
    write_architecture_bundle(root, snapshot)
    selected = metrics["trajectory"]["selected_source_chunk"]
    read_chunk = metrics["trajectory"]["read_chunk"]
    decision = metrics["decisions"]
    styles = """
body{font-family:system-ui,'Noto Sans SC',sans-serif;margin:0;background:#f4f6fa;color:#172033}
main{max-width:1500px;margin:auto;padding:28px}section{background:white;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 3px 16px #17203312}
h1,h2,h3{margin-top:.3em}.focus{border-left:6px solid #6b46c1;background:#f5f0ff;padding:12px 16px}.status{display:inline-block;padding:8px 13px;border-radius:99px;background:#e9f7ee;color:#17633a;font-weight:750;margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d9e0ea;padding:8px;text-align:left}th{background:#eef2f7}
.images,.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:13px}figure{margin:0;background:#f7f9fc;padding:10px;border-radius:10px}figcaption{font-weight:650;margin-bottom:8px}
img,video{width:100%;max-height:520px;object-fit:contain;background:#0c111b;border-radius:8px}.architecture{overflow-x:auto}.architecture img{min-width:1450px;max-height:none}
button{padding:8px 14px;margin:4px;border:0;border-radius:7px;background:#2d5bd1;color:white;cursor:pointer}code{background:#eef2f7;padding:2px 5px;border-radius:4px}
"""
    script = """
function videos(g){return Array.from(document.querySelectorAll('video[data-group="'+g+'"]'))}
function playGroup(g){let v=videos(g);if(!v.length)return;let t=v[0].currentTime;v.forEach(x=>{x.currentTime=t;x.play()})}
function pauseGroup(g){videos(g).forEach(x=>x.pause())}
function resetGroup(g){videos(g).forEach(x=>{x.pause();x.currentTime=0})}
"""
    edge_methods = (
        "anchor_b1",
        "selected_observation",
        "selected_warped",
        "baseline",
        "current_source_protected",
        "reentry_only",
        "view_adaptive",
        "edge_safe",
        "final_step",
    )
    edge_labels = {
        "anchor_b1": "B1 anchor +45°",
        "selected_observation": f"自适应历史 chunk {selected}",
        "selected_warped": "选中历史 warp 到 B2",
        **LABELS,
    }
    right_edge = "".join(
        "<figure><figcaption>"
        + html.escape(edge_labels[name])
        + "</figcaption><img src='assets/reentry_refinement/right_edge/"
        + name
        + ".png'></figure>"
        for name in edge_methods
    )
    status_html = "".join(
        f"<span class='status'>{html.escape(value)}</span>"
        for value in metrics["statuses"]
    )
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>MapKV Re-entry Refinement</title>
<style>{styles}</style></head><body><main>
<h1>MapKV Re-entry Memory Refinement</h1>{status_html}
<p class='focus'><b>本次最新方法 / Focus：</b>
E4 Edge-Safe View-Adaptive Re-entry RGB-Warp WRE。长期 memory 不是“可见就持续读”，
而是在 surface group 缺席 2 blocks 后，于真正 re-entry 只读一次，再交回 native Recent。</p>
<p>Anchor chunk {metrics['trajectory']['anchor_chunk']}；自适应选择 chunk {selected}；
实际 read chunk {read_chunk}；B2 chunk {metrics['trajectory']['target_chunk']}。</p>

<section><h2>A. 完整 Pipeline / Framework 与架构修改</h2>
<div class='architecture'><img src='assets/architecture_graph.svg'></div>
{render_pipeline_table_html(snapshot)}<h3>Architecture Changes</h3>
{render_changes_html(snapshot)}</section>

<section><h2>B. Lifecycle 与 Observation Selection</h2>
<div class='images'><figure><figcaption>生命周期时间线（yaw / visibility / one-shot read）</figcaption>
<img src='assets/reentry_refinement/lifecycle_timeline.png'></figure>
<figure><figcaption>Anchor B1 +45°</figcaption><img src='assets/reentry_refinement/anchor_b1_chunk.png'></figure>
<figure><figcaption>选中的首次 episode observation：chunk {selected}</figcaption>
<img src='assets/reentry_refinement/selected_observation.png'></figure></div>
{_lifecycle_table(metrics)}<h3>Edge-Safe 方法 Top-3 observation scores</h3>
{_selection_table(metrics)}
<p><b>重要限制：</b>本轮按指令使用 global best historical chunk，而不是
per-surfel multi-view fusion。Top-1 chunk {selected} 与 anchor chunk 的已成功
fusion surfels 数为
{metrics['observation_selection']['edge_safe_top3'][0]['shared_anchor_surfels']}；
因此它是 pose/coverage 匹配的首次 episode observation，但不能据此宣称
surfel-level identity correspondence 已建立。这与 E3/E4 的 duplicate appearance
是同一处需要继续定位的证据。</p></section>

<section><h2>C. Geometry / Safe Support</h2><div class='images'>
<figure><figcaption>真实历史 RGB surfel splats</figcaption><img src='surfel_rgb_options/A_rgb_world_splats.png'></figure>
<figure><figcaption>真实历史 RGB oriented disks</figcaption><img src='surfel_rgb_options/B_rgb_oriented_disks.png'></figure>
<figure><figcaption>选中 observation warp 到 B2</figcaption><img src='assets/reentry_refinement/selected_observation_warped_to_b2.png'></figure>
<figure><figcaption>Anchor chunk 11 warp 到 B2</figcaption><img src='assets/reentry_refinement/anchor_b1_warped_to_b2.png'></figure>
<figure><figcaption>B2 historical visibility</figcaption><img src='assets/reentry_refinement/M_history_b2.png'></figure>
<figure><figcaption>eroded warp-valid</figcaption><img src='assets/reentry_refinement/M_warp_valid_eroded.png'></figure>
<figure><figcaption>true revisit eval mask</figcaption><img src='assets/reentry_refinement/M_revisit_eval.png'></figure>
<figure><figcaption>anchor true revisit eval mask</figcaption><img src='assets/reentry_refinement/M_anchor_revisit_eval.png'></figure>
</div></section>

<section><h2>D. 完整回访同步视频（B1→首次离开→Leave→Return→B2）</h2>
<p>{_controls('full')}</p>{_videos('full_revisit_', 'full')}</section>

<section><h2>E. 首次离开窗口（B1 hold→Leave ramp）</h2>
<figure><figcaption>同帧首次离开 strip</figcaption><img src='assets/reentry_refinement/departure_review_small.jpg'></figure>
<p>{_controls('leave')}</p>{_videos('departure_', 'leave')}</section>

<section><h2>F. Re-entry 窗口（Leave→B2）</h2>
<figure><figcaption>同帧 re-entry strip</figcaption><img src='assets/reentry_refinement/reentry_review_small.jpg'></figure>
<p>{_controls('reentry')}</p>{_videos('reentry_', 'reentry')}</section>

<section><h2>G. B2 右侧 Identity Overlap 对比</h2>
<figure><figcaption>B2 全帧方法对比</figcaption><img src='assets/reentry_refinement/b2_full_review_small.jpg'></figure>
<div class='images'>{right_edge}</div></section>

<section><h2>H. Core Metrics</h2>{_metric_table(metrics)}
<p><b>归因规则：</b>E0/E1/E2 使用固定 anchor chunk 11 的 warp/mask；
E3/E4/E5 使用实际锁定的 adaptive chunk {selected} 的 warp/mask。两类 source
不再混用同一个 revisit target。</p>
<p>True revisit mask 占比 {metrics['regions']['true_revisit_fraction']:.3f}；
source region 占比 {metrics['regions']['source_fraction']:.3f}；
right-edge revisit 占比 {metrics['regions']['right_edge_revisit_fraction']:.3f}。</p></section>

<section><h2>I. 本轮结论</h2>
<p><b>Primary status：{html.escape(metrics['status'])}</b></p>
<p>Q1 lifecycle policy works：<b>{decision['reentry_policy_works']}</b>。</p>
<p>首次离开已恢复 baseline：<b>{decision['first_departure_fixed']}</b>；但 group-level
one-shot handoff 只保留当前持续读取 memory gain 的
<b>{100*decision['memory_gain_retention_ratio']:.1f}%</b>，因此不能判定成功。</p>
<p>Q2 view-adaptive + edge-safe works：<b>{decision['view_adaptive_edge_fix_works']}</b>。</p>
<p>Final-step stabilization useful：<b>{decision['final_step_stabilization_useful']}</b>。</p>
<p><b>下一步唯一优先任务：</b>
{('在第二个静态室内场景复现 E4，不进入 Canonical-K。' if decision['view_adaptive_edge_fix_works'] else '把 group-level SERVED 改为 per-surfel one-shot lifecycle：每个新 re-entered surfel 只服务一次，尚未进入视野的 surfels 不提前 handoff。')}</p>
</section></main><script>{script}</script></body></html>"""
    report = root / "report.html"
    report.write_text(document, encoding="utf-8")

    lines = [
        "# MapKV Re-entry Memory Refinement",
        "",
        f"- Statuses: {', '.join(metrics['statuses'])}",
        "- 本次 Focus: lifecycle → stable observation → edge-safe support",
        f"- Anchor/source/read/B2: {metrics['trajectory']['anchor_chunk']} → "
        f"{selected} → {read_chunk} → {metrics['trajectory']['target_chunk']}",
        "",
        "## Metrics",
        "",
        "| Method | History ref | Revisit ↓ | Source Δ ↓ | Leave peak ↓ | Re-entry peak ↓ | Right edge ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in LABELS.items():
        value = metrics["methods"][method]
        right = value["right_edge_revisit_l1"]
        lines.append(
            f"| {label} | chunk {value['revisit_reference_chunk']} | "
            f"{value['revisit_region_b1_to_b2_l1']:.5f} | "
            f"{value['source_region_delta_vs_baseline_l1']:.5f} | "
            f"{value['leave_window_peak_l1']:.5f} | "
            f"{value['reentry_window_peak_l1']:.5f} | "
            f"{'—' if right is None else f'{right:.5f}'} |"
        )
    lines += [
        "",
        "## Decisions",
        "",
        f"- Re-entry policy: {decision['reentry_policy_works']}",
        f"- View-adaptive edge fix: {decision['view_adaptive_edge_fix_works']}",
        f"- Final-step stabilization: {decision['final_step_stabilization_useful']}",
        f"- First departure fixed: {decision['first_departure_fixed']}",
        f"- Continuous-memory gain retained after group handoff: "
        f"{100*decision['memory_gain_retention_ratio']:.1f}%",
        "",
        "## Next action",
        "",
        "Implement per-surfel one-shot lifecycle so later re-entering surfaces "
        "are not marked SERVED by the first partial read.",
        "",
        "## Complete videos",
        "",
    ]
    lines += [
        f"- {label}: videos/report/full_revisit_{method}.mp4"
        for method, label in LABELS.items()
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


__all__ = ["build_report"]
