from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import yaml

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
    "baseline": "原始基线（Baseline）",
    "block_on_wre": "固定块开启 WRE（记忆保真参考）",
    "continuous_raw_recent": "连续 RawRecent（隔离 recent warp）",
    "masked_continuous_wre": "掩码连续 WRE（本次最新方法 / Focus）",
}


def _architecture_snapshot(validity: dict) -> ArchitectureSnapshot:
    return ArchitectureSnapshot(
        name="MapKV — Masked Continuous Warp-Reencode Recent",
        focus_zh=(
            "保持原生短期 Recent，并让同一几何 mask 同时控制历史融合与 query 校正"
        ),
        focus_en=(
            "Raw native short-term Recent + geometry-localized historical correction"
        ),
        nodes=(
            node(
                id="experiment_inputs",
                label_zh="静态源与精确相机轨迹",
                label_en="Static source + exact c2w trajectory",
                role="input",
                column=0,
                row=0,
                summary="固定 source、0→30→0→20 pure-yaw、render/mask/prompt。",
                files=("mapkv/trajectory.py", "datasets/test_dataset.py"),
            ),
            node(
                id="deterministic_noise",
                label_zh="确定性噪声回放",
                label_en="Initial noise + per-step re-noise bundle",
                role="input",
                column=0,
                row=1,
                summary="所有方法共享初始噪声和四步 re-noise。",
                files=("mapkv_proto/deterministic_noise.py",),
            ),
            node(
                id="baseline_generation",
                label_zh="原生 InSpatio 因果生成",
                label_en="Frozen InSpatio block-wise generation",
                role="generation",
                column=1,
                row=0,
                summary="保留 Ref + Recent + Current 和原始四步 denoising。",
                files=("pipeline/causal_inference.py", "wan/modules/causal_model.py"),
            ),
            node(
                id="cut3r_geometry",
                label_zh="Known-pose CUT3R 几何",
                label_en="Causal prefix depth/pointmaps with known c2w",
                role="geometry",
                column=1,
                row=2,
                summary="仅使用 target 前历史生成帧；CUT3R 不提供 map pose。",
                files=("mapkv/cut3r_adapter.py",),
            ),
            node(
                id="historical_payload",
                label_zh="B1 历史干净 latent",
                label_en="Fixed clean historical B1 chunk 8 latent",
                role="payload",
                column=2,
                row=0,
                summary="本轮固定 chunk 8，以隔离 injection architecture。",
                files=("mapkv/warp_reencode.py",),
            ),
            node(
                id="surfel_address",
                label_zh="Radius-normal Surfel 地址",
                label_en="Surfel index with observing chunk metadata",
                role="address",
                column=2,
                row=2,
                summary="几何仅保存 address metadata；payload 不存入 surfel。",
                files=("mapkv/surfel_index.py",),
            ),
            node(
                id="historical_warp",
                label_zh="历史 B1 相机重投影",
                label_en="Exact B1-to-current camera latent warp",
                role="geometry",
                column=3,
                row=0,
                summary="仅 long-term historical latent 做 pure-rotation warp。",
                files=("mapkv/warp_reencode.py",),
            ),
            node(
                id="raw_recent",
                label_zh="原生短期 last_pred",
                label_en="Raw native short-term Recent fallback",
                role="context",
                column=3,
                row=1,
                summary="不再把上一 block Recent 主动 warp 到当前相机。",
                change_type="modified",
                files=("mapkv/warp_reencode.py", "inference_mapkv_proto.py"),
            ),
            node(
                id="visible_memory_mask",
                label_zh="可见历史表面 M_history",
                label_en="Projected source-surfel visibility mask",
                role="address",
                column=3,
                row=2,
                summary="同一 mask 同时控制 latent fusion 与 query correction。",
                change_type="modified",
                files=(
                    "mapkv/warp_reencode.py",
                    "mapkv_proto/memory_context.py",
                ),
            ),
            node(
                id="virtual_recent",
                label_zh="目标对齐 Virtual Recent",
                label_en="M*warped history + (1-M)*raw last_pred",
                role="context",
                column=4,
                row=0,
                summary="历史区使用 warped B1；其余区域使用原生 last_pred。",
                change_type="modified",
                files=("mapkv/warp_reencode.py",),
            ),
            node(
                id="base_attention",
                label_zh="原始 Base Attention",
                label_en="Attention(Q, [K_ref, K_recent, K_current])",
                role="attention",
                column=4,
                row=2,
                summary="原始 reference/recent cache 完整保留且不被改写。",
                files=("wan/modules/causal_model.py",),
            ),
            node(
                id="native_writer",
                label_zh="原生 t=0 Recent Writer",
                label_en="Native [Ref, Virtual Recent] context writer",
                role="context",
                column=5,
                row=0,
                summary="产生 target-layout、recent temporal phase 的合法 K/V。",
                files=("pipeline/causal_inference.py",),
            ),
            node(
                id="masked_attention",
                label_zh="几何掩码历史 Attention 校正",
                label_en="A_base + M_query * (A_virtual - A_base)",
                role="attention",
                column=6,
                row=0,
                summary="只有 M_history 对应的 current query 接受历史校正。",
                change_type="added",
                focus=True,
                files=(
                    "mapkv/warp_reencode.py",
                    "mapkv_proto/memory_context.py",
                    "pipeline/causal_inference.py",
                ),
            ),
            node(
                id="normal_denoise",
                label_zh="正常四步 Current 去噪",
                label_en="Normal 4-step current-block denoising",
                role="generation",
                column=7,
                row=0,
                summary="current block 仍从相同噪声开始，不做 output replacement。",
                files=("pipeline/causal_inference.py",),
            ),
            node(
                id="video_output",
                label_zh="完整回访视频输出",
                label_en="Decoded B1-leave-return-B2 video",
                role="output",
                column=8,
                row=0,
                summary="保存完整回访视频和 re-entry 辅助 clip。",
                files=("mapkv/continuous_cavr_stage.py",),
            ),
            node(
                id="evaluation",
                label_zh="局部性与过渡评估",
                label_en="Overlap / non-overlap / transition evaluation",
                role="evaluation",
                column=8,
                row=1,
                summary="比较 memory fidelity、非重叠污染和 re-entry 峰值。",
                files=("mapkv/continuous_cavr_evaluation.py",),
            ),
            node(
                id="report_framework",
                label_zh="统一架构报告框架",
                label_en="Validated graph + architecture change artifacts",
                role="evaluation",
                column=8,
                row=2,
                summary="输出完整 graph、状态 JSON、变更 JSON 和 Markdown。",
                change_type="added",
                files=(
                    "mapkv/report_framework.py",
                    "mapkv/continuous_cavr_report.py",
                    "AGENTS.md",
                    "mapkv/report_preferences.yaml",
                ),
            ),
        ),
        edges=(
            ArchitectureEdge("experiment_inputs", "baseline_generation", "条件"),
            ArchitectureEdge("deterministic_noise", "baseline_generation", "replay"),
            ArchitectureEdge("experiment_inputs", "cut3r_geometry", "known c2w"),
            ArchitectureEdge("baseline_generation", "historical_payload", "B1"),
            ArchitectureEdge("baseline_generation", "raw_recent", "last_pred"),
            ArchitectureEdge("baseline_generation", "cut3r_geometry", "历史 RGB"),
            ArchitectureEdge("cut3r_geometry", "surfel_address", "pointmaps"),
            ArchitectureEdge("surfel_address", "visible_memory_mask", "target view"),
            ArchitectureEdge("historical_payload", "historical_warp", "source latent"),
            ArchitectureEdge("experiment_inputs", "historical_warp", "camera_t"),
            ArchitectureEdge("historical_warp", "virtual_recent", "warped B1"),
            ArchitectureEdge("raw_recent", "virtual_recent", "fallback"),
            ArchitectureEdge("visible_memory_mask", "virtual_recent", "M_history"),
            ArchitectureEdge("virtual_recent", "native_writer", "clean context"),
            ArchitectureEdge("native_writer", "masked_attention", "virtual K/V"),
            ArchitectureEdge("baseline_generation", "base_attention", "runtime cache"),
            ArchitectureEdge("base_attention", "masked_attention", "A_base"),
            ArchitectureEdge("visible_memory_mask", "masked_attention", "M_query"),
            ArchitectureEdge("masked_attention", "normal_denoise", "condition"),
            ArchitectureEdge("deterministic_noise", "normal_denoise", "same noise"),
            ArchitectureEdge("normal_denoise", "video_output", "VAE decode"),
            ArchitectureEdge("video_output", "evaluation", "videos/latents"),
            ArchitectureEdge("evaluation", "report_framework", "metrics"),
        ),
        changes=(
            ArchitectureChange(
                component_id="raw_recent",
                change_type="modified",
                before="把 last_pred 从 camera_(t-1) warp 到 camera_t。",
                after="保持 InSpatio 原生 raw last_pred，不做 short-term warp。",
                affected_files=("mapkv/warp_reencode.py", "inference_mapkv_proto.py"),
                rationale="避免破坏模型训练时的短期 Recent 分布和 warp 空洞。",
            ),
            ArchitectureChange(
                component_id="visible_memory_mask",
                change_type="modified",
                before="M_history 只用于 Virtual Recent latent 融合。",
                after="同一 M_history 还被 token 化为 M_query，限制 attention delta。",
                affected_files=(
                    "mapkv/warp_reencode.py",
                    "mapkv_proto/memory_context.py",
                ),
                rationale="少量可见历史表面不能再触发全局 current-query 修改。",
            ),
            ArchitectureChange(
                component_id="virtual_recent",
                change_type="modified",
                before="M*warped_history + (1-M)*warped_recent。",
                after="M*warped_history + (1-M)*raw_last_pred。",
                affected_files=("mapkv/warp_reencode.py",),
                rationale="让 continuous 方法成为 successful WRE 的单变量扩展。",
            ),
            ArchitectureChange(
                component_id="masked_attention",
                change_type="added",
                before="Virtual Recent counterfactual 以 alpha=1 全局替换 A_base。",
                after="A_base + M_query*(A_virtual-A_base)。",
                affected_files=(
                    "mapkv/warp_reencode.py",
                    "mapkv_proto/memory_context.py",
                    "pipeline/causal_inference.py",
                ),
                rationale="在保留历史恢复的同时约束 non-overlap 污染和 re-entry popping。",
            ),
            ArchitectureChange(
                component_id="report_framework",
                change_type="added",
                before="报告用局部伪代码描述 architecture，缺少完整 graph 和统一变更 schema。",
                after="所有报告输出完整 pipeline SVG、state/change JSON、Markdown 和变更表。",
                affected_files=(
                    "mapkv/report_framework.py",
                    "mapkv/continuous_cavr_report.py",
                    "AGENTS.md",
                    "mapkv/report_preferences.yaml",
                ),
                rationale="让每次架构演进易理解、可追踪、可由后续 agent 复用。",
            ),
        ),
        metadata={
            "backbone": "InSpatio-World-1.3B frozen student",
            "source_chunk": 8,
            "active_chunks": validity["active_chunks"],
            "inactive_no_support_chunks": validity[
                "inactive_no_support_chunks"
            ],
            "injection": "replace_recent_delta; alpha=1; all layers; all steps",
            "base_runtime_cache_replaced": False,
        },
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 5) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _video(name: str, *, full: bool = False) -> str:
    prefix = "full_revisit" if full else "transition_window"
    return (
        f"<figure><figcaption>{html.escape(LABELS[name])}</figcaption>"
        f"<video class='sync' controls preload='metadata' "
        f"poster='assets/posters/{name}.jpg' "
        f"src='videos/report/{prefix}_{name}.mp4'></video></figure>"
    )


def build_report(run_root: str | Path) -> str:
    root = Path(run_root).resolve()
    metrics = _json(root / "metrics.json")
    status = metrics["status"]
    methods = metrics["methods"]
    decision = metrics["decision"]
    validity = metrics["validity"]
    if status == "MASKED_CONTINUOUS_WRE_WORKS":
        conclusion = (
            "Preserving raw short-term Recent and applying the same geometry "
            "mask to latent composition and attention correction retained "
            "historical recovery while improving locality and re-entry."
        )
        next_action = (
            "Implement canonical pre-RoPE K/V spatial re-addressing as an "
            "efficient approximation of this validated quality path."
        )
    elif status == "CONTINUOUS_WRE_LOCALITY_INCOMPLETE":
        conclusion = (
            "The repaired continuous formulation remains interpretable, but "
            "did not jointly meet memory-fidelity, non-overlap, and transition "
            "criteria in this controlled case."
        )
        next_action = (
            "Use correspondence-aware/canonical-K spatial re-addressing to "
            "localize the historical payload itself; do not tune retrieval."
        )
    else:
        conclusion = "The run failed an implementation-validity invariant."
        next_action = "Repair the failed invariant and rerun the same case."

    architecture = _architecture_snapshot(validity)
    write_architecture_bundle(root, architecture)
    architecture_changes_html = render_changes_html(architecture)
    pipeline_table_html = render_pipeline_table_html(architecture)
    resolved = {
        "case": metrics["case"],
        "source_chunk": metrics["source_chunk"],
        "target_chunks": metrics["target_chunks"],
        "camera": metrics["camera"],
        "coverage": metrics["coverage"],
        "architecture": architecture.to_dict(),
    }
    (root / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    repo = Path(__file__).resolve().parents[1]
    git_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    )
    status_payload = {
        "status": status,
        "git_commit": git_commit,
        "working_tree_clean": not bool(git_status),
        "report": str(root / "report.html"),
        "decision": decision,
    }
    (root / "status.json").write_text(
        json.dumps(status_payload, indent=2), encoding="utf-8"
    )
    if git_status:
        (root / "git_diff_stat.txt").write_text(
            subprocess.check_output(
                ["git", "-C", str(repo), "diff", "--stat"], text=True
            ),
            encoding="utf-8",
        )

    metric_rows = "".join(
        "<tr>"
        f"<td>{LABELS[name]}</td>"
        f"<td>{_fmt(values['overlap_b1_to_b2_l1'])}</td>"
        f"<td>{_fmt(values['nonoverlap_delta_vs_baseline_l1'])}</td>"
        f"<td>{_fmt(values['reentry_window_mean_l1'])}</td>"
        f"<td>{_fmt(values['reentry_window_peak_l1'])}</td>"
        f"<td>{values['reentry_window_peak_rgb_frame']}</td>"
        "</tr>"
        for name, values in methods.items()
    )
    timeline_rows = "".join(
        "<tr>"
        f"<td>{item['chunk']}</td>"
        f"<td>{item['yaw_degrees']:.2f}°</td>"
        f"<td>{100.0 * item['coverage']:.2f}%</td>"
        f"<td>{_fmt(item['query_gate_fraction'])}</td>"
        f"<td>{'on' if item['active'] else 'off'}</td>"
        "</tr>"
        for item in metrics["coverage"]["timeline"]
    )
    validity_rows = "".join(
        f"<li><b>{html.escape(key.replace('_', ' '))}:</b> "
        f"{html.escape(str(value))}</li>"
        for key, value in validity.items()
        if key != "gpu_by_method"
    )
    full_videos = "".join(_video(name, full=True) for name in LABELS)
    videos = "".join(_video(name) for name in LABELS)
    previous = metrics.get("previous_failed_cavr") or {}
    previous_summary = (
        "unavailable"
        if not previous
        else (
            f"non-overlap={_fmt(previous['nonoverlap_delta_vs_baseline_l1'])}, "
            f"mean={_fmt(previous['transition_window_mean_frame_l1'])}, "
            f"peak={_fmt(previous['transition_window_peak_frame_l1'])}"
        )
    )
    html_payload = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MapKV Masked Continuous WRE</title>
<style>
:root{{--bg:#f4f6fa;--card:#fff;--ink:#172033;--accent:#2759c7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:24px}}
section{{background:var(--card);margin:16px 0;padding:20px;border-radius:12px;
box-shadow:0 2px 10px #2233aa12}}h1,h2{{margin-top:0}}.status{{font-size:24px;
font-weight:800;color:var(--accent)}}.grid{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}}
figure{{margin:0}}figcaption{{font-weight:650;margin-bottom:6px}}img,video{{
width:100%;border-radius:8px;background:#111}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px;border-bottom:1px solid #e5e8ef;text-align:right}}
th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#eef1f8;
padding:14px;border-radius:8px}}button{{padding:8px 12px;margin:0 6px 10px 0}}
.yes{{color:#147a3d;font-weight:700}}.no{{color:#a13c2f;font-weight:700}}
.architecture{{overflow-x:auto}}.architecture img{{min-width:1180px;background:#f8fafc}}
.architecture table td,.architecture table th{{text-align:left;vertical-align:top}}
.architecture small{{color:#65758b}}
</style></head><body><main>
<section><h1>MapKV Masked Continuous Warp-Reencode Recent</h1>
<div class="status">{status}</div><p>{html.escape(conclusion)}</p>
<p><b>本次关注：</b>历史 B1 只做相机对齐，短期 Recent 保持原生
last_pred，并由同一个 surfel mask 同时限制 Virtual Recent 与 query attention。
最新方法是<b>掩码连续 WRE（Masked Continuous WRE）</b>。</p>
<p>Q1 raw Recent improves failed CAVR: <b>{decision['raw_recent_improves_failed_cavr']}</b>;
Q2 memory fidelity: <b>{decision['memory_fidelity_preserved']}</b>;
non-overlap locality: <b>{decision['nonoverlap_locality_improved']}</b>;
transition: <b>{decision['transition_improved']}</b>.</p></section>
<section class="architecture"><h2>完整 Pipeline / Framework</h2>
<p><b>本次架构 Focus：</b>{html.escape(architecture.focus_zh)}</p>
<a href="assets/architecture_graph.svg"><img src="assets/architecture_graph.svg"></a>
<h3>完整模块表</h3>{pipeline_table_html}
<h3>本次架构与模块修改（Before → After）</h3>{architecture_changes_html}
<p>Previous failed CAVR: {previous_summary}</p><ul>{validity_rows}</ul>
<p>机器可读快照：<a href="architecture_state.json">architecture_state.json</a> ·
<a href="architecture_changes.json">architecture_changes.json</a> ·
<a href="architecture.md">architecture.md</a></p></section>
<section><h2>Geometry and masks</h2><div class="grid">
<figure><figcaption>RGB surfel 可视化候选（A–E）</figcaption>
<a href="surfel_rgb_options/report.html"><img src="surfel_rgb_options/options_contact_sheet.jpg"></a></figure>
<figure><figcaption>旧 chunk-ID disk（仅 secondary audit）</figcaption><img src="surfel/surfel_disk_preview.png"></figure>
<figure><figcaption>Coverage/yaw timeline</figcaption><img src="assets/masked_continuous/coverage_timeline.png"></figure>
<figure><figcaption>B1 source</figcaption><img src="assets/masked_continuous/b1_source.png"></figure>
<figure><figcaption>B1 warped to B2</figcaption><img src="assets/masked_continuous/b1_warped_to_b2.png"></figure>
<figure><figcaption>B2 coverage</figcaption><img src="assets/masked_continuous/b2_memory_coverage_overlay.png"></figure>
<figure><figcaption>C3 Virtual Recent @ chunk22</figcaption><img src="generation/masked_continuous_wre/warp/target_0022/virtual_recent.png"></figure>
<figure><figcaption>C3 query gate @ first re-entry chunk17</figcaption><img src="generation/masked_continuous_wre/masks/chunk_0017_query_gate_overlay.png"></figure>
<figure><figcaption>C3 query gate @ B2 chunk22</figcaption><img src="generation/masked_continuous_wre/masks/chunk_0022_query_gate_overlay.png"></figure>
</div><h3>Per-block activation</h3>
<p><a href="surfel_rgb_options/report.html"><b>打开 RGB surfel A–E 选择页</b></a>；
颜色均来自真实 generated historical observations。</p>
<table><tr><th>Chunk</th><th>Yaw</th><th>M_history</th><th>M_query</th><th>Memory</th></tr>{timeline_rows}</table></section>
<section><h2>完整回访视频对比：B1 首访 → 离开 → 返回 → B2 回访</h2>
<button onclick="playGroup('full')">全部播放</button>
<button onclick="pauseGroup('full')">全部暂停</button>
<button onclick="resetGroup('full')">全部复位</button>
<div class="grid full">{full_videos}</div></section>
<section><h2>回访进入窗口（补充诊断）</h2>
<button onclick="playAll()">Play all</button><button onclick="pauseAll()">Pause all</button>
<button onclick="resetAll()">Reset all</button><div class="grid">{videos}</div>
<img src="assets/masked_continuous_transition_small.jpg">
<h3>Old failure → repaired causality check</h3>
<img src="assets/repair_causality_review.jpg">
<h3>Dense RGB 218–240 re-entry check</h3>
<img src="assets/masked_continuous_dense_reentry.jpg">
<p>{html.escape(decision['visual_review']['finding'])}</p></section>
<section><h2>Main metrics</h2>
<table><tr><th>Method</th><th>Overlap B1→B2 ↓</th>
<th>Non-overlap Δ ↓</th><th>Re-entry mean ↓</th>
<th>Re-entry peak ↓</th><th>Peak RGB</th></tr>{metric_rows}</table></section>
<section><h2>Conclusion</h2><p>{html.escape(conclusion)}</p>
<p><b>Next single action:</b> {html.escape(next_action)}</p></section>
</main><script>
const vids=()=>[...document.querySelectorAll('video.sync')];
const group=(name)=>[...document.querySelectorAll('.'+name+' video.sync')];
function playGroup(name){{let v=group(name);let t=v.length?v[0].currentTime:0;
v.forEach(x=>{{x.currentTime=t;x.play()}})}}
function pauseGroup(name){{group(name).forEach(x=>x.pause())}}
function resetGroup(name){{group(name).forEach(x=>{{x.pause();x.currentTime=0}})}}
function playAll(){{let v=vids();let t=v.length?v[0].currentTime:0;
v.forEach(x=>{{x.currentTime=t;x.play()}})}}function pauseAll(){{vids().forEach(x=>x.pause())}}
function resetAll(){{vids().forEach(x=>{{x.pause();x.currentTime=0}})}}
</script></body></html>"""
    (root / "report.html").write_text(html_payload, encoding="utf-8")

    table_rows = "\n".join(
        f"| {LABELS[name]} | {_fmt(values['overlap_b1_to_b2_l1'])} | "
        f"{_fmt(values['nonoverlap_delta_vs_baseline_l1'])} | "
        f"{_fmt(values['reentry_window_mean_l1'])} | "
        f"{_fmt(values['reentry_window_peak_l1'])} |"
        for name, values in methods.items()
    )
    markdown = f"""# MapKV Masked Continuous Warp-Reencode Recent

Status: **{status}**

## Architecture

本次架构 Focus：**{architecture.focus_zh}**

![完整 Pipeline](assets/architecture_graph.svg)

完整模块与 before/after 变更说明见 [architecture.md](architecture.md)、
[architecture_state.json](architecture_state.json) 和
[architecture_changes.json](architecture_changes.json)。

- Active chunks: {validity['active_chunks']}
- No-support chunks: {validity['inactive_no_support_chunks']}
- Raw Recent improves failed CAVR: {decision['raw_recent_improves_failed_cavr']}
- Memory fidelity preserved: {decision['memory_fidelity_preserved']}
- Non-overlap locality improved: {decision['nonoverlap_locality_improved']}
- Transition improved: {decision['transition_improved']}

## Metrics

| Method | Overlap B1→B2 ↓ | Non-overlap Δ ↓ | Re-entry mean ↓ | Re-entry peak ↓ |
|---|---:|---:|---:|---:|
{table_rows}

## Videos

- Full revisit Baseline: videos/report/full_revisit_baseline.mp4
- Full revisit Block-on WRE: videos/report/full_revisit_block_on_wre.mp4
- Full revisit Continuous RawRecent: videos/report/full_revisit_continuous_raw_recent.mp4
- Full revisit Masked Continuous WRE: videos/report/full_revisit_masked_continuous_wre.mp4
- Baseline: videos/report/transition_window_baseline.mp4
- Block-on WRE: videos/report/transition_window_block_on_wre.mp4
- Continuous RawRecent: videos/report/transition_window_continuous_raw_recent.mp4
- Masked Continuous WRE: videos/report/transition_window_masked_continuous_wre.mp4

## Conclusion

{conclusion}

## Next action

{next_action}
"""
    (root / "report.md").write_text(markdown, encoding="utf-8")
    return str(root / "report.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Masked Continuous WRE report"
    )
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    print(build_report(args.run_root))


if __name__ == "__main__":
    main()
