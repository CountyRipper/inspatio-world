from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import yaml


LABELS = {
    "baseline": "原始基线（Baseline）",
    "block_on_wre": "固定块开启 WRE（记忆保真参考）",
    "continuous_raw_recent": "连续 RawRecent（隔离 recent warp）",
    "masked_continuous_wre": "掩码连续 WRE（本次最新方法 / Focus）",
}


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

    architecture = {
        "backbone": "InSpatio-World-1.3B frozen student",
        "base_context": "Ref + native Recent + Current",
        "historical_source": "fixed clean B1 chunk 8 latent",
        "geometry": "known-pose CUT3R radius-normal surfels",
        "activation": "per-block projected source-surfel visibility",
        "continuous_raw_recent": (
            "M_history*warp(B1->camera_t) + "
            "(1-M_history)*raw_last_pred; global attention delta"
        ),
        "masked_continuous_wre": (
            "same Virtual Recent; M_query=tokenize(M_history); "
            "A_base + M_query*(A_virtual-A_base)"
        ),
        "writer": "isolated native timestep-0 [Ref, Virtual Recent] writer",
        "injection": "replace_recent_delta, alpha=1, all layers, all 4 steps",
        "base_runtime_cache_replaced": False,
        "active_chunks": validity["active_chunks"],
        "inactive_no_support_chunks": validity[
            "inactive_no_support_chunks"
        ],
    }
    (root / "architecture_state.json").write_text(
        json.dumps(architecture, indent=2), encoding="utf-8"
    )
    resolved = {
        "case": metrics["case"],
        "source_chunk": metrics["source_chunk"],
        "target_chunks": metrics["target_chunks"],
        "camera": metrics["camera"],
        "coverage": metrics["coverage"],
        "architecture": architecture,
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
<section><h2>Architecture</h2>
<pre>Original InSpatio
Ref + native Recent + Current

Failed CAVR
warp(history) + warp(last_pred) → Virtual Recent → GLOBAL delta

Continuous RawRecent (C2)
warp(history) + RAW last_pred → Virtual Recent → GLOBAL delta

Masked Continuous WRE (C3)
warp(history) + RAW last_pred → Virtual Recent
M_query = tokenize(the same M_history)
A_out = A_base + M_query × (A_virtual - A_base)</pre>
<p><b>Architecture changes:</b> short-term Recent is again the native raw
last_pred, while geometry controls both what enters Virtual Recent and which
current query tokens may receive its correction. The runtime base cache remains intact.</p>
<p>Previous failed CAVR: {previous_summary}</p><ul>{validity_rows}</ul></section>
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

Only the historical B1 latent is camera-warped. The short-term fallback remains
raw native `last_pred`. The feathered surfel mask composes Virtual Recent and,
after tokenization, gates `A_virtual - A_base`. The runtime cache is unchanged.

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
