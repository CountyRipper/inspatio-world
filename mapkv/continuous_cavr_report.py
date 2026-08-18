from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import yaml


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 5) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _video(label: str, name: str) -> str:
    return (
        f"<figure><figcaption>{html.escape(label)}</figcaption>"
        f"<video class='sync' controls preload='metadata' "
        f"poster='assets/posters/{name}.jpg' "
        f"src='videos/report/transition_window_{name}.mp4'></video></figure>"
    )


def build_report(run_root: str | Path) -> str:
    root = Path(run_root).resolve()
    metrics = _json(root / "metrics.json")
    status = metrics["status"]
    methods = metrics["methods"]
    decision = metrics["decision"]
    visual_review = decision["visual_review"]
    validity = metrics["validity"]
    if status == "CONTINUOUS_CAVR_WORKS":
        conclusion = (
            "Continuous geometry visibility preserves B1 memory recovery and "
            "new-region behavior while reducing the B2 entrance discontinuity."
        )
        next_action = (
            "Implement canonical pre-RoPE K/V spatial re-addressing as an "
            "efficiency approximation of this validated Virtual Recent oracle."
        )
    elif status == "CONTINUOUS_CAVR_MIXED_TRANSITION":
        conclusion = (
            "Continuous CAVR preserves memory fidelity/locality, but its measured "
            "entrance is not smoother than block-on Warp-Reencode."
        )
        next_action = (
            "Inspect the first re-entry blocks and short-term warp holes before "
            "attempting canonical-K re-addressing."
        )
    else:
        conclusion = (
            "Continuous Virtual Recent did not jointly preserve memory fidelity, "
            "new regions, and transition behavior in this controlled case."
        )
        next_action = (
            "Run one RecentWarpOnly control (historical mask forced to zero) "
            "to isolate short-term warp/re-encode from historical fusion."
        )
    architecture = {
        "backbone": "InSpatio-World-1.3B frozen student",
        "base_context": "Ref + Recent + Current",
        "historical_source": "fixed clean B1 chunk 8 latent",
        "geometry": "known-pose CUT3R radius-normal surfels",
        "activation": "per-block projected source-surfel visibility",
        "virtual_recent": (
            "M_mem * warp(B1 -> camera_t) + "
            "(1-M_mem) * warp(last_pred@(t-1) -> camera_t)"
        ),
        "writer": "native timestep-0 [Ref, Virtual Recent] writer",
        "injection": "replace_recent_delta, alpha=1, all layers, all 4 steps",
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

    labels = {
        "baseline": "Baseline",
        "block_on_wre": "Block-on Warp-Reencode",
        "continuous_cavr": "Continuous CAVR",
    }
    rows = "".join(
        "<tr>"
        f"<td>{labels[name]}</td>"
        f"<td>{_fmt(values['overlap_b1_to_b2_l1'])}</td>"
        f"<td>{_fmt(values['nonoverlap_delta_vs_baseline_l1'])}</td>"
        f"<td>{_fmt(values['b2_entry_boundary_l1'])}</td>"
        f"<td>{_fmt(values['entrance_frame_l1'])}</td>"
        f"<td>{_fmt(values['transition_window_peak_frame_l1'])}</td>"
        "</tr>"
        for name, values in methods.items()
    )
    validity_rows = "".join(
        f"<li><b>{html.escape(key.replace('_', ' '))}:</b> "
        f"{html.escape(str(value))}</li>"
        for key, value in validity.items()
        if key != "gpu_by_method"
    )
    videos = "".join(
        _video(labels[name], name)
        for name in ("baseline", "block_on_wre", "continuous_cavr")
    )
    html_payload = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MapKV Continuous CAVR</title>
<style>
:root{{--bg:#f4f6fa;--card:#fff;--ink:#172033;--accent:#2759c7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:24px}}
section{{background:var(--card);margin:16px 0;padding:20px;border-radius:12px;
box-shadow:0 2px 10px #2233aa12}}h1,h2{{margin-top:0}}.status{{font-size:24px;
font-weight:800;color:var(--accent)}}.grid{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
figure{{margin:0}}figcaption{{font-weight:650;margin-bottom:6px}}img,video{{
width:100%;border-radius:8px;background:#111}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px;border-bottom:1px solid #e5e8ef;text-align:right}}
th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#eef1f8;
padding:14px;border-radius:8px}}button{{padding:8px 12px;margin:0 6px 10px 0}}
</style></head><body><main>
<section><h1>MapKV Continuous Geometry-Reprojected Virtual Recent</h1>
<div class="status">{status}</div><p>{html.escape(conclusion)}</p>
<p>Memory fidelity: <b>{decision['memory_fidelity_preserved']}</b>;
new-region preservation: <b>{decision['new_region_preserved']}</b>;
transition improved: <b>{decision['transition_improved']}</b>.</p></section>
<section><h2>Architecture</h2>
<pre>Original InSpatio
Ref + Recent + Current

Block-on WRE
Ref + [warped B1 only at fixed B2 chunks] + Current

Continuous CAVR
target camera → projected B1 surfel support M_mem
B1 latent → camera_t warp
last_pred@(t-1) → camera_t warp
→ M_mem * historical + (1-M_mem) * short-term recent
→ native t=0 [Ref, Virtual Recent] writer
→ fixed-size Recent K/V → normal 4-step denoising</pre>
<p><b>Architecture changes:</b> fixed B2 activation was replaced by per-block
geometry visibility, and the short-term recent fallback is now camera-aligned.
No third persistent cache and no output replacement were added.</p>
<ul>{validity_rows}</ul></section>
<section><h2>Geometry and Virtual Recent</h2><div class="grid">
<figure><figcaption>Surfel disk map</figcaption><img src="surfel/surfel_disk_preview.png"></figure>
<figure><figcaption>Coverage over trajectory</figcaption><img src="assets/cavr/coverage_timeline.png"></figure>
<figure><figcaption>B1 warped to B2</figcaption><img src="generation/continuous_cavr/warp/target_0022/warped.png"></figure>
<figure><figcaption>Historical surfel coverage</figcaption><img src="generation/continuous_cavr/warp/target_0022/coverage.png"></figure>
<figure><figcaption>Raw last_pred</figcaption><img src="generation/continuous_cavr/warp/target_0022/current_recent.png"></figure>
<figure><figcaption>Warped short-term recent</figcaption><img src="generation/continuous_cavr/warp/target_0022/warped_recent.png"></figure>
<figure><figcaption>Short-term warp coverage</figcaption><img src="generation/continuous_cavr/warp/target_0022/recent_coverage.png"></figure>
<figure><figcaption>Virtual Recent</figcaption><img src="generation/continuous_cavr/warp/target_0022/virtual_recent.png"></figure>
</div></section>
<section><h2>Synchronized revisit transition</h2>
<button onclick="playAll()">Play all</button><button onclick="pauseAll()">Pause all</button>
<button onclick="resetAll()">Reset all</button><div class="grid">{videos}</div>
<img src="assets/cavr_transition_filmstrip_small.jpg">
<p>{html.escape(visual_review['finding'])}</p></section>
<section><h2>Main metrics</h2>
<table><tr><th>Method</th><th>Overlap B1→B2 ↓</th>
<th>Non-overlap Δ ↓</th><th>Chunk entrance ↓</th>
<th>Exact entrance frame ↓</th><th>Window peak ↓</th></tr>{rows}</table></section>
<section><h2>Conclusion</h2><p>{html.escape(conclusion)}</p>
<p><b>Next single action:</b> {html.escape(next_action)}</p></section>
</main><script>
const vids=()=>[...document.querySelectorAll('video.sync')];
function playAll(){{let v=vids();let t=v.length?v[0].currentTime:0;
v.forEach(x=>{{x.currentTime=t;x.play()}})}}function pauseAll(){{vids().forEach(x=>x.pause())}}
function resetAll(){{vids().forEach(x=>{{x.pause();x.currentTime=0}})}}
</script></body></html>"""
    (root / "report.html").write_text(html_payload, encoding="utf-8")
    markdown = f"""# MapKV Continuous Geometry-Reprojected Virtual Recent

Status: **{status}**

## Architecture

Known-pose CUT3R source-chunk surfel visibility dynamically reconstructs the
existing Recent slot. Both B1 and runtime last_pred are warped to camera_t,
fused by M_mem, and re-encoded by the native timestep-zero writer.

- Active chunks: {validity['active_chunks']}
- No-support fallback chunks: {validity['inactive_no_support_chunks']}
- Memory fidelity preserved: {decision['memory_fidelity_preserved']}
- New region preserved: {decision['new_region_preserved']}
- Transition improved: {decision['transition_improved']}

## Metrics

| Method | Overlap B1→B2 ↓ | Non-overlap Δ ↓ | Chunk entrance ↓ | Exact entrance frame ↓ |
|---|---:|---:|---:|---:|
| Baseline | {_fmt(methods['baseline']['overlap_b1_to_b2_l1'])} | 0 | {_fmt(methods['baseline']['b2_entry_boundary_l1'])} | {_fmt(methods['baseline']['entrance_frame_l1'])} |
| Block-on WRE | {_fmt(methods['block_on_wre']['overlap_b1_to_b2_l1'])} | {_fmt(methods['block_on_wre']['nonoverlap_delta_vs_baseline_l1'])} | {_fmt(methods['block_on_wre']['b2_entry_boundary_l1'])} | {_fmt(methods['block_on_wre']['entrance_frame_l1'])} |
| Continuous CAVR | {_fmt(methods['continuous_cavr']['overlap_b1_to_b2_l1'])} | {_fmt(methods['continuous_cavr']['nonoverlap_delta_vs_baseline_l1'])} | {_fmt(methods['continuous_cavr']['b2_entry_boundary_l1'])} | {_fmt(methods['continuous_cavr']['entrance_frame_l1'])} |

## Videos

- Baseline: videos/report/transition_window_baseline.mp4
- Block-on WRE: videos/report/transition_window_block_on_wre.mp4
- Continuous CAVR: videos/report/transition_window_continuous_cavr.mp4

## Conclusion

{conclusion}

## Next action

{next_action}
"""
    (root / "report.md").write_text(markdown, encoding="utf-8")
    return str(root / "report.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Continuous CAVR report")
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    print(build_report(args.run_root))


if __name__ == "__main__":
    main()
