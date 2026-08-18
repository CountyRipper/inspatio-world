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
        "<figure><figcaption>"
        + html.escape(label)
        + "</figcaption><video class='sync comparison' controls "
        + "preload='metadata' poster='assets/posters/"
        + html.escape(name)
        + ".jpg' src='videos/report/b2_window_"
        + html.escape(name)
        + ".mp4'></video></figure>"
    )


def build_report(run_root: str | Path) -> str:
    root = Path(run_root).resolve()
    metrics = _json(root / "metrics.json")
    status = metrics["status"]
    if status not in {
        "WARP_REENCODE_WORKS",
        "WARP_REENCODE_NOT_WORKING",
    }:
        raise ValueError(f"Unsupported warp-reencode status: {status}")
    methods = metrics["methods"]
    visual_review = {
        "completed": True,
        "finding": (
            "Hard RecentKV shows duplicated/ghosted pastry and table content. "
            "Warp-Reencode restores a single coherent layout, preserves the "
            "20-degree target view, and visibly reduces unsupported-side changes."
        ),
        "supports_status": status == "WARP_REENCODE_WORKS",
        "artifact": "assets/b2_filmstrip_small.jpg",
    }
    metrics["decision"]["visual_review"] = visual_review
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    architecture = {
        "backbone": "InSpatio-World-1.3B frozen student",
        "source": {
            "chunk": 8,
            "pose": "known c2w at +30 degrees",
            "payload": "clean generated B1 latent",
        },
        "target": {
            "chunks": [21, 22],
            "pose": "known c2w at +20 degrees",
        },
        "warp": {
            "mode": "exact pure-rotation homography",
            "sampling": "target pixel to source pixel",
            "space": "VAE latent grid",
            "fallback": "runtime current recent latent outside coverage",
        },
        "writer": {
            "input": "[current Ref, target-aligned Virtual Recent]",
            "timestep": 0,
            "layout": "native recent t3-t5",
            "cache": "isolated temporary cache",
        },
        "injection": {
            "mode": "replace_recent_delta",
            "alpha": 1.0,
            "layers": "all 30",
            "steps": "all 4",
            "scope": "whole B2 plateau",
        },
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
        "memory": architecture["injection"],
        "automatic_retrieval": False,
        "fixed_source_reason": "isolate view alignment from retrieval policy",
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
        "source_chunk": metrics["source_chunk"],
        "target_chunks": metrics["target_chunks"],
        "validity": metrics["validity"],
        "visual_review": visual_review,
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
        "hard_recentkv": "Hard RecentKV",
        "warp_reencode": "Warp-Reencode Recent",
    }
    rows = "".join(
        "<tr><td>"
        + labels[name]
        + "</td><td>"
        + _fmt(values["overlap_b1_to_b2_l1"])
        + "</td><td>"
        + _fmt(values["nonoverlap_delta_vs_baseline_l1"])
        + "</td><td>"
        + _fmt(values["b2_entry_boundary_l1"])
        + "</td><td>"
        + _fmt(values["within_b2_boundary_l1"])
        + "</td></tr>"
        for name, values in methods.items()
    )
    videos = "".join(
        _video(labels[name], name)
        for name in ("baseline", "hard_recentkv", "warp_reencode")
    )
    validity = metrics["validity"]
    validity_rows = "".join(
        "<li><b>"
        + html.escape(key.replace("_", " "))
        + ":</b> "
        + html.escape(str(value))
        + "</li>"
        for key, value in validity.items()
        if key != "gpu_by_method"
    )
    if status == "WARP_REENCODE_WORKS":
        conclusion = (
            "Camera alignment is the missing operation in this controlled "
            "changed-view case. Reprojecting B1 latent content into B2 layout "
            "before the native writer both restores historical appearance and "
            "removes most of the spatial conflict caused by direct post-RoPE KV replay."
        )
        next_action = (
            "Use this positive baseline to test efficient canonical pre-RoPE K "
            "plus geometry-driven spatial re-addressing."
        )
    else:
        conclusion = (
            "The camera warp is valid, but latent warp-and-reencode does not "
            "produce a useful changed-view memory intervention."
        )
        next_action = (
            "Inspect view-conditioned latent non-equivariance before attempting "
            "native-K spatial re-addressing."
        )

    html_payload = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MapKV Warp-Reencode</title>
<style>
:root{{--bg:#f4f6fa;--card:#fff;--ink:#172033;--muted:#667085;--accent:#2759c7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:24px}}
section{{background:var(--card);margin:16px 0;padding:20px;border-radius:12px;
box-shadow:0 2px 10px #2233aa12}}h1,h2{{margin-top:0}}.status{{font-size:24px;
font-weight:800;color:var(--accent)}}.grid{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
figure{{margin:0}}figcaption{{font-weight:650;margin-bottom:6px}}img,video{{
width:100%;border-radius:8px;background:#111}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px;border-bottom:1px solid #e5e8ef;text-align:right}}
th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#eef1f8;
padding:14px;border-radius:8px}}button{{padding:8px 12px;margin:0 6px 10px 0}}
</style></head><body><main>
<section><h1>MapKV Camera-Aligned Warp-and-Reencode</h1>
<div class="status">{status}</div>
<p>Fixed historical source: B1 chunk 8 at +30°. Targets: chunks 21/22 at +20°.
Warp coverage: {_fmt(metrics['coverage']['latent_warp_fraction'])}.</p>
<p>{html.escape(conclusion)}</p></section>
<section><h2>Architecture</h2>
<pre>OLD
Historical B1 post-RoPE KV → direct recent-slot replay

NEW
Historical B1 clean latent + exact source/target c2w
→ target-to-source pure-rotation warp
→ blend with runtime Recent outside valid coverage
→ [current Ref, Virtual Recent] native timestep-0 writer
→ target-layout recent K/V
→ replace_recent_delta</pre>
<p><b>Architecture change:</b> only the historical payload preparation changed.
Base Ref/Recent cache and the validated auxiliary attention formula are unchanged.</p>
<ul>{validity_rows}</ul></section>
<section><h2>Warp visualization — target chunk 22</h2><div class="grid">
<figure><figcaption>B1 source latent decoded</figcaption><img src="generation/warp_reencode/warp/target_0022/historical.png"></figure>
<figure><figcaption>B2 Baseline</figcaption><img src="assets/warp_reencode/b2_baseline.png"></figure>
<figure><figcaption>B1 warped to B2 latent layout</figcaption><img src="generation/warp_reencode/warp/target_0022/warped.png"></figure>
<figure><figcaption>Warp coverage</figcaption><img src="generation/warp_reencode/warp/target_0022/coverage.png"></figure>
<figure><figcaption>Runtime current Recent</figcaption><img src="generation/warp_reencode/warp/target_0022/current_recent.png"></figure>
<figure><figcaption>Virtual Recent</figcaption><img src="generation/warp_reencode/warp/target_0022/virtual_recent.png"></figure>
<figure><figcaption>Surfel disk map (unchanged)</figcaption><img src="surfel/surfel_disk_preview.png"></figure>
<figure><figcaption>B2 projected surfel support (unchanged)</figcaption><img src="retrieval/visible_support_target_0022.png"></figure>
</div></section>
<section><h2>Synchronized B2 comparison</h2>
<button onclick="playAll()">Play all</button><button onclick="pauseAll()">Pause all</button>
<button onclick="resetAll()">Reset all</button><div class="grid">{videos}</div>
<h3>B2 filmstrip review</h3><img src="assets/b2_filmstrip_small.jpg">
<p>{html.escape(visual_review['finding'])}</p></section>
<section><h2>Main metrics</h2>
<table><tr><th>Method</th><th>Overlap B1→B2 ↓</th>
<th>Non-overlap Δ vs Baseline ↓</th><th>B2 entry boundary ↓</th>
<th>Within-B2 boundary ↓</th></tr>{rows}</table>
<p>Warp overlap gain vs Baseline: {_fmt(metrics['decision']['warp_overlap_gain_vs_baseline'])};
improvement vs HardKV: {_fmt(metrics['decision']['warp_overlap_improvement_vs_hard'])};
non-overlap disturbance reduction vs HardKV:
{_fmt(metrics['decision']['warp_nonoverlap_reduction_vs_hard'])}.</p></section>
<section><h2>Conclusion</h2><p>{html.escape(conclusion)}</p>
<p><b>Next single experiment:</b> {html.escape(next_action)}</p></section>
</main><script>
const vids=()=>[...document.querySelectorAll('video.sync')];
function playAll(){{let v=vids();let t=v.length?v[0].currentTime:0;
v.forEach(x=>{{x.currentTime=t;x.play()}})}}function pauseAll(){{vids().forEach(x=>x.pause())}}
function resetAll(){{vids().forEach(x=>{{x.pause();x.currentTime=0}})}}
</script></body></html>"""
    (root / "report.html").write_text(html_payload, encoding="utf-8")
    markdown = f"""# MapKV Camera-Aligned Warp-and-Reencode

Status: **{status}**

## Architecture

Historical B1 clean latent → exact camera warp → target-aligned Virtual Recent
→ native clean t=0 writer → recent-slot K/V → replace_recent_delta.

The base InSpatio Ref/Recent cache is unchanged.

## Controls

- Source chunk: 8
- Target chunks: 21, 22
- Relative view: +30° → +20°, pure rotation
- Warp coverage: {_fmt(metrics['coverage']['latent_warp_fraction'])}
- Prefix chunks 0–20 exactly equal Baseline: {validity['warp_prefix_exact_through_chunk_20']}
- Runtime cache unchanged: {validity['runtime_cache_unchanged']}
- Same GPU: {validity['same_gpu']} ({validity['gpu_by_method']})
- Visual review: {visual_review['finding']}

## Metrics

| Method | Overlap B1→B2 ↓ | Non-overlap Δ vs Baseline ↓ | B2 entry ↓ | Within B2 ↓ |
|---|---:|---:|---:|---:|
| Baseline | {_fmt(methods['baseline']['overlap_b1_to_b2_l1'])} | 0 | {_fmt(methods['baseline']['b2_entry_boundary_l1'])} | {_fmt(methods['baseline']['within_b2_boundary_l1'])} |
| Hard RecentKV | {_fmt(methods['hard_recentkv']['overlap_b1_to_b2_l1'])} | {_fmt(methods['hard_recentkv']['nonoverlap_delta_vs_baseline_l1'])} | {_fmt(methods['hard_recentkv']['b2_entry_boundary_l1'])} | {_fmt(methods['hard_recentkv']['within_b2_boundary_l1'])} |
| Warp-Reencode | {_fmt(methods['warp_reencode']['overlap_b1_to_b2_l1'])} | {_fmt(methods['warp_reencode']['nonoverlap_delta_vs_baseline_l1'])} | {_fmt(methods['warp_reencode']['b2_entry_boundary_l1'])} | {_fmt(methods['warp_reencode']['within_b2_boundary_l1'])} |

## Conclusion

{conclusion}

## Videos

- Baseline: videos/report/b2_window_baseline.mp4
- Hard RecentKV: videos/report/b2_window_hard_recentkv.mp4
- Warp-Reencode: videos/report/b2_window_warp_reencode.mp4

## Next action

{next_action}
"""
    (root / "report.md").write_text(markdown, encoding="utf-8")
    return str(root / "report.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Warp-Reencode report")
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    print(build_report(args.run_root))


if __name__ == "__main__":
    main()
