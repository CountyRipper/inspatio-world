from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from pathlib import Path

import yaml


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(
        os.path.relpath(source.resolve(), destination.parent.resolve())
    )


def _fmt(value: object, digits: int = 5) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _video_grid(group: str, items: list[tuple[str, str]]) -> str:
    return "".join(
        "<figure><figcaption>"
        + html.escape(label)
        + "</figcaption><video class='sync "
        + html.escape(group)
        + "' controls preload='metadata' src='"
        + html.escape(path)
        + "'></video></figure>"
        for label, path in items
    )


def build_report(
    *,
    output_root: str | Path,
    replication_root: str | Path,
    partial_root: str | Path,
    layer_root: str | Path,
) -> str:
    output = Path(output_root).resolve()
    replication_root = Path(replication_root).resolve()
    partial_root = Path(partial_root).resolve()
    layer_root = Path(layer_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    replication = _json(replication_root / "metrics.json")
    locality = _json(partial_root / "locality_metrics.json")
    layers = _json(layer_root / "metrics.json")
    status = (
        locality["status"]
        if replication["replication"]["status"] == "PASS"
        else "REPLICATION_FAILED"
    )
    allowed = {
        "QUERY_GATING_SUFFICIENT",
        "TOKEN_SELECTION_NEEDED",
        "PARTIAL_OVERLAP_NOT_WORKING",
        "REPLICATION_FAILED",
    }
    if status not in allowed:
        raise ValueError(f"Unsupported next-stage status: {status}")

    for method in ("baseline", "manualcorrect", "wrongkv", "surfelkv"):
        _link(
            replication_root
            / "videos/report"
            / f"b2_window_{method}.mp4",
            output / "videos/replication" / f"{method}.mp4",
        )
    partial_methods = [
        "baseline", "global_surfelkv", "gated_surfelkv"
    ]
    if (
        partial_root
        / "videos/report/b2_window_token_selected_surfelkv.mp4"
    ).exists():
        partial_methods.append("token_selected_surfelkv")
    for method in partial_methods:
        _link(
            partial_root
            / "videos/report"
            / f"b2_window_{method}.mp4",
            output / "videos/partial" / f"{method}.mp4",
        )
    image_links = {
        "replication/surfel_center.png": replication_root
        / "surfel/surfel_center_preview.png",
        "replication/surfel_disk.png": replication_root
        / "surfel/surfel_disk_preview.png",
        "replication/support.png": replication_root
        / "retrieval/visible_support.png",
        "partial/surfel_disk.png": partial_root
        / "surfel/surfel_disk_preview.png",
        "partial/b1.png": partial_root
        / "assets/locality/b1_source.png",
        "partial/b1_warped.png": partial_root
        / "assets/locality/b1_warped_to_b2.png",
        "partial/b2.png": partial_root
        / "assets/locality/b2_baseline.png",
        "partial/support.png": partial_root
        / "assets/locality/b2_support_overlay.png",
        "partial/gate.png": partial_root
        / "generation/gated_surfelkv/masks"
        / f"chunk_{locality['target_chunk']:04d}_query_gate_overlay.png",
        "partial/retrieval.png": partial_root
        / "retrieval/retrieval_timeline.png",
        "partial/token_selection.png": partial_root
        / "retrieval"
        / f"selected_tokens_source_target_{locality['target_chunk']:04d}.png",
    }
    for relative, source in image_links.items():
        if source.exists():
            _link(source, output / "assets" / relative)

    architecture = {
        "backbone": "InSpatio-World-1.3B frozen student",
        "geometry": {
            "backend": "official CUT3R offline causal prefix",
            "pose": "known control c2w",
            "address": "radius-normal surfel",
            "retrieval": "eligibility-first visible observation voting",
        },
        "memory": {
            "payload": "whole clean post-RoPE recent-slot K/V",
            "injection": "replace_recent_delta",
            "alpha": 1.0,
            "steps": [0, 1, 2, 3],
            "target_scope": "whole B2 plateau",
        },
        "locality": {
            "global": "all query tokens receive A_hist - A_base",
            "gated": "projected selected-surfel support, dilation + smoothing",
            "token_selection_implemented": (
                "token_selected_surfelkv" in locality["methods"]
            ),
        },
    }
    (output / "architecture_state.json").write_text(
        json.dumps(architecture, indent=2), encoding="utf-8"
    )
    combined = {
        "status": status,
        "replication": replication,
        "partial_overlap": locality,
        "layer_budget": layers,
    }
    (output / "metrics.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
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
        "scene02_replication": replication["replication"]["status"],
        "partial_overlap": locality["status"],
        "git_commit": git_commit,
        "working_tree_clean": not bool(git_status),
        "report": str(output / "report.html"),
    }
    (output / "status.json").write_text(
        json.dumps(status_payload, indent=2), encoding="utf-8"
    )
    resolved = {
        "replication_root": str(replication_root),
        "partial_root": str(partial_root),
        "layer_root": str(layer_root),
        "seed": 0,
        "memory": {
            "mode": "replace_recent_delta",
            "alpha": 1.0,
            "steps": [0, 1, 2, 3],
            "target_scope": "whole_B2_plateau",
        },
        "partial_query_pose": locality["query_pose_mode"],
        "partial_locality_control": locality["addressing_context"],
    }
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    if git_status:
        (output / "git_diff_stat.txt").write_text(
            subprocess.check_output(
                ["git", "-C", str(repo), "diff", "--stat"], text=True
            ),
            encoding="utf-8",
        )

    replication_rows = "".join(
        "<tr><td>"
        + html.escape(method)
        + "</td><td>"
        + _fmt(
            replication["generation"][method][
                "b1_b2_generated_region_l1"
            ]
        )
        + "</td><td>"
        + _fmt(
            replication["generation"][method][
                "b2_vs_baseline_generated_region_l1"
            ]
        )
        + "</td><td>"
        + _fmt(replication["generation"][method]["target_boundary_l1"])
        + "</td></tr>"
        for method in ("baseline", "manualcorrect", "wrongkv", "surfelkv")
    )
    locality_labels = [
        ("baseline", "Baseline"),
        ("global_surfelkv", "Global RecentKV"),
        ("gated_surfelkv", "Surfel-Gated RecentKV"),
    ]
    if "token_selected_surfelkv" in locality["methods"]:
        locality_labels.append(
            (
                "token_selected_surfelkv",
                "Gated + Token-Selected RecentKV",
            )
        )
    locality_rows = "".join(
        "<tr><td>"
        + html.escape(label)
        + "</td><td>"
        + _fmt(locality["methods"][method]["overlap_b1_to_b2_l1"])
        + "</td><td>"
        + _fmt(
            locality["methods"][method][
                "nonoverlap_delta_vs_baseline_l1"
            ]
        )
        + "</td><td>"
        + _fmt(locality["methods"][method]["boundary_l1"])
        + "</td></tr>"
        for method, label in locality_labels
    )
    layer_rows = "".join(
        "<tr><td>"
        + html.escape(name)
        + "</td><td>"
        + str(values["num_layers"])
        + "</td><td>"
        + _fmt(values["b1_to_b2_generated_region_l1"])
        + "</td><td>"
        + _fmt(values["boundary_l1"])
        + "</td><td>"
        + _fmt(values["runtime_relative_to_all"], 3)
        + "×</td><td>"
        + _fmt(values["memory_relative_to_all"], 3)
        + "×</td></tr>"
        for name, values in layers["methods"].items()
    )
    selected = locality["retrieval"]["selected_chunks"]
    top3 = locality["retrieval"]["top3_chunks"]
    addressing = locality.get("addressing_context", {})
    unconstrained = addressing.get("unconstrained_selected_chunks")
    coverage = locality["coverage"]["raw_fraction"]
    decision = locality["decision"]
    if status == "QUERY_GATING_SUFFICIENT":
        interpretation = (
            "Query-side surfel gating retains most historical overlap recovery "
            "while reducing changes outside projected support. Token-level "
            "payload selection is not justified by this case."
        )
        next_action = (
            "Test the same gate on one translated or occluded partial-overlap case."
        )
    elif status == "TOKEN_SELECTION_NEEDED":
        interpretation = (
            "The gate is geometrically meaningful, but whole-chunk KV still "
            "mixes historical content outside the intended support."
        )
        next_action = "Implement the minimal surfel-to-historical-token selector."
    elif status == "REPLICATION_FAILED":
        interpretation = (
            "The second-scene same-pose control did not reproduce the established "
            "source-specific RecentKV pattern."
        )
        next_action = "Localize the scene02 capture or injection discrepancy."
    else:
        interpretation = (
            "At the changed B2 pose, whole-chunk B1 KV creates spatial duplication. "
            "Query gating and surfel-selected historical tokens reduce disturbance "
            "but neither beats the already-consistent baseline in the overlap."
        )
        next_action = (
            "Test target-position RoPE re-addressing for the selected historical "
            "tokens in this fixed partial-overlap case."
        )

    rep_videos = _video_grid(
        "rep",
        [
            ("Baseline", "videos/replication/baseline.mp4"),
            ("ManualRecentB1", "videos/replication/manualcorrect.mp4"),
            ("WrongRecent", "videos/replication/wrongkv.mp4"),
            ("SurfelKV", "videos/replication/surfelkv.mp4"),
        ],
    )
    partial_videos = _video_grid(
        "partial",
        [
            ("Baseline", "videos/partial/baseline.mp4"),
            ("Global SurfelKV", "videos/partial/global_surfelkv.mp4"),
            ("Surfel-Gated SurfelKV", "videos/partial/gated_surfelkv.mp4"),
        ]
        + (
            [
                (
                    "Gated + Token-Selected",
                    "videos/partial/token_selected_surfelkv.mp4",
                )
            ]
            if "token_selected_surfelkv" in locality["methods"]
            else []
        ),
    )
    html_payload = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MapKV locality stage</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#667085;--accent:#3157c8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:24px}}
section{{background:var(--card);margin:16px 0;padding:20px;border-radius:12px;
box-shadow:0 2px 10px #2233aa12}}h1,h2{{margin-top:0}}.status{{font-size:22px;
font-weight:750;color:var(--accent)}}.grid{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
figure{{margin:0}}figcaption{{font-weight:650;margin:0 0 6px}}
video,img{{width:100%;border-radius:8px;background:#10131a}}table{{width:100%;
border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e5e8ef;
text-align:right}}th:first-child,td:first-child{{text-align:left}}
button{{padding:8px 12px;margin:0 6px 10px 0}}code,pre{{background:#eef1f8;
padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<section><h1>MapKV: Replication → Partial-Overlap Locality</h1>
<div class="status">{status}</div>
<p><b>Scene02 replication:</b> {replication['replication']['status']} ·
<b>Partial locality payload:</b> selected {selected}, coverage {_fmt(coverage)} ·
<b>Best locality method:</b> {'Surfel-Gated RecentKV' if status == 'QUERY_GATING_SUFFICIENT' else 'Unresolved'}</p>
<p><b>Addressing context:</b> unconstrained top-1 {unconstrained}; the where-only
ablation uses the declared B1 candidate cluster {locality['retrieval'].get('candidate_chunks')}.
This control is reported separately and is not claimed as unconstrained retrieval.</p>
<p>{html.escape(interpretation)}</p></section>
<section><h2>Architecture</h2>
<pre>Known control c2w → fixed-pose CUT3R → radius-normal surfel
→ eligible-first chunk vote → clean historical recent-slot KV
→ replace_recent_delta (global or projected-surfel query gate)</pre>
<p><b>Architecture changes:</b> the partial-overlap query uses the actual known B2
pose. P0–P2 keep the whole-chunk payload and gate only A_hist − A_base. Conditional
P3 subselects B1 tokens through visible surfels without spatial RoPE re-addressing.</p></section>
<section><h2>Scene02 same-pose replication</h2>
<table><tr><th>Method</th><th>B1→B2 generated L1 ↓</th><th>Δ vs baseline</th>
<th>Boundary ↓</th></tr>{replication_rows}</table>
<p>Retrieved chunk {replication['replication']['selected_chunk']}; B1 cluster
{replication['replication']['positive_cluster']}; SurfelKV↔Manual L1
{_fmt(replication['replication']['surfel_vs_manual_generated_region_l1'])}.</p>
<button onclick="playGroup('rep')">Play all</button><button onclick="pauseGroup('rep')">Pause all</button>
<button onclick="resetGroup('rep')">Reset all</button><div class="grid">{rep_videos}</div></section>
<section><h2>Partial-overlap geometry</h2><div class="grid">
<figure><figcaption>Surfel disks</figcaption><img src="assets/partial/surfel_disk.png"></figure>
<figure><figcaption>Retrieved B1</figcaption><img src="assets/partial/b1.png"></figure>
<figure><figcaption>B1 warped to B2 for metric</figcaption><img src="assets/partial/b1_warped.png"></figure>
<figure><figcaption>B2 baseline</figcaption><img src="assets/partial/b2.png"></figure>
<figure><figcaption>Projected support</figcaption><img src="assets/partial/support.png"></figure>
<figure><figcaption>Actual token query gate</figcaption><img src="assets/partial/gate.png"></figure>
<figure><figcaption>Selected historical KV tokens</figcaption><img src="assets/partial/token_selection.png"></figure></div>
<p>Query pose mode: <code>{locality['query_pose_mode']}</code>; visible surfels
{locality['retrieval']['num_visible_surfels']}; raw coverage {_fmt(coverage)}.</p></section>
<section><h2>Global vs spatially gated RecentKV</h2>
<table><tr><th>Method</th><th>Overlap B1→B2 ↓</th>
<th>Non-overlap Δ vs baseline ↓</th><th>Boundary ↓</th></tr>{locality_rows}</table>
<p>Gated recovery retained: {_fmt(decision['gated_recovery_retained'])};
non-overlap disturbance ratio: {_fmt(decision['gated_to_global_disturbance_ratio'])}.</p>
<button onclick="playGroup('partial')">Play all</button><button onclick="pauseGroup('partial')">Pause all</button>
<button onclick="resetGroup('partial')">Reset all</button><div class="grid">{partial_videos}</div></section>
<section><h2>Layer budget</h2><table><tr><th>Budget</th><th>Layers</th>
<th>B1→B2 generated L1 ↓</th><th>Boundary ↓</th><th>Runtime / All</th>
<th>KV bytes / All</th></tr>{layer_rows}</table></section>
<section><h2>Final interpretation</h2><p>{html.escape(interpretation)}</p>
<p><b>Next single experiment:</b> {html.escape(next_action)}</p></section>
</main><script>
function group(n){{return [...document.querySelectorAll('video.'+n)]}}
function playGroup(n){{let v=group(n);if(!v.length)return;let t=v[0].currentTime;
v.forEach(x=>{{x.currentTime=t;x.play()}})}}function pauseGroup(n){{group(n).forEach(x=>x.pause())}}
function resetGroup(n){{group(n).forEach(x=>{{x.pause();x.currentTime=0}})}}
</script></body></html>"""
    (output / "report.html").write_text(html_payload, encoding="utf-8")
    markdown = f"""# MapKV Next-Stage Report

Status: **{status}**

## Summary

- Scene02 replication: **{replication['replication']['status']}**
- Partial unconstrained retrieval: {unconstrained}
- Partial B1-locality control: selected {selected}, candidates {locality['retrieval'].get('candidate_chunks')}, coverage {_fmt(coverage)}
- Query pose: {locality['query_pose_mode']}
- Conclusion: {interpretation}

## Architecture

known c2w → fixed-pose CUT3R → radius-normal surfel → chunk vote
→ clean recent KV → replace_recent_delta

## Partial-overlap locality

| Method | Overlap B1→B2 ↓ | Non-overlap Δ vs baseline ↓ | Boundary ↓ |
|---|---:|---:|---:|
| Baseline | {_fmt(locality['methods']['baseline']['overlap_b1_to_b2_l1'])} | 0 | {_fmt(locality['methods']['baseline']['boundary_l1'])} |
| Global RecentKV | {_fmt(locality['methods']['global_surfelkv']['overlap_b1_to_b2_l1'])} | {_fmt(locality['methods']['global_surfelkv']['nonoverlap_delta_vs_baseline_l1'])} | {_fmt(locality['methods']['global_surfelkv']['boundary_l1'])} |
| Surfel-Gated | {_fmt(locality['methods']['gated_surfelkv']['overlap_b1_to_b2_l1'])} | {_fmt(locality['methods']['gated_surfelkv']['nonoverlap_delta_vs_baseline_l1'])} | {_fmt(locality['methods']['gated_surfelkv']['boundary_l1'])} |
{"| Token-Selected | " + _fmt(locality['methods']['token_selected_surfelkv']['overlap_b1_to_b2_l1']) + " | " + _fmt(locality['methods']['token_selected_surfelkv']['nonoverlap_delta_vs_baseline_l1']) + " | " + _fmt(locality['methods']['token_selected_surfelkv']['boundary_l1']) + " |" if 'token_selected_surfelkv' in locality['methods'] else ""}

## Next action

{next_action}
"""
    (output / "report.md").write_text(markdown, encoding="utf-8")
    return str(output / "report.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MapKV next-stage report")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--replication_root", required=True)
    parser.add_argument("--partial_root", required=True)
    parser.add_argument("--layer_root", required=True)
    args = parser.parse_args()
    print(
        build_report(
            output_root=args.output_root,
            replication_root=args.replication_root,
            partial_root=args.partial_root,
            layer_root=args.layer_root,
        )
    )


if __name__ == "__main__":
    main()
