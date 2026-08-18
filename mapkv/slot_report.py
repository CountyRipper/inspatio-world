from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import subprocess
from pathlib import Path


VALID_STATUS = {
    "KV_RECENT_VIABLE",
    "KV_REFERENCE_VIABLE",
    "KV_MULTI_SLOT_VIABLE",
    "KV_TRAINING_FREE_LIMITED",
    "INCONCLUSIVE",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "recentzero": "RecentZero",
    "recentwrong": "RecentWrong",
    "recentb1": "RecentB1",
    "refzero": "RefZero",
    "refwrong": "RefWrong",
    "refb1": "RefB1",
    "bothwrong": "BothWrong",
    "bothb1": "BothB1",
    "latenthard": "LatentHard",
    "latentsoft": "LatentSoft",
    "surfelkv": "SurfelKV(best slot)",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, str)):
        return str(value)
    return f"{float(value):.{digits}f}"


def _video(root: Path, name: str) -> str:
    path = root / "videos/report" / f"b2_window_{name}.mp4"
    if not path.exists():
        return ""
    poster = root / "assets/posters" / f"{name}.jpg"
    poster_attr = (
        f" poster='assets/posters/{html.escape(name)}.jpg'" if poster.exists() else ""
    )
    return (
        "<figure><figcaption>"
        + html.escape(METHOD_LABELS.get(name, name))
        + "</figcaption><video class='sync' controls preload='metadata'"
        + poster_attr
        + f" src='videos/report/b2_window_{html.escape(name)}.mp4'></video></figure>"
    )


def _grid(root: Path, names: list[str]) -> str:
    return "<div class='video-grid'>" + "".join(_video(root, name) for name in names) + "</div>"


def generate_report(
    *, run_root: str | Path, status: str, conclusion: str, next_action: str
) -> None:
    if status not in VALID_STATUS:
        raise ValueError(f"Unsupported context-slot status: {status}")
    root = Path(run_root).resolve()
    config = _json(root / "config_resolved.json")
    architecture = _json(root / "architecture_state.json")
    metrics = _json(root / "metrics.json")
    retrieval_payload = _json(root / "retrieval/retrieval.json")
    retrieval = next(
        item
        for item in retrieval_payload["targets"]
        if int(item["target_chunk"]) == int(config["target_chunk"])
    )
    slot = metrics["slot_ablation"]
    generated = metrics["generation"]
    groups = slot["groups"]
    best_slot = slot["most_influential_context_channel"]
    best_method = slot["best_manual_method"]
    baseline_error = generated["baseline"]["b1_b2_generated_region_l1"]

    order = [
        "baseline", "recentzero", "recentwrong", "recentb1",
        "refzero", "refwrong", "refb1", "bothwrong", "bothb1",
        "latenthard", "latentsoft", "surfelkv",
    ]
    rows = []
    for name in order:
        if name not in generated:
            continue
        item = generated[name]
        visual = (
            "baseline"
            if name == "baseline"
            else (
                "upper bound"
                if name.startswith("latent")
                else "inspect synchronized clip"
            )
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(METHOD_LABELS.get(name, name))}</td>"
            f"<td>{_fmt(item['b1_b2_generated_region_l1'])}</td>"
            f"<td>{_fmt(item['b2_vs_baseline_generated_region_l1'])}</td>"
            f"<td>{_fmt(item['target_boundary_l1'])}</td>"
            f"<td>{html.escape(visual)}</td></tr>"
        )
    group_rows = []
    for name in ("recent", "reference", "both"):
        if name not in groups:
            continue
        item = groups[name]
        group_rows.append(
            "<tr>"
            f"<td>{name.title()}</td>"
            f"<td>{_fmt(item['correct_intervention_l1'])}</td>"
            f"<td>{_fmt(item['wrong_intervention_l1'])}</td>"
            f"<td>{_fmt(item['source_specificity_error_margin'])}</td>"
            f"<td>{_fmt(item['correct_vs_wrong_generated_region_l1'])}</td>"
            f"<td>{_fmt(item['correct_improvement_vs_baseline'])}</td></tr>"
        )

    latent_hard = generated.get("latenthard", {})
    best = groups[best_slot]
    latent_gap = (
        float(latent_hard.get("b1_b2_generated_region_l1", baseline_error))
        - float(generated[best_method]["b1_b2_generated_region_l1"])
    )
    surfel_match = slot.get("surfel_vs_best_manual") or {}
    source_separation = best["source_specificity_error_margin"]
    git_commit = subprocess.check_output(
        ["git", "-C", str(root.parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    report = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>MapKV context-slot ablation</title><style>
body{{margin:0;background:#f2f5f7;color:#18232d;font:15px/1.48 system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:24px}}section{{background:#fff;margin:14px 0;padding:20px;border-radius:10px;box-shadow:0 1px 5px #ccd4da}}
h1,h2,h3{{margin-top:0}}.status{{display:inline-block;background:#183b56;color:#fff;padding:6px 11px;border-radius:20px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:7px;border-bottom:1px solid #dde2e6;text-align:left}}th{{background:#f7f9fa}}
.video-grid,.image-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
video,img{{width:100%;border-radius:6px;background:#111}}figure{{margin:0}}figcaption{{font-weight:650;margin:5px 0}}
button{{margin:0 8px 10px 0;padding:7px 12px}}code,pre{{background:#f0f3f5}}pre{{padding:12px;white-space:pre-wrap}}.note{{color:#51616d}}
</style></head><body><main>
<section><h1>MapKV Context-Slot Ablation</h1><p class='status'>{html.escape(status)}</p>
<p><b>Most influential frozen context channel:</b> {html.escape(best_slot.title())}</p>
<p>{html.escape(conclusion)}</p><p class='note'>Run {html.escape(root.name)} · commit {git_commit} · {generated_at}</p></section>

<section><h2>Architecture</h2><pre>Known camera pose
→ frozen fixed-pose CUT3R / surfel retrieval
→ Manual B1 chunk for slot topology
→ post-RoPE native KV
→ Recent / Reference / Both counterfactual replacement
→ alpha=1, all {len(architecture['memory']['layers'])} layers, all 4 steps, B2 chunks {architecture['intervention']['target_chunks']}</pre>
<p>Recent payload layout: <code>{html.escape(architecture['memory']['recent_layout'])}</code>. Reference payload is the same clean B1 generated latent re-encoded as <code>{html.escape(architecture['memory']['reference_layout'])}</code>.</p>
<h3>Architecture Changes</h3><table><tr><th>Component</th><th>Before</th><th>After</th></tr>
<tr><td>Context intervention</td><td>recent slot only, uniform8</td><td>Recent / Reference / Both, all layers</td></tr>
<tr><td>Reference payload</td><td>none</td><td>clean B1 re-encoded with t0–t2 reference RoPE</td></tr>
<tr><td>Geometry</td><td>correct B1 retrieval</td><td>unchanged; frozen plan checksum <code>{html.escape(config['retrieval_plan_sha256'][:16])}…</code></td></tr>
<tr><td>Latent upper bound</td><td>not in MapKV branch</td><td>direct clean-x0 block Hard/Soft diagnostic; explicitly not native KV nor the separately trained Conv3D sidecar</td></tr></table></section>

<section><h2>Context-slot summary</h2><table><tr><th>Method</th><th>B1→B2 generated L1 ↓</th><th>Δ vs baseline</th><th>Boundary L1</th><th>Visual</th></tr>{''.join(rows)}</table>
<h3>Source specificity by channel</h3><table><tr><th>Channel</th><th>Correct effect</th><th>Wrong effect</th><th>Wrong−Correct B1 error</th><th>Correct↔Wrong L1</th><th>Correct improvement</th></tr>{''.join(group_rows)}</table></section>

<section><h2>Synchronized B2 videos</h2><button onclick='playAll()'>Play all</button><button onclick='pauseAll()'>Pause all</button><button onclick='resetAll()'>Reset all</button>
<h3>Recent slot</h3>{_grid(root, ['baseline','recentzero','recentwrong','recentb1'])}
<h3>Reference slot</h3>{_grid(root, ['baseline','refzero','refwrong','refb1'])}
<h3>Best native KV vs latent upper bound</h3>{_grid(root, ['baseline',best_method,'latenthard','latentsoft'])}
<h3>Joint slot and restored SurfelKV</h3>{_grid(root, ['baseline','bothwrong','bothb1','surfelkv'])}</section>

<section><h2>Surfel (visualization only; algorithm frozen)</h2><div class='image-grid'>
<figure><figcaption>Center scatter</figcaption><img src='surfel/surfel_center_preview.png'></figure>
<figure><figcaption>Oriented disks</figcaption><img src='surfel/surfel_disk_preview.png'></figure>
<figure><figcaption>B2 projected coverage</figcaption><img src='retrieval/visible_support.png'></figure></div>
<p>Retrieved chunk <b>{retrieval['selected_chunks']}</b>; B1 cluster {retrieval.get('positive_cluster')}; coverage {_fmt(retrieval.get('coverage_fraction'))}; positive-cluster hit={_fmt(retrieval.get('positive_cluster_hit'))}. SurfelKV↔BestManual generated-region L1: {_fmt(surfel_match.get('generated_region_l1'))}.</p></section>

<section><h2>Final interpretation</h2><ol>
<li><b>Recent slot:</b> correct effect {_fmt(groups['recent']['correct_intervention_l1'])}, Wrong−Correct source margin {_fmt(groups['recent']['source_specificity_error_margin'])}.</li>
<li><b>Reference slot:</b> correct effect {_fmt(groups['reference']['correct_intervention_l1'])}, Wrong−Correct source margin {_fmt(groups['reference']['source_specificity_error_margin'])}.</li>
<li><b>Correct vs Wrong:</b> best channel {html.escape(best_slot)}, source-dependent error separation {_fmt(source_separation)}.</li>
<li><b>Native KV vs LatentHard:</b> best-KV error {_fmt(generated[best_method]['b1_b2_generated_region_l1'])}; LatentHard error {_fmt(latent_hard.get('b1_b2_generated_region_l1'))}; signed latent-minus-KV gap {_fmt(latent_gap)}.</li>
<li><b>Path:</b> {html.escape(next_action)}</li></ol>
<p><b>Final status:</b> {html.escape(status)}</p></section>
</main><script>
const vids=[...document.querySelectorAll('video.sync')];
function playAll(){{const t=Math.min(...vids.map(v=>v.currentTime));vids.forEach(v=>{{v.currentTime=t;v.play();}})}}
function pauseAll(){{vids.forEach(v=>v.pause())}}
function resetAll(){{vids.forEach(v=>{{v.pause();v.currentTime=0}})}}
</script></body></html>"""
    (root / "report.html").write_text(report, encoding="utf-8")

    markdown_rows = "\n".join(
        "| " + METHOD_LABELS.get(name, name) + " | "
        + _fmt(generated[name]["b1_b2_generated_region_l1"]) + " | "
        + _fmt(generated[name]["b2_vs_baseline_generated_region_l1"]) + " | "
        + _fmt(generated[name]["target_boundary_l1"]) + " |"
        for name in order if name in generated
    )
    markdown = f"""# MapKV Context-Slot Ablation

- Status: **{status}**
- Most influential context channel: **{best_slot.title()}**
- Best native-KV method: `{best_method}`
- Retrieval: `{retrieval['selected_chunks']}`; B1 cluster `{retrieval.get('positive_cluster')}`; coverage `{_fmt(retrieval.get('coverage_fraction'))}`
- Config: alpha=1, all layers, all steps, whole B2, global gate
- Conclusion: {conclusion}

| Method | B1→B2 generated L1 | Δ vs baseline | Boundary L1 |
|---|---:|---:|---:|
{markdown_rows}

## Interpretation

1. Recent source margin: `{_fmt(groups['recent']['source_specificity_error_margin'])}`.
2. Reference source margin: `{_fmt(groups['reference']['source_specificity_error_margin'])}`.
3. Best channel: `{best_slot}`.
4. BestKV / LatentHard B1→B2 error: `{_fmt(generated[best_method]['b1_b2_generated_region_l1'])}` / `{_fmt(latent_hard.get('b1_b2_generated_region_l1'))}`.
5. SurfelKV / BestManual difference: `{_fmt(surfel_match.get('generated_region_l1'))}`.

## Videos

- `videos/report/b2_window_recentb1.mp4`
- `videos/report/b2_window_refb1.mp4`
- `videos/report/b2_window_bothb1.mp4`
- `videos/report/b2_window_latenthard.mp4`
- `videos/report/b2_window_latentsoft.mp4`
- `videos/report/b2_window_surfelkv.mp4`

## Next action

{next_action}
"""
    (root / "report.md").write_text(markdown, encoding="utf-8")
    (root / "status.json").write_text(
        json.dumps(
            {
                "status": status,
                "most_influential_context_channel": best_slot,
                "best_manual_method": best_method,
                "conclusion": conclusion,
                "next_action": next_action,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    architecture_text = (
        "# Architecture\n\nKnown-pose frozen CUT3R/surfel retrieval → chunk-native "
        "post-RoPE KV → recent/reference/both delta intervention. Geometry and "
        "retrieval are unchanged in this run.\n"
    )
    (root / "architecture.md").write_text(architecture_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MapKV context-slot report")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--status", choices=sorted(VALID_STATUS), required=True)
    parser.add_argument("--conclusion", required=True)
    parser.add_argument("--next_action", required=True)
    args = parser.parse_args()
    generate_report(
        run_root=args.run_root,
        status=args.status,
        conclusion=args.conclusion,
        next_action=args.next_action,
    )


if __name__ == "__main__":
    main()
