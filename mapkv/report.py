from __future__ import annotations

import argparse
import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


VALID_STATUS = {
    "SMOKE_OK", "SANITY_OK", "CLOSED_LOOP_OK", "GO", "CONTINUE",
    "NO_GO", "INVALID", "IMPLEMENTATION_BLOCKED",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def generate_report(
    *, run_root: str | Path, status: str, conclusion: str, next_action: str
) -> None:
    if status not in VALID_STATUS:
        raise ValueError(f"Unsupported status: {status}")
    root = Path(run_root).resolve()
    metrics = _json(root / "metrics.json")
    architecture = _json(root / "architecture_state.json")
    retrieval = metrics["retrieval"]
    cut3r = metrics["cut3r"]
    surfel = metrics["surfel"]
    bank = _json(root / "kv" / "bank_stats.json")
    capture = _json(root / "kv" / "capture_manifest.json")
    config = _json(root / "config_resolved.json")
    baseline_metadata = _json(root / "baseline" / "run_metadata.json")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    generated = metrics["generation"]
    generated_at = datetime.now(timezone.utc).isoformat()
    methods = [
        name for name in ("baseline", "surfelkv", "wrongkv", "posekv", "manualcorrect")
        if name in generated
    ]
    metrics["status"] = status
    metrics["conclusion"] = conclusion
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (root / "status.json").write_text(
        json.dumps(
            {
                "status": status,
                "conclusion": conclusion,
                "next_action": next_action,
                "generated_at": generated_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    architecture_text = (
        "# Architecture snapshot\n\n"
        "InSpatio base attention + selected-layer residual memory attention\n"
        "<- whole historical native post-RoPE K/V chunks\n"
        "<- visible CUT3R voxel-surfel chunk vote.\n\n"
        f"- Memory layers: {architecture['memory']['layers']}\n"
        f"- Alpha: {architecture['injection']['alpha']}\n"
        f"- Gate: {architecture['injection']['gate']}\n"
        f"- Top-K: {architecture['retrieval']['top_k']}\n"
        f"- Query pose: {architecture['retrieval']['query_pose_mode']}\n"
        f"- Prefix cutoff: chunk {architecture['geometry']['prefix_last_chunk']}\n"
    )
    (root / "architecture.md").write_text(architecture_text, encoding="utf-8")

    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{_fmt(values['b1_b2_generated_region_l1'])}</td>"
        f"<td>{_fmt(values['b1_b2_whole_frame_l1'])}</td>"
        f"<td>{_fmt(values['b1_b2_pooled_feature_cosine'])}</td>"
        f"<td>{_fmt(values['target_boundary_l1'])}</td>"
        f"<td>{_fmt(values['generation_seconds'], 2)}</td>"
        "</tr>"
        for name, values in generated.items()
    )
    videos = "\n".join(
        "<figure><figcaption>"
        f"{html.escape(name)}</figcaption><video class='sync' controls preload='metadata' "
        f"poster='assets/posters/{html.escape(name)}.jpg' "
        f"src='videos/report/{html.escape(name)}.mp4'></video></figure>"
        for name in methods
        if (root / "videos" / "report" / f"{name}.mp4").exists()
    )
    b2_videos = "\n".join(
        "<figure><figcaption>"
        f"{html.escape(name)} B2 window</figcaption><video class='sync-b2' controls "
        f"preload='metadata' src='videos/report/b2_window_{html.escape(name)}.mp4'></video></figure>"
        for name in methods
        if (root / "videos" / "report" / f"b2_window_{name}.mp4").exists()
    )
    retrieved_rows = "\n".join(
        "<tr>"
        f"<td>{item['chunk_id']}</td><td>{_fmt(item['score'])}</td>"
        f"<td>{item['visible_support']}</td><td>{_fmt(item['mean_confidence'])}</td>"
        f"<td>{item['temporal_gap_chunks']}</td></tr>"
        for item in retrieval.get("retrieved", [])
    )
    capture_by_chunk = {int(item["chunk_id"]): item for item in capture["chunks"]}
    capture_rows = []
    for label, chunk_id in (
        ("B1/correct", int(config["source_chunk"])),
        ("wrong", int(config["wrong_chunk"])),
    ):
        for layer, values in capture_by_chunk[chunk_id]["layers"].items():
            capture_rows.append(
                "<tr>"
                f"<td>{label} ({chunk_id})</td><td>{layer}</td>"
                f"<td>{_fmt(values['k_stats']['mean'])} / {_fmt(values['k_stats']['std'])} / {_fmt(values['k_stats']['l2_norm'], 2)}</td>"
                f"<td>{_fmt(values['v_stats']['mean'])} / {_fmt(values['v_stats']['std'])} / {_fmt(values['v_stats']['l2_norm'], 2)}</td>"
                f"<td><code>{values['sha256'][:16]}...</code></td></tr>"
            )
    capture_rows_html = "\n".join(capture_rows)
    dirty_note = "dirty (git_diff_stat.txt saved)" if git_status else "clean"
    report_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MapKV {html.escape(root.name)}</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#17212b}}
main{{max-width:1180px;margin:auto;padding:28px}}section{{background:white;padding:20px;margin:16px 0;border-radius:10px;box-shadow:0 1px 5px #ccd2d8}}
h1,h2{{margin-top:0}}.status{{display:inline-block;padding:5px 10px;border-radius:20px;background:#19324a;color:white}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}}video,img{{width:100%;border-radius:6px;background:#111}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#eef;padding:2px 4px}}
.controls button{{margin-right:8px;padding:7px 12px}}small{{color:#536273}}pre{{white-space:pre-wrap}}
</style></head><body><main>
<section><h1>MapKV rapid prototype</h1><p class="status">{status}</p>
<p><b>Run:</b> {html.escape(root.name)} - <b>Case:</b> {html.escape(config['case'])} - <b>Seed:</b> {config['seed']}</p>
<p><b>Date:</b> {generated_at} UTC</p>
<p><b>Conclusion:</b> {html.escape(conclusion)}</p>
<small>Commit {git_commit} - branch {html.escape(git_branch)} - worktree {dirty_note}</small></section>

<section><h2>Reproducibility</h2><table>
<tr><th>InSpatio GPU</th><td>{html.escape(baseline_metadata['gpu'])} (single GPU)</td></tr>
<tr><th>CUT3R GPU</th><td>{html.escape(cut3r['gpu'])} (separate process, single GPU)</td></tr>
<tr><th>Baseline generation commit</th><td><code>{baseline_metadata['git_commit']}</code>; dirty={str(bool(baseline_metadata['git_status'])).lower()} (diff stat saved)</td></tr>
<tr><th>Report/code commit</th><td><code>{git_commit}</code>; dirty={str(bool(git_status)).lower()}</td></tr>
<tr><th>InSpatio checkpoint</th><td><code>{html.escape(config['inspatio_checkpoint'])}</code></td></tr>
<tr><th>CUT3R</th><td><code>{html.escape(cut3r['cut3r_commit'])}</code>; dirty={str(cut3r['cut3r_dirty']).lower()}</td></tr>
<tr><th>CUT3R checkpoint</th><td><code>{html.escape(config['cut3r_checkpoint'])}</code><br><small>sha256 {cut3r['checkpoint_sha256']}</small></td></tr>
<tr><th>Trajectory sha256</th><td><code>{config['trajectory_sha256']}</code></td></tr>
<tr><th>Source sha256</th><td><code>{config['source_sha256']}</code></td></tr>
<tr><th>Seed</th><td>{config['seed']}</td></tr></table></section>

<section><h2>Architecture snapshot</h2>
<pre>InSpatio base attention
    + alpha * residual memory attention
              ^ native post-RoPE K/V, {len(architecture['memory']['layers'])} layers
              ^ top-{architecture['retrieval']['top_k']} historical chunk
              ^ CUT3R visible voxel-surfel vote</pre>
<table><tr><th>Payload</th><td>{architecture['memory']['payload']}</td></tr>
<tr><th>Geometry</th><td>{architecture['geometry']['backend']} / {architecture['geometry']['mode']}</td></tr>
<tr><th>Query pose</th><td>{architecture['retrieval']['query_pose_mode']}</td></tr>
<tr><th>Alpha / gate</th><td>{architecture['injection']['alpha']} / {architecture['injection']['gate']}</td></tr>
<tr><th>Layers</th><td>{architecture['memory']['layers']}</td></tr></table></section>

<section><h2>Architecture changes in this run</h2>
<table><tr><th>Component</th><th>Previous</th><th>Current</th><th>Files</th><th>Why</th></tr>
<tr><td>Self-attention</td><td>base or replace-recent delta</td><td>base + residual MapKV branch</td><td>causal_model.py, memory_attention.py</td><td>v0.4 closed-loop diagnostic</td></tr>
<tr><td>History</td><td>manual chunk only</td><td>CPU KVChunkBank, runtime uniform8</td><td>kv_bank.py</td><td>native payload lookup</td></tr>
<tr><td>Retrieval</td><td>manual/pose</td><td>official CUT3R voxel-surfel vote</td><td>cut3r_adapter.py, surfel_index.py, retrieval.py</td><td>geometry address</td></tr></table></section>

<section><h2>Input and trajectory</h2><div class="grid">
<img src="assets/posters/source.jpg" alt="source frame">
<img src="assets/plots/trajectory.png" alt="exact trajectory">
<img src="assets/plots/pair_contact_sheet.png" alt="B1 B2 pair">
<img src="baseline/masks/chunk_{metrics['trajectory']['b2_chunk']:04d}_generated_region.png" alt="B2 generated-region mask"></div>
<p>Pure yaw 0 to {metrics['trajectory']['angle_deg']} deg to 0 to {metrics['trajectory']['angle_deg']} deg; pitch/roll/translation = 0.
B1 chunk {metrics['trajectory']['b1_chunk']}; B2 chunk {metrics['trajectory']['b2_chunk']}; gap {metrics['trajectory']['history_gap_chunks']} chunks.</p>
<p>Reference-blind fraction: {_fmt(config.get('reference_blind_fraction'))}. CUT3R prefix last chunk:
{cut3r['prefix_last_chunk']} &lt; target {cut3r['target_chunk']}; future leakage = {str(cut3r['future_leakage']).lower()}.</p></section>

<section><h2>KV sanity</h2>
<p>Alpha=0 max difference: {_fmt(metrics['kv_sanity']['alpha0_vs_baseline'])}.
Preserved baseline max difference: {_fmt(metrics['kv_sanity'].get('preserved_baseline_max_abs_diff'))}.
Memory branch effect: {metrics['kv_sanity']['memory_branch_effect']}. Capture: clean context / post-RoPE.
Chunks: {bank['num_chunks']}; layers: {bank['selected_layers']}; KV bytes: {bank['memory_bytes']}.
Runtime cache unchanged: {metrics['kv_sanity']['runtime_cache_unchanged']}.</p>
<table><tr><th>Chunk</th><th>Layer</th><th>K mean / std / L2</th><th>V mean / std / L2</th><th>File SHA256</th></tr>{capture_rows_html}</table></section>

<section><h2>CUT3R / surfel diagnostics</h2><div class="grid">
<img src="cut3r/pointcloud_preview.png" alt="point cloud">
<img src="cut3r/camera_trajectory.png" alt="CUT3R camera trajectory">
<img src="cut3r/confidence_histogram.png" alt="confidence">
<img src="surfel/surfel_preview.png" alt="surfel"></div>
<p>CUT3R frames: {cut3r['frames']}; accepted ratio: {_fmt(cut3r['accepted_point_ratio'])};
surfel cells: {surfel['num_cells']}; voxel size: {_fmt(surfel['voxel_size'])};
mean observing chunks/cell: {_fmt(surfel['mean_observing_chunks_per_cell'])}.</p>
<p><a href="cut3r/coordinate_convention.md">Coordinate convention</a></p></section>

<section><h2>Retrieval explanation</h2><div class="grid">
<img src="retrieval/retrieval_timeline.png" alt="retrieval timeline">
<img src="retrieval/visible_support.png" alt="visible support"></div>
<p>Target {retrieval['target_chunk']}; visible surfels {retrieval['num_visible_surfels']};
selected {retrieval['selected_chunks']}; latency {_fmt(retrieval['retrieval_ms'], 2)} ms.</p>
<table><tr><th>Chunk</th><th>Score</th><th>Visible support</th><th>Mean confidence</th><th>Gap</th></tr>{retrieved_rows}</table></section>

<section><h2>Video comparison</h2><p>B1/B2 RGB centers: {config['source_rgb_index']} / {config['target_rgb_index']}.</p>
<div class="controls"><button onclick="allPlay()">Play all</button><button onclick="allPause()">Pause all</button><button onclick="allReset()">Reset all</button></div>
<div class="grid">{videos}</div>
<h3>B2 windows</h3><div class="grid">{b2_videos}</div></section>

<section><h2>Metrics</h2><table><tr><th>Method</th><th>Generated-mask L1 (lower)</th><th>Whole L1 (lower)</th><th>Feature cosine (higher)</th><th>Boundary L1 (lower)</th><th>Total s</th></tr>{rows}</table></section>

<section><h2>Findings and next action</h2>
<p><b>What worked:</b> causal CUT3R to surfel to chunk to KV to B2 generation completed.</p>
<p><b>What failed:</b> see conclusion above; retrieval and video evidence are shown without an Oracle gate.</p>
<p><b>What is uncertain:</b> whole-chunk post-RoPE KV may be too coarse even when addressing is correct.</p>
<p><b>Next:</b> {html.escape(next_action)}</p></section>
</main><script>
const vids=[...document.querySelectorAll('video.sync')];
const b2vids=[...document.querySelectorAll('video.sync-b2')];
function allPlay(){{const t=Math.min(...vids.map(v=>v.currentTime));vids.forEach(v=>{{v.currentTime=t;v.play();}});b2vids.forEach(v=>{{v.currentTime=0;v.play();}})}}
function allPause(){{[...vids,...b2vids].forEach(v=>v.pause())}}
function allReset(){{[...vids,...b2vids].forEach(v=>{{v.pause();v.currentTime=0;}})}}
</script></body></html>"""
    (root / "report.html").write_text(report_html, encoding="utf-8")

    video_lines = "\n".join(
        f"- {name}: videos/original/{name}.mp4" for name in methods
    )
    markdown = f"""# MapKV rapid prototype

- Status: **{status}**
- Run: `{root.name}`
- Commit: `{git_commit}`
- Baseline generation commit / dirty: `{baseline_metadata['git_commit']}` / `{bool(baseline_metadata['git_status'])}`
- InSpatio / CUT3R GPU: `{baseline_metadata['gpu']}` / `{cut3r['gpu']}` (separate single-GPU processes)
- InSpatio checkpoint: `{config['inspatio_checkpoint']}`
- CUT3R commit / checkpoint SHA256: `{cut3r['cut3r_commit']}` / `{cut3r['checkpoint_sha256']}`
- Trajectory / source SHA256: `{config['trajectory_sha256']}` / `{config['source_sha256']}`
- Case / seed: `{config['case']}` / `{config['seed']}`
- Conclusion: {conclusion}

## Architecture snapshot

{architecture_text}

## Retrieval

- Target chunk: {retrieval['target_chunk']}
- Visible surfels: {retrieval['num_visible_surfels']}
- Retrieved chunks: {retrieval['selected_chunks']}
- Scores: {retrieval['scores']}
- Prefix cutoff: {cut3r['prefix_last_chunk']}
- Future leakage: {cut3r['future_leakage']}

## Key metrics

| Method | Generated-mask L1 | Whole L1 | Feature cosine | Boundary L1 |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {name} | {_fmt(value['b1_b2_generated_region_l1'])} | "
        f"{_fmt(value['b1_b2_whole_frame_l1'])} | "
        f"{_fmt(value['b1_b2_pooled_feature_cosine'])} | "
        f"{_fmt(value['target_boundary_l1'])} |"
        for name, value in generated.items()
    ) + f"""

## Videos

{video_lines}

## Next action

{next_action}
"""
    (root / "report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact MapKV HTML/Markdown reports")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--status", choices=sorted(VALID_STATUS), default="CLOSED_LOOP_OK")
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
