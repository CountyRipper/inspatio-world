from __future__ import annotations

import html
import json
from pathlib import Path

from .memory_interface_evaluation import METHOD_LABELS_ZH, METHOD_ROOTS
from .report_framework import (
    ArchitectureChange, ArchitectureEdge, ArchitectureSnapshot, node,
    render_changes_html, render_pipeline_table_html, write_architecture_bundle,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _architecture() -> ArchitectureSnapshot:
    nodes = (
        node(id="input", label_zh="受控静态回访输入", label_en="Controlled revisit", role="input", column=0, row=0, summary="0°→+45°→−20°→+35°，固定 seed/noise/pose。"),
        node(id="generation", label_zh="InSpatio 原生生成", label_en="Frozen generator", role="generation", column=1, row=0, summary="Frozen 1.3B、原生 4-step scheduler、raw Recent。"),
        node(id="geometry", label_zh="Known-pose CUT3R", label_en="Geometry", role="geometry", column=0, row=1, summary="沿用已验证 fixed-pose depth/surfel state，本轮未修改。"),
        node(id="address", label_zh="固定历史地址", label_en="Address", role="address", column=1, row=1, summary="固定 canonical chunk 11 与 episode-continuous re-entry。"),
        node(id="payload", label_zh="Target-aligned 历史 payload", label_en="Payload", role="payload", column=2, row=1, summary="chunk 11 RGB camera warp → Wan VAE；所有方法共用 L_mem。"),
        node(id="context", label_zh="Memory Interface Ladder", label_en="Interface ladder", role="context", column=3, row=0, summary="HardX0 / coherent dual Recent / native Render / latent anchor。", change_type="added", focus=True, files=("mapkv/memory_interface.py", "pipeline/causal_inference.py")),
        node(id="attention", label_zh="模型内控制边界", label_en="Control boundary", role="attention", column=4, row=0, summary="仅 M3 双完整 forward；M4 无额外 attention；M5 在 x0/re-noise 边界。", change_type="modified", files=("pipeline/causal_inference.py",)),
        node(id="output", label_zh="完整回访生成", label_en="Revisit output", role="output", column=5, row=0, summary="每个方法保存完整 B1→离开→回访→B2 视频。"),
        node(id="evaluation", label_zh="结构化 Identity 评估", label_en="Identity evaluation", role="evaluation", column=5, row=1, summary="自动选 generated-history 结构区域；L1 仅称 appearance。", change_type="added", files=("mapkv/memory_interface_evaluation.py", "mapkv/memory_interface_report.py")),
    )
    changes = (
        ArchitectureChange(component_id="context", change_type="added", before="仅 layer-wise replace_recent_delta。", after="同一 L_mem/M_need 通过四级 frozen-model interface 对照。", affected_files=("mapkv/memory_interface.py", "inference_mapkv_proto.py"), rationale="隔离 memory 内容与模型控制入口。"),
        ArchitectureChange(component_id="attention", change_type="modified", before="单 forward 内混合 attention delta。", after="M3 两套独立 Recent cache 完整 x0；M4/M5 不走额外 attention。", affected_files=("pipeline/causal_inference.py",), rationale="测试 self-consistency、原生 spatial condition 与 latent trajectory。"),
        ArchitectureChange(component_id="evaluation", change_type="added", before="平均 revisit L1 易被误读为 identity。", after="自动结构 crop + 同步视频 + STRONG/PARTIAL/NONE。", affected_files=("mapkv/memory_interface_evaluation.py", "mapkv/memory_interface_report.py"), rationale="明确分开 appearance proximity 与 instance identity。"),
    )
    return ArchitectureSnapshot(
        name="MapKV — Frozen Memory Interface Convergence",
        focus_zh="同一 target-aligned 历史记忆应进入哪个 frozen InSpatio inference interface",
        focus_en="Matched Hard X0 → coherent Recent → native Render → noise-consistent latent anchoring",
        nodes=nodes,
        edges=(
            ArchitectureEdge("input", "generation", "native inference"),
            ArchitectureEdge("geometry", "address", "visible history"),
            ArchitectureEdge("address", "payload", "chunk 11"),
            ArchitectureEdge("payload", "context", "same L_mem + M_need"),
            ArchitectureEdge("generation", "context", "same noisy x_t"),
            ArchitectureEdge("context", "attention", "interface boundary"),
            ArchitectureEdge("attention", "output", "x0 / re-noise"),
            ArchitectureEdge("payload", "evaluation", "warped anchor"),
            ArchitectureEdge("output", "evaluation", "videos + crops"),
        ),
        changes=changes,
        metadata={"geometry_changed": False, "retrieval_changed": False, "lifecycle_changed": False, "source_chunk": 11},
    )


def _video_cards(root: Path, methods: list[str], prefix: str) -> str:
    cards = []
    for method in methods:
        relative = f"videos/report/{prefix}_{method}.mp4"
        if (root / relative).exists():
            cards.append(
                f"<article class='card'><h3>{html.escape(METHOD_LABELS_ZH[method])}</h3>"
                f"<video controls preload='metadata' src='{relative}'></video></article>"
            )
    return "<div class='video-grid'>" + "".join(cards) + "</div>"


def _metrics_table(metrics: dict) -> str:
    rows = []
    for method, values in metrics["methods"].items():
        rows.append(
            f"<tr><td>{html.escape(METHOD_LABELS_ZH[method])}</td>"
            f"<td>{values['warped_historical_appearance_l1']:.5f}</td>"
            f"<td>{values['source_region_delta_vs_baseline']:.5f}</td>"
            f"<td>{values['first_departure_peak']:.5f}</td>"
            f"<td>{values['reentry_peak']:.5f}</td>"
            f"<td>{values['right_edge_l1']:.5f}</td>"
            f"<td><b>{html.escape(values['identity'])}</b></td></tr>"
        )
    return "<table><tr><th>方法</th><th>Warped historical appearance L1 ↓</th><th>Source Δ ↓</th><th>First-departure peak ↓</th><th>Re-entry peak ↓</th><th>Right-edge L1 ↓</th><th>Identity</th></tr>" + "".join(rows) + "</table>"


def build_report(run_root: str | Path) -> Path:
    root = Path(run_root).resolve()
    metrics = _json(root / "metrics.json")
    snapshot = _architecture()
    write_architecture_bundle(root, snapshot)
    methods = [method for method in METHOD_ROOTS if method in metrics["methods"]]
    identity_images = "".join(
        f"<figure><figcaption>自动结构 identity region {index}</figcaption><img src='assets/memory_interface/identity_region_{index}.jpg'></figure>"
        for index in range(1, len(metrics["identity_regions"]) + 1)
    )
    status = metrics["status"]
    strongest = metrics.get("strongest_viable_interface") or "尚待肉眼评级"
    controls = metrics["controls"]
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>MapKV Memory Interface</title><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f4f7fb;color:#172033;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}section{{background:white;border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 3px 14px #1f293714}}h1,h2{{margin-top:0}}.focus{{background:#f2eaff;border-left:5px solid #6b46c1;padding:12px}}.controls button{{margin:4px;padding:8px 13px}}.video-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}}.card{{border:1px solid #d7deea;border-radius:10px;padding:10px}}video,img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d8dee9;padding:8px;text-align:left}}th{{background:#edf2f7}}code{{background:#edf2f7;padding:2px 5px}}.status{{font-size:20px;font-weight:bold;color:#6b46c1}}
</style></head><body><main>
<section><h1>MapKV Memory Interface Convergence</h1><p class='focus'><b>本次最新 method / focus：</b>固定同一 chunk-11、同一 RGB-Warp→VAE <code>L_mem</code> 与同一 <code>M_need</code>，只比较 frozen InSpatio 的控制入口。</p><p class='status'>状态：{html.escape(status)}</p><p>当前最强可行接口：<b>{html.escape(str(strongest))}</b></p><p>Hard upper bound: {metrics['decisions']['hard_upper_bound_works']}；Identity 人工 review: {metrics['decisions']['identity_review_present']}；M_need coverage: {controls['revisit_fraction']:.2%}。</p></section>
<section><h2>A. 完整 Pipeline / Architecture Changes</h2><img src='assets/architecture_graph.svg'>{render_pipeline_table_html(snapshot)}<h3>本次架构修改</h3>{render_changes_html(snapshot)}</section>
<section><h2>B. 固定的历史记忆与评价区域</h2><div class='video-grid'><figure><figcaption>Canonical B1 chunk 11</figcaption><img src='assets/memory_interface/canonical_b1_anchor.png'></figure><figure><figcaption>B1 camera-warp 到 B2</figcaption><img src='assets/memory_interface/canonical_b1_warped_to_b2.png'></figure><figure><figcaption>M_need：generated-history × current ref-blind</figcaption><img src='assets/memory_interface/M_need_b2.png'></figure><figure><figcaption>Source-valid protected region</figcaption><img src='assets/memory_interface/M_source_b2.png'></figure></div>{identity_images}</section>
<section><h2>C. 完整回访同步视频（B1 → 离开 → 回访 → B2）</h2><p>这是主对比，不是只截 B2。中文标题标出每个接口。</p><div class='controls'><button onclick='playAll("full")'>Play all</button><button onclick='pauseAll("full")'>Pause all</button><button onclick='resetAll("full")'>Reset all</button></div><div id='full'>{_video_cards(root, methods, 'full_revisit')}</div></section>
<section><h2>D. Re-entry 局部同步视频</h2><div class='controls'><button onclick='playAll("reentry")'>Play all</button><button onclick='pauseAll("reentry")'>Pause all</button><button onclick='resetAll("reentry")'>Reset all</button></div><div id='reentry'>{_video_cards(root, methods, 'reentry')}</div></section>
<section><h2>E. 指标与 Identity 结论</h2><p><b>注意：</b>historical appearance L1 不能单独证明 instance identity；Identity 来自同步视频与自动结构 crop 的人工评级。</p>{_metrics_table(metrics)}</section>
<script>function vids(id){{return [...document.querySelectorAll('#'+id+' video')]}}function playAll(id){{let v=vids(id);if(!v.length)return;let t=Math.min(...v.map(x=>x.currentTime));v.forEach(x=>{{x.currentTime=t;x.play()}})}}function pauseAll(id){{vids(id).forEach(x=>x.pause())}}function resetAll(id){{vids(id).forEach(x=>{{x.pause();x.currentTime=0}})}}</script></main></body></html>"""
    report_path = root / "report.html"
    report_path.write_text(page, encoding="utf-8")
    rows = "\n".join(
        f"- {METHOD_LABELS_ZH[method]}: appearance L1={values['warped_historical_appearance_l1']:.5f}, source Δ={values['source_region_delta_vs_baseline']:.5f}, identity={values['identity']}"
        for method, values in metrics["methods"].items()
    )
    (root / "report.md").write_text(
        f"# MapKV Memory Interface Convergence\n\n状态：**{status}**  \n本次 focus：同一 target-aligned chunk-11 memory 经不同 frozen inference interface 的 identity 控制力。  \n最强接口：**{strongest}**\n\n## Architecture\n\n![完整 Pipeline](assets/architecture_graph.svg)\n\n## Results\n\n{rows}\n\n## Videos\n\n完整回访视频：`videos/report/full_revisit_*.mp4`；re-entry：`videos/report/reentry_*.mp4`。\n\nWarped historical appearance L1 仅衡量外观接近度；Identity 依据同步视频和自动结构 crop。\n",
        encoding="utf-8",
    )
    return report_path


__all__ = ["build_report"]
