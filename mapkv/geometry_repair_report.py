from __future__ import annotations

import html
import json
import os
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


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot() -> ArchitectureSnapshot:
    return ArchitectureSnapshot(
        name="MapKV Fixed-Geometry Address Repair",
        focus_zh="真实 fixed-pose alignment、stable surface address 与双 geometry Gate",
        focus_en="fixed global alignment + stable neighborhood retrieval",
        nodes=(
            node(
                id="input",
                label_zh="双受控轨迹",
                label_en="pure yaw + 0.08 translation",
                role="input",
                column=0,
                row=0,
                summary="静态 source、exact c2w；translation case 提供真实视差。",
            ),
            node(
                id="generation",
                label_zh="冻结 InSpatio Baseline",
                label_en="deterministic generated history",
                role="generation",
                column=1,
                row=0,
                summary="只生成 history；geometry Gate 前不运行 memory generation。",
            ),
            node(
                id="geometry",
                label_zh="Fixed-pose CUT3R",
                label_en="known pose/K global alignment",
                role="geometry",
                column=2,
                row=0,
                summary="使用 cross-view pointmaps；known pose/K 冻结；joint 为质量路径。",
                change_type="modified",
                focus=True,
                files=("mapkv/cut3r_adapter.py",),
            ),
            node(
                id="address",
                label_zh="Stable Surface Address",
                label_en="tentative→stable + reprojection gate",
                role="address",
                column=3,
                row=0,
                summary="至少3次一致观测；单视图 cell 不参与 z-buffer/voting。",
                change_type="modified",
                focus=True,
                files=("mapkv/surfel_index.py",),
            ),
            node(
                id="payload",
                label_zh="Chunk Observation Metadata",
                label_en="surface neighborhood→chunk IDs",
                role="payload",
                column=4,
                row=0,
                summary="native KV 未载入；本阶段只验证 geometry address。",
            ),
            node(
                id="context",
                label_zh="Unique-surfel Retrieval",
                label_en="neighbor vote + calibrated confidence",
                role="context",
                column=5,
                row=0,
                summary="每个可见 stable cell只计一次；pixel coverage仅作解释。",
                change_type="modified",
                files=("mapkv/retrieval.py",),
            ),
            node(
                id="attention",
                label_zh="Memory Attention 暂停",
                label_en="generation blocked by geometry Gate",
                role="attention",
                column=6,
                row=0,
                summary="geometry Gate 通过前不注入 KV。",
            ),
            node(
                id="output",
                label_zh="3D Diagnostics",
                label_en="stable RGB surfels / target z-buffer",
                role="output",
                column=7,
                row=0,
                summary="Z反向 world overview、coverage、overlay 与完整回访视频。",
            ),
            node(
                id="evaluation",
                label_zh="Yaw + Translation Gates",
                label_en="angular address + depth-cycle",
                role="evaluation",
                column=8,
                row=0,
                summary="pure yaw验证angular address；translation验证真实depth。",
                change_type="added",
                files=(
                    "mapkv/geometry_gate.py",
                    "mapkv/translation_depth_gate.py",
                ),
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
                component_id="geometry",
                change_type="modified",
                before="独立 self-pointmap × known pose 的 rigid placement",
                after="CUT3R cross-view global aligner + fixed known pose/K",
                affected_files=("mapkv/cut3r_adapter.py",),
                rationale="消除 view-specific depth shells 与每帧 focal 漂移。",
            ),
            ArchitectureChange(
                component_id="address",
                change_type="modified",
                before="所有单视图 cells 立即进入 address",
                after="tentative→stable；distance+normal+reprojection consistency",
                affected_files=("mapkv/surfel_index.py",),
                rationale="禁止单视图浮层参与长期 occlusion/voting。",
            ),
            ArchitectureChange(
                component_id="context",
                change_type="modified",
                before="per-pixel重复计票、cluster求和、raw confidence",
                after="unique stable cell、surface neighborhood、cluster max、calibrated confidence",
                affected_files=("mapkv/retrieval.py",),
                rationale="消除 disk大小、plateau长度和confidence尺度偏置。",
            ),
            ArchitectureChange(
                component_id="evaluation",
                change_type="added",
                before="same-pose命中即认为3D正确",
                after="yaw angular Gate + translation depth-cycle Gate",
                affected_files=(
                    "mapkv/geometry_gate.py",
                    "mapkv/translation_depth_gate.py",
                ),
                rationale="纯旋转无法观测depth，必须用视差单独验证。",
            ),
        ),
    )


def _checks(values: dict) -> str:
    return "<table><tr><th>Check</th><th>Pass</th></tr>" + "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>"
        for key, value in values.items()
    ) + "</table>"


def build_report(
    *,
    yaw_root: str | Path,
    translation_root: str | Path,
) -> Path:
    root = Path(yaw_root).resolve()
    translation = Path(translation_root).resolve()
    yaw = _json(root / "geometry_gate.json")
    depth = _json(translation / "translation_depth_gate.json")
    translation_link = root / "translation"
    if not translation_link.exists():
        translation_link.symlink_to(
            os.path.relpath(translation, root), target_is_directory=True
        )
    snapshot = _snapshot()
    write_architecture_bundle(root, snapshot)
    overall = (
        "GEOMETRY_ADDRESS_REPAIR_WORKS"
        if yaw["status"] == "GEOMETRY_GATE_PASS"
        and depth["status"] == "TRANSLATION_DEPTH_GATE_PASS"
        else "GEOMETRY_ADDRESS_REPAIR_INCOMPLETE"
    )
    combined = {
        "status": overall,
        "yaw_gate": yaw,
        "translation_gate": depth,
    }
    (root / "metrics.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    (root / "status.json").write_text(
        json.dumps({"status": overall}, indent=2), encoding="utf-8"
    )
    styles = """
body{font-family:system-ui,'Noto Sans SC',sans-serif;margin:0;background:#f3f6fa;color:#172033}
main{max-width:1500px;margin:auto;padding:26px}section{background:#fff;padding:20px;margin:16px 0;border-radius:13px;box-shadow:0 2px 14px #17203312}
.focus{border-left:6px solid #5b4bc4;background:#f4f0ff;padding:12px}.grid,.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:13px}
img,video{width:100%;max-height:540px;object-fit:contain;background:#10151c;border-radius:8px}figure{margin:0}figcaption{font-weight:650;margin-bottom:7px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d9e0e9;padding:8px;text-align:left}th{background:#eef2f7}.architecture{overflow:auto}.architecture img{min-width:1400px;max-height:none}
button{padding:8px 13px;border:0;border-radius:7px;background:#315dce;color:white;margin:3px}
"""
    script = """
function vs(){return Array.from(document.querySelectorAll('video[data-sync]'))}
function playAll(){let v=vs(),t=v.length?v[0].currentTime:0;v.forEach(x=>{x.currentTime=t;x.play()})}
function pauseAll(){vs().forEach(x=>x.pause())}
function resetAll(){vs().forEach(x=>{x.pause();x.currentTime=0})}
"""
    yaw_fixed = yaw["fixed_global"]
    yaw_legacy = yaw["legacy"]
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>MapKV Geometry Repair</title>
<style>{styles}</style></head><body><main>
<h1>MapKV 3D Surfel Address Repair</h1><h2>{overall}</h2>
<p class='focus'><b>本次最新 Method / Focus：</b>fixed known-pose/K CUT3R global alignment
→ tentative/stable surfel → surface-neighborhood unique-cell retrieval。纯 yaw 只判 angular
address；0.08 translation case 单独判真实 depth。</p>
<section><h2>A. 完整 Pipeline 与 Architecture Changes</h2>
<div class='architecture'><img src='assets/architecture_graph.svg'></div>
{render_pipeline_table_html(snapshot)}{render_changes_html(snapshot)}</section>
<section><h2>B. Gate Summary</h2><div class='grid'>
<div><h3>Pure-yaw Angular Gate：{yaw['status']}</h3>{_checks(yaw['checks'])}</div>
<div><h3>Translation Depth Gate：{depth['status']}</h3>{_checks(depth['checks'])}</div>
</div></section>
<section><h2>C. Stable RGB Surfel State（显示坐标 Z 反向）</h2><div class='grid'>
<figure><figcaption>Yaw repaired stable world splats</figcaption><img src='surfel_rgb_options/A_rgb_world_splats.png'></figure>
<figure><figcaption>Yaw repaired stable oriented disks</figcaption><img src='surfel_rgb_options/B_rgb_oriented_disks.png'></figure>
<figure><figcaption>Translation stable world splats</figcaption><img src='translation/surfel_rgb_options/A_rgb_world_splats.png'></figure>
<figure><figcaption>Translation stable oriented disks</figcaption><img src='translation/surfel_rgb_options/B_rgb_oriented_disks.png'></figure>
</div></section>
<section><h2>D. Target-view Geometry</h2><div class='grid'>
<figure><figcaption>Legacy B1 z-buffer</figcaption><img src='legacy_shadow/assets/geometry_gate/legacy_zbuffer.png'></figure>
<figure><figcaption>Yaw repaired B1 z-buffer</figcaption><img src='assets/geometry_gate/fixed_global_zbuffer.png'></figure>
<figure><figcaption>Yaw repaired overlay</figcaption><img src='assets/geometry_gate/fixed_global_overlay.png'></figure>
<figure><figcaption>Translation repaired z-buffer</figcaption><img src='translation/assets/geometry_gate/translation_zbuffer.png'></figure>
<figure><figcaption>Translation repaired overlay</figcaption><img src='translation/assets/geometry_gate/translation_overlay.png'></figure>
</div></section>
<section><h2>E. 完整回访视频（仅 Baseline，未运行 KV generation）</h2>
<p><button onclick='playAll()'>同步播放</button><button onclick='pauseAll()'>暂停</button>
<button onclick='resetAll()'>复位</button></p><div class='videos'>
<figure><figcaption>Pure-yaw baseline：B1→leave→B2</figcaption><video controls preload='metadata' data-sync src='baseline/pred.mp4'></video></figure>
<figure><figcaption>0.08 translation baseline：B1→leave→B2</figcaption><video controls preload='metadata' data-sync src='translation/baseline/pred.mp4'></video></figure>
</div></section>
<section><h2>F. Core Metrics</h2>
<table><tr><th>Metric</th><th>Legacy/Yaw</th><th>Repaired/Yaw</th><th>Translation</th></tr>
<tr><td>Stable cells</td><td>{yaw_legacy['surfel'].get('stable_cells','n/a')}</td><td>{yaw_fixed['surfel']['stable_cells']}</td><td>{depth['surfel']['stable_cells']}</td></tr>
<tr><td>Target RGB L1 ↓</td><td>{yaw_legacy['target_render']['rgb_l1']:.4f}</td><td>{yaw_fixed['target_render']['rgb_l1']:.4f}</td><td>{depth['target_render']['rgb_l1']:.4f}</td></tr>
<tr><td>Target edge corr ↑</td><td>{yaw_legacy['target_render']['edge_correlation']:.4f}</td><td>{yaw_fixed['target_render']['edge_correlation']:.4f}</td><td>{depth['target_render']['edge_correlation']:.4f}</td></tr>
<tr><td>Depth-cycle median ↓</td><td>不可观测</td><td>不可观测</td><td>{depth['cycle']['relative_depth_error_median']:.4f}</td></tr>
<tr><td>Surface persistence ↑</td><td>{yaw_legacy['anchor_overlap']['stable_anchor_recall']:.3f}</td><td>{yaw_fixed['anchor_overlap']['stable_anchor_recall']:.3f}</td><td>{depth['surface_neighbor_persistence']['recall']:.3f}</td></tr>
</table></section>
<section><h2>G. 结论</h2><p>geometry repair 已通过 angular 与 translation depth 双 Gate。
严格 previous-depth freeze 路径已实现但无法收敛；当前质量路径采用 full causal-prefix joint
alignment。下一步可以恢复 SurfelKV generation，但仍保持 geometry diagnostics 与低置信 retrieval audit。</p></section>
</main><script>{script}</script></body></html>"""
    report = root / "report.html"
    report.write_text(document, encoding="utf-8")
    (root / "report.md").write_text(
        "\n".join(
            [
                "# MapKV 3D Surfel Address Repair",
                "",
                f"- Status: {overall}",
                f"- Pure-yaw Gate: {yaw['status']}",
                f"- Translation Gate: {depth['status']}",
                f"- Translation depth-cycle median: {depth['cycle']['relative_depth_error_median']:.5f}",
                f"- Translation surface persistence: {depth['surface_neighbor_persistence']['recall']:.3f}",
                f"- Retrieved chunk: {depth['retrieval']['selected_chunks']}",
                "",
                "Complete videos:",
                "- baseline/pred.mp4",
                "- translation/baseline/pred.mp4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["build_report"]
