from __future__ import annotations

import html
import json
import subprocess
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
    "baseline": "原始 InSpatio（Baseline）",
    "current_masked": "旧双软化掩码 WRE（Current Masked WRE）",
    "strong_latent": "强内核 Latent-Warp WRE（Priority 1）",
    "rgb_warp_vae": "RGB-Warp→VAE 质量上界（Priority 2 / 本次最佳）",
    "canonical_kv": "Canonical-K/V 重寻址（Priority 3 / 本次最新 Focus）",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value) -> str:
    return "—" if value is None else f"{float(value):.5f}"


def _architecture() -> ArchitectureSnapshot:
    nodes = (
        node(id="input", label_zh="静态源 + 已知轨迹", label_en="Static source + exact c2w", role="input", column=0, row=1, summary="固定 scene01、seed0、0→30→0→20、B1 chunk8。", files=("artifacts/control/yaw30to20_scene01",)),
        node(id="generation", label_zh="InSpatio 基础生成", label_en="Frozen InSpatio generation", role="generation", column=1, row=0, summary="原始 Ref + Recent + Current，4-step bf16。", files=("pipeline/causal_inference.py",)),
        node(id="geometry", label_zh="Known-pose CUT3R", label_en="Fixed-pose CUT3R prefix", role="geometry", column=1, row=2, summary="几何与 retrieval 参数冻结，严格 causal prefix。", files=("mapkv/cut3r_adapter.py",)),
        node(id="address", label_zh="RGB Surfel 地址", label_en="Visible surfel support", role="address", column=2, row=2, summary="固定 B1 chunk8；surfel visibility 只提供 M_hard。", files=("mapkv/surfel_index.py",)),
        node(id="payload", label_zh="B1 历史状态", label_en="Historical B1 payload", role="payload", column=2, row=0, summary="clean B1 latent，或 writer 中 pre-norm K_projected + V。", files=("mapkv/warp_reencode.py", "mapkv/canonical_kv.py")),
        node(id="masks", label_zh="强 M_memory + 软 M_query", label_en="Strong core + soft query boundary", role="context", column=3, row=2, summary="max/dilation 保留支持；只在 query 边界 feather。", change_type="modified", files=("mapkv/warp_reencode.py", "mapkv_proto/memory_context.py")),
        node(id="rgb_oracle", label_zh="RGB Warp→VAE 质量上界", label_en="RGB warp then native VAE encode", role="context", column=3, row=0, summary="先在严格几何成立的 RGB 空间 warp，再原生 VAE encode。", change_type="added", files=("mapkv/warp_reencode.py",), focus=True),
        node(id="canonical_capture", label_zh="Canonical K/V 捕获", label_en="Pre-norm projected K + V capture", role="payload", column=3, row=1, summary="仅 clean writer 捕获；source post-RoPE 重建误差必须为 0。", change_type="added", files=("wan/modules/causal_model.py", "utils/wan_wrapper.py", "pipeline/causal_inference.py"), focus=True),
        node(id="native_writer", label_zh="原生 t=0 Recent Writer", label_en="Native clean Recent writer", role="context", column=4, row=0, summary="Latent/RGB oracle 生成合法 target-layout Recent K/V。", files=("pipeline/causal_inference.py",)),
        node(id="canonical_readdress", label_zh="Target-grid K/V 重寻址", label_en="Canonical target-grid re-addressing", role="context", column=4, row=1, summary="warp projected K/V → norm K → target Recent T/H/W RoPE。", change_type="added", files=("mapkv/canonical_kv.py",), focus=True),
        node(id="attention", label_zh="固定 Recent-slot Counterfactual", label_en="Geometry-gated Recent delta", role="attention", column=5, row=1, summary="M_memory_token 保留 runtime Recent fallback；M_query 限制当前 query。", change_type="modified", files=("wan/modules/causal_model.py", "mapkv_proto/memory_context.py")),
        node(id="output", label_zh="正常噪声起步与 4-step 去噪", label_en="Normal denoising and decoded video", role="output", column=6, row=1, summary="memory 是 context，不直接覆盖 current output。", files=("pipeline/causal_inference.py",)),
        node(id="evaluation", label_zh="Identity/Locality/Transition", label_en="Common-mask controlled evaluation", role="evaluation", column=7, row=1, summary="同一 overlap/non-overlap mask、完整回访视频与目标物体 crops。", change_type="modified", files=("mapkv/identity_recovery_evaluation.py", "mapkv/identity_recovery_report.py")),
    )
    edges = (
        ArchitectureEdge("input", "generation"), ArchitectureEdge("input", "geometry"),
        ArchitectureEdge("generation", "payload", "B1"), ArchitectureEdge("geometry", "address"),
        ArchitectureEdge("address", "masks", "M_hard"), ArchitectureEdge("payload", "rgb_oracle"),
        ArchitectureEdge("payload", "canonical_capture"), ArchitectureEdge("rgb_oracle", "native_writer"),
        ArchitectureEdge("canonical_capture", "canonical_readdress"), ArchitectureEdge("masks", "native_writer"),
        ArchitectureEdge("masks", "canonical_readdress"), ArchitectureEdge("native_writer", "attention"),
        ArchitectureEdge("canonical_readdress", "attention"), ArchitectureEdge("attention", "output"),
        ArchitectureEdge("output", "evaluation"),
    )
    changes = (
        ArchitectureChange("masks", "modified", "同一个 feathered mask 同时削弱 latent composition 与 query delta。", "二值/膨胀 M_memory 保留历史内核；support-preserving tokenization 后只 feather M_query 边界。", ("mapkv/warp_reencode.py", "mapkv_proto/memory_context.py"), "隔离 double attenuation 对 identity 的影响。"),
        ArchitectureChange("rgb_oracle", "added", "B1 VAE latent 直接做 bilinear camera warp。", "B1 lossless RGB 先做 exact camera warp，再由原生 WanVAE encode。", ("mapkv/warp_reencode.py",), "建立几何上严格的 representation quality oracle。"),
        ArchitectureChange("canonical_capture", "added", "历史 bank 只保存 post-RoPE K/V。", "clean Recent writer 同时捕获 K_projected_pre_norm 与 V，并精确重建 native K。", ("wan/modules/causal_model.py", "utils/wan_wrapper.py", "pipeline/causal_inference.py"), "允许先 warp projected K，再 norm 和应用 target RoPE。"),
        ArchitectureChange("canonical_readdress", "added", "无 target-layout native K/V reconstruction。", "构建固定 F×Htok×Wtok 的 target-aligned Canonical K/V grid。", ("mapkv/canonical_kv.py",), "测试直接 native feature re-address 能否逼近 RGB oracle。"),
        ArchitectureChange("attention", "modified", "replace_recent_delta 只支持完整外部 Recent K/V。", "canonical_recent_delta 在 M_memory_token 内用历史 K/V，外部精确保留 runtime Recent，并继续 M_query gate。", ("wan/modules/causal_model.py", "mapkv_proto/memory_context.py"), "固定 slot 长度并保持 base path/cache 不变。"),
        ArchitectureChange("evaluation", "modified", "主要看 layout/locality。", "增加糕点、杯子、盘中物体 crop 以及 P1/P2/P3 common-mask 比较。", ("mapkv/identity_recovery_evaluation.py", "mapkv/identity_recovery_report.py"), "直接评估 identity/fine appearance。"),
    )
    return ArchitectureSnapshot(
        name="MapKV Identity Recovery — complete pipeline",
        focus_zh="RGB-Warp→VAE 身份质量上界与 Canonical-K/V 原生重寻址差距",
        focus_en="Strong memory core -> RGB warp oracle -> canonical K/V re-addressing",
        nodes=nodes,
        edges=edges,
        changes=changes,
        metadata={"retrieval": "frozen manual B1 chunk8", "geometry": "frozen known-pose CUT3R"},
    )


def _video(method: str, *, full: bool) -> str:
    prefix = "full_revisit" if full else "reentry"
    return (
        f"<figure><figcaption>{html.escape(LABELS[method])}</figcaption>"
        f"<video class='sync' controls preload='metadata' src='videos/report/{prefix}_{method}.mp4'></video></figure>"
    )


def build_report(run_root: str | Path) -> str:
    root = Path(run_root).resolve()
    metrics = _json(root / "metrics.json")
    architecture = _architecture()
    write_architecture_bundle(root, architecture)
    pipeline_table = render_pipeline_table_html(architecture)
    changes_table = render_changes_html(architecture)
    methods = metrics["methods"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(LABELS[name])}</td>"
        f"<td>{_fmt(value['overlap_b1_to_b2_l1'])}</td>"
        f"<td>{_fmt(value['nonoverlap_delta_vs_baseline_l1'])}</td>"
        f"<td>{_fmt(value['reentry_window_mean_l1'])}</td>"
        f"<td>{_fmt(value['reentry_window_peak_l1'])}</td>"
        f"<td>{value['generation_seconds']:.1f}s</td></tr>"
        for name, value in methods.items()
    )
    priority1_rows = "".join(
        "<tr>"
        f"<td>{html.escape(LABELS[name])}</td>"
        f"<td>{_fmt(methods[name]['overlap_b1_to_b2_l1'])}</td>"
        f"<td>{_fmt(methods[name]['nonoverlap_delta_vs_baseline_l1'])}</td>"
        f"<td>{_fmt(methods[name]['reentry_window_peak_l1'])}</td></tr>"
        for name in ("baseline", "current_masked", "strong_latent")
    )
    crop_methods = ("warped_b1", "baseline", "strong_latent", "rgb_warp_vae", "canonical_kv")
    crop_labels = {"warped_b1": "B1 warp 到 B2", **LABELS}
    crop_sections = "".join(
        "<section><h3>" + title + "</h3><div class='cropgrid'>" + "".join(
            f"<figure><figcaption>{html.escape(crop_labels[name])}</figcaption>"
            f"<img src='assets/identity_recovery/identity_crops/{key}_{name}.png'></figure>"
            for name in crop_methods
        ) + "</div></section>"
        for key, title in (
            ("pastry", "糕点 / 中央小物体 identity"),
            ("cup", "杯子 / 右侧物体 identity"),
            ("plate_local", "盘中局部物体与纹理"),
        )
    )
    full_videos = "".join(_video(name, full=True) for name in LABELS)
    reentry_videos = "".join(_video(name, full=False) for name in ("baseline", "strong_latent", "rgb_warp_vae", "canonical_kv"))
    validity = "".join(
        f"<li><b>{html.escape(key)}:</b> {html.escape(str(value))}</li>"
        for key, value in metrics["validity"].items()
    )
    conclusion = (
        "P1 证明 double attenuation 会削弱 identity，但简单扩大 strong core 会增加 locality/transition 代价；"
        "P2 的 RGB-Warp→VAE 在相同 mask 下成为最强 changed-view quality oracle；"
        "P3 的 Canonical-K 实现数值与 cache 均正确，却明显不能逼近该 oracle，说明直接插值 projected K/V "
        "并重新施加 target RoPE 仍缺少 writer 内部的 camera-conditioned feature reconstruction。"
    )
    next_action = (
        "以 RGB-Warp→VAE 为固定 oracle，做一次 writer-depth localization：在少量早/中/晚层捕获并重投影 hidden state，"
        "从该层继续运行剩余 writer，定位最早能逼近 oracle 的 camera-aware re-encode 边界。"
    )
    html_payload = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>MapKV Identity Recovery</title><style>
:root{{--bg:#f3f6fb;--card:#fff;--ink:#172033;--accent:#6b46c1}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1440px;margin:auto;padding:24px}}section{{background:var(--card);margin:16px 0;padding:20px;border-radius:12px;box-shadow:0 2px 10px #2233aa12}}
h1,h2{{margin-top:0}}.status{{font-size:23px;font-weight:800;color:var(--accent)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.cropgrid{{display:grid;grid-template-columns:repeat(5,minmax(160px,1fr));gap:8px}}
figure{{margin:0}}figcaption{{font-weight:650;margin-bottom:6px}}img,video{{width:100%;border-radius:8px;background:#111}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e2e7f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}
button{{padding:8px 12px;margin:0 6px 10px 0}}.architecture{{overflow-x:auto}}.architecture>a>img{{min-width:1500px}}
.architecture table td,.architecture table th{{text-align:left;vertical-align:top}}code{{font-size:12px}}
@media(max-width:900px){{.cropgrid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<section><h1>MapKV Identity Recovery Stage</h1><div class='status'>{metrics['status']}</div>
<p><b>本次最新方法 / Focus：</b>Canonical-K/V target-grid 重寻址；<b>本次最佳质量方法：</b>RGB-Warp→VAE WRE。</p>
<p>{conclusion}</p><ul>{validity}</ul></section>
<section class='architecture'><h2>完整 Pipeline / Framework</h2><p><b>本次架构 Focus：</b>{html.escape(architecture.focus_zh)}</p>
<a href='assets/architecture_graph.svg'><img src='assets/architecture_graph.svg'></a><h3>完整模块表</h3>{pipeline_table}<h3>架构修改 Before → After</h3>{changes_table}
<p><a href='architecture_state.json'>architecture_state.json</a> · <a href='architecture_changes.json'>architecture_changes.json</a> · <a href='architecture.md'>architecture.md</a></p></section>
<section><h2>Priority 1 — Strong Memory Core + Soft Boundary</h2><div class='grid'>
<figure><figcaption>原始 surfel 支持 M_hard</figcaption><img src='assets/identity_recovery/M_hard.png'></figure>
<figure><figcaption>强历史内核 M_memory</figcaption><img src='assets/identity_recovery/M_memory.png'></figure>
<figure><figcaption>soft-boundary query gate M_query</figcaption><img src='assets/identity_recovery/M_query.png'></figure></div>
<table><tr><th>Method</th><th>Overlap B1→B2 ↓</th><th>Non-overlap Δ ↓</th><th>Re-entry peak ↓</th></tr>{priority1_rows}</table>
<p>判断：mask attenuation 存在，但扩大 support 的 locality/transition 代价也真实存在。</p></section>
<section><h2>Priority 2 — RGB-Warp→VAE Quality Oracle</h2><div class='grid'>
<figure><figcaption>B1 generated RGB</figcaption><img src='generation/rgb_warp_vae_wre/warp/target_0022/rgb_history_source.png'></figure>
<figure><figcaption>B1 RGB exact-warp 到 B2</figcaption><img src='generation/rgb_warp_vae_wre/warp/target_0022/rgb_history_warped_to_target.png'></figure>
<figure><figcaption>Latent-Warp decoded history</figcaption><img src='generation/strong_core_latent_wre/warp/target_0022/warped.png'></figure>
<figure><figcaption>RGB-Warp→VAE decoded history</figcaption><img src='generation/rgb_warp_vae_wre/warp/target_0022/historical.png'></figure></div>
<p>Overlap L1: Latent-Warp {_fmt(methods['strong_latent']['overlap_b1_to_b2_l1'])} → RGB-Warp {_fmt(methods['rgb_warp_vae']['overlap_b1_to_b2_l1'])}。</p></section>
<section><h2>Identity-focused crops</h2><figure><figcaption>crop 位置总览</figcaption><img src='assets/identity_recovery/identity_crops/crop_overview.png'></figure>{crop_sections}</section>
<section><h2>Priority 3 — Canonical-K/V</h2><p><code>Historical hidden → K_projected/V → geometry warp → norm_k + target Recent RoPE → fixed Virtual Recent K/V → M_query delta</code></p>
<p>Source native reconstruction max abs diff = <b>{metrics['priority_3']['source_reconstruction_max_abs_diff']}</b>；
与原始 B1 clean-context bank 的全层 K/V abs-mean 最大差 = <b>{metrics['priority_3']['original_b1_bank_abs_mean_max_diff']:.2e}</b>；payload bytes = {metrics['priority_3']['memory_bytes']:,}。</p>
<p>Canonical overlap gap to RGB oracle = <b>{_fmt(metrics['priority_3']['canonical_overlap_gap_to_rgb'])}</b>；结论：<b>未逼近 quality oracle</b>。</p></section>
<section><h2>RGB Surfel（真实历史观测颜色）</h2><div class='grid'>
<figure><figcaption>RGB world splats（显示坐标 Z 反向）</figcaption><img src='surfel_rgb_options/A_rgb_world_splats.png'></figure>
<figure><figcaption>RGB B1 target z-buffer</figcaption><img src='surfel_rgb_options/D_rgb_b1_target_zbuffer.png'></figure>
<figure><figcaption>B1→B2 RGB support overlay</figcaption><img src='surfel_rgb_options/E_rgb_b1_target_overlay.png'></figure></div>
<p><a href='surfel_rgb_options/report.html'>打开全部 RGB surfel 选项</a>；chunk-ID/disk 仅作 secondary audit。</p></section>
<section><h2>完整回访视频：B1 首访 → 离开 → 返回 → B2</h2><p>这些是完整 273-frame 回访，不是只截 B2。</p>
<button onclick="playGroup('full')">全部播放</button><button onclick="pauseGroup('full')">全部暂停</button><button onclick="resetGroup('full')">全部复位</button><div class='grid full'>{full_videos}</div></section>
<section><h2>Return / re-entry window（补充）</h2><button onclick="playGroup('short')">全部播放</button><button onclick="pauseGroup('short')">全部暂停</button><button onclick="resetGroup('short')">全部复位</button><div class='grid short'>{reentry_videos}</div></section>
<section><h2>Main metrics（同一 hard-overlap / strong-core non-overlap）</h2><table><tr><th>Method</th><th>Overlap B1→B2 ↓</th><th>Non-overlap Δ ↓</th><th>Re-entry mean ↓</th><th>Re-entry peak ↓</th><th>Total</th></tr>{rows}</table></section>
<section><h2>Final interpretation</h2><p>{conclusion}</p><p><b>下一步唯一最高优先级实验：</b>{next_action}</p></section>
</main><script>const group=n=>[...document.querySelectorAll('.'+n+' video.sync')];
function playGroup(n){{const v=group(n),t=v.length?v[0].currentTime:0;v.forEach(x=>{{x.currentTime=t;x.play()}})}}
function pauseGroup(n){{group(n).forEach(x=>x.pause())}}function resetGroup(n){{group(n).forEach(x=>{{x.pause();x.currentTime=0}})}}</script></body></html>"""
    (root / "report.html").write_text(html_payload, encoding="utf-8")
    table = "\n".join(
        f"| {LABELS[name]} | {_fmt(value['overlap_b1_to_b2_l1'])} | {_fmt(value['nonoverlap_delta_vs_baseline_l1'])} | {_fmt(value['reentry_window_mean_l1'])} | {_fmt(value['reentry_window_peak_l1'])} |"
        for name, value in methods.items()
    )
    markdown = f"""# MapKV Identity Recovery Stage

Status: **{metrics['status']}**

本次最新方法：**Canonical-K/V target-grid 重寻址**<br>
本次最佳质量方法：**RGB-Warp→VAE WRE**

## Architecture

![完整 Pipeline](assets/architecture_graph.svg)

完整变更见 [architecture.md](architecture.md)、[architecture_state.json](architecture_state.json) 与 [architecture_changes.json](architecture_changes.json)。

## Results

| Method | Overlap B1→B2 ↓ | Non-overlap Δ ↓ | Re-entry mean ↓ | Re-entry peak ↓ |
|---|---:|---:|---:|---:|
{table}

### Q1 — Mask attenuation

Strong core identity gain: `{metrics['priority_1']['overlap_gain']:.5f}`；同时存在 locality cost = `{metrics['priority_1']['strong_core_has_locality_cost']}`。

### Q2 — Representation bottleneck

RGB-Warp→VAE better than Latent-Warp = `{metrics['priority_2']['rgb_warp_vae_better_than_latent_warp']}`；quality oracle = `{metrics['priority_2']['quality_oracle']}`。

### Q3 — Native re-addressing

Canonical source reconstruction max diff = `{metrics['priority_3']['source_reconstruction_max_abs_diff']}`，但 Canonical→RGB overlap gap = `{metrics['priority_3']['canonical_overlap_gap_to_rgb']:.5f}`；does not approximate oracle.

## Complete videos

完整 273-frame B1→leave→return→B2 视频位于 `videos/report/full_revisit_*.mp4`。

## Conclusion

{conclusion}

## Next action

{next_action}
"""
    (root / "report.md").write_text(markdown, encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    git_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True)
    (root / "status.json").write_text(json.dumps({"status": metrics["status"], "git_commit": git_commit, "working_tree_clean": not bool(git_status), "report": str(root / "report.html")}, indent=2), encoding="utf-8")
    if git_status:
        (root / "git_diff_stat.txt").write_text(subprocess.check_output(["git", "-C", str(repo), "diff", "--stat"], text=True), encoding="utf-8")
    return str(root / "report.html")


__all__ = ["build_report"]
