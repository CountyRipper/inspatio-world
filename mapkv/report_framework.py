from __future__ import annotations

import html
import json
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


CHANGE_TYPES = {"unchanged", "added", "modified", "removed"}
REQUIRED_PIPELINE_ROLES = {
    "input",
    "generation",
    "geometry",
    "address",
    "payload",
    "context",
    "attention",
    "output",
    "evaluation",
}


@dataclass(frozen=True)
class ArchitectureNode:
    id: str
    label_zh: str
    label_en: str
    role: str
    column: int
    row: int
    summary: str
    change_type: str = "unchanged"
    focus: bool = False
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchitectureEdge:
    source: str
    target: str
    label: str = ""


@dataclass(frozen=True)
class ArchitectureChange:
    component_id: str
    change_type: str
    before: str
    after: str
    affected_files: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class ArchitectureSnapshot:
    name: str
    focus_zh: str
    focus_en: str
    nodes: tuple[ArchitectureNode, ...]
    edges: tuple[ArchitectureEdge, ...]
    changes: tuple[ArchitectureChange, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("Architecture graph must contain nodes")
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Architecture node IDs must be unique")
        node_by_id = {node.id: node for node in self.nodes}
        for node in self.nodes:
            if node.change_type not in CHANGE_TYPES:
                raise ValueError(
                    f"Unsupported change type for {node.id}: {node.change_type}"
                )
            if node.column < 0 or node.row < 0:
                raise ValueError(f"Negative graph position for {node.id}")
        roles = {node.role for node in self.nodes}
        missing_roles = REQUIRED_PIPELINE_ROLES - roles
        if missing_roles:
            raise ValueError(
                "Incomplete pipeline graph; missing roles: "
                + ", ".join(sorted(missing_roles))
            )
        if not any(node.focus for node in self.nodes):
            raise ValueError("Architecture graph must mark the current focus")
        for edge in self.edges:
            if edge.source not in node_by_id or edge.target not in node_by_id:
                raise ValueError(
                    f"Architecture edge references unknown node: {edge}"
                )
        change_by_component = {change.component_id: change for change in self.changes}
        if len(change_by_component) != len(self.changes):
            raise ValueError("Architecture changes must have unique component IDs")
        for change in self.changes:
            if change.component_id not in node_by_id:
                raise ValueError(
                    f"Architecture change references unknown node: {change.component_id}"
                )
            if change.change_type not in {"added", "modified", "removed"}:
                raise ValueError(
                    f"Invalid architecture change type: {change.change_type}"
                )
            if not change.before or not change.after:
                raise ValueError(
                    f"Architecture change {change.component_id} needs before/after"
                )
            if not change.affected_files:
                raise ValueError(
                    f"Architecture change {change.component_id} needs affected files"
                )
            if not change.rationale:
                raise ValueError(
                    f"Architecture change {change.component_id} needs rationale"
                )
            if node_by_id[change.component_id].change_type != change.change_type:
                raise ValueError(
                    f"Graph/table change type mismatch for {change.component_id}"
                )
        for node in self.nodes:
            if node.change_type != "unchanged" and node.id not in change_by_component:
                raise ValueError(
                    f"Changed graph node {node.id} lacks a change annotation"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _svg_text_lines(value: str, width: int) -> list[str]:
    return textwrap.wrap(value, width=width, break_long_words=False) or [""]


def render_architecture_svg(snapshot: ArchitectureSnapshot) -> str:
    snapshot.validate()
    node_width = 230
    node_height = 104
    x_gap = 62
    y_gap = 64
    margin_x = 44
    margin_y = 118
    max_column = max(node.column for node in snapshot.nodes)
    max_row = max(node.row for node in snapshot.nodes)
    width = margin_x * 2 + (max_column + 1) * node_width + max_column * x_gap
    height = margin_y + (max_row + 1) * node_height + max_row * y_gap + 48
    positions = {
        node.id: (
            margin_x + node.column * (node_width + x_gap),
            margin_y + node.row * (node_height + y_gap),
        )
        for node in snapshot.nodes
    }
    palette = {
        "unchanged": ("#edf2f7", "#718096", "#263445"),
        "added": ("#e8f7ed", "#2f855a", "#184f32"),
        "modified": ("#fff4df", "#d97706", "#7a3e00"),
        "removed": ("#fdecec", "#c53030", "#7a1d1d"),
    }
    badges = {
        "unchanged": "未修改",
        "added": "新增",
        "modified": "已修改",
        "removed": "已移除",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="Noto Sans CJK SC, Source Han Sans SC, sans-serif">',
        "<style>text{font-family:'Noto Sans CJK SC','Source Han Sans SC',"
        "'WenQuanYi Zen Hei','Microsoft YaHei',sans-serif}</style>",
        "<defs><marker id='arrow' markerWidth='10' markerHeight='8' "
        "refX='9' refY='4' orient='auto'><path d='M0,0 L10,4 L0,8 z' "
        "fill='#65758b'/></marker></defs>",
        "<rect width='100%' height='100%' fill='#f8fafc'/>",
        f"<text x='{margin_x}' y='32' font-size='19' font-weight='700' "
        f"fill='#172033'>{html.escape(snapshot.name)}</text>",
        f"<text x='{margin_x}' y='58' font-size='15' font-weight='650' "
        f"fill='#6b46c1'>本次关注：{html.escape(snapshot.focus_zh)}</text>",
        f"<text x='{margin_x}' y='80' font-size='12' fill='#596579'>"
        f"{html.escape(snapshot.focus_en)}</text>",
    ]
    legend_x = max(margin_x, width - 440)
    for offset, change_type in enumerate(("unchanged", "added", "modified", "removed")):
        fill, stroke, _ = palette[change_type]
        x = legend_x + offset * 105
        parts.extend(
            [
                f"<rect x='{x}' y='27' width='15' height='15' rx='3' "
                f"fill='{fill}' stroke='{stroke}'/>",
                f"<text x='{x + 21}' y='40' font-size='11' fill='#46556a'>"
                f"{badges[change_type]}</text>",
            ]
        )
    for edge in snapshot.edges:
        sx, sy = positions[edge.source]
        tx, ty = positions[edge.target]
        x1 = sx + node_width
        y1 = sy + node_height / 2
        x2 = tx
        y2 = ty + node_height / 2
        middle = (x1 + x2) / 2
        parts.append(
            f"<path d='M{x1},{y1} C{middle},{y1} {middle},{y2} {x2},{y2}' "
            "fill='none' stroke='#65758b' stroke-width='2' marker-end='url(#arrow)'/>"
        )
        if edge.label:
            parts.append(
                f"<text x='{middle}' y='{(y1 + y2) / 2 - 7}' text-anchor='middle' "
                f"font-size='10' fill='#596579'>{html.escape(edge.label)}</text>"
            )
    for node in snapshot.nodes:
        x, y = positions[node.id]
        fill, stroke, text_color = palette[node.change_type]
        stroke_width = 4 if node.focus else 2
        if node.focus:
            stroke = "#6b46c1"
        parts.append(
            f"<rect x='{x}' y='{y}' width='{node_width}' height='{node_height}' "
            f"rx='12' fill='{fill}' stroke='{stroke}' stroke-width='{stroke_width}'/>"
        )
        parts.append(
            f"<text x='{x + 12}' y='{y + 21}' font-size='10' font-weight='700' "
            f"fill='{stroke}'>{badges[node.change_type]}"
            f"{' · 本次 Focus' if node.focus else ''}</text>"
        )
        y_text = y + 44
        for line in _svg_text_lines(node.label_zh, 16)[:2]:
            parts.append(
                f"<text x='{x + 12}' y='{y_text}' font-size='14' font-weight='700' "
                f"fill='{text_color}'>{html.escape(line)}</text>"
            )
            y_text += 17
        parts.append(
            f"<text x='{x + 12}' y='{y + 88}' font-size='10.5' fill='#596579'>"
            f"{html.escape(node.label_en[:37])}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def render_changes_html(snapshot: ArchitectureSnapshot) -> str:
    snapshot.validate()
    if not snapshot.changes:
        return "<p><b>本次无架构变化</b></p>"
    nodes = {node.id: node for node in snapshot.nodes}
    rows = []
    for change in snapshot.changes:
        node = nodes[change.component_id]
        files = "<br>".join(
            f"<code>{html.escape(path)}</code>" for path in change.affected_files
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(node.label_zh)}<br><small>{html.escape(node.label_en)}</small></td>"
            f"<td>{html.escape(change.change_type)}</td>"
            f"<td>{html.escape(change.before)}</td>"
            f"<td>{html.escape(change.after)}</td>"
            f"<td>{files}</td>"
            f"<td>{html.escape(change.rationale)}</td>"
            "</tr>"
        )
    return (
        "<table><tr><th>组件</th><th>类型</th><th>修改前</th><th>修改后</th>"
        "<th>影响文件</th><th>原因</th></tr>" + "".join(rows) + "</table>"
    )


def render_pipeline_table_html(snapshot: ArchitectureSnapshot) -> str:
    snapshot.validate()
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(node.label_zh)}</td>"
        f"<td>{html.escape(node.role)}</td>"
        f"<td>{html.escape(node.change_type)}</td>"
        f"<td>{html.escape(node.summary)}</td>"
        f"<td>{'<br>'.join(html.escape(path) for path in node.files) or '—'}</td>"
        "</tr>"
        for node in sorted(snapshot.nodes, key=lambda item: (item.column, item.row))
    )
    return (
        "<table><tr><th>模块</th><th>职责</th><th>状态</th><th>当前行为</th>"
        "<th>实现文件</th></tr>" + rows + "</table>"
    )


def _markdown_changes(snapshot: ArchitectureSnapshot) -> str:
    if not snapshot.changes:
        return "本次无架构变化。"
    lines = [
        "| 组件 | 类型 | 修改前 | 修改后 | 影响文件 | 原因 |",
        "|---|---|---|---|---|---|",
    ]
    nodes = {node.id: node for node in snapshot.nodes}
    for change in snapshot.changes:
        node = nodes[change.component_id]
        files = "<br>".join(f"`{path}`" for path in change.affected_files)
        lines.append(
            f"| {node.label_zh} | {change.change_type} | {change.before} | "
            f"{change.after} | {files} | {change.rationale} |"
        )
    return "\n".join(lines)


def write_architecture_bundle(
    root: str | Path, snapshot: ArchitectureSnapshot
) -> dict[str, str]:
    snapshot.validate()
    root = Path(root).resolve()
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    state_path = root / "architecture_state.json"
    changes_path = root / "architecture_changes.json"
    graph_path = assets / "architecture_graph.svg"
    markdown_path = root / "architecture.md"
    state_path.write_text(
        json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    changes_path.write_text(
        json.dumps(
            [asdict(change) for change in snapshot.changes],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    graph_path.write_text(render_architecture_svg(snapshot), encoding="utf-8")
    pipeline_lines = "\n".join(
        f"- **{node.label_zh}** (`{node.role}`, {node.change_type}): {node.summary}"
        for node in sorted(snapshot.nodes, key=lambda item: (item.column, item.row))
    )
    markdown_path.write_text(
        f"# Architecture — {snapshot.name}\n\n"
        f"本次关注：**{snapshot.focus_zh}**  \n{snapshot.focus_en}\n\n"
        "![完整 Pipeline](assets/architecture_graph.svg)\n\n"
        f"## 完整 Pipeline\n\n{pipeline_lines}\n\n"
        f"## 本次架构变化\n\n{_markdown_changes(snapshot)}\n",
        encoding="utf-8",
    )
    return {
        "state": str(state_path),
        "changes": str(changes_path),
        "graph": str(graph_path),
        "markdown": str(markdown_path),
    }


def node(
    *,
    id: str,
    label_zh: str,
    label_en: str,
    role: str,
    column: int,
    row: int,
    summary: str,
    change_type: str = "unchanged",
    focus: bool = False,
    files: Iterable[str] = (),
) -> ArchitectureNode:
    return ArchitectureNode(
        id=id,
        label_zh=label_zh,
        label_en=label_en,
        role=role,
        column=column,
        row=row,
        summary=summary,
        change_type=change_type,
        focus=focus,
        files=tuple(files),
    )


__all__ = [
    "ArchitectureChange",
    "ArchitectureEdge",
    "ArchitectureNode",
    "ArchitectureSnapshot",
    "REQUIRED_PIPELINE_ROLES",
    "node",
    "render_architecture_svg",
    "render_changes_html",
    "render_pipeline_table_html",
    "write_architecture_bundle",
]
