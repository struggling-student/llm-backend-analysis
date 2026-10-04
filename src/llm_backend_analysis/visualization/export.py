"""Static export of selected notebook figures (presentation artifacts only).

Notebooks call `export_figure(fig, name, ...)` on the *same* figure object they display,
so an export can never diverge from the canonical analysis. Files go to
results/figures/<topic>/<name>.{svg,png} and are listed with their source notebook and
caption in results/figures/<topic>/figures.json. Exports are deterministic (Kaleido's
random SVG ids and attribute order are normalized), so regenerating unchanged figures
leaves git clean. Set LBA_EXPORT=0 to skip writing files during exploratory runs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import plotly.graph_objects as go

from .. import paths

DEFAULT_FORMATS = ("svg", "png")
PNG_SCALE = 2

_UID = re.compile(rb'id="topdefs-([0-9a-f]{6})"')
_TAG = re.compile(rb'<([A-Za-z][\w:-]*)((?:\s+[\w:-]+="[^"]*")+)\s*(/?)>')
_ATTR = re.compile(rb'([\w:-]+)="([^"]*)"')


def _sort_attributes(m: re.Match[bytes]) -> bytes:
    attrs = sorted(_ATTR.findall(m.group(2)))
    return b"<" + m.group(1) + b"".join(b" " + k + b'="' + v + b'"' for k, v in attrs) + m.group(3) + b">"


def _normalize_svg(svg: bytes) -> bytes:
    """Replace Kaleido's random figure id and sort attributes (their order varies between renders)."""
    m = _UID.search(svg)
    if m:
        svg = svg.replace(m.group(1), b"lbafig")
    return _TAG.sub(_sort_attributes, svg)


def exports_enabled() -> bool:
    return os.environ.get("LBA_EXPORT", "1") != "0"


def export_figure(fig: go.Figure, name: str, *, topic: str = "backend_selection", source: str, caption: str,
                  formats: tuple[str, ...] = DEFAULT_FORMATS, width: int | None = None,
                  height: int | None = None) -> list[Path]:
    """Write `fig` as results/figures/<topic>/<name>.<fmt> and record it in figures.json.

    `source` is the notebook that owns the figure (e.g. "04_concurrency_scaling.ipynb");
    `caption` says what the figure shows, for whoever embeds it (Notion, thesis, README)."""
    if not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", name):
        raise ValueError(f"figure name {name!r} must be snake_case")
    if not exports_enabled():
        return []
    out = paths.FIGURES / topic
    out.mkdir(parents=True, exist_ok=True)
    w = width or fig.layout.width
    h = height or fig.layout.height
    fig = go.Figure(fig)  # copy: fixed trace uids make the SVG reproducible without touching the notebook figure
    for i, trace in enumerate(fig.data):
        trace.uid = f"t{i:03d}"
    written = []
    for fmt in formats:
        data = fig.to_image(format=fmt, width=w, height=h, scale=PNG_SCALE if fmt == "png" else 1)
        if fmt == "svg":
            data = _normalize_svg(data)
        path = out / f"{name}.{fmt}"
        path.write_bytes(data)
        written.append(path)
    manifest_path = out / "figures.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[name] = {"source_notebook": f"notebooks/{topic}/{source}", "caption": caption,
                      "files": [p.name for p in written], "width": w, "height": h}
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2, ensure_ascii=False) + "\n")
    return written
