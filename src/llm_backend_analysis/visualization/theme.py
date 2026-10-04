"""Visual language shared by every figure.

Encoding (fixed across notebooks and exports):
  colour        = backend      (vLLM blue, llama.cpp orange)
  line/marker   = precision    (BF16 solid line + filled marker, INT8 dashed line + open marker)
  marker shape  = backend      (vLLM circle, llama.cpp square), so identity never relies on colour alone

Colours are the first two slots of a validated categorical palette (CVD ΔE 24.7, contrast
>= 3:1 on the light surface). Workload-coloured figures (ratios) use slots 3-7 with
direct labels, because some of those slots are below 3:1 contrast.
Styles derive from (backend, precision) metadata, so a new configuration such as vLLM
INT8 is styled automatically: blue, dashed, open circles.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio

from ..data import registry

# Surfaces and ink
PAPER = "#ffffff"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e6e5e0"
REFERENCE = "#52514e"  # reference lines (STREAM ceiling, ratio = 1, linear scaling)

BACKEND_COLORS = {"vllm": "#2a78d6", "llama_cpp": "#eb6834"}
BACKEND_SYMBOLS = {"vllm": "circle", "llama_cpp": "square"}
EXTRA_COLORS = ["#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300"]  # slots 3-7, never reused for backends
PRECISION_STYLES = {"bf16": {"dash": "solid", "open": False}, "int8": {"dash": "dash", "open": True}}
WORKLOAD_SYMBOLS = ["circle", "diamond", "triangle-up", "square", "x"]

FONT_FAMILY = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT_SIZE = 13
WIDTH = 1100
HEIGHT = 460
TEMPLATE_NAME = "llm_backend_analysis"


@dataclass(frozen=True)
class TraceStyle:
    name: str
    color: str
    dash: str
    symbol: str

    def line(self, width: float = 2.0) -> dict:
        return {"color": self.color, "width": width, "dash": self.dash}

    def marker(self, size: int = 8) -> dict:
        return {"color": self.color, "size": size, "symbol": self.symbol, "line": {"color": self.color, "width": 1.5}}


def configuration_style(configuration_id: str) -> TraceStyle:
    c = registry.configurations()[configuration_id]
    p = PRECISION_STYLES.get(c.precision, {"dash": "dot", "open": True})
    symbol = BACKEND_SYMBOLS.get(c.backend, "diamond") + ("-open" if p["open"] else "")
    return TraceStyle(name=c.label, color=BACKEND_COLORS.get(c.backend, EXTRA_COLORS[-1]), dash=p["dash"], symbol=symbol)


def workload_style(workload: str) -> TraceStyle:
    order = registry.workload_order()
    i = order.index(workload) if workload in order else len(order)
    return TraceStyle(name=registry.workload_label(workload), color=EXTRA_COLORS[i % len(EXTRA_COLORS)],
                      dash="solid", symbol=WORKLOAD_SYMBOLS[i % len(WORKLOAD_SYMBOLS)])


def _template() -> go.layout.Template:
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=TEXT_MUTED, ticks="outside", tickcolor=TEXT_MUTED,
                title_font=dict(size=FONT_SIZE, color=TEXT_SECONDARY), tickfont=dict(color=TEXT_SECONDARY),
                automargin=True)
    return go.layout.Template(layout=go.Layout(
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=TEXT_PRIMARY),
        title=dict(x=0.012, xref="container", xanchor="left", font=dict(size=17, color=TEXT_PRIMARY)),
        paper_bgcolor=PAPER, plot_bgcolor=SURFACE, width=WIDTH, height=HEIGHT,
        margin=dict(l=70, r=30, t=95, b=115),
        # Legend under the plot area, anchored to the figure (container) so it never meets titles or axes.
        legend=dict(orientation="h", yref="container", y=0.005, yanchor="bottom", xanchor="left", x=0.0,
                    title_text="", font=dict(size=12, color=TEXT_PRIMARY)),
        hovermode="closest",
        hoverlabel=dict(bgcolor=PAPER, bordercolor=TEXT_MUTED, font=dict(family=FONT_FAMILY, size=12, color=TEXT_PRIMARY)),
        xaxis=axis, yaxis=axis,
        annotationdefaults=dict(font=dict(size=11, color=TEXT_SECONDARY)),
    ))


def register_theme() -> None:
    """Register the template and make it the default (idempotent)."""
    pio.templates[TEMPLATE_NAME] = _template()
    pio.templates.default = f"plotly_white+{TEMPLATE_NAME}"


def setup_notebook(static_fallback: bool = True) -> None:
    """Theme + renderer for notebooks: interactive Plotly output for JupyterLab/VS Code, plus a PNG
    copy so the executed notebook also shows its figures where JavaScript is not run (GitHub)."""
    register_theme()
    pio.renderers.default = "plotly_mimetype+png" if static_fallback else "plotly_mimetype"
    pio.renderers["png"].width = WIDTH
    pio.renderers["png"].scale = 1


register_theme()
