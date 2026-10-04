"""Figure builders. Each takes an analysis-ready DataFrame and returns a go.Figure; the same
object is displayed in the notebook and, for selected figures, exported by export.py.

Conventions: concurrency on a log2 axis with the measured C values as ticks; error bars are
±1 standard deviation over repetitions; no dual y axes (different quantities get separate
rows sharing x); reference lines are grey and labelled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data import registry
from . import theme


@dataclass
class Panel:
    """One row of a concurrency grid: which metric, how to label and scale it."""
    metric: str                       # column, e.g. "total_tok_s_mean"
    title: str                        # y-axis title incl. unit
    unit: str = ""                    # unit shown in the hover label
    scale: float = 1.0                # e.g. 1e3 to show seconds as ms
    log_y: bool = False
    error: str | None = "std"         # suffix of the spread column, None for no error bars
    share_y: bool = False             # one y range across facets of this row
    fmt: str = ".3g"
    reference: tuple[float, str] | None = None   # horizontal line (value in plotted units, label)
    range_to_zero: bool = True
    extra_hover: dict[str, str] = field(default_factory=dict)  # column -> label


def _finish(fig: go.Figure) -> go.Figure:
    """Subplot titles in secondary ink and a size below the figure title."""
    fig.update_annotations(selector=dict(xref="paper", yanchor="bottom"), font_size=13,
                           font_color=theme.TEXT_PRIMARY)
    return fig


def _reference_line(fig: go.Figure, *, axis: str, value: float, label: str | None, row: int, col: int,
                    log: bool = False) -> None:
    """Dotted reference line with an optional label.

    Shapes on log axes take data coordinates, annotations take log10 coordinates, so the label
    is added separately instead of through add_hline/add_vline's annotation_text."""
    line = dict(color=theme.REFERENCE, dash="dot", width=1.3)
    (fig.add_hline(y=value, row=row, col=col, line=line) if axis == "y"
     else fig.add_vline(x=value, row=row, col=col, line=line))
    if not label:
        return
    pos = np.log10(value) if log else value
    ref = fig.get_subplot(row, col)
    xa, ya = ref.xaxis.plotly_name.replace("axis", ""), ref.yaxis.plotly_name.replace("axis", "")
    if axis == "y":
        fig.add_annotation(x=0.01, xref=f"{xa} domain", y=pos, yref=ya, text=label, showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=11, color=theme.TEXT_SECONDARY))
    else:
        fig.add_annotation(x=pos, xref=xa, y=0.99, yref=f"{ya} domain", text=label, showarrow=False, xanchor="right",
                           yanchor="top", xshift=-4, font=dict(size=11, color=theme.TEXT_SECONDARY))


def title_text(title: str, subtitle: str | None = None) -> str:
    """Bold title with an optional grey subtitle line."""
    return f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{theme.TEXT_SECONDARY}'>{subtitle}</span>"
                                if subtitle else "")


def _facet_values(df: pd.DataFrame, facet: str) -> list:
    present = list(dict.fromkeys(df[facet]))
    if facet == "workload":
        return [w for w in registry.workload_order() if w in present]
    if facet == "model":
        return [m for m in registry.models()["model"] if m in present]
    return sorted(present)


def _facet_title(facet: str, value, compact: bool = True) -> str:
    if facet == "workload":
        label = registry.workload_label(value)
        head, _, shape = label.rpartition(" ")
        return f"{head}<br>{shape}" if compact and head else label
    if facet == "model":
        return registry.model_label(value)
    if facet == "configuration_id":
        return registry.configuration_label(value)
    return f"{facet} = {value}"


def log_ticks(values: Sequence[float] | pd.Series) -> list[float]:
    """1-2-5 tick values covering the data range, for readable log axes."""
    v = pd.Series(values, dtype=float)
    v = v[v > 0].dropna()
    if v.empty:
        return []
    lo, hi = np.floor(np.log10(v.min())), np.ceil(np.log10(v.max()))
    ticks = [m * 10 ** e for e in range(int(lo), int(hi) + 1) for m in (1, 2, 5)]
    return [t for t in ticks if v.min() / 1.6 <= t <= v.max() * 1.6]


def _apply_log_ticks(fig: go.Figure, values, **where) -> None:
    ticks = log_ticks(values)
    if ticks:
        fig.update_yaxes(tickvals=ticks, ticktext=[f"{t:g}" for t in ticks], **where)


def _configurations(df: pd.DataFrame, configurations: Sequence[str] | None) -> list[str]:
    present = set(df["configuration_id"])
    return [c for c in (configurations or registry.configuration_order()) if c in present]


def concurrency_grid(points: pd.DataFrame, panels: Sequence[Panel], *, title: str, subtitle: str | None = None,
                     facet: str = "workload", configurations: Sequence[str] | None = None,
                     height: int | None = None, width: int | None = None) -> go.Figure:
    """Rows = panels (metrics), columns = facet values, one trace per configuration."""
    facets = _facet_values(points, facet)
    confs = _configurations(points, configurations)
    fig = make_subplots(rows=len(panels), cols=len(facets), shared_xaxes=True,
                        subplot_titles=[_facet_title(facet, f) for f in facets],
                        horizontal_spacing=0.035 if len(facets) > 3 else 0.06, vertical_spacing=0.07)
    for r, p in enumerate(panels, start=1):
        spread = p.metric.replace("_mean", f"_{p.error}") if (p.error and p.metric.endswith("_mean")) else None
        for j, fv in enumerate(facets, start=1):
            for cid in confs:
                g = points[(points[facet] == fv) & (points["configuration_id"] == cid)].sort_values("concurrency")
                if g.empty or g[p.metric].isna().all():
                    continue
                st = theme.configuration_style(cid)
                y = g[p.metric] * p.scale
                err = g[spread] * p.scale if spread and spread in g else None
                extra = list(p.extra_hover.items())
                custom = np.column_stack([err if err is not None else np.full(len(g), np.nan),
                                          g["n_reps"] if "n_reps" in g else np.full(len(g), np.nan),
                                          *[g[c] for c, _ in extra]])
                hover = (f"<b>{st.name}</b><br>C = %{{x}}<br>{p.title.split('<br>')[0]}: %{{y:{p.fmt}}} {p.unit}"
                         + (f" ± %{{customdata[0]:{p.fmt}}}" if err is not None else "")
                         + ("<br>repetitions: %{customdata[1]}" if "n_reps" in g else "")
                         + "".join(f"<br>{lab}: %{{customdata[{k + 2}]:.3g}}" for k, (_, lab) in enumerate(extra))
                         + "<extra></extra>")
                fig.add_trace(go.Scatter(
                    x=g["concurrency"], y=y, mode="lines+markers", name=st.name, legendgroup=cid,
                    showlegend=(r == 1 and j == 1), line=st.line(), marker=st.marker(),
                    error_y=dict(type="data", array=err, thickness=1.1, width=3, color=st.color) if err is not None else None,
                    customdata=custom, hovertemplate=hover), row=r, col=j)
            if p.reference is not None:
                _reference_line(fig, axis="y", value=p.reference[0], label=p.reference[1] if j == 1 else None,
                                row=r, col=j, log=p.log_y)
        fig.update_yaxes(title_text=p.title, row=r, col=1)
        fig.update_yaxes(type="log" if p.log_y else "linear", row=r)
        if p.log_y:
            _apply_log_ticks(fig, points[p.metric] * p.scale, row=r)
        if not p.log_y and p.range_to_zero:
            fig.update_yaxes(rangemode="tozero", row=r)
        if p.share_y:
            v = points[p.metric] * p.scale
            hi = v + (points[spread].fillna(0) * p.scale if spread and spread in points else 0)
            top = max(float(np.nanmax(hi)), p.reference[0] if p.reference else 0.0)
            if p.log_y:
                lo = float(np.nanmin(v[v > 0]))
                fig.update_yaxes(range=[np.log10(lo) - 0.08, np.log10(top) + 0.08], row=r)
            else:
                fig.update_yaxes(range=[0, 1.08 * top], row=r)
    cs = sorted(points["concurrency"].unique())
    fig.update_xaxes(type="log", tickvals=cs, ticktext=[str(int(c)) for c in cs])
    for j in range(1, len(facets) + 1):
        fig.update_xaxes(title_text="Concurrent requests (C)", row=len(panels), col=j)
    fig.update_layout(title_text=title_text(title, subtitle), height=height or (210 + 330 * len(panels)),
                      width=width or theme.WIDTH)
    return _finish(fig)


def concurrency_lines(points: pd.DataFrame, panel: Panel, **kw) -> go.Figure:
    return concurrency_grid(points, [panel], **kw)


def ratio_vs_concurrency(ratio: pd.DataFrame, *, title: str, subtitle: str | None = None, y_title: str,
                         facet: str = "model", log_y: bool = True) -> go.Figure:
    """Ratio between two configurations per workload (colour = workload, direct end labels)."""
    facets = _facet_values(ratio, facet)
    fig = make_subplots(rows=1, cols=len(facets), subplot_titles=[_facet_title(facet, f) for f in facets],
                        shared_yaxes=True, horizontal_spacing=0.06)
    for j, fv in enumerate(facets, start=1):
        d = ratio[ratio[facet] == fv]
        labels = []
        for wl in _facet_values(d, "workload"):
            g = d[d["workload"] == wl].sort_values("concurrency")
            st = theme.workload_style(wl)
            err = g["ratio"] * g["ratio_rel_err"] if "ratio_rel_err" in g else None
            fig.add_trace(go.Scatter(
                x=g["concurrency"], y=g["ratio"], mode="lines+markers", name=registry.workload_label(wl),
                legendgroup=wl, showlegend=(j == 1), line=dict(color=st.color, width=2), marker=dict(
                    color=st.color, symbol=st.symbol, size=8),
                error_y=dict(type="data", array=err, thickness=1, width=3) if err is not None else None,
                hovertemplate=f"<b>{registry.workload_label(wl)}</b><br>C = %{{x}}<br>ratio: %{{y:.2f}}×<extra></extra>"),
                row=1, col=j)
            last = g.iloc[-1]
            labels.append([np.log10(last["ratio"]) if log_y else last["ratio"], np.log10(last["concurrency"]),
                           f"{registry.workload_label(wl, short=True)}  {last['ratio']:.1f}×"])
        # Direct end labels, pushed apart vertically so they never overlap.
        labels.sort()
        gap = 0.042 if log_y else 0.05 * (d["ratio"].max() - d["ratio"].min())
        for k in range(1, len(labels)):
            labels[k][0] = max(labels[k][0], labels[k - 1][0] + gap)
        for y_pos, x_pos, text in labels:
            fig.add_annotation(x=x_pos, y=y_pos, text=text, showarrow=False, xanchor="left", xshift=7,
                               font=dict(size=11, color=theme.TEXT_SECONDARY), row=1, col=j)
        _reference_line(fig, axis="y", value=1, label="parity (1×)" if j == 1 else None, row=1, col=j, log=log_y)
    cs = sorted(ratio["concurrency"].unique())
    fig.update_xaxes(type="log", tickvals=cs, ticktext=[str(int(c)) for c in cs], title_text="Concurrent requests (C)")
    fig.update_xaxes(range=[np.log10(min(cs)) - 0.1, np.log10(max(cs)) + 0.45])
    fig.update_yaxes(type="log" if log_y else "linear")
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    if log_y:
        ticks = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8]
        fig.update_yaxes(tickvals=ticks, ticktext=[f"{t:g}×" for t in ticks])
    fig.update_layout(title_text=title_text(title, subtitle), height=520)
    return _finish(fig)


def grouped_bars(summary: pd.DataFrame, *, x: str, y: str, error: str | None, title: str, y_title: str | Sequence[str],
                 facet: str, subtitle: str | None = None, configurations: Sequence[str] | None = None,
                 show_values: bool = True, value_fmt: str = ".0f", log_y: bool = False,
                 facet_labels: dict | None = None, height: int = 500) -> go.Figure:
    """Grouped bars per configuration (INT8 bars hatched), one panel per facet value.

    y_title may be a list (one per facet) when facets hold different quantities; each panel
    then keeps its own y axis."""
    facets = [f for f in (facet_labels or {}) if f in set(summary[facet])] or _facet_values(summary, facet)
    confs = _configurations(summary, configurations)
    titles = [(facet_labels or {}).get(f) or _facet_title(facet, f) for f in facets]
    fig = make_subplots(rows=1, cols=len(facets), subplot_titles=titles, horizontal_spacing=0.07)
    xs = list(dict.fromkeys(summary.sort_values([c for c in ("workload_order",) if c in summary] or [x])[x]))
    for j, fv in enumerate(facets, start=1):
        for cid in confs:
            g = summary[(summary[facet] == fv) & (summary["configuration_id"] == cid)].set_index(x).reindex(xs)
            if g[y].isna().all():
                continue
            st = theme.configuration_style(cid)
            fig.add_trace(go.Bar(
                x=xs, y=g[y], name=st.name, legendgroup=cid, showlegend=(j == 1), marker_color=st.color,
                marker_line=dict(color=theme.PAPER, width=1.5),
                marker_pattern=dict(shape="/" if st.dash != "solid" else "", fgcolor=theme.PAPER, solidity=0.25),
                error_y=dict(type="data", array=g[error], thickness=1.1, width=3, color=theme.TEXT_SECONDARY)
                if error else None,
                text=g[y].map(lambda v: f"{v:{value_fmt}}" if pd.notna(v) else "") if show_values else None,
                textposition="outside", textfont=dict(size=10, color=theme.TEXT_SECONDARY), cliponaxis=False,
                hovertemplate=f"<b>{st.name}</b><br>%{{x}}<br>{titles[j - 1]}: %{{y:.4g}}<extra></extra>"), row=1, col=j)
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.05, title_text=title_text(title, subtitle),
                      height=height)
    if isinstance(y_title, str):
        fig.update_yaxes(title_text=y_title, row=1, col=1)
    else:
        for j, t in enumerate(y_title, start=1):
            fig.update_yaxes(title_text=t, row=1, col=j)
    fig.update_yaxes(type="log" if log_y else "linear", rangemode="tozero")
    return _finish(fig)


def path_scatter(points: pd.DataFrame, *, x: str, y: str, x_title: str, y_title: str, title: str,
                 subtitle: str | None = None, facet: str = "workload", x_scale: float = 1.0, y_scale: float = 1.0,
                 log_x: bool = False, log_y: bool = False, label_cs: Sequence[int] = (1, 8, 64),
                 configurations: Sequence[str] | None = None, reference_x: tuple[float, str] | None = None,
                 height: int = 530) -> go.Figure:
    """One path per configuration through its concurrency points (labelled C=...), e.g. latency vs throughput."""
    facets = _facet_values(points, facet)
    confs = _configurations(points, configurations)
    fig = make_subplots(rows=1, cols=len(facets), subplot_titles=[_facet_title(facet, f) for f in facets],
                        horizontal_spacing=0.05)
    for j, fv in enumerate(facets, start=1):
        for cid in confs:
            g = points[(points[facet] == fv) & (points["configuration_id"] == cid)].sort_values("concurrency")
            if g.empty:
                continue
            st = theme.configuration_style(cid)
            last_c = int(g["concurrency"].max())
            text = [f"C{int(c)}" if (int(c) in label_cs or int(c) == last_c) else "" for c in g["concurrency"]]
            fig.add_trace(go.Scatter(
                x=g[x] * x_scale, y=g[y] * y_scale, mode="lines+markers+text", text=text, textposition="top center",
                textfont=dict(size=10, color=theme.TEXT_SECONDARY), name=st.name, legendgroup=cid,
                showlegend=(j == 1), line=st.line(), marker=st.marker(), customdata=g["concurrency"],
                hovertemplate=f"<b>{st.name}</b><br>C = %{{customdata}}<br>{x_title}: %{{x:.3g}}<br>"
                              f"{y_title}: %{{y:.3g}}<extra></extra>"), row=1, col=j)
        if reference_x is not None:
            _reference_line(fig, axis="x", value=reference_x[0], label=reference_x[1] if j == 1 else None,
                            row=1, col=j, log=log_x)
    fig.update_xaxes(title_text=x_title, type="log" if log_x else "linear")
    if log_x:
        ticks = log_ticks(points[x] * x_scale)
        fig.update_xaxes(tickvals=ticks, ticktext=[f"{v:g}" for v in ticks])
    fig.update_yaxes(type="log" if log_y else "linear")
    if log_y:
        _apply_log_ticks(fig, points[y] * y_scale)
    if not log_x:
        fig.update_xaxes(rangemode="tozero")
    if not log_y:
        fig.update_yaxes(rangemode="tozero")
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_layout(title_text=title_text(title, subtitle), height=height)
    return _finish(fig)


def request_distribution(requests: pd.DataFrame, *, metric: str, y_title: str, title: str, subtitle: str | None = None,
                         facet: str = "concurrency", scale: float = 1.0, log_y: bool = True,
                         configurations: Sequence[str] | None = None) -> go.Figure:
    """Box plots of per-request latency (pooled steady requests) per configuration, one panel per facet."""
    facets = _facet_values(requests, facet)
    confs = _configurations(requests, configurations)
    fig = make_subplots(rows=1, cols=len(facets), shared_yaxes=True,
                        subplot_titles=[f"C = {int(f)}" if facet == "concurrency" else _facet_title(facet, f)
                                        for f in facets], horizontal_spacing=0.03)
    for j, fv in enumerate(facets, start=1):
        for cid in confs:
            g = requests[(requests[facet] == fv) & (requests["configuration_id"] == cid)]
            if g.empty:
                continue
            st = theme.configuration_style(cid)
            fig.add_trace(go.Box(y=g[metric] * scale, name=st.name, legendgroup=cid, showlegend=(j == 1),
                                 marker_color=st.color, line=dict(color=st.color, width=1.5), boxpoints=False,
                                 hovertemplate=f"<b>{st.name}</b><br>%{{y:.3g}}<extra></extra>"), row=1, col=j)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(type="log" if log_y else "linear")
    if log_y:
        _apply_log_ticks(fig, requests[metric] * scale)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_layout(title_text=title_text(title, subtitle), height=490, boxgap=0.25)
    return _finish(fig)


def stream_calibration_figure(stream: pd.DataFrame, threads: int, *, title: str, subtitle: str | None = None,
                              hbm_cache_gib: float = 64) -> go.Figure:
    """Left: sustainable bandwidth vs working set (STREAM vs PMU proxy). Right: proxy ÷ STREAM model."""
    s = stream[stream["threads"] == threads].sort_values("working_set_gib")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["Bandwidth vs working set", "PMU counters ÷ STREAM-modelled traffic"])
    blue, green, violet = theme.BACKEND_COLORS["vllm"], theme.EXTRA_COLORS[0], theme.EXTRA_COLORS[1]
    fig.add_trace(go.Scatter(x=s["working_set_gib"], y=s["stream_triad_gbs"], mode="lines+markers",
                             name="STREAM triad (modelled bytes)", line=dict(color=theme.TEXT_PRIMARY, width=2),
                             marker=dict(size=8, color=theme.TEXT_PRIMARY),
                             hovertemplate="%{x} GiB: %{y:.0f} GB/s<extra>STREAM</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s["working_set_gib"], y=s["mem_total_gbs"], mode="lines+markers",
                             name="PMU proxy reads + writes", line=dict(color=blue, width=2, dash="dash"),
                             marker=dict(size=8, color=blue, symbol="circle-open"),
                             hovertemplate="%{x} GiB: %{y:.0f} GB/s<extra>PMU proxy</extra>"), row=1, col=1)
    for col, name, color, dash in (("pmu_total_over_stream", "total", theme.TEXT_PRIMARY, "solid"),
                                   ("pmu_read_over_stream_read", "reads", green, "dash"),
                                   ("pmu_write_over_stream_write", "writes", violet, "dot")):
        fig.add_trace(go.Scatter(x=s["working_set_gib"], y=s[col], mode="lines+markers", name=f"ratio: {name}",
                                 line=dict(color=color, width=2, dash=dash), marker=dict(size=7, color=color),
                                 hovertemplate=f"%{{x}} GiB: %{{y:.3f}}<extra>{name}</extra>"), row=1, col=2)
    _reference_line(fig, axis="x", value=hbm_cache_gib, label=f"HBM cache {hbm_cache_gib:g} GiB", row=1, col=1, log=True)
    fig.add_hline(y=1, row=1, col=2, line=dict(color=theme.REFERENCE, dash="dot", width=1.3))
    ws = s["working_set_gib"].tolist()
    fig.update_xaxes(type="log", tickvals=ws, ticktext=[f"{w:g}" for w in ws], title_text="Working set (GiB)")
    fig.update_yaxes(title_text="GB/s", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="ratio", range=[0.85, 1.1], row=1, col=2)
    fig.update_layout(title_text=title_text(title, subtitle), height=500)
    return _finish(fig)


def traffic_components(points: pd.DataFrame, configuration_id: str, *, title: str, subtitle: str | None = None,
                       facet: str = "workload") -> go.Figure:
    """Measured read traffic of one configuration next to the model's weight and KV terms."""
    d = points[points["configuration_id"] == configuration_id]
    facets = _facet_values(d, facet)
    st = theme.configuration_style(configuration_id)
    series = [("mem_read_gbs_mean", "measured reads (PMU)", st.color, "solid", st.symbol),
              ("model_weight_read_gbs", "model: weight reads", theme.TEXT_SECONDARY, "dash", "diamond-open"),
              ("model_kv_read_gbs", "model: KV reads", theme.EXTRA_COLORS[0], "dot", "triangle-up")]
    fig = make_subplots(rows=1, cols=len(facets), shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=[_facet_title(facet, f) for f in facets])
    for j, fv in enumerate(facets, start=1):
        g = d[d[facet] == fv].sort_values("concurrency")
        for col, name, color, dash, symbol in series:
            fig.add_trace(go.Scatter(x=g["concurrency"], y=g[col], mode="lines+markers", name=name, legendgroup=name,
                                     showlegend=(j == 1), line=dict(color=color, width=2, dash=dash),
                                     marker=dict(color=color, size=7, symbol=symbol),
                                     hovertemplate=f"{name}<br>C = %{{x}}<br>%{{y:.0f}} GB/s<extra></extra>"),
                          row=1, col=j)
    cs = sorted(d["concurrency"].unique())
    fig.update_xaxes(type="log", tickvals=cs, ticktext=[str(int(c)) for c in cs], title_text="Concurrent requests (C)")
    fig.update_yaxes(title_text="Read traffic (GB/s)", row=1, col=1)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(title_text=title_text(title, subtitle), height=500)
    return _finish(fig)


def add_note(fig: go.Figure, text: str, *, x: float, y: float, row: int = 1, col: int = 1, ax: int = 0,
             ay: int = -40, log_x: bool = True, log_y: bool = False) -> None:
    """Arrow annotation at a data point (handles log axes)."""
    fig.add_annotation(x=np.log10(x) if log_x else x, y=np.log10(y) if log_y else y, text=text, showarrow=True,
                       arrowhead=0, arrowwidth=1, arrowcolor=theme.TEXT_MUTED, ax=ax, ay=ay,
                       font=dict(size=11, color=theme.TEXT_PRIMARY), bgcolor="rgba(255,255,255,0.85)",
                       row=row, col=col)
