#!/usr/bin/env python3
"""Plotly figures for the backend-selection report (run after process.py).

Every figure is written as interactive HTML and static PNG into plots/, and the
exact data behind it as CSV into plots/data/. Conventions:
- colour = backend (blue vLLM, orange llama.cpp), style = precision pair (BF16
  solid with filled markers, INT8 dashed with open markers), fixed across figures;
- concurrency on a log2 axis; panels that are compared share their y axis;
- error bars are +-1 standard deviation over repetitions (3 per point);
- no dual y axes: throughput and bandwidth are stacked panels sharing x.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "main"
ARMS = {
    # colour = backend, line/marker style = precision pair (BF16 solid/filled, INT8 dashed/open)
    "vllm-bf16": {"name": "vLLM · BF16", "color": "#2a78d6", "dash": "solid", "symbol": "circle"},
    "llamacpp-bf16": {"name": "llama.cpp · BF16", "color": "#eb6834", "dash": "solid", "symbol": "square"},
    "vllm-w8a8": {"name": "vLLM · INT8 (W8A8)", "color": "#2a78d6", "dash": "dash", "symbol": "circle-open"},
    "llamacpp-q8_0": {"name": "llama.cpp · INT8 (Q8_0)", "color": "#eb6834", "dash": "dash", "symbol": "square-open"},
}
WORKLOADS = {"decode": "Decode-heavy 128→512", "balanced": "Balanced 512→512", "prefill": "Prefill-heavy 4096→128",
             "longctx": "Long context 8192→256", "kvdecode": "Long-KV decode 4096→1024"}
MODELS = {"llama31_8b": "Llama-3.1-8B", "llama32_1b": "Llama-3.2-1B"}
FONT = dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color="#0b0b0b")
GRID = "#e6e5e0"


def style(fig: go.Figure, title: str, height: int = 460, width: int = 1200) -> go.Figure:
    fig.update_layout(template="plotly_white", font=FONT, title=dict(text=title, x=0.01, font=dict(size=16)),
                      height=height, width=width, legend=dict(orientation="h", y=-0.18, x=0),
                      margin=dict(l=70, r=20, t=80, b=90), hovermode="closest", plot_bgcolor="#fcfcfb",
                      paper_bgcolor="#ffffff")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#8a8984")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#8a8984")
    return fig


def save(fig: go.Figure, name: str, data: pd.DataFrame, out: Path) -> None:
    (out / "data").mkdir(parents=True, exist_ok=True)
    fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / f"{name}.png", scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"PNG export failed for {name}: {exc}")
    data.to_csv(out / "data" / f"{name}.csv", index=False)


def conc_axis(fig: go.Figure, cs: list[int], **kw) -> None:
    fig.update_xaxes(type="log", tickvals=cs, ticktext=[str(c) for c in cs], title_text="Concurrent requests (C)", **kw)


def line_panels(points: pd.DataFrame, model: str, metric: str, ylabel: str, title: str, name: str, out: Path,
                log_y: bool = False, err: str | None = "std", workloads: list[str] | None = None,
                yfmt: str | None = None, scale: float = 1.0, ref_line: float | None = None,
                share_y: bool = True) -> None:
    df = points[(points["model"] == model) & (points["campaign"] == CAMPAIGN)].copy()
    if df.empty or f"{metric}_mean" not in df and metric not in df:
        return
    wls = [w for w in (workloads or WORKLOADS) if w in set(df["workload"])]
    fig = make_subplots(rows=1, cols=len(wls), shared_yaxes=share_y, subplot_titles=[WORKLOADS[w] for w in wls],
                        horizontal_spacing=0.03 if share_y else 0.05)
    col_mean = f"{metric}_mean" if f"{metric}_mean" in df else metric
    for j, wl in enumerate(wls, start=1):
        for arm, a in ARMS.items():
            g = df[(df["workload"] == wl) & (df["arm"] == arm) & (df["arm_dir"] == arm)].sort_values("concurrency")
            if g.empty or g[col_mean].isna().all():
                continue
            y = g[col_mean] * scale
            e = g[f"{metric}_{err}"] * scale if (err and f"{metric}_{err}" in g) else None
            fig.add_trace(go.Scatter(
                x=g["concurrency"], y=y, mode="lines+markers", name=a["name"], legendgroup=arm, showlegend=(j == 1),
                line=dict(color=a["color"], width=2, dash=a["dash"]), marker=dict(size=8, symbol=a["symbol"]),
                error_y=dict(type="data", array=e, thickness=1.2, width=4) if e is not None else None,
                hovertemplate=f"{a['name']}<br>C=%{{x}}<br>{ylabel}=%{{y:.4g}}<extra></extra>"), row=1, col=j)
        if ref_line is not None:
            fig.add_hline(y=ref_line, line=dict(color="#8a8984", dash="dot", width=1), row=1, col=j)
    cs = sorted(df["concurrency"].unique())
    conc_axis(fig, cs)
    fig.update_yaxes(title_text=ylabel, row=1, col=1, type="log" if log_y else "linear", tickformat=yfmt)
    if log_y:
        fig.update_yaxes(type="log")
    elif not share_y:
        fig.update_yaxes(rangemode="tozero")
    style(fig, f"{title} — {MODELS[model]}")
    keep = ["model", "arm", "precision", "workload", "concurrency", "n_reps", col_mean] + \
           [c for c in (f"{metric}_std", f"{metric}_ci95", f"{metric}_min", f"{metric}_max") if c in df]
    save(fig, f"{name}_{model}", df[keep], out)


def offline_bars(off: pd.DataFrame, test: str, name: str, title: str, out: Path) -> None:
    if off.empty:
        return
    d = off[(off["campaign"] == CAMPAIGN) & (off["test"] == test) & (off["experiment"].isin(["A1", "A2"]))]
    if d.empty:
        return
    agg = d.groupby(["model", "arm", "workload", "batch"])["tok_s"].agg(["mean", "std", "count"]).reset_index()
    for model in agg["model"].unique():
        m = agg[agg["model"] == model]
        batches = sorted(m["batch"].unique())
        fig = make_subplots(rows=1, cols=len(batches), subplot_titles=[f"Batch {b}" for b in batches],
                            horizontal_spacing=0.06)
        for j, b in enumerate(batches, start=1):
            for arm, a in ARMS.items():
                g = m[(m["batch"] == b) & (m["arm"] == arm)]
                g = g.set_index("workload").reindex([w for w in WORKLOADS if w in set(m["workload"])]).reset_index()
                fig.add_trace(go.Bar(x=[WORKLOADS[w] for w in g["workload"]], y=g["mean"], name=a["name"],
                                     legendgroup=arm, showlegend=(j == 1), marker_color=a["color"],
                                     marker_pattern_shape="/" if a["dash"] == "dash" else "",
                                     error_y=dict(type="data", array=g["std"], thickness=1.2, width=4),
                                     hovertemplate=f"{a['name']}<br>%{{x}}<br>%{{y:.4g}} tok/s<extra></extra>"),
                              row=1, col=j)
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
        fig.update_yaxes(title_text="tokens/s", row=1, col=1)
        style(fig, f"{title} — {MODELS[model]} (engine microbenchmark; tools differ in scope)")
        save(fig, f"{name}_{model}", m, out)


def key_figure(points: pd.DataFrame, model: str, out: Path) -> None:
    """Throughput (top) and memory bandwidth (bottom) vs concurrency, per workload."""
    df = points[(points["model"] == model) & (points["campaign"] == CAMPAIGN)]
    if df.empty:
        return
    wls = [w for w in WORKLOADS if w in set(df["workload"])]
    # Throughput: own y scale per workload (backends share it inside a panel). Bandwidth:
    # one common scale for all panels, since every panel is read against the same STREAM line.
    fig = make_subplots(rows=2, cols=len(wls), shared_xaxes=True, vertical_spacing=0.08,
                        horizontal_spacing=0.05, subplot_titles=[WORKLOADS[w] for w in wls])
    for j, wl in enumerate(wls, start=1):
        for arm, a in ARMS.items():
            g = df[(df["workload"] == wl) & (df["arm"] == arm) & (df["arm_dir"] == arm)].sort_values("concurrency")
            if g.empty:
                continue
            for row, metric, lab in ((1, "out_tok_s", "output tok/s"), (2, "mem_total_gbs", "GB/s")):
                if f"{metric}_mean" not in g:
                    continue
                fig.add_trace(go.Scatter(
                    x=g["concurrency"], y=g[f"{metric}_mean"], mode="lines+markers", name=a["name"], legendgroup=arm,
                    showlegend=(j == 1 and row == 1), line=dict(color=a["color"], width=2, dash=a["dash"]),
                    marker=dict(size=8, symbol=a["symbol"]),
                    error_y=dict(type="data", array=g[f"{metric}_std"], thickness=1.2, width=4),
                    hovertemplate=f"{a['name']}<br>C=%{{x}}<br>%{{y:.4g}} {lab}<extra></extra>"), row=row, col=j)
        if "stream_pmu_total_gbs_max" in df and df["stream_pmu_total_gbs_max"].notna().any():
            kw = dict(annotation_text="STREAM sustainable (same counters)", annotation_position="top left",
                      annotation_font_size=11) if j == 1 else {}
            fig.add_hline(y=float(df["stream_pmu_total_gbs_max"].max()), row=2, col=j,
                          line=dict(color="#52514e", dash="dot", width=1.5), **kw)
    cs = sorted(df["concurrency"].unique())
    fig.update_xaxes(type="log", tickvals=cs, ticktext=[str(c) for c in cs])
    bw_top = 1.1 * max(float(df["stream_pmu_total_gbs_max"].max()) if "stream_pmu_total_gbs_max" in df else 0,
                       float((df["mem_total_gbs_mean"] + df["mem_total_gbs_std"].fillna(0)).max()))
    for j in range(1, len(wls) + 1):
        fig.update_yaxes(range=[0, bw_top], row=2, col=j)
        fig.update_yaxes(rangemode="tozero", row=1, col=j)
    for j in range(1, len(wls) + 1):
        fig.update_xaxes(title_text="Concurrent requests (C)", row=2, col=j)
    fig.update_yaxes(title_text="Aggregate output tokens/s", row=1, col=1)
    fig.update_yaxes(title_text="Memory traffic, GB/s<br>(reads+writes beyond L3)", row=2, col=1)
    style(fig, f"Turning concurrency into throughput and memory traffic — {MODELS[model]}", height=760)
    keep = [c for c in ["model", "arm", "workload", "concurrency", "out_tok_s_mean", "out_tok_s_std",
                        "mem_total_gbs_mean", "mem_total_gbs_std", "stream_pmu_total_gbs_max"] if c in df]
    save(fig, f"key_throughput_and_bandwidth_{model}", df[keep], out)


def tput_vs_bw(points: pd.DataFrame, model: str, out: Path) -> None:
    df = points[(points["model"] == model) & (points["campaign"] == CAMPAIGN)]
    if df.empty or "mem_total_gbs_mean" not in df:
        return
    wls = [w for w in WORKLOADS if w in set(df["workload"])]
    fig = make_subplots(rows=1, cols=len(wls), subplot_titles=[WORKLOADS[w] for w in wls], horizontal_spacing=0.05)
    for j, wl in enumerate(wls, start=1):
        for arm, a in ARMS.items():
            g = df[(df["workload"] == wl) & (df["arm"] == arm) & (df["arm_dir"] == arm)].sort_values("concurrency")
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g["mem_total_gbs_mean"], y=g["total_tok_s_mean"], mode="lines+markers+text", name=a["name"],
                legendgroup=arm, showlegend=(j == 1),
                text=[f"C{c}" if (c in (1, 4, 16, 64) or c == g["concurrency"].max()) else "" for c in g["concurrency"]],
                textposition="top center", textfont=dict(size=10, color="#52514e"),
                line=dict(color=a["color"], width=2, dash=a["dash"]), marker=dict(size=8, symbol=a["symbol"]),
                hovertemplate=f"{a['name']}<br>%{{text}}<br>%{{x:.1f}} GB/s<br>%{{y:.4g}} tok/s<extra></extra>"),
                row=1, col=j)
        if "stream_pmu_total_gbs_max" in df and df["stream_pmu_total_gbs_max"].notna().any():
            fig.add_vline(x=float(df["stream_pmu_total_gbs_max"].max()), row=1, col=j,
                          line=dict(color="#52514e", dash="dot", width=1.5))
        fig.update_xaxes(rangemode="tozero", row=1, col=j)
    fig.update_xaxes(title_text="Memory traffic, GB/s (PMU)")
    fig.update_yaxes(title_text="Total tokens/s", row=1, col=1)
    style(fig, f"Throughput vs measured memory traffic — {MODELS[model]} (dotted: STREAM sustainable)")
    save(fig, f"throughput_vs_bandwidth_{model}", df[["model", "arm", "workload", "concurrency",
                                                     "mem_total_gbs_mean", "total_tok_s_mean"]], out)


def stream_figure(stream: pd.DataFrame, out: Path) -> None:
    if stream.empty:
        return
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Sustainable bandwidth vs working set (cache mode, socket 0)",
                                                        "PMU proxy ÷ STREAM-modeled traffic"], horizontal_spacing=0.1)
    colors = {55: "#2a78d6", 56: "#eb6834"}
    for t, g in stream.groupby("threads"):
        g = g.sort_values("working_set_gib")
        fig.add_trace(go.Scatter(x=g["working_set_gib"], y=g["stream_triad_gbs"], mode="lines+markers",
                                 name=f"STREAM triad, {t} threads", line=dict(color=colors.get(t, "#4a3aa7"), width=2)),
                      row=1, col=1)
        if "mem_total_gbs" in g:
            fig.add_trace(go.Scatter(x=g["working_set_gib"], y=g["mem_total_gbs"], mode="lines+markers",
                                     name=f"PMU reads+writes, {t} threads",
                                     line=dict(color=colors.get(t, "#4a3aa7"), width=2, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=g["working_set_gib"], y=g["pmu_read_over_stream_read"], mode="lines+markers",
                                     name=f"read ratio, {t} t", line=dict(color=colors.get(t), width=2)), row=1, col=2)
            fig.add_trace(go.Scatter(x=g["working_set_gib"], y=g["pmu_write_over_stream_write"], mode="lines+markers",
                                     name=f"write ratio, {t} t", line=dict(color=colors.get(t), width=2, dash="dash")),
                          row=1, col=2)
    fig.add_vline(x=64, line=dict(color="#8a8984", dash="dot"), annotation_text="HBM cache 64 GiB", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Working set (GiB)", tickvals=[4, 8, 16, 32, 48, 64, 96],
                     ticktext=["4", "8", "16", "32", "48", "64", "96"])
    fig.update_yaxes(title_text="GB/s", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=1, col=2)
    style(fig, "Experiment 0: STREAM triad baseline and PMU-proxy validation")
    save(fig, "stream_calibration", stream, out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--campaign", default="main")
    p.add_argument("--out", help="output directory (default plots/)")
    args = p.parse_args()
    global CAMPAIGN
    CAMPAIGN = args.campaign
    root = Path(args.root)
    proc, out = root / "data" / "processed", Path(args.out) if args.out else root / "plots"
    out.mkdir(exist_ok=True)
    stream = pd.read_csv(proc / "stream_calibration.csv") if (proc / "stream_calibration.csv").stat().st_size > 1 else pd.DataFrame()
    stream_figure(stream, out)
    off = pd.read_csv(proc / "offline.csv") if (proc / "offline.csv").stat().st_size > 1 else pd.DataFrame()
    offline_bars(off, "prefill", "f01_offline_prefill", "Offline prompt-processing throughput", out)
    offline_bars(off, "decode", "f02_offline_decode", "Offline generation throughput", out)
    offline_bars(off, "total", "f02b_offline_combined", "Offline prompt + generation throughput", out)
    pts = pd.read_csv(proc / "online_points.csv") if (proc / "online_points.csv").stat().st_size > 1 else pd.DataFrame()
    if pts.empty:
        print("no online points yet")
        return 0
    for model in pts["model"].unique():
        line_panels(pts, model, "out_tok_s", "Output tokens/s", "Online output throughput", "f03_output_throughput", out,
                    share_y=False)
        line_panels(pts, model, "total_tok_s", "Total tokens/s", "Online total throughput", "f04_total_throughput", out,
                    share_y=False)
        line_panels(pts, model, "ttft_s_p50", "Median TTFT (s)", "Time to first token", "f05_ttft", out, log_y=True)
        line_panels(pts, model, "tpot_s_p50", "Median TPOT (s)", "Time per output token", "f06_tpot", out, log_y=True)
        line_panels(pts, model, "e2e_s_p50", "Median end-to-end latency (s)", "End-to-end latency", "f07_e2e", out,
                    log_y=True)
        line_panels(pts, model, "mem_read_gbs", "Memory read GB/s (beyond L3)",
                    "Memory read bandwidth (HBM cache + DDR, core PMU)", "f08_mem_read_bw", out)
        if "uncore_ddr_cas_rd_gbs_mean" in pts:
            line_panels(pts, model, "uncore_ddr_cas_rd_gbs", "DDR read GB/s", "DDR read bandwidth (IMC)",
                        "f09_ddr_read_bw", out)
        line_panels(pts, model, "mem_write_gbs", "Memory write GB/s (estimate)", "Memory write bandwidth (core PMU)",
                    "f09b_mem_write_bw", out)
        line_panels(pts, model, "bw_util_vs_stream_max", "Fraction of sustainable", "Memory-bandwidth utilization "
                    "(application ÷ STREAM, same counters)", "f10_bw_utilization", out, ref_line=1.0)
        key_figure(pts, model, out)
        line_panels(pts, model, "scaling_efficiency", "Speedup ÷ C", "Scaling efficiency (total tok/s)",
                    "f12_scaling_efficiency", out, err=None, log_y=True)
        tput_vs_bw(pts, model, out)
        line_panels(pts, model, "tokens_per_step", "Tokens per forward step", "Effective batch (server counters)",
                    "f14_tokens_per_step", out, log_y=True)
        line_panels(pts, model, "amx_busy_frac", "AMX busy / cycles", "AMX utilization (EXE.AMX_BUSY)",
                    "f15_amx_busy", out)
        line_panels(pts, model, "cpu_busy_compute_mean", "Busy fraction", "CPU utilization of compute cores 0–54",
                    "f16_cpu_util", out)
    print(f"figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
