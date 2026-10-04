"""Derived per-curve metrics: scaling with concurrency, saturation, single-request values.

A *curve* is one configuration on one model and workload across concurrency C
(CURVE_KEYS). Everything is computed from the processed point table; nothing is
read from raw files or typed by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CURVE_KEYS = ["campaign", "model", "configuration_id", "arm_dir", "workload"]
# Sweep rule declared before the campaign (configs/experiment.toml [online]).
PLATEAU_GAIN = 0.05
KNEE_FRACTION = 0.90


def add_scaling_metrics(points: pd.DataFrame, throughput: str = "total_tok_s_mean") -> pd.DataFrame:
    """Add, per curve and point:

    speedup_vs_c1        throughput(C) / throughput(1)
    scaling_efficiency   speedup_vs_c1 / C (1.0 = linear scaling)
    gain_vs_prev         throughput(C) / throughput(C/2) - 1 (only for a doubling)
    tpot_change_vs_prev  median TPOT(C) / median TPOT(previous C) - 1
    mem_change_vs_prev   memory traffic(C) / memory traffic(previous C) - 1
    fraction_of_peak     throughput(C) / max over the curve
    """
    out = []
    for _, g in points.groupby(CURVE_KEYS, sort=False):
        g = g.sort_values("concurrency").copy()
        base = g.loc[g["concurrency"] == 1, throughput]
        c1 = base.iloc[0] if len(base) else np.nan
        g["speedup_vs_c1"] = g[throughput] / c1
        g["scaling_efficiency"] = g["speedup_vs_c1"] / g["concurrency"]
        prev, prev_c = g[throughput].shift(1), g["concurrency"].shift(1)
        g["gain_vs_prev"] = np.where(prev_c * 2 == g["concurrency"], g[throughput] / prev - 1, np.nan)
        g["tpot_change_vs_prev"] = g["tpot_s_p50_mean"] / g["tpot_s_p50_mean"].shift(1) - 1
        g["mem_change_vs_prev"] = g["mem_total_gbs_mean"] / g["mem_total_gbs_mean"].shift(1) - 1
        g["fraction_of_peak"] = g[throughput] / g[throughput].max()
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else points.copy()


def _plateau_concurrency(g: pd.DataFrame) -> float:
    """First C after which two consecutive doublings each add < PLATEAU_GAIN (NaN if not reached)."""
    gains = g.set_index("concurrency")["gain_vs_prev"]
    cs = list(gains.index)
    for i in range(1, len(cs) - 1):
        if gains.iloc[i] < PLATEAU_GAIN and gains.iloc[i + 1] < PLATEAU_GAIN:
            return float(cs[i - 1])
    return np.nan


def curve_summary(points: pd.DataFrame, throughput: str = "total_tok_s_mean") -> pd.DataFrame:
    """One row per curve: C=1 behaviour, peak, scaling factor, knee and plateau."""
    pts = add_scaling_metrics(points, throughput) if "speedup_vs_c1" not in points else points
    label_cols = [c for c in ("backend", "precision", "quantization", "isl", "osl", "configuration_label",
                              "configuration_order", "workload_label", "workload_shape", "workload_order",
                              "model_label") if c in pts]
    rows = []
    for key, g in pts.groupby(CURVE_KEYS, sort=False):
        g = g.sort_values("concurrency")
        c1 = g[g["concurrency"] == 1]
        peak = g.loc[g[throughput].idxmax()]
        knee = g[g[throughput] >= KNEE_FRACTION * peak[throughput]].iloc[0]
        row = dict(zip(CURVE_KEYS, key)) | {c: g[c].iloc[0] for c in label_cols}
        row |= {
            "n_points": len(g), "max_concurrency": int(g["concurrency"].max()),
            "c1_output_tok_s": c1["output_tok_s_mean"].iloc[0] if len(c1) else np.nan,
            "c1_total_tok_s": c1[throughput].iloc[0] if len(c1) else np.nan,
            "c1_tpot_s": c1["tpot_s_p50_mean"].iloc[0] if len(c1) else np.nan,
            "c1_ttft_s": c1["ttft_s_p50_mean"].iloc[0] if len(c1) else np.nan,
            "c1_mem_total_gbs": c1["mem_total_gbs_mean"].iloc[0] if len(c1) else np.nan,
            "peak_total_tok_s": peak[throughput], "peak_concurrency": int(peak["concurrency"]),
            "peak_output_tok_s": peak["output_tok_s_mean"], "peak_over_c1": peak["speedup_vs_c1"],
            "tpot_at_peak_s": peak["tpot_s_p50_mean"], "e2e_p95_at_peak_s": peak.get("e2e_s_pooled_p95", np.nan),
            "mem_total_gbs_at_peak": peak["mem_total_gbs_mean"],
            "knee_concurrency": int(knee["concurrency"]),
            "plateau_concurrency": _plateau_concurrency(g),
            "last_doubling_gain": g["gain_vs_prev"].dropna().iloc[-1] if g["gain_vs_prev"].notna().any() else np.nan,
            "max_mem_total_gbs": g["mem_total_gbs_mean"].max(),
            "concurrency_at_max_mem": int(g.loc[g["mem_total_gbs_mean"].idxmax(), "concurrency"]),
            "max_bw_util_vs_stream": g["bw_util_vs_stream_max_mean"].max(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_offline(samples: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and count of tokens/s per offline measurement cell."""
    keys = [c for c in ("model", "configuration_id", "backend", "precision", "quantization", "configuration_label",
                        "configuration_order", "workload", "workload_label", "workload_shape", "workload_order",
                        "benchmark", "tool", "batch", "test") if c in samples]
    return (samples.groupby(keys, dropna=False)["tok_s"].agg(tok_s_mean="mean", tok_s_std="std", n="count")
            .reset_index())


def relative_difference(a: float | pd.Series, b: float | pd.Series) -> float | pd.Series:
    """(a - b) / b: positive when a is larger."""
    return a / b - 1
