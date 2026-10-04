"""Comparisons between configurations, defined by metadata rather than by name.

Backend comparisons are only formed between configurations of the *same precision*
(BF16 vs BF16, INT8 vs INT8). Cross-precision comparisons are formed explicitly with
`precision_ratio` and are always labelled as such. When vLLM INT8 data appears, the INT8
backend pair becomes available automatically through `backend_pairs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MATCH_KEYS = ["campaign", "model", "workload", "concurrency"]


def backend_pairs(points: pd.DataFrame, numerator: str = "vllm", denominator: str = "llama_cpp") -> pd.DataFrame:
    """(model, precision) combinations for which both backends have measurements."""
    have = points.groupby(["model", "precision"])["backend"].agg(set).reset_index()
    have = have[have["backend"].map(lambda s: {numerator, denominator} <= s)]
    return have[["model", "precision"]].reset_index(drop=True)


def _ratio(points: pd.DataFrame, metric: str, num_mask: pd.Series, den_mask: pd.Series,
           keys: list[str], std: str | None) -> pd.DataFrame:
    cols = keys + [metric] + ([std] if std and std in points else [])
    num = points.loc[num_mask, cols + ["configuration_id"]]
    den = points.loc[den_mask, cols + ["configuration_id"]]
    m = num.merge(den, on=keys, suffixes=("_num", "_den"))
    m = m.rename(columns={f"{metric}_num": "value_num", f"{metric}_den": "value_den"})
    m["ratio"] = m["value_num"] / m["value_den"]
    if std and f"{std}_num" in m:
        # First-order propagation of the repetition spread of both means.
        m["ratio_rel_err"] = np.sqrt((m[f"{std}_num"] / m["value_num"]) ** 2 + (m[f"{std}_den"] / m["value_den"]) ** 2)
        m = m.drop(columns=[f"{std}_num", f"{std}_den"])
    return m


def backend_ratio(points: pd.DataFrame, metric: str = "total_tok_s_mean", precision: str = "bf16",
                  numerator: str = "vllm", denominator: str = "llama_cpp",
                  keys: list[str] | None = None) -> pd.DataFrame:
    """numerator backend ÷ denominator backend at equal precision, matched on campaign/model/workload/C."""
    keys = keys or MATCH_KEYS
    std = metric.replace("_mean", "_std") if metric.endswith("_mean") else None
    same = points["precision"] == precision
    out = _ratio(points, metric, same & (points["backend"] == numerator), same & (points["backend"] == denominator),
                 keys, std)
    return out.assign(precision=precision, numerator=numerator, denominator=denominator, metric=metric)


def precision_ratio(points: pd.DataFrame, backend: str, metric: str = "total_tok_s_mean",
                    numerator: str = "int8", denominator: str = "bf16", keys: list[str] | None = None) -> pd.DataFrame:
    """Same backend, numerator precision ÷ denominator precision (a quantization effect, not an engine effect)."""
    keys = keys or MATCH_KEYS
    std = metric.replace("_mean", "_std") if metric.endswith("_mean") else None
    same = points["backend"] == backend
    out = _ratio(points, metric, same & (points["precision"] == numerator), same & (points["precision"] == denominator),
                 keys, std)
    return out.assign(backend=backend, numerator=numerator, denominator=denominator, metric=metric)


def crossover_summary(ratio: pd.DataFrame) -> pd.DataFrame:
    """Per (model, workload): ratio at C=1, at the highest common C, and the first C where ratio >= 1."""
    rows = []
    for (model, workload), g in ratio.groupby(["model", "workload"], sort=False):
        g = g.sort_values("concurrency")
        above = g[g["ratio"] >= 1]
        rows.append({"model": model, "workload": workload,
                     "ratio_c1": g.loc[g["concurrency"] == 1, "ratio"].iloc[0] if (g["concurrency"] == 1).any() else np.nan,
                     "highest_common_c": int(g["concurrency"].max()),
                     "ratio_at_highest_common_c": g["ratio"].iloc[-1],
                     "max_ratio": g["ratio"].max(),
                     "first_c_ratio_ge_1": int(above["concurrency"].iloc[0]) if len(above) else np.nan,
                     "ratio_monotone_from_c2": bool(g[g["concurrency"] >= 2]["ratio"].is_monotonic_increasing)})
    return pd.DataFrame(rows)


def peak_ratio(summary: pd.DataFrame, precision: str = "bf16", numerator: str = "vllm",
               denominator: str = "llama_cpp", metric: str = "peak_total_tok_s") -> pd.DataFrame:
    """Ratio of each backend's best throughput (each at its own best C) per model and workload."""
    s = summary[summary["precision"] == precision]
    num = s[s["backend"] == numerator][["model", "workload", metric, "peak_concurrency"]]
    den = s[s["backend"] == denominator][["model", "workload", metric, "peak_concurrency"]]
    m = num.merge(den, on=["model", "workload"], suffixes=("_num", "_den"))
    m["ratio"] = m[f"{metric}_num"] / m[f"{metric}_den"]
    return m.assign(precision=precision)
