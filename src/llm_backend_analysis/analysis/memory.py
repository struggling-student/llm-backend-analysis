"""Memory-system quantities derived from the point table.

The traffic model is a first-principles lower bound on the read traffic of serving:
every forward step streams all weights once (except the gathered input-embedding table),
and every decoded token reads the KV cache of its sequence at the mean context
ISL + OSL/2. Activations, attention intermediates and prefill attention are ignored,
so measured/model > 1 is expected and grows where prefill dominates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import registry


def add_traffic_model(points: pd.DataFrame) -> pd.DataFrame:
    """Add model_weight_read_gbs, model_kv_read_gbs, model_read_gbs and measured_over_model_read."""
    out = points.copy()
    confs = registry.configurations()
    models = registry.experiment_config()["models"]
    weight = out.apply(lambda r: registry.step_weight_bytes(r["model"], confs[r["configuration_id"]].harness_precision),
                       axis=1)
    kv = out["model"].map(lambda m: models[m]["kv_bytes_per_token"])
    mean_ctx = out["isl"] + out["osl"] / 2
    out["step_weight_gb"] = weight / 1e9
    out["model_weight_read_gbs"] = out["engine_steps_per_s_mean"] * weight / 1e9
    out["model_kv_read_gbs"] = out["output_tok_s_mean"] * mean_ctx * kv / 1e9
    out["model_read_gbs"] = out["model_weight_read_gbs"] + out["model_kv_read_gbs"]
    out["model_kv_share"] = out["model_kv_read_gbs"] / out["model_read_gbs"]
    out["measured_over_model_read"] = out["mem_read_gbs_mean"] / out["model_read_gbs"]
    out["decode_dominated"] = out["osl"] >= out["isl"]
    return out


def traffic_trend(points: pd.DataFrame) -> pd.DataFrame:
    """Per curve: memory traffic at C=1 and at the highest C, and whether it falls with C."""
    rows = []
    for key, g in points.groupby(["model", "configuration_id", "workload"], sort=False):
        g = g.sort_values("concurrency")
        first, last = g.iloc[0], g.iloc[-1]
        rows.append({"model": key[0], "configuration_id": key[1], "workload": key[2],
                     "c_first": int(first["concurrency"]), "c_last": int(last["concurrency"]),
                     "mem_total_gbs_first": first["mem_total_gbs_mean"], "mem_total_gbs_last": last["mem_total_gbs_mean"],
                     "bw_util_first": first["bw_util_vs_stream_max_mean"], "bw_util_last": last["bw_util_vs_stream_max_mean"],
                     "max_bw_util": g["bw_util_vs_stream_max_mean"].max(),
                     "c_at_max_traffic": int(g.loc[g["mem_total_gbs_mean"].idxmax(), "concurrency"]),
                     "traffic_falls_with_c": bool(last["mem_total_gbs_mean"] < first["mem_total_gbs_mean"]),
                     "traffic_ratio_last_over_first": last["mem_total_gbs_mean"] / first["mem_total_gbs_mean"]})
    return pd.DataFrame(rows)


def stream_ceiling(points: pd.DataFrame) -> float:
    """Sustainable traffic of the server layout measured with the same counters (GB/s)."""
    v = points["stream_pmu_total_gbs_max"].dropna()
    return float(v.max()) if len(v) else np.nan
