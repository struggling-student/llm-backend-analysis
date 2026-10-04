"""Experiment 0: STREAM triad under the same telemetry -> sustainable-bandwidth baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from ... import paths
from .. import registry
from ..telemetry import read_telemetry, window_aggregate

# Triad a[i] = b[i] + s*c[i] with non-temporal stores: per iteration it reads two arrays
# and writes one, so 2/3 of STREAM's modeled bytes are reads and 1/3 are writes.
# (Until 2026-10-04 the processing compared PMU reads with the full modeled traffic,
# which made the read capture ratio look like ~0.65 instead of ~0.98.)
TRIAD_READ_SHARE, TRIAD_WRITE_SHARE = 2 / 3, 1 / 3


def process_stream() -> pd.DataFrame:
    rows = []
    for f in sorted((paths.RAW / "stream").glob("*/stream_calibration.json")):
        campaign = f.parent.name
        d = json.loads(f.read_text())
        for r in d["runs"]:
            if not r.get("stream"):
                continue
            _, tel = read_telemetry(paths.local_telemetry_path(r["telemetry_file"]))
            # Steady part: the last `steady_seconds_total` seconds before the process ended.
            end = r["t_end_mono"]
            agg = window_aggregate(tel, end - r["steady_seconds_total"] + 0.5, end - 0.25)
            bw = r["bandwidth_gbs_median"]
            read_model, write_model = bw * TRIAD_READ_SHARE, bw * TRIAD_WRITE_SHARE
            rows.append({
                "campaign": campaign, "campaign_role": registry.campaign_role("stream", campaign),
                "host": d["hostname"].split(".")[0], "slurm_job_id": str(d["slurm_job_id"]),
                "run": r["name"], "threads": r["threads"], "cpus": r["cpus"], "working_set_gib": r["working_set_gib"],
                "stream_triad_gbs": bw, "stream_read_model_gbs": read_model, "stream_write_model_gbs": write_model,
                **{k: v for k, v in agg.items() if k.startswith(("mem_", "tel_", "runfrac", "cpu_busy"))},
                "pmu_total_over_stream": (agg.get("mem_total_gbs") or np.nan) / bw,
                "pmu_read_over_stream_read": (agg.get("mem_read_gbs") or np.nan) / read_model,
                "pmu_write_over_stream_write": (agg.get("mem_write_gbs") or np.nan) / write_model,
                "source": paths.relative(f),
            })
    return pd.DataFrame(rows).sort_values(["campaign", "threads", "working_set_gib"], ignore_index=True)


def sustainable_lookup(stream: pd.DataFrame, threads: int | None = None,
                       campaign: str = registry.ANALYSIS_CAMPAIGN["stream"]) -> Callable[[float], dict[str, float]]:
    """Baselines measured with the server's thread count, interpolated in log(working set).

    Returns f(working_set_gib) -> {stream_triad_gbs_at_ws, stream_pmu_total_gbs_at_ws, ..._max}."""
    threads = threads or registry.topology()["compute_threads"]
    s = (stream[(stream["threads"] == threads) & (stream["campaign"] == campaign)]
         .groupby("working_set_gib").median(numeric_only=True).sort_index())
    if s.empty:
        return lambda ws: {}
    x = np.log(s.index.values.astype(float))

    def f(ws_gib: float) -> dict[str, float]:
        xi = math.log(max(ws_gib, 1e-3))
        out = {name: float(np.interp(xi, x, s[col].values))
               for col, name in (("stream_triad_gbs", "stream_triad_gbs_at_ws"),
                                 ("mem_total_gbs", "stream_pmu_total_gbs_at_ws")) if col in s}
        out["stream_triad_gbs_max"] = float(s["stream_triad_gbs"].max())
        if "mem_total_gbs" in s:
            out["stream_pmu_total_gbs_max"] = float(s["mem_total_gbs"].max())
            out["stream_pmu_read_gbs_max"] = float(s["mem_read_gbs"].max())
        return out
    return f


def read_node_checks() -> pd.DataFrame:
    """Per-job 10 s STREAM health check (8 GiB, 55 threads, socket 0) written by job.sbatch."""
    rows = []
    for f in sorted((paths.RAW_ENVIRONMENT / "node_checks").glob("*.json")):
        host, _, job = f.stem.rpartition("-")
        lines = [ln for ln in f.read_text().splitlines() if ln.startswith("{")]
        if not lines:
            continue
        o = json.loads(lines[-1])
        steady = sorted(o["steady_elapsed_seconds"])
        rows.append({"host": host, "slurm_job_id": job, "threads": o["threads"],
                     "working_set_gib": o["working_set_bytes"] / 2**30,
                     "triad_gbs": o["modeled_bytes_per_iteration"] / steady[len(steady) // 2] / 1e9,
                     "valid": o.get("valid"), "source": paths.relative(f)})
    return pd.DataFrame(rows).sort_values(["host", "slurm_job_id"], ignore_index=True)

