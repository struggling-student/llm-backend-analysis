"""Read the 1 s telemetry files written by scripts/benchmark/telemetry.py and aggregate them
over a time window (idle baseline, steady-state window of a repetition, STREAM run).

Memory traffic comes from per-thread core offcore-response counters (64 B per count):
reads = OCR.READS_TO_CORE.L3_MISS_LOCAL (+ OCR.HWPF_L3.L3_MISS_LOCAL where recorded),
writes = OCR.WRITE_ESTIMATE.MEMORY. In HBM cache mode this is HBM-cache + DDR traffic
as a whole; it cannot be split (uncore PMUs are not readable at perf_event_paranoid=2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import registry

LINE_BYTES = 64


def read_telemetry(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (header, samples). Missing files give an empty frame."""
    header: dict[str, Any] = {}
    rows = []
    if not path.exists():
        return header, pd.DataFrame()
    topo = registry.topology()
    compute = set(topo["compute_cpu_list"])
    frontend = topo["frontend_cpu"]
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec["type"] == "header":
            header = rec
        elif rec["type"] == "sample":
            row = {"t": rec["t_mono"], "dt": rec["dt"], "threads": rec.get("threads"), "rss_bytes": rec.get("rss_bytes")}
            for k, v in (rec.get("pmu") or {}).items():
                row[f"pmu_{k}"] = v
            for k, v in (rec.get("pmu_running_fraction") or {}).items():
                row[f"runfrac_{k}"] = v
            for k, v in (rec.get("uncore") or {}).items():
                row[f"unc_{k}"] = v
            for k, v in (rec.get("node_used_bytes") or {}).items():
                row[f"used_{k}"] = v
            busy = rec.get("cpu_busy") or {}
            if busy:
                vals = {int(c): b for c, b in busy.items()}
                comp = [b for c, b in vals.items() if c in compute]
                row["cpu_busy_compute_mean"] = float(np.mean(comp)) if comp else None
                row["cpu_busy_frontend"] = vals.get(frontend)
            for k, v in (rec.get("metrics") or {}).items():
                row[f"m_{k}"] = v
            rows.append(row)
    return header, pd.DataFrame(rows)


def window_aggregate(tel: pd.DataFrame, start: float, end: float) -> dict[str, Any]:
    """Aggregate samples over [start, end], weighting each sample by its overlap with the window.

    Rates are bytes/s converted to GB/s; fractions are cycle-weighted; server counters
    (forward steps, tokens per step) are differenced across the window."""
    if tel.empty or end <= start:
        return {}
    t0 = tel["t"] - tel["dt"]
    ov = (np.minimum(tel["t"], end) - np.maximum(t0, start)).clip(lower=0)
    w = ov / tel["dt"]
    sel = w > 0
    if not sel.any():
        return {}
    tw, ww = tel[sel], w[sel]
    span = float(ov[sel].sum())
    out: dict[str, Any] = {"tel_seconds": span, "tel_samples": int(sel.sum())}

    def rate(col: str) -> float | None:
        return float((tw[col] * ww).sum() / span) if col in tw else None

    rd = (rate("pmu_ocr_rd_l3miss_local") or 0) + (rate("pmu_ocr_hwpf_l3_l3miss_local") or 0)
    wr = rate("pmu_ocr_wr_est_memory") or 0
    out["mem_read_gbs"] = rd * LINE_BYTES / 1e9
    out["mem_write_gbs"] = wr * LINE_BYTES / 1e9
    out["mem_total_gbs"] = (rd + wr) * LINE_BYTES / 1e9
    out["llc_miss_gbs"] = (rate("pmu_llc_miss") or 0) * LINE_BYTES / 1e9
    cyc = rate("pmu_cycles")
    out["amx_busy_frac"] = (rate("pmu_exe_amx_busy") or 0) / cyc if cyc else None
    out["ipc"] = (rate("pmu_instructions") or 0) / cyc if cyc else None
    out["ghz_effective_per_core"] = cyc / registry.topology()["compute_threads"] / 1e9 if cyc else None
    for c in [c for c in tw.columns if c.startswith("unc_")]:
        out[c.replace("unc_", "uncore_") + "_gbs"] = rate(c) * LINE_BYTES / 1e9
    for c in ("cpu_busy_compute_mean", "cpu_busy_frontend", "threads"):
        if c in tw:
            out[c] = float(np.average(tw[c].astype(float), weights=ww))
    for c in [c for c in tw.columns if c.startswith("runfrac_")]:
        out[c] = float(tw[c].min())
    out["rss_gib_max"] = float(tw["rss_bytes"].max()) / 2**30 if "rss_bytes" in tw else None
    if "used_node0" in tw:
        out["node0_used_gib_max"] = float(tw["used_node0"].max()) / 2**30

    inside = tel[(tel["t"] >= start) & (tel["t"] <= end)]

    def delta(col: str) -> float | None:
        if col not in tel:
            return None
        v = inside[col].dropna()
        return float(v.iloc[-1] - v.iloc[0]) if len(v) >= 2 else None

    sp = float(inside["t"].iloc[-1] - inside["t"].iloc[0]) if len(inside) >= 2 else None
    if "m_vllm:iteration_tokens_total_count" in tel:
        steps, toks = delta("m_vllm:iteration_tokens_total_count"), delta("m_vllm:iteration_tokens_total_sum")
        out["engine_steps_per_s"] = steps / sp if (steps and sp) else None
        out["tokens_per_step"] = toks / steps if (steps and toks) else None
        out["running_requests_mean"] = rate("m_vllm:num_requests_running")
        out["waiting_requests_mean"] = rate("m_vllm:num_requests_waiting")
        out["kv_usage_mean"] = rate("m_vllm:kv_cache_usage_perc")
        out["preemptions"] = delta("m_vllm:num_preemptions_total")
    if "m_llamacpp:n_decode_total" in tel:
        steps = delta("m_llamacpp:n_decode_total")
        toks = (delta("m_llamacpp:prompt_tokens_total") or 0) + (delta("m_llamacpp:tokens_predicted_total") or 0)
        out["engine_steps_per_s"] = steps / sp if (steps and sp) else None
        out["tokens_per_step"] = toks / steps if (steps and toks) else None
        out["running_requests_mean"] = rate("m_llamacpp:requests_processing")
        out["waiting_requests_mean"] = rate("m_llamacpp:requests_deferred")
        out["busy_slots_per_decode"] = rate("m_llamacpp:n_busy_slots_per_decode")
    return out
