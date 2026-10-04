#!/usr/bin/env python3
"""Normalize raw results into tidy tables (run locally after scripts/sync_results.sh).

Outputs in processed_results/:
  stream_calibration.csv   STREAM bandwidth and the PMU proxy's view of the same traffic
  online_reps.csv          one row per (model, arm, workload, C, repetition)
  online_points.csv        per point: mean/std/CI over repetitions, pooled latency percentiles,
                           scaling metrics, bandwidth utilization
  online_requests.parquet  every measured request (pooled distributions)
  telemetry.parquet        every telemetry sample, labelled with its point and phase
  offline.csv              offline microbenchmark results, one row per measured sample
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
LINE = 64


# --------------------------------------------------------------------------- telemetry
def read_telemetry(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    header: dict[str, Any] = {}
    rows = []
    if not path.exists():
        return header, pd.DataFrame()
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec["type"] == "header":
            header = rec
        elif rec["type"] == "sample":
            row = {"t": rec["t_mono"], "dt": rec["dt"], "threads": rec.get("threads"),
                   "rss_bytes": rec.get("rss_bytes")}
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
                compute = [b for c, b in vals.items() if c <= 54]
                row["cpu_busy_compute_mean"] = float(np.mean(compute)) if compute else None
                row["cpu_busy_frontend"] = vals.get(55)
            for k, v in (rec.get("metrics") or {}).items():
                row[f"m_{k}"] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    return header, df


def window_aggregate(tel: pd.DataFrame, start: float, end: float) -> dict[str, Any]:
    """Aggregate samples over [start, end], weighting each sample by its overlap."""
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
    out["mem_read_gbs"] = rd * LINE / 1e9
    out["mem_write_gbs"] = wr * LINE / 1e9
    out["mem_total_gbs"] = (rd + wr) * LINE / 1e9
    out["mem_read_demand_core_gbs"] = (rate("pmu_ocr_rd_l3miss_local") or 0) * LINE / 1e9
    out["llc_miss_gbs"] = (rate("pmu_llc_miss") or 0) * LINE / 1e9
    cyc = rate("pmu_cycles")
    out["amx_busy_frac"] = (rate("pmu_exe_amx_busy") or 0) / cyc if cyc else None
    out["ipc"] = (rate("pmu_instructions") or 0) / cyc if cyc else None
    out["ghz_effective_per_core"] = cyc / 55 / 1e9 if cyc else None
    for c in [c for c in tw.columns if c.startswith("unc_")]:
        out[c.replace("unc_", "uncore_") + "_gbs"] = rate(c) * LINE / 1e9
    for c in ("cpu_busy_compute_mean", "cpu_busy_frontend", "threads"):
        if c in tw:
            out[c] = float(np.average(tw[c].astype(float), weights=ww))
    for c in [c for c in tw.columns if c.startswith("runfrac_")]:
        out[c] = float(tw[c].min())
    out["rss_gib_max"] = float(tw["rss_bytes"].max()) / 2**30 if "rss_bytes" in tw else None
    if "used_node0" in tw:
        out["node0_used_gib_max"] = float(tw["used_node0"].max()) / 2**30
    # Server-side batching from the servers' own counters (difference over the window).
    def delta(col: str) -> float | None:
        if col not in tel:
            return None
        inside = tel[(tel["t"] >= start) & (tel["t"] <= end)][col].dropna()
        return float(inside.iloc[-1] - inside.iloc[0]) if len(inside) >= 2 else None

    def span_inside() -> float | None:
        inside = tel[(tel["t"] >= start) & (tel["t"] <= end)]["t"]
        return float(inside.iloc[-1] - inside.iloc[0]) if len(inside) >= 2 else None

    sp = span_inside()
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


# --------------------------------------------------------------------------- STREAM
def process_stream(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "data" / "raw" / "stream").glob("*/stream_calibration.json")):
        d = json.loads(f.read_text())
        for r in d["runs"]:
            if not r.get("stream"):
                continue
            tel_path = root / "data" / "raw" / "telemetry" / "stream" / f.parent.name / Path(r["telemetry_file"]).name
            _, tel = read_telemetry(tel_path)
            # steady part = the last `steady_seconds_total` seconds before the process ended
            end = r["t_end_mono"]
            start = end - r["steady_seconds_total"]
            agg = window_aggregate(tel, start + 0.5, end - 0.25)
            stream_bw = r["bandwidth_gbs_median"]
            rows.append({"campaign": f.parent.name, "host": d["hostname"].split(".")[0], "threads": r["threads"],
                         "working_set_gib": r["working_set_gib"], "stream_triad_gbs": stream_bw,
                         "stream_read_model_gbs": stream_bw * 3 / 3, "stream_write_model_gbs": stream_bw / 3,
                         **{k: v for k, v in agg.items() if k.startswith(("mem_", "llc_", "uncore_", "tel_", "runfrac", "cpu_busy"))},
                         "pmu_total_over_stream": (agg.get("mem_total_gbs") or np.nan) / stream_bw,
                         "pmu_read_over_stream_read": (agg.get("mem_read_gbs") or np.nan) / stream_bw,
                         "pmu_write_over_stream_write": (agg.get("mem_write_gbs") or np.nan) / (stream_bw / 3)})
    return pd.DataFrame(rows)


def sustainable_lookup(stream: pd.DataFrame, threads: int = 55, campaign: str = "calibration"):
    """Interpolate (in log working-set) the STREAM baselines measured with the
    same thread count. Returns f(ws_gib) -> dict of baselines."""
    s = stream[(stream["threads"] == threads) & (stream["campaign"] == campaign)].groupby("working_set_gib").median(numeric_only=True).sort_index()
    if s.empty:
        return lambda ws: {}
    x = np.log(s.index.values.astype(float))

    def f(ws_gib: float) -> dict[str, float]:
        xi = math.log(max(ws_gib, 1e-3))
        out = {}
        for col, name in (("stream_triad_gbs", "stream_triad_gbs_at_ws"), ("mem_total_gbs", "stream_pmu_total_gbs_at_ws"),
                          ("mem_read_gbs", "stream_pmu_read_gbs_at_ws")):
            if col in s:
                out[name] = float(np.interp(xi, x, s[col].values))
        out["stream_triad_gbs_max"] = float(s["stream_triad_gbs"].max())
        if "mem_total_gbs" in s:
            out["stream_pmu_total_gbs_max"] = float(s["mem_total_gbs"].max())
            out["stream_pmu_read_gbs_max"] = float(s["mem_read_gbs"].max())
        return out
    return f


# --------------------------------------------------------------------------- online
def ci95(values: pd.Series) -> float:
    v = values.dropna()
    if len(v) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def process_online(root: Path, cfg: dict[str, Any], lookup) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rep_rows, req_rows, tel_frames = [], [], []
    for meta_path in sorted((root / "data" / "raw" / "online").glob("*/*/*/*/c*/metadata.json")):
        if ".attempt-" in str(meta_path):
            continue
        meta = json.loads(meta_path.read_text())
        if not str(meta.get("status", "")).startswith("ok"):
            continue
        campaign, model, arm_dir, workload, cdir = meta_path.parts[-6:-1]
        c = meta["concurrency"]
        mcfg = cfg["models"][model]
        tel_path = root / "data" / "raw" / "telemetry" / "online" / campaign / model / arm_dir / workload / f"{cdir}.jsonl"
        _, tel = read_telemetry(tel_path)
        keys = {"campaign": campaign, "model": model, "arm": meta["arm"], "arm_dir": arm_dir, "backend": meta["backend"],
                "precision": meta["precision"], "workload": workload, "isl": meta["isl"], "osl": meta["osl"],
                "concurrency": c, "host": meta["hostname"].split(".")[0]}
        if not tel.empty:
            tel_frames.append(tel.assign(**keys))
        idle = next((p for p in meta.get("phases", []) if p["phase"] == "idle"), None)
        idle_agg = window_aggregate(tel, idle["t_start"], idle["t_end"]) if idle else {}
        for rep in meta.get("reps", []):
            s = rep["summary"]
            lat = s["latency"]
            win = s["window"]
            agg = window_aggregate(tel, win["start"], win["end"]) if win["start"] else {}
            # Active working set: weights + live KV of C sequences at mean context length.
            mean_ctx = meta["isl"] + meta["osl"] / 2
            ws_gib = ((mcfg.get("step_weight_bytes", {}).get(meta["precision"]) or
                       mcfg["weight_bytes_bf16"] * (1.0 if meta["precision"] == "bf16" else 34 / 64))
                      + c * mean_ctx * mcfg["kv_bytes_per_token"]) / 2**30
            base = lookup(ws_gib)
            row = {**keys, "rep": rep["rep"], "requests": s["requests"], "completed": s["completed"],
                   "failed": s["failed"], "prompt_tokens_ok": s["prompt_tokens_ok"],
                   "completion_tokens_ok": s["completion_tokens_ok"], "chunks_equal_tokens": s["chunks_equal_tokens"],
                   "steady_window_s": s["steady_window_s"], "out_tok_s": s["steady_output_tok_s"],
                   "in_tok_s": s["steady_input_tok_s"], "total_tok_s": s["steady_total_tok_s"],
                   "req_s": s["steady_request_s"], "overall_out_tok_s": s["overall_output_tok_s"],
                   "overall_req_s": s["overall_request_s"], "makespan_s": s["makespan_s"],
                   "latency_sample": s["latency_sample"],
                   "client_cpu_frac": rep["client_resources"]["cpu_fraction_of_one_core"],
                   "client_loop_lag_max_ms": rep["timing"].get("loop_lag_max_ms"),
                   "prompt_manifest_sha256": rep["prompt_manifest_sha256"], "working_set_gib_model": ws_gib,
                   **base, **agg,
                   "idle_mem_total_gbs": idle_agg.get("mem_total_gbs"), "idle_cpu_busy": idle_agg.get("cpu_busy_compute_mean")}
            for metric in ("ttft_s", "tpot_s", "e2e_s", "itl_s"):
                for q in ("mean", "p50", "p95", "p99"):
                    row[f"{metric}_{q}"] = lat[metric][q]
                row[f"{metric}_n"] = lat[metric]["n"]
            if agg.get("mem_total_gbs") and base.get("stream_pmu_total_gbs_at_ws"):
                row["bw_util_vs_stream_at_ws"] = agg["mem_total_gbs"] / base["stream_pmu_total_gbs_at_ws"]
                row["bw_util_vs_stream_max"] = agg["mem_total_gbs"] / base["stream_pmu_total_gbs_max"]
                row["read_bw_util_vs_stream_max"] = agg["mem_read_gbs"] / base["stream_pmu_read_gbs_max"]
            if agg.get("mem_total_gbs") and row["out_tok_s"]:
                row["bytes_per_output_token_mb"] = agg["mem_total_gbs"] * 1e3 / row["out_tok_s"]
            # First-principles read traffic: every forward step reads all weights once; every
            # decoded token reads the KV of its sequence (mean context ISL + OSL/2). Prefill
            # attention and activations are ignored, so this is a lower-bound-style estimate.
            wbytes = mcfg.get("step_weight_bytes", {}).get(meta["precision"]) or \
                mcfg["weight_bytes_bf16"] * (1.0 if meta["precision"] == "bf16" else 34 / 64)
            if agg.get("engine_steps_per_s") and row["out_tok_s"]:
                row["model_weight_read_gbs"] = agg["engine_steps_per_s"] * wbytes / 1e9
                row["model_kv_read_gbs"] = row["out_tok_s"] * mean_ctx * mcfg["kv_bytes_per_token"] / 1e9
                row["model_read_gbs"] = row["model_weight_read_gbs"] + row["model_kv_read_gbs"]
                if agg.get("mem_read_gbs"):
                    row["measured_over_model_read"] = agg["mem_read_gbs"] / row["model_read_gbs"]
            rep_rows.append(row)
            # pooled requests
            raw = meta_path.parent / rep["raw_file"]
            if raw.exists():
                with gzip.open(raw, "rt") as f:
                    for line in f:
                        r = json.loads(line)
                        if r["error"] or not r["n_chunks"]:
                            continue
                        steady = (win["start"] is not None and r["t_send"] >= (rep["timing"].get("t_all_workers_started") or 0)
                                  and r["t_end"] <= (win["end"] or 0)) or c == 1
                        n_out = r["completion_tokens"] or r["n_chunks"]
                        req_rows.append({**keys, "rep": rep["rep"], "index": r["index"], "steady": steady,
                                         "ttft_s": r["t_first"] - r["t_send"], "e2e_s": r["t_end"] - r["t_send"],
                                         "tpot_s": (r["t_end"] - r["t_first"]) / max(1, n_out - 1),
                                         "completion_tokens": n_out, "prompt_tokens": r["prompt_tokens"],
                                         "t_send": r["t_send"]})
    reps = pd.DataFrame(rep_rows)
    reqs = pd.DataFrame(req_rows)
    tel_all = pd.concat(tel_frames, ignore_index=True) if tel_frames else pd.DataFrame()
    points = aggregate_points(reps, reqs) if not reps.empty else pd.DataFrame()
    return reps, points, reqs, tel_all


POINT_METRICS = ["out_tok_s", "in_tok_s", "total_tok_s", "req_s", "mem_read_gbs", "mem_write_gbs", "mem_total_gbs",
                 "llc_miss_gbs", "amx_busy_frac", "ipc", "cpu_busy_compute_mean", "cpu_busy_frontend",
                 "bw_util_vs_stream_at_ws", "bw_util_vs_stream_max", "read_bw_util_vs_stream_max",
                 "tokens_per_step", "engine_steps_per_s", "running_requests_mean", "bytes_per_output_token_mb",
                 "ttft_s_p50", "tpot_s_p50", "e2e_s_p50", "client_cpu_frac", "rss_gib_max",
                 "ghz_effective_per_core", "model_read_gbs", "model_weight_read_gbs", "model_kv_read_gbs",
                 "measured_over_model_read"]
GROUP = ["campaign", "model", "arm", "arm_dir", "backend", "precision", "workload", "isl", "osl", "concurrency"]


def aggregate_points(reps: pd.DataFrame, reqs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in reps.groupby(GROUP):
        row = dict(zip(GROUP, key))
        row["n_reps"] = len(g)
        row["hosts"] = ",".join(sorted(g["host"].unique()))
        row["failed_requests"] = int(g["failed"].sum())
        row["tokens_ok"] = bool(g["prompt_tokens_ok"].all() and g["completion_tokens_ok"].all())
        row["manifest_consistent"] = g["prompt_manifest_sha256"].nunique() == 1
        for m in POINT_METRICS:
            if m in g:
                v = g[m].astype(float)
                row[f"{m}_mean"], row[f"{m}_std"] = v.mean(), v.std(ddof=1)
                row[f"{m}_min"], row[f"{m}_max"], row[f"{m}_ci95"] = v.min(), v.max(), ci95(v)
                row[f"{m}_cv"] = v.std(ddof=1) / v.mean() if v.mean() else np.nan
        for k in ("working_set_gib_model", "stream_triad_gbs_at_ws", "stream_pmu_total_gbs_at_ws", "stream_pmu_total_gbs_max"):
            if k in g:
                row[k] = g[k].iloc[0]
        if not reqs.empty:
            q = reqs[np.logical_and.reduce([reqs[k] == v for k, v in row.items() if k in GROUP])]
            qs = q[q["steady"]] if q["steady"].sum() >= 3 else q
            row["latency_n"] = len(qs)
            for metric in ("ttft_s", "tpot_s", "e2e_s"):
                for name, quant in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
                    row[f"{metric}_pooled_{name}"] = qs[metric].quantile(quant) if len(qs) else np.nan
                row[f"{metric}_pooled_mean"] = qs[metric].mean() if len(qs) else np.nan
            row["p99_meaningful"] = len(qs) >= 100
        rows.append(row)
    pts = pd.DataFrame(rows).sort_values(GROUP)
    # scaling metrics within each (campaign, model, arm_dir, workload) curve
    out = []
    for _, g in pts.groupby(["campaign", "model", "arm_dir", "workload"]):
        g = g.sort_values("concurrency").copy()
        base = g.loc[g["concurrency"] == 1, "total_tok_s_mean"]
        base_out = g.loc[g["concurrency"] == 1, "out_tok_s_mean"]
        if len(base):
            g["speedup_vs_c1"] = g["total_tok_s_mean"] / base.iloc[0]
            g["scaling_efficiency"] = g["speedup_vs_c1"] / g["concurrency"]
            g["out_speedup_vs_c1"] = g["out_tok_s_mean"] / base_out.iloc[0]
        prev = g["total_tok_s_mean"].shift(1)
        prev_c = g["concurrency"].shift(1)
        g["gain_vs_prev"] = np.where(prev_c * 2 == g["concurrency"], g["total_tok_s_mean"] / prev - 1, np.nan)
        g["tpot_increase_vs_prev"] = g["tpot_s_p50_mean"] / g["tpot_s_p50_mean"].shift(1) - 1
        g["bw_gain_vs_prev"] = g["mem_total_gbs_mean"] / g["mem_total_gbs_mean"].shift(1) - 1 if "mem_total_gbs_mean" in g else np.nan
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- offline
def process_offline(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "data" / "raw" / "offline").glob("*/*/*/*/offline.json")):
        d = json.loads(f.read_text())
        keys = {"campaign": f.parts[-5], "model": d["model"], "arm": d["arm"], "backend": d["backend"],
                "precision": d["precision"], "workload": d["workload"], "isl": d["isl"], "osl": d["osl"]}
        vllm_lat: dict[tuple[int, str], list[float]] = {}
        for run in d["runs"]:
            tel_path = root / "data" / "raw" / "telemetry" / Path(run["telemetry_file"]).relative_to(Path(run["telemetry_file"]).parents[5])
            tool = run["tool"]
            if tool == "llama-bench":
                for r in run.get("rows", []):
                    test = run["test"]
                    ntok = r["n_prompt"] + r["n_gen"] if test == "pg" else (r["n_prompt"] or r["n_gen"])
                    for i, ns in enumerate(r["samples_ns"]):
                        rows.append({**keys, "tool": tool, "experiment": "A1", "batch": 1, "test": test, "sample": i,
                                     "seconds": ns / 1e9, "tokens": ntok, "tok_s": ntok / (ns / 1e9)})
            elif tool == "llama-batched-bench":
                for r in run.get("rows", []):
                    b = r.get("pl") or run["batch"]
                    rows.append({**keys, "tool": tool, "experiment": "A2", "batch": b, "test": "prefill",
                                 "sample": run["rep"], "seconds": r["t_pp"], "tokens": r["pp"] * b, "tok_s": r["speed_pp"]})
                    rows.append({**keys, "tool": tool, "experiment": "A2", "batch": b, "test": "decode",
                                 "sample": run["rep"], "seconds": r["t_tg"], "tokens": r["tg"] * b, "tok_s": r["speed_tg"]})
                    rows.append({**keys, "tool": tool, "experiment": "A2", "batch": b, "test": "total",
                                 "sample": run["rep"], "seconds": r["t"], "tokens": r["n_kv"], "tok_s": r["speed"]})
            elif tool == "vllm-bench-latency" and run.get("result"):
                vllm_lat[(run["batch"], run["test"])] = run["result"]["latencies"]
            elif tool == "vllm-bench-throughput" and run.get("result"):
                r = run["result"]
                rows.append({**keys, "tool": tool, "experiment": "A3", "batch": run["batch"], "test": "total",
                             "sample": 0, "seconds": r["elapsed_time"], "tokens": r["total_num_tokens"],
                             "tok_s": r["tokens_per_second"]})
        for (b, test), lats in vllm_lat.items():
            if test != "prefill" or (b, "full") not in vllm_lat:
                continue
            full = vllm_lat[(b, "full")]
            exp = "A1" if b == 1 else "A2"
            for i, tp in enumerate(lats):
                rows.append({**keys, "tool": "vllm-bench-latency", "experiment": exp, "batch": b, "test": "prefill",
                             "sample": i, "seconds": tp, "tokens": d["isl"] * b, "tok_s": d["isl"] * b / tp})
            # decode time = full - prefill, using the median prefill (the two runs are separate engines)
            tp_med = float(np.median(lats))
            for i, tf in enumerate(full):
                tp = tp_med
                rows.append({**keys, "tool": "vllm-bench-latency", "experiment": exp, "batch": b, "test": "decode",
                             "sample": i, "seconds": tf - tp, "tokens": (d["osl"] - 1) * b,
                             "tok_s": (d["osl"] - 1) * b / (tf - tp)})
                rows.append({**keys, "tool": "vllm-bench-latency", "experiment": exp, "batch": b, "test": "total",
                             "sample": i, "seconds": tf, "tokens": (d["isl"] + d["osl"]) * b,
                             "tok_s": (d["isl"] + d["osl"]) * b / tf})
    df = pd.DataFrame(rows)
    if not df.empty:
        # llama-bench "pg" is the combined test; map names onto the common vocabulary
        df["test"] = df["test"].replace({"pp": "prefill", "tg": "decode", "pg": "total"})
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args()
    root = Path(args.root)
    cfg = tomllib.loads((root / "configs" / "experiment.toml").read_text())
    out = root / "data" / "processed"
    out.mkdir(exist_ok=True)
    stream = process_stream(root)
    stream.to_csv(out / "stream_calibration.csv", index=False)
    lookup = sustainable_lookup(stream)
    reps, points, reqs, tel = process_online(root, cfg, lookup)
    reps.to_csv(out / "online_reps.csv", index=False)
    points.to_csv(out / "online_points.csv", index=False)
    if not reqs.empty:
        reqs.to_parquet(out / "online_requests.parquet", index=False)
    if not tel.empty:
        tel.to_parquet(out / "telemetry.parquet", index=False)
    offline = process_offline(root)
    offline.to_csv(out / "offline.csv", index=False)
    print(f"stream rows {len(stream)}, online reps {len(reps)}, points {len(points)}, requests {len(reqs)}, "
          f"telemetry samples {len(tel)}, offline rows {len(offline)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
