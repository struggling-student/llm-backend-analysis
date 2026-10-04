"""Experiment B (online serving): raw point directories -> repetition, point and request tables.

Raw layout: data/raw/online/<campaign>/<model>/<arm_dir>/<workload>/c<C>/
    metadata.json      server/client commands, settings, per-repetition summaries
    rep<N>.jsonl.gz    one record per request (token timestamps)
Telemetry:  data/raw/telemetry/online/<campaign>/<model>/<arm_dir>/<workload>/c<C>.jsonl

Only measurement aggregation happens here. Cross-point quantities (speedup over C=1,
scaling efficiency, backend ratios, traffic model) are computed in
llm_backend_analysis.analysis so the notebooks show them explicitly.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ... import paths
from .. import registry
from ..telemetry import read_telemetry, window_aggregate

POINT_KEYS = ["campaign", "campaign_role", "model", "configuration_id", "backend", "precision", "quantization",
              "arm", "arm_dir", "variant", "workload", "isl", "osl", "concurrency"]

# Per-repetition metrics summarized per point as mean/std/min/max/ci95/cv.
REPETITION_METRICS = [
    "output_tok_s", "input_tok_s", "total_tok_s", "request_s",
    "mem_read_gbs", "mem_write_gbs", "mem_total_gbs", "amx_busy_frac", "ipc", "ghz_effective_per_core",
    "cpu_busy_compute_mean", "cpu_busy_frontend", "tokens_per_step", "engine_steps_per_s",
    "running_requests_mean", "rss_gib_max", "client_cpu_frac",
    "ttft_s_p50", "tpot_s_p50", "e2e_s_p50", "itl_s_p50",
]


def _ci95(values: pd.Series) -> float:
    v = values.dropna()
    if len(v) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def _working_set_gib(model: str, harness_precision: str, isl: int, osl: int, c: int) -> float:
    """Active working set: streamed weights + live KV of C sequences at mean context ISL + OSL/2."""
    kv = registry.experiment_config()["models"][model]["kv_bytes_per_token"]
    return (registry.step_weight_bytes(model, harness_precision) + c * (isl + osl / 2) * kv) / 2**30


def process_online(lookup: Callable[[float], dict[str, float]], node_checks: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (repetitions, requests)."""
    node_gbs = dict(zip(node_checks["slurm_job_id"], node_checks["triad_gbs"])) if not node_checks.empty else {}
    rep_rows: list[dict[str, Any]] = []
    req_rows: list[dict[str, Any]] = []
    for meta_path in sorted((paths.RAW / "online").glob("*/*/*/*/c*/metadata.json")):
        if ".attempt-" in str(meta_path):
            continue
        meta = json.loads(meta_path.read_text())
        if not str(meta.get("status", "")).startswith("ok"):
            continue
        campaign, model, arm_dir, workload, cdir = meta_path.parts[-6:-1]
        arm, variant = registry.split_arm_dir(arm_dir)
        conf = registry.configuration_for_arm(arm)
        c = int(meta["concurrency"])
        _, tel = read_telemetry(paths.RAW_TELEMETRY / "online" / campaign / model / arm_dir / workload / f"{cdir}.jsonl")
        keys = {"campaign": campaign, "campaign_role": registry.campaign_role("online", campaign), "model": model,
                **registry.configuration_columns(arm), "arm": arm, "arm_dir": arm_dir, "variant": variant,
                "workload": workload, "isl": int(meta["isl"]), "osl": int(meta["osl"]), "concurrency": c}
        idle = next((p for p in meta.get("phases", []) if p["phase"] == "idle"), None)
        idle_agg = window_aggregate(tel, idle["t_start"], idle["t_end"]) if idle else {}
        ws_gib = _working_set_gib(model, conf.harness_precision, keys["isl"], keys["osl"], c)
        base = lookup(ws_gib)
        job = str(meta.get("slurm_job_id", ""))
        for rep in meta.get("reps", []):
            s, win = rep["summary"], rep["summary"]["window"]
            agg = window_aggregate(tel, win["start"], win["end"]) if win["start"] else {}
            row = {**keys, "rep": rep["rep"], "host": meta["hostname"].split(".")[0], "slurm_job_id": job,
                   "node_check_triad_gbs": node_gbs.get(job),
                   "requests": s["requests"], "completed": s["completed"], "failed": s["failed"],
                   "prompt_tokens_ok": s["prompt_tokens_ok"], "completion_tokens_ok": s["completion_tokens_ok"],
                   "chunks_equal_tokens": s["chunks_equal_tokens"], "steady_window_s": s["steady_window_s"],
                   "output_tok_s": s["steady_output_tok_s"], "input_tok_s": s["steady_input_tok_s"],
                   "total_tok_s": s["steady_total_tok_s"], "request_s": s["steady_request_s"],
                   "overall_output_tok_s": s["overall_output_tok_s"], "makespan_s": s["makespan_s"],
                   "latency_sample": s["latency_sample"],
                   "client_cpu_frac": rep["client_resources"]["cpu_fraction_of_one_core"],
                   "client_loop_lag_max_ms": rep["timing"].get("loop_lag_max_ms"),
                   "prompt_manifest_sha256": rep["prompt_manifest_sha256"],
                   "working_set_gib_model": ws_gib, **base, **agg,
                   "idle_mem_total_gbs": idle_agg.get("mem_total_gbs"),
                   "idle_cpu_busy_compute": idle_agg.get("cpu_busy_compute_mean"),
                   "source": paths.relative(meta_path.parent)}
            for metric in ("ttft_s", "tpot_s", "e2e_s", "itl_s"):
                for q in ("mean", "p50", "p95", "p99"):
                    row[f"{metric}_{q}"] = s["latency"][metric][q]
                row[f"{metric}_n"] = s["latency"][metric]["n"]
            if agg.get("mem_total_gbs") and base.get("stream_pmu_total_gbs_at_ws"):
                row["bw_util_vs_stream_at_ws"] = agg["mem_total_gbs"] / base["stream_pmu_total_gbs_at_ws"]
                row["bw_util_vs_stream_max"] = agg["mem_total_gbs"] / base["stream_pmu_total_gbs_max"]
            if agg.get("mem_total_gbs") and row["output_tok_s"]:
                row["bytes_per_output_token_mb"] = agg["mem_total_gbs"] * 1e3 / row["output_tok_s"]
            rep_rows.append(row)
            raw = meta_path.parent / rep["raw_file"]
            if not raw.exists():
                continue
            t_started = rep["timing"].get("t_all_workers_started") or 0
            with gzip.open(raw, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if r["error"] or not r["n_chunks"]:
                        continue
                    # Steady request: sent after every worker started and finished inside the window.
                    steady = (win["start"] is not None and r["t_send"] >= t_started and r["t_end"] <= (win["end"] or 0)) \
                        or c == 1
                    n_out = r["completion_tokens"] or r["n_chunks"]
                    req_rows.append({**{k: keys[k] for k in ("campaign", "model", "configuration_id", "arm_dir",
                                                             "workload", "concurrency")},
                                     "rep": rep["rep"], "index": r["index"], "steady": steady,
                                     "ttft_s": r["t_first"] - r["t_send"], "e2e_s": r["t_end"] - r["t_send"],
                                     "tpot_s": (r["t_end"] - r["t_first"]) / max(1, n_out - 1),
                                     "completion_tokens": n_out, "prompt_tokens": r["prompt_tokens"],
                                     "t_send": r["t_send"]})
    reps = pd.DataFrame(rep_rows).sort_values(POINT_KEYS + ["rep"], ignore_index=True)
    reqs = pd.DataFrame(req_rows).sort_values(["campaign", "model", "arm_dir", "workload", "concurrency", "rep", "index"],
                                              ignore_index=True)
    return reps, reqs


def aggregate_points(reps: pd.DataFrame, reqs: pd.DataFrame) -> pd.DataFrame:
    """One row per (campaign, model, arm_dir, workload, C): statistics over repetitions plus
    latency percentiles pooled over the steady requests of all repetitions."""
    req_groups = {k: g for k, g in reqs.groupby(["campaign", "model", "arm_dir", "workload", "concurrency"])}
    rows = []
    for key, g in reps.groupby(POINT_KEYS, sort=True):
        row = dict(zip(POINT_KEYS, key))
        row["n_reps"] = len(g)
        row["hosts"] = ",".join(sorted(g["host"].unique()))
        row["failed_requests"] = int(g["failed"].sum())
        row["tokens_ok"] = bool(g["prompt_tokens_ok"].all() and g["completion_tokens_ok"].all())
        row["manifest_consistent"] = g["prompt_manifest_sha256"].nunique() == 1
        row["node_check_triad_gbs_min"] = g["node_check_triad_gbs"].min()
        for m in REPETITION_METRICS:
            if m not in g:
                continue
            v = g[m].astype(float)
            row[f"{m}_mean"], row[f"{m}_std"] = v.mean(), v.std(ddof=1)
            row[f"{m}_min"], row[f"{m}_max"], row[f"{m}_ci95"] = v.min(), v.max(), _ci95(v)
            row[f"{m}_cv"] = v.std(ddof=1) / v.mean() if v.mean() else np.nan
        for k in ("working_set_gib_model", "stream_triad_gbs_at_ws", "stream_pmu_total_gbs_at_ws",
                  "stream_pmu_total_gbs_max"):
            if k in g:
                row[k] = g[k].iloc[0]
        for k in ("bw_util_vs_stream_at_ws", "bw_util_vs_stream_max", "bytes_per_output_token_mb"):
            if k in g:
                row[f"{k}_mean"] = g[k].mean()
        row["idle_cpu_busy_compute_max"] = g["idle_cpu_busy_compute"].max()
        row["client_loop_lag_ms_max"] = g["client_loop_lag_max_ms"].max()
        q = req_groups.get((row["campaign"], row["model"], row["arm_dir"], row["workload"], row["concurrency"]))
        if q is not None:
            qs = q[q["steady"]] if q["steady"].sum() >= 3 else q
            row["latency_n"] = len(qs)
            for metric in ("ttft_s", "tpot_s", "e2e_s"):
                for name, quant in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
                    row[f"{metric}_pooled_{name}"] = qs[metric].quantile(quant)
                row[f"{metric}_pooled_mean"] = qs[metric].mean()
            row["p99_meaningful"] = len(qs) >= 100
        rows.append(row)
    return pd.DataFrame(rows).sort_values(POINT_KEYS, ignore_index=True)


def read_sweeps() -> pd.DataFrame:
    """Why each concurrency sweep ended (sweep.json): last point status and declared rule."""
    rows = []
    for f in sorted((paths.RAW / "online").glob("*/*/*/*/sweep.json")):
        campaign, model, arm_dir, workload = f.parts[-5:-1]
        arm, variant = registry.split_arm_dir(arm_dir)
        d = json.loads(f.read_text())
        pts = d.get("points", [])
        last = dict(pts[-1]) if pts else {}
        point_meta = f.parent / f"c{last.get('concurrency', 0):03d}" / "metadata.json"
        if str(last.get("status", "")).startswith("skipped") and point_meta.exists():
            m = json.loads(point_meta.read_text())
            last.update({k: m.get(k) for k in ("skip_reason", "projected_point_s")})
        rows.append({"campaign": campaign, "model": model, **registry.configuration_columns(arm), "arm_dir": arm_dir,
                     "variant": variant, "workload": workload, "planned": ",".join(str(c) for c in d.get("plan", [])),
                     "last_concurrency": last.get("concurrency"), "last_status": last.get("status"),
                     "skip_reason": last.get("skip_reason"),
                     "projected_point_s": last.get("projected_point_s"),
                     "rule_plateau_gain": d.get("rule", {}).get("plateau_gain"),
                     "rule_extend_min_gain": d.get("rule", {}).get("extend_min_gain"),
                     "rule_max_point_s": d.get("rule", {}).get("max_point_seconds"), "source": paths.relative(f)})
    return pd.DataFrame(rows)
