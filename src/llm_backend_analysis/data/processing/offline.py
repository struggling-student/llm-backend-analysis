"""Experiment A (offline engine microbenchmarks): offline.json -> one row per measured sample.

Benchmarks (column `benchmark`) and the tools behind them:
  batch1      llama-bench pp/tg/pg              vs  vllm bench latency, batch 1
  batched     llama-batched-bench (B = 8, 32)   vs  vllm bench latency, batch B
  throughput  vllm bench throughput (vLLM-only reference, 1 sample per shape)

The tools differ in scope: llama-bench times forward passes only, vllm bench latency
times LLM.generate() (scheduler, input preparation and sampling included).
vLLM decode time is derived as full-run latency minus the median prefill-only latency
(two separate engine runs), so its spread understates the uncertainty.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ... import paths
from .. import registry

TEST_NAMES = {"pp": "prefill", "tg": "decode", "pg": "total"}


def process_offline() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for f in sorted((paths.RAW / "offline").glob("*/*/*/*/offline.json")):
        d = json.loads(f.read_text())
        campaign = f.parts[-5]
        isl, osl = int(d["isl"]), int(d["osl"])
        keys = {"campaign": campaign, "campaign_role": registry.campaign_role("offline", campaign), "model": d["model"],
                **registry.configuration_columns(d["arm"]), "arm": d["arm"], "workload": d["workload"],
                "isl": isl, "osl": osl, "host": d["hostname"].split(".")[0], "slurm_job_id": str(d["slurm_job_id"]),
                "source": paths.relative(f)}
        vllm_lat: dict[tuple[int, str], list[float]] = {}
        for run in d["runs"]:
            tool = run["tool"]
            if tool == "llama-bench":
                for r in run.get("rows", []):
                    test = run["test"]
                    ntok = r["n_prompt"] + r["n_gen"] if test == "pg" else (r["n_prompt"] or r["n_gen"])
                    for i, ns in enumerate(r["samples_ns"]):
                        rows.append({**keys, "tool": tool, "benchmark": "batch1", "batch": 1,
                                     "test": TEST_NAMES[test], "sample": i, "seconds": ns / 1e9, "tokens": ntok,
                                     "tok_s": ntok / (ns / 1e9)})
            elif tool == "llama-batched-bench":
                for r in run.get("rows", []):
                    b = r.get("pl") or run["batch"]
                    for test, sec, tok, speed in (("prefill", r["t_pp"], r["pp"] * b, r["speed_pp"]),
                                                  ("decode", r["t_tg"], r["tg"] * b, r["speed_tg"]),
                                                  ("total", r["t"], r["n_kv"], r["speed"])):
                        rows.append({**keys, "tool": tool, "benchmark": "batched", "batch": b, "test": test,
                                     "sample": run["rep"], "seconds": sec, "tokens": tok, "tok_s": speed})
            elif tool == "vllm-bench-latency" and run.get("result"):
                vllm_lat[(run["batch"], run["test"])] = run["result"]["latencies"]
            elif tool == "vllm-bench-throughput" and run.get("result"):
                r = run["result"]
                rows.append({**keys, "tool": tool, "benchmark": "throughput", "batch": run["batch"], "test": "total",
                             "sample": 0, "seconds": r["elapsed_time"], "tokens": r["total_num_tokens"],
                             "tok_s": r["tokens_per_second"]})
        for (b, test), lats in vllm_lat.items():
            if test != "prefill" or (b, "full") not in vllm_lat:
                continue
            bench = "batch1" if b == 1 else "batched"
            for i, tp in enumerate(lats):
                rows.append({**keys, "tool": "vllm-bench-latency", "benchmark": bench, "batch": b, "test": "prefill",
                             "sample": i, "seconds": tp, "tokens": isl * b, "tok_s": isl * b / tp})
            tp_med = float(np.median(lats))
            for i, tf in enumerate(vllm_lat[(b, "full")]):
                rows.append({**keys, "tool": "vllm-bench-latency", "benchmark": bench, "batch": b, "test": "decode",
                             "sample": i, "seconds": tf - tp_med, "tokens": (osl - 1) * b,
                             "tok_s": (osl - 1) * b / (tf - tp_med)})
                rows.append({**keys, "tool": "vllm-bench-latency", "benchmark": bench, "batch": b, "test": "total",
                             "sample": i, "seconds": tf, "tokens": (isl + osl) * b, "tok_s": (isl + osl) * b / tf})
    df = pd.DataFrame(rows)
    return df.sort_values(["campaign", "model", "configuration_id", "workload", "benchmark", "batch", "test", "tool",
                           "sample"], ignore_index=True)
