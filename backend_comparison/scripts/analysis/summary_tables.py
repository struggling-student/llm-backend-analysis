#!/usr/bin/env python3
"""Markdown tables for the report, computed from processed_results/ (no hand-typed numbers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "processed_results"
ARMS = ["vllm-bf16", "llamacpp-bf16", "vllm-w8a8", "llamacpp-q8_0"]
NAMES = {"vllm-bf16": "vLLM BF16", "llamacpp-bf16": "llama.cpp BF16", "vllm-w8a8": "vLLM INT8 (W8A8)",
         "llamacpp-q8_0": "llama.cpp INT8 (Q8_0)"}
PAIRS = [("bf16", "vllm-bf16", "llamacpp-bf16"), ("int8", "vllm-w8a8", "llamacpp-q8_0")]
WL = ["decode", "balanced", "prefill", "longctx", "kvdecode"]
WLN = {"decode": "128→512", "balanced": "512→512", "prefill": "4096→128", "longctx": "8192→256", "kvdecode": "4096→1024"}


def f(x, nd=0):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def main() -> None:
    pts = pd.read_csv(P / "online_points.csv")
    pts = pts[(pts.campaign == "main") & (pts.arm == pts.arm_dir)]
    off = pd.read_csv(P / "offline.csv")
    off = off[off.campaign == "main"]
    out = []
    for model in ["llama31_8b", "llama32_1b"]:
        m = pts[pts.model == model]
        if m.empty:
            continue
        out.append(f"\n### {model}: C=1 and peak throughput per arm\n")
        out.append("| Workload | Arm | C=1 output tok/s | C=1 TPOT p50 (ms) | C=1 TTFT p50 (s) | Peak total tok/s (at C) | "
                   "Peak / C=1 | TPOT p50 at peak (ms) | E2E p95 at peak (s) | Highest C run | Why the sweep ended |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for wl in WL:
            for arm in ARMS:
                g = m[(m.workload == wl) & (m.arm == arm)].sort_values("concurrency")
                if g.empty:
                    continue
                c1 = g[g.concurrency == 1].iloc[0] if (g.concurrency == 1).any() else None
                pk = g.loc[g.total_tok_s_mean.idxmax()]
                last = g.iloc[-1]
                sweep = ROOT / "raw_results" / "online" / "main" / model / arm / wl / "sweep.json"
                sw = __import__("json").loads(sweep.read_text()) if sweep.exists() else None
                if not sweep.exists():
                    reason = "sweep incomplete (job running or not yet run)"
                elif sw["points"][-1]["status"] == "skipped_budget":
                    reason = f"next point (C={sw['points'][-1]['concurrency']}) projected > {sw['rule']['max_point_seconds'] // 3600} h budget"
                elif (g.gain_vs_prev.tail(2) < 0.05).all() and len(g) > 2:
                    reason = "plateau (2 doublings < +5%)"
                elif last.concurrency >= 256:
                    reason = "maximum configured C"
                elif last.concurrency >= 64:
                    reason = "last doubling < +10%: no extension"
                else:
                    reason = "see sweep.json"
                out.append(f"| {WLN[wl]} | {NAMES[arm]} | {f(c1.out_tok_s_mean, 1) if c1 is not None else '–'} | "
                           f"{f(c1.tpot_s_p50_mean * 1e3, 1) if c1 is not None else '–'} | {f(c1.ttft_s_p50_mean, 2) if c1 is not None else '–'} | "
                           f"{f(pk.total_tok_s_mean)} (C={int(pk.concurrency)}) | {f(pk.total_tok_s_mean / c1.total_tok_s_mean, 1) if c1 is not None else '–'}× | "
                           f"{f(pk.tpot_s_p50_mean * 1e3)} | {f(pk.e2e_s_pooled_p95, 1)} | {int(last.concurrency)} | {reason} |")
        cs = sorted(m.concurrency.unique())
        for pair, va, la in PAIRS:
            if not ((m.arm == va).any() and (m.arm == la).any()):
                continue
            out.append(f"\n### {model}: {NAMES[va]} ÷ {NAMES[la]} ({pair} pair), total tokens/s\n")
            out.append("| Workload | " + " | ".join(f"C={c}" for c in cs) + " |")
            out.append("|---|" + "---|" * len(cs))
            for wl in WL:
                v = m[(m.workload == wl) & (m.arm == va)].set_index("concurrency").total_tok_s_mean
                l = m[(m.workload == wl) & (m.arm == la)].set_index("concurrency").total_tok_s_mean
                if v.empty or l.empty:
                    continue
                out.append(f"| {WLN[wl]} | " + " | ".join(f"{v[c] / l[c]:.2f}×" if (c in v and c in l) else "–" for c in cs) + " |")
        out.append(f"\n### {model}: memory traffic (GB/s, core PMU) and share of STREAM-sustainable traffic\n")
        out.append("| Workload | Arm | " + " | ".join(f"C={c}" for c in cs) + " |")
        out.append("|---|---|" + "---|" * len(cs))
        for wl in WL:
            for arm in ARMS:
                g = m[(m.workload == wl) & (m.arm == arm)].set_index("concurrency")
                if g.empty:
                    continue
                out.append(f"| {WLN[wl]} | {NAMES[arm]} | " + " | ".join(
                    f"{g.mem_total_gbs_mean[c]:.0f} ({g.bw_util_vs_stream_max_mean[c]:.0%})" if c in g.index else "–" for c in cs) + " |")
    # offline
    out.append("\n### Offline engine microbenchmarks (tokens/s, mean of repetitions)\n")
    o = off[off.experiment.isin(["A1", "A2"])].groupby(["model", "workload", "batch", "test", "arm"]).tok_s.mean().unstack("arm")
    o = o.reindex(columns=[a for a in ARMS if a in o.columns])
    out.append("| Model | Workload | Batch | Phase | " + " | ".join(NAMES[a] for a in o.columns) + " |")
    out.append("|---|---|---|---|" + "---|" * len(o.columns))
    for (model, wl, b, t), row in o.iterrows():
        out.append(f"| {model} | {WLN[wl]} | {b} | {t} | " + " | ".join(f(v, 1) for v in row.values) + " |")
    (ROOT / "report" / "tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
