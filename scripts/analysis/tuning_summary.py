#!/usr/bin/env python3
"""Summarize the tuning phase and apply the declared selection rule.

Rule (scripts/tuning.py): per arm, pick the configuration with the highest steady
total tokens/s at C=16, unless its C=1 median TPOT is more than 5% worse than the
default configuration's; then keep the default. Writes processed_results/tuning.csv
and prints the choice. The choice is copied into configs/experiment.toml [tuned.*]
by hand, so the config file stays the single, reviewed source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {"vllm-bf16": "mnbt4096", "vllm-w8a8": "mnbt4096", "llamacpp-bf16": "fa-on_ub512", "llamacpp-q8_0": "fa-on_ub512"}


def main() -> int:
    rows = []
    for meta_path in sorted((ROOT / "data" / "raw" / "online" / "tuning").glob("*/*/*/c*/metadata.json")):
        m = json.loads(meta_path.read_text())
        arm_dir = meta_path.parts[-4]
        tag = arm_dir.split("__", 1)[1] if "__" in arm_dir else "default"
        row = {"model": m["model"], "arm": m["arm"], "tag": tag, "concurrency": m["concurrency"], "status": m["status"],
               "host": m["hostname"].split(".")[0], "settings": json.dumps(m.get("backend_settings", {}), sort_keys=True),
               "server_command": m.get("server_command_str")}
        if m.get("reps"):
            s = m["reps"][0]["summary"]
            row.update({"total_tok_s": s["steady_total_tok_s"], "out_tok_s": s["steady_output_tok_s"],
                        "tpot_p50_ms": 1e3 * s["latency"]["tpot_s"]["p50"], "ttft_p50_s": s["latency"]["ttft_s"]["p50"],
                        "failed": s["failed"]})
        else:
            row["error"] = m.get("error") or m.get("skip_reason")
        rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "processed"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "tuning.csv", index=False)
    if df.empty:
        print("no tuning results")
        return 0
    pd.set_option("display.width", 200)
    print(df.drop(columns=["settings", "server_command"]).sort_values(["arm", "tag", "concurrency"]).to_string(index=False))
    for arm, g in df.groupby("arm"):
        c16 = g[(g["concurrency"] == 16) & g["total_tok_s"].notna()].set_index("tag")
        c1 = g[(g["concurrency"] == 1) & g["tpot_p50_ms"].notna()].set_index("tag")
        if c16.empty:
            continue
        best = c16["total_tok_s"].idxmax()
        default = DEFAULTS.get(arm)
        reason = f"highest C=16 throughput ({c16.loc[best, 'total_tok_s']:.0f} tok/s)"
        if default in c1.index and best in c1.index and c1.loc[best, "tpot_p50_ms"] > 1.05 * c1.loc[default, "tpot_p50_ms"]:
            reason = f"{best} would worsen C=1 TPOT >5%; keeping default"
            best = default
        print(f"\n{arm}: choose {best} -- {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
