"""Runs that check the method rather than compare backends: the tuning grid, the PMU-overhead
A/B test and the INT8 quality/AMX sanity check."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ... import paths
from .. import registry


def process_tuning() -> pd.DataFrame:
    """One row per tuning point (campaigns with role `tuning` or a superseded tuning attempt)."""
    rows: list[dict[str, Any]] = []
    tuning = registry.campaigns()
    names = tuning[(tuning["kind"] == "online") & tuning["campaign"].str.startswith("tuning")]["campaign"]
    for campaign in names:
        for meta_path in sorted((paths.RAW / "online" / campaign).glob("*/*/*/c*/metadata.json")):
            m = json.loads(meta_path.read_text())
            model, arm_dir, workload = meta_path.parts[-5:-2]
            arm, variant = registry.split_arm_dir(arm_dir)
            row = {"campaign": campaign, "campaign_role": registry.campaign_role("online", campaign), "model": model,
                   **registry.configuration_columns(arm), "arm": arm, "variant": variant or "default",
                   "is_default_variant": variant == registry.configuration_for_arm(arm).tuning_default_variant,
                   "workload": workload, "concurrency": m["concurrency"], "status": m["status"],
                   "host": m["hostname"].split(".")[0],
                   "settings": json.dumps(m.get("backend_settings", {}), sort_keys=True),
                   "server_command": m.get("server_command_str"), "source": paths.relative(meta_path.parent)}
            if m.get("reps"):
                s = m["reps"][0]["summary"]
                row.update({"total_tok_s": s["steady_total_tok_s"], "output_tok_s": s["steady_output_tok_s"],
                            "tpot_s_p50": s["latency"]["tpot_s"]["p50"], "ttft_s_p50": s["latency"]["ttft_s"]["p50"],
                            "steady_window_s": s["steady_window_s"], "failed": s["failed"]})
            else:
                row["error"] = m.get("error") or m.get("skip_reason")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["campaign", "model", "configuration_id", "variant", "concurrency"],
                                          ignore_index=True)


def process_pmu_overhead(reps: pd.DataFrame) -> pd.DataFrame:
    """PMU on/off A/B/A/B runs: variant pmu<i>/nopmu<i> -> telemetry flag and run order."""
    d = reps[reps["campaign"].str.startswith("pmu-overhead")].copy()
    d["pmu_enabled"] = ~d["variant"].str.startswith("nopmu")
    d["run_order"] = d["variant"].str.extract(r"(\d+)$")[0].astype(int)
    cols = ["campaign", "campaign_role", "model", "configuration_id", "backend", "precision", "quantization", "variant", "run_order",
            "pmu_enabled", "workload", "concurrency", "host", "total_tok_s", "output_tok_s", "tpot_s_p50", "ttft_s_p50",
            "source"]
    return d[cols].sort_values(["campaign", "configuration_id", "run_order"], ignore_index=True)


def process_quality() -> pd.DataFrame:
    """INT8 quality/AMX sanity check: per configuration, agreement with a BF16 reference."""
    rows: list[dict[str, Any]] = []
    for f in sorted((paths.RAW / "quality").glob("*/*/quality.json")):
        q = json.loads(f.read_text())
        campaign, model = f.parts[-3], f.parts[-2]
        for arm, s in q["summary"].items():
            base = {"campaign": campaign, "campaign_role": registry.campaign_role("quality", campaign), "model": model,
                    **registry.configuration_columns(arm), "arm": arm, "host": q["hostname"].split(".")[0],
                    "cpu_model": q["cpu_model"], "prompts": len(q["prompts"]), "max_tokens": q["max_tokens"],
                    "amx_busy_frac": s.get("amx_busy_fraction"), "source": paths.relative(f)}
            comparisons = {k: v for k, v in s.items() if k.startswith("vs_")}
            if not comparisons:
                rows.append({**base, "comparison": "reference"})
            for name, v in comparisons.items():
                rows.append({**base, "comparison": name.removeprefix("vs_"),
                             "reference_configuration_id": registry.configuration_for_arm(v["reference"]).configuration_id,
                             "identical_continuations": v["identical_continuations"],
                             "mean_common_prefix_words": v["mean_common_prefix_words"],
                             "min_common_prefix_words": v["min_common_prefix_words"]})
    return pd.DataFrame(rows)
