"""Contract of the processed tables: identity columns, key columns and consistency checks.

The full column reference is data/README.md; this module holds what is machine-checked
by `make validate` and the tests.
"""

from __future__ import annotations

import pandas as pd

from . import registry

CONFIGURATION_COLUMNS = ["configuration_id", "backend", "precision", "quantization"]

# table -> (columns that must exist, columns that identify a row uniquely)
TABLES: dict[str, tuple[list[str], list[str]]] = {
    "stream_calibration": (["campaign", "threads", "working_set_gib", "stream_triad_gbs", "mem_total_gbs"],
                           ["campaign", "threads", "working_set_gib"]),
    "node_checks": (["host", "slurm_job_id", "triad_gbs"], ["host", "slurm_job_id"]),
    "online_repetitions": ([*CONFIGURATION_COLUMNS, "campaign", "campaign_role", "model", "arm_dir", "workload",
                            "concurrency", "rep", "total_tok_s", "output_tok_s", "mem_total_gbs", "source"],
                           ["campaign", "model", "arm_dir", "workload", "concurrency", "rep"]),
    "online_points": ([*CONFIGURATION_COLUMNS, "campaign", "campaign_role", "model", "arm_dir", "variant", "workload",
                       "isl", "osl", "concurrency", "n_reps", "total_tok_s_mean", "total_tok_s_std",
                       "output_tok_s_mean", "tpot_s_p50_mean", "ttft_s_p50_mean", "mem_total_gbs_mean",
                       "bw_util_vs_stream_max_mean", "tokens_ok", "manifest_consistent", "failed_requests"],
                      ["campaign", "model", "arm_dir", "workload", "concurrency"]),
    "online_requests": (["campaign", "model", "configuration_id", "arm_dir", "workload", "concurrency", "rep",
                         "index", "steady", "ttft_s", "tpot_s", "e2e_s"],
                        ["campaign", "model", "arm_dir", "workload", "concurrency", "rep", "index"]),
    "online_sweeps": ([*CONFIGURATION_COLUMNS, "campaign", "model", "workload", "last_status"],
                      ["campaign", "model", "arm_dir", "workload"]),
    "offline_samples": ([*CONFIGURATION_COLUMNS, "campaign", "model", "workload", "tool", "benchmark", "batch",
                         "test", "sample", "tok_s"],
                        ["campaign", "model", "configuration_id", "workload", "tool", "benchmark", "batch", "test",
                         "sample"]),
    "tuning_runs": ([*CONFIGURATION_COLUMNS, "campaign", "variant", "concurrency", "total_tok_s"],
                    ["campaign", "model", "configuration_id", "variant", "workload", "concurrency"]),
    "pmu_overhead": ([*CONFIGURATION_COLUMNS, "campaign", "variant", "pmu_enabled", "total_tok_s"],
                     ["campaign", "configuration_id", "variant"]),
    "quality_check": ([*CONFIGURATION_COLUMNS, "model", "comparison", "amx_busy_frac"],
                      ["campaign", "model", "configuration_id", "comparison"]),
}


def check_table(name: str, df: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty if the table satisfies its contract)."""
    required, key = TABLES[name]
    problems = [f"{name}: missing column {c}" for c in required if c not in df]
    if problems:
        return problems
    dup = df.duplicated(key).sum()
    if dup:
        problems.append(f"{name}: {dup} duplicate rows for key {key}")
    if "configuration_id" in df:
        known = set(registry.configurations())
        unknown = sorted(set(df["configuration_id"].dropna()) - known)
        if unknown:
            problems.append(f"{name}: unknown configuration ids {unknown}")
        confs = registry.configurations()
        for col in ("backend", "precision", "quantization"):
            if col in df:
                bad = df[df["configuration_id"].map(lambda c, col=col: getattr(confs[c], col)) != df[col]]
                if len(bad):
                    problems.append(f"{name}: {len(bad)} rows whose {col} disagrees with configurations.toml")
    if "campaign_role" in df:
        roles = set(registry.campaigns()["role"])
        if not set(df["campaign_role"].dropna()) <= roles:
            problems.append(f"{name}: unknown campaign roles")
    return problems
