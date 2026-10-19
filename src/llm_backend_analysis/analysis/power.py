"""RAPL and BMC power of vLLM BF16 on the flat-mode node cresco8-hbm14, server memory on HBM vs DDR (notebook 14).

Inputs are the processed `power_phases` table (one row per idle window and per repetition window) and the online points
of the same campaign. Power domains:

- `pkg0` RAPL package 0, the server socket: cores, uncore and the on-package HBM (no separate HBM domain exists);
- `ram0` RAPL DRAM of socket 0: the DDR5 attached to the server socket;
- `pkg1 + ram1` the client socket (load client, telemetry, meters);
- `sys` the BMC's Sys Power (whole node); the BMC's CPU and Mem Power are the RAPL sums, so `platform` = sys − (CPU + Mem)
  is the part outside both sockets and their DRAM (fans, voltage regulators, PSU losses, NIC, disks).

Idle levels are the settled part of the job's opening idle window; "above idle" subtracts them domain by domain.
"""

from __future__ import annotations

import pandas as pd

CAMPAIGN = "power-flat-20261019"
PLACEMENTS = {"vllm-bf16__mem2": "HBM (node 2)", "vllm-bf16__mem0": "DDR (node 0)"}
IDLE_PHASE = "idle-start-settled"
DOMAINS = {"pkg0": "rapl_pkg0_w", "ram0": "rapl_ram0_w", "pkg1": "rapl_pkg1_w", "ram1": "rapl_ram1_w", "sys": "ipmi_sys_w",
           "ipmi_cpu": "ipmi_cpu_w", "ipmi_mem": "ipmi_mem_w", "unc_mhz": "rapl_unc0_mhz", "pkg0_tmp_c": "rapl_pkg0_tmp_c"}
POWER_LIMIT_W = 350.0     # Xeon CPU Max 9480 TDP = package power limit PL1


def _derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["server_w"] = df["pkg0"] + df["ram0"]
    df["client_w"] = df["pkg1"] + df["ram1"]
    df["platform_w"] = df["sys"] - df["ipmi_cpu"] - df["ipmi_mem"]
    return df


def idle_levels(phases: pd.DataFrame, phase: str = IDLE_PHASE) -> pd.Series:
    """Mean power per domain over one idle window (default: the settled part of the opening idle)."""
    row = phases[(phases["window"] == "idle") & (phases["phase"] == phase)].iloc[0]
    return _derived(pd.DataFrame([{k: row[v] for k, v in DOMAINS.items()}])).iloc[0]


def point_power(phases: pd.DataFrame, points: pd.DataFrame, window: str = "steady") -> pd.DataFrame:
    """One row per (placement, C): duration-weighted means over the repetitions' `window`, idle-subtracted levels and
    J per output token at four boundaries (server socket, server socket above idle, node, node above idle)."""
    rows = []
    idle = idle_levels(phases)
    for (arm_dir, c), g in phases[phases["window"] == window].groupby(["arm_dir", "concurrency"]):
        w = g["duration_s"]
        row = {"arm_dir": arm_dir, "placement": PLACEMENTS.get(arm_dir, arm_dir), "concurrency": int(c),
               "duration_s": w.sum(), "ipmi_reads": int(g["ipmi_n"].sum())}
        for k, col in DOMAINS.items():
            v = g[col]
            row[k] = (v * w).sum() / w[v.notna()].sum() if v.notna().any() else float("nan")
        rows.append(row)
    df = _derived(pd.DataFrame(rows))
    p = points.set_index(["arm_dir", "concurrency"])
    df["output_tok_s"] = [p.loc[(a, c), "output_tok_s_mean"] for a, c in zip(df["arm_dir"], df["concurrency"])]
    df["ghz_core"] = [p.loc[(a, c), "ghz_effective_per_core_mean"] for a, c in zip(df["arm_dir"], df["concurrency"])]
    df["server_above_idle_w"] = df["server_w"] - idle["server_w"]
    df["client_above_idle_w"] = df["client_w"] - idle["client_w"]
    df["platform_above_idle_w"] = df["platform_w"] - idle["platform_w"]
    df["sys_above_idle_w"] = df["sys"] - idle["sys"]
    for name, col in (("server", "server_w"), ("server_above_idle", "server_above_idle_w"), ("node", "sys"),
                      ("node_above_idle", "sys_above_idle_w")):
        df[f"j_per_token_{name}"] = df[col] / df["output_tok_s"]
    order = list(PLACEMENTS)
    return df.sort_values(["arm_dir", "concurrency"], key=lambda s: s.map(order.index) if s.name == "arm_dir" else s,
                          ignore_index=True)


def meter_agreement(phases: pd.DataFrame, min_reads: int = 3) -> pd.DataFrame:
    """Per window with at least `min_reads` BMC reads: BMC CPU and Mem Power against the RAPL sums of both sockets."""
    d = phases[phases["ipmi_n"] >= min_reads].copy()
    d["rapl_cpu_w"] = d["rapl_pkg0_w"] + d["rapl_pkg1_w"]
    d["rapl_mem_w"] = d["rapl_ram0_w"] + d["rapl_ram1_w"]
    d["cpu_diff_w"] = d["ipmi_cpu_w"] - d["rapl_cpu_w"]
    d["mem_diff_w"] = d["ipmi_mem_w"] - d["rapl_mem_w"]
    return d


def hbm_vs_ddr(pp: pd.DataFrame) -> pd.DataFrame:
    """Per C: HBM ÷ DDR throughput, J/token and clock; HBM − DDR power per domain."""
    h = pp[pp["arm_dir"] == "vllm-bf16__mem2"].set_index("concurrency")
    d = pp[pp["arm_dir"] == "vllm-bf16__mem0"].set_index("concurrency")
    out = pd.DataFrame({"throughput_ratio": h["output_tok_s"] / d["output_tok_s"],
                        "server_j_ratio": h["j_per_token_server"] / d["j_per_token_server"],
                        "node_j_ratio": h["j_per_token_node"] / d["j_per_token_node"],
                        "ghz_hbm": h["ghz_core"], "ghz_ddr": d["ghz_core"]})
    for k in ("pkg0", "ram0", "client_w", "platform_w", "sys", "unc_mhz"):
        out[f"delta_{k}"] = h[k] - d[k]
    return out.reset_index()
