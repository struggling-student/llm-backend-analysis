"""Memory-node sensitivity (the review's Experiment A): the same server, cores on socket 0, memory on node 0
(reference) or node 1 (remote, over the socket interconnect).

Arm directories of campaign `membind` are `<arm>__mem<node>_p<pass>`; processing stores the suffix as `variant`.
Each cell ran node 0, node 1, node 0, node 1 inside one job, so the ratio of a pass pair is formed on one node and
one day. Remote memory lowers the bandwidth AND raises the latency: a throughput ratio below 1 is sensitivity to
the memory system, not to bandwidth alone.

The core PMU traffic proxy counts local reads only (`OCR.READS_TO_CORE.L3_MISS_LOCAL`); under remote memory it reads
zero (STREAM check, `stream/calibration-membind1`). Traffic of remote cells is therefore *modelled* from the
throughput here (`implied_bandwidth_gbs`), never taken from `mem_*_gbs`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

VARIANT = re.compile(r"^mem(?P<node>\d+)_p(?P<pass>\d+)$")
CELL = ["model", "configuration_id", "workload", "concurrency"]
CAMPAIGN = "membind-v2"   # the corrected run (image LD_PRELOAD kept); `membind` is the first attempt, confounded
PLACEMENT_MIN = 0.90   # share of the server's memory that must sit on the requested node


def parse_variant(variant: str) -> tuple[int, int]:
    m = VARIANT.match(variant)
    if not m:
        raise ValueError(f"not a memory-node variant: {variant!r}")
    return int(m["node"]), int(m["pass"])


def with_node(points: pd.DataFrame) -> pd.DataFrame:
    """Add `mem_node` and `pass_no` from the variant of the campaign's points."""
    out = points.copy()
    parsed = out["variant"].map(parse_variant)
    out["mem_node"] = [p[0] for p in parsed]
    out["pass_no"] = [p[1] for p in parsed]
    return out


def paired_ratios(points: pd.DataFrame, reference: int = 0, remote: int = 1) -> pd.DataFrame:
    """One row per cell and pass: remote ÷ reference for throughput, and reference ÷ remote for TPOT and TTFT
    (so every column reads 'fraction of the reference performance')."""
    p = with_node(points)
    metrics = {"total_tok_s_mean": "throughput", "output_tok_s_mean": "output_throughput"}
    inverse = {"tpot_s_p50_mean": "tpot", "ttft_s_p50_mean": "ttft"}
    ref = p[p["mem_node"] == reference].set_index(CELL + ["pass_no"])
    rem = p[p["mem_node"] == remote].set_index(CELL + ["pass_no"])
    both = ref.index.intersection(rem.index)
    out = pd.DataFrame(index=both)
    for col, name in metrics.items():
        out[name] = rem.loc[both, col] / ref.loc[both, col]
    for col, name in inverse.items():
        out[name] = ref.loc[both, col] / rem.loc[both, col]
    out["reference_total_tok_s"] = ref.loc[both, "total_tok_s_mean"]
    out["remote_total_tok_s"] = rem.loc[both, "total_tok_s_mean"]
    out["reference_tpot_s"] = ref.loc[both, "tpot_s_p50_mean"]
    out["remote_tpot_s"] = rem.loc[both, "tpot_s_p50_mean"]
    return out.reset_index()


def cell_summary(points: pd.DataFrame) -> pd.DataFrame:
    """Per cell: mean ratio over the pass pairs, its range, and the within-node repeatability of the passes."""
    pr = paired_ratios(points)
    rows = []
    for key, g in pr.groupby(CELL, sort=False):
        p = with_node(points)
        pc = p[(p[CELL] == pd.Series(dict(zip(CELL, key)))).all(axis=1)]
        cv = {n: pc[pc["mem_node"] == n]["total_tok_s_mean"].pipe(lambda s: s.std(ddof=1) / s.mean() if len(s) > 1 else np.nan)
              for n in (0, 1)}
        rows.append({**dict(zip(CELL, key)), "passes": len(g), "throughput_ratio": g["throughput"].mean(),
                     "throughput_ratio_min": g["throughput"].min(), "throughput_ratio_max": g["throughput"].max(),
                     "tpot_ratio": g["tpot"].mean(), "ttft_ratio": g["ttft"].mean(),
                     "reference_total_tok_s": g["reference_total_tok_s"].mean(), "remote_total_tok_s": g["remote_total_tok_s"].mean(),
                     "reference_tpot_s": g["reference_tpot_s"].mean(), "remote_tpot_s": g["remote_tpot_s"].mean(),
                     "pass_cv_reference": cv[0], "pass_cv_remote": cv[1]})
    return pd.DataFrame(rows)


def implied_bandwidth_gbs(bytes_per_step: float, step_time_s: float | pd.Series) -> float | pd.Series:
    """Bandwidth a decode step needs if it streams `bytes_per_step` once per step (C = 1: weights + KV per token)."""
    return bytes_per_step / step_time_s / 1e9


def remote_roof_gbs(stream: pd.DataFrame, working_set_gib: float, campaign: str = "calibration-membind1",
                    threads: int = 55) -> float:
    """STREAM triad bandwidth with remote memory at a working set, log-linear interpolation between the measured sizes."""
    s = stream[(stream["campaign"] == campaign) & (stream["threads"] == threads)].sort_values("working_set_gib")
    if s.empty:
        raise ValueError(f"no STREAM rows for campaign {campaign!r}, {threads} threads")
    return float(np.interp(np.log(working_set_gib), np.log(s["working_set_gib"]), s["stream_triad_gbs"]))


def numastat_share(path: Path, node: int) -> float | None:
    """Share of the server's resident memory on `node`, from the `Total` rows of the `numastat -p` tables."""
    if not Path(path).exists():
        return None
    per_node: list[float] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("Total"):
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
            if len(nums) >= 3:
                per_node = [a + b for a, b in zip(per_node or [0.0] * (len(nums) - 1), nums[:-1])]
    total = sum(per_node)
    return per_node[node] / total if total and node < len(per_node) else None


def placement_table(raw_online: Path, campaign: str = CAMPAIGN) -> pd.DataFrame:
    """Share of memory on the requested node for every point of a campaign, from the recorded numastat files."""
    rows = []
    for f in sorted((raw_online / campaign).glob("*/*/*/*/numastat.txt")):
        model, arm_dir, workload, point = f.parts[-5], f.parts[-4], f.parts[-3], f.parts[-2]
        arm, _, variant = arm_dir.partition("__")
        if variant:
            node, pass_no = parse_variant(variant)
        else:   # untagged point (the shim check): the node is in the recorded numa policy
            meta = json.loads((f.parent / "metadata.json").read_text())
            node, pass_no = int(re.search(r"membind=(\d+)", meta["numa_policy"]).group(1)), 0
        share = numastat_share(f, node)
        rows.append({"model": model, "arm": arm, "workload": workload, "concurrency": int(point[1:]), "mem_node": node,
                     "pass_no": pass_no, "share_on_requested_node": share,
                     "valid": share is not None and share >= PLACEMENT_MIN})
    return pd.DataFrame(rows)


def sensitivity_table(points: pd.DataFrame, stream: pd.DataFrame, reference: int = 0, remote: int = 1) -> pd.DataFrame:
    """One row per cell: what remote memory does to the server, and how much of the remote bandwidth that needs.

    Bytes per step come from the *reference* runs (local memory, where the PMU proxy is validated) and are assumed
    unchanged with remote memory: the same tokens per step are processed (`tokens_per_step` columns show it) and the
    weights and KV read per step do not depend on where they sit. The remote traffic is then modelled as
    bytes-per-step × remote steps/s and compared with the remote STREAM roof at the cell's working set."""
    p = with_node(points)
    keys = CELL
    agg = {"total_tok_s_mean": "mean", "tpot_s_p50_mean": "mean", "ttft_s_p50_mean": "mean", "engine_steps_per_s_mean": "mean",
           "tokens_per_step_mean": "mean", "mem_total_gbs_mean": "mean", "working_set_gib_model": "mean",
           "cpu_busy_compute_mean_mean": "mean"}
    ref = p[p["mem_node"] == reference].groupby(keys, sort=False).agg(agg)
    rem = p[p["mem_node"] == remote].groupby(keys, sort=False).agg(agg)
    both = ref.index.intersection(rem.index)
    ref, rem = ref.loc[both], rem.loc[both]
    out = pd.DataFrame(index=both)
    out["working_set_gib"] = ref["working_set_gib_model"]
    out["reference_tok_s"], out["remote_tok_s"] = ref["total_tok_s_mean"], rem["total_tok_s_mean"]
    out["throughput_ratio"] = rem["total_tok_s_mean"] / ref["total_tok_s_mean"]
    out["tpot_ratio_remote_over_reference"] = rem["tpot_s_p50_mean"] / ref["tpot_s_p50_mean"]
    out["ttft_ratio_remote_over_reference"] = rem["ttft_s_p50_mean"] / ref["ttft_s_p50_mean"]
    out["step_rate_ratio"] = rem["engine_steps_per_s_mean"] / ref["engine_steps_per_s_mean"]
    out["tokens_per_step_reference"], out["tokens_per_step_remote"] = ref["tokens_per_step_mean"], rem["tokens_per_step_mean"]
    out["compute_core_busy_reference"], out["compute_core_busy_remote"] = (ref["cpu_busy_compute_mean_mean"],
                                                                         rem["cpu_busy_compute_mean_mean"])
    out["bytes_per_step_gb"] = ref["mem_total_gbs_mean"] / ref["engine_steps_per_s_mean"]
    out["reference_traffic_gbs"] = ref["mem_total_gbs_mean"]
    out["implied_remote_traffic_gbs"] = out["bytes_per_step_gb"] * rem["engine_steps_per_s_mean"]
    out["reference_roof_gbs"] = [remote_roof_gbs(stream, w, campaign="calibration") for w in out["working_set_gib"]]
    out["remote_roof_gbs"] = [remote_roof_gbs(stream, w) for w in out["working_set_gib"]]
    out["roof_ratio"] = out["remote_roof_gbs"] / out["reference_roof_gbs"]
    # Upper bound on the throughput ratio if only the remote bandwidth limited the server: it needs the reference
    # traffic and the remote roof is lower, so it cannot do better than roof ÷ traffic.
    out["bandwidth_only_ratio_bound"] = np.minimum(1.0, out["remote_roof_gbs"] / out["reference_traffic_gbs"])
    out["reference_roof_share"] = out["reference_traffic_gbs"] / out["reference_roof_gbs"]
    out["implied_remote_roof_share"] = out["implied_remote_traffic_gbs"] / out["remote_roof_gbs"]
    return out.reset_index()
