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

import re
from pathlib import Path

import numpy as np
import pandas as pd

VARIANT = re.compile(r"^mem(?P<node>\d+)_p(?P<pass>\d+)$")
CELL = ["model", "configuration_id", "workload", "concurrency"]
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


def placement_table(raw_online: Path, campaign: str = "membind") -> pd.DataFrame:
    """Share of memory on the requested node for every point of a campaign, from the recorded numastat files."""
    rows = []
    for f in sorted((raw_online / campaign).glob("*/*/*/*/numastat.txt")):
        model, arm_dir, workload, point = f.parts[-5], f.parts[-4], f.parts[-3], f.parts[-2]
        arm, _, variant = arm_dir.partition("__")
        node, pass_no = parse_variant(variant)
        share = numastat_share(f, node)
        rows.append({"model": model, "arm": arm, "workload": workload, "concurrency": int(point[1:]), "mem_node": node,
                     "pass_no": pass_no, "share_on_requested_node": share,
                     "valid": share is not None and share >= PLACEMENT_MIN})
    return pd.DataFrame(rows)
