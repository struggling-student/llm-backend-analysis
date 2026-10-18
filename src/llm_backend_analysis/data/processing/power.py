"""Whole-node power with nothing running (scripts/benchmark/idle_power.py, `submit.sh idle-power`)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ... import paths
from .. import registry

COLUMNS = ["campaign", "campaign_role", "host", "slurm_job_id", "t_s", "energy_j", "current_watts", "cpu_busy_share",
           "loadavg_1m", "source"]


def process_idle_power() -> pd.DataFrame:
    """One row per sample: the job's consumed energy (sstat, J), the node's latest power reading (scontrol
    CurrentWatts, W) and the all-CPU busy share since the previous sample. Both readings come from the IPMI DCMI
    sensor of the whole node, which Slurm updates every 60 s."""
    rows: list[dict[str, Any]] = []
    for f in sorted((paths.RAW / "power").glob("*/idle-*.json")):
        r = json.loads(f.read_text())
        campaign = f.parent.name
        for s in r["samples"]:
            rows.append({"campaign": campaign, "campaign_role": registry.campaign_role("power", campaign),
                         "host": r["hostname"].split(".")[0], "slurm_job_id": str(r["slurm_job_id"]), "t_s": s["t_s"],
                         "energy_j": s["energy_j"], "current_watts": s["current_watts"],
                         "cpu_busy_share": s["cpu_busy_share"], "loadavg_1m": s["loadavg_1m"], "source": paths.relative(f)})
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["campaign", "host", "slurm_job_id", "t_s"], ignore_index=True)
