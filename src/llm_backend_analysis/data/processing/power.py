"""Node power: the idle baseline of the Slurm IPMI counter (scripts/benchmark/idle_power.py, `submit.sh idle-power`) and
the RAPL/BMC meters of the flat-mode power campaign (scripts/benchmark/power_sampler.py, power_flat.py)."""

from __future__ import annotations

import json
from pathlib import Path
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


# --------------------------------------------------------------------------- RAPL and BMC meters (power_flat.py)
PHASE_COLUMNS = ["campaign", "campaign_role", "host", "slurm_job_id", "model", "configuration_id", "arm_dir", "variant",
                 "workload", "concurrency", "phase", "window", "t_start", "t_end", "duration_s",
                 "rapl_pkg0_w", "rapl_pkg1_w", "rapl_ram0_w", "rapl_ram1_w", "rapl_unc0_mhz", "rapl_pkg0_tmp_c", "rapl_n",
                 "ipmi_sys_w", "ipmi_cpu_w", "ipmi_mem_w", "ipmi_psu_w", "ipmi_n", "slurm_node_w", "output_tok_s",
                 "total_tok_s", "source"]
IPMI_SENSORS = {"ipmi_sys_w": "Sys Power", "ipmi_cpu_w": "CPU Power", "ipmi_mem_w": "Mem Power", "ipmi_psu_w": "PSU Power"}
IDLE_SETTLE_S = 300.0   # the first minutes after the job's STREAM node check still carry its heat and uncore clock


def read_meters(path) -> tuple[list[dict], list[dict]]:
    recs = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in recs if r["src"] == "rapl"], [r for r in recs if r["src"] == "ipmi"]


def meter_means(rapl: list[dict], ipmi: list[dict], t0: float, t1: float) -> dict[str, Any]:
    """Means over [t0, t1]: RAPL intervals that lie inside the window, BMC reads whose whole read lies inside it."""
    r = [x for x in rapl if x["t_mono"] - x["interval_s"] >= t0 - 1e-6 and x["t_mono"] <= t1 + 1e-6]
    q = [x for x in ipmi if x["t_mono_start"] >= t0 and x["t_mono_end"] <= t1]

    def mean(vals):
        v = [x for x in vals if x is not None]
        return sum(v) / len(v) if v else None

    out = {"rapl_pkg0_w": mean([x["pkg_w"].get("0") for x in r]), "rapl_pkg1_w": mean([x["pkg_w"].get("1") for x in r]),
           "rapl_ram0_w": mean([x["ram_w"].get("0") for x in r]), "rapl_ram1_w": mean([x["ram_w"].get("1") for x in r]),
           "rapl_unc0_mhz": mean([x["unc_mhz"].get("0") for x in r]), "rapl_pkg0_tmp_c": mean([x["pkg_tmp_c"].get("0") for x in r]),
           "rapl_n": len(r), "ipmi_n": len(q)}
    for col, sensor in IPMI_SENSORS.items():
        out[col] = mean([x["watts"].get(sensor) for x in q])
    return out


def _slurm_power(p: dict) -> float | None:
    e0, e1 = p.get("energy_j_start"), p.get("energy_j_end")
    d = p["t_end"] - p["t_start"]
    return (e1 - e0) / d if e0 is not None and e1 is not None and d > 0 and e1 > e0 else None


def process_power_phases() -> pd.DataFrame:
    """One row per measured window of the power-meter campaigns (scripts/benchmark/power_flat.py):

    - idle windows of the job (`idle-start`, its settled part `idle-start-settled` after IDLE_SETTLE_S, `idle-end`);
    - every repetition of every online point recorded with `run_online.py --power`, twice: over the whole repetition
      (`window` = rep) and over its steady window (`window` = steady), with that repetition's steady tok/s.
    RAPL means are over the 1 s turbostat intervals inside the window (package 0 = the server socket; the package domain
    includes the on-package HBM, the RAM domain the DDR5); BMC means are over the ipmitool reads inside the window."""
    rows: list[dict[str, Any]] = []
    for f in sorted((paths.RAW / "power").glob("*/run-*.json")):
        run = json.loads(f.read_text())
        campaign = f.parent.name
        base = {"campaign": campaign, "campaign_role": registry.campaign_role("power", campaign),
                "host": run["hostname"].split(".")[0], "slurm_job_id": str(run["slurm_job_id"])}
        for p in run["phases"]:
            meters = f.parent / Path(p["file"]).name
            if not meters.exists():
                continue
            rapl, ipmi = read_meters(meters)
            windows = [(p["phase"], p["t_start"], p["t_end"])]
            if p["phase"] == "idle-start":
                windows.append(("idle-start-settled", p["t_start"] + IDLE_SETTLE_S, p["t_end"]))
            for name, t0, t1 in windows:
                rows.append({**base, "phase": name, "window": "idle", "t_start": t0, "t_end": t1, "duration_s": t1 - t0,
                             **meter_means(rapl, ipmi, t0, t1), "slurm_node_w": _slurm_power(p) if name == p["phase"] else None,
                             "source": paths.relative(meters)})
    for meters in sorted((paths.RAW_TELEMETRY / "online").glob("*/*/*/*/c*.power.jsonl")):
        campaign, model, arm_dir, workload = meters.parts[-5:-1]
        point = paths.RAW / "online" / campaign / model / arm_dir / workload / meters.name.replace(".power.jsonl", "")
        if not (point / "metadata.json").exists():
            continue
        meta = json.loads((point / "metadata.json").read_text())
        if not str(meta.get("status", "")).startswith("ok"):
            continue
        rapl, ipmi = read_meters(meters)
        arm, variant = registry.split_arm_dir(arm_dir)
        base = {"campaign": campaign, "campaign_role": registry.campaign_role("online", campaign),
                "host": meta["hostname"].split(".")[0], "slurm_job_id": str(meta.get("slurm_job_id")), "model": model,
                "configuration_id": registry.configuration_for_arm(arm).configuration_id, "arm_dir": arm_dir,
                "variant": variant, "workload": workload, "concurrency": int(meta["concurrency"]), "source": paths.relative(meters)}
        phases = {p["phase"]: p for p in meta.get("phases", [])}
        for rep in meta.get("reps", []):
            p = phases.get(f"rep{rep['rep']}")
            if p is None:
                continue
            s, win = rep["summary"], rep["summary"]["window"]
            spans = [("rep", p["t_start"], p["t_end"])]
            if win.get("start") and win.get("end"):
                spans.append(("steady", win["start"], win["end"]))
            for name, t0, t1 in spans:
                rows.append({**base, "phase": p["phase"], "window": name, "t_start": t0, "t_end": t1, "duration_s": t1 - t0,
                             **meter_means(rapl, ipmi, t0, t1), "slurm_node_w": _slurm_power(p) if name == "rep" else None,
                             "output_tok_s": s["steady_output_tok_s"], "total_tok_s": s["steady_total_tok_s"]})
    if not rows:
        return pd.DataFrame(columns=PHASE_COLUMNS)
    df = pd.DataFrame(rows).reindex(columns=PHASE_COLUMNS)
    return df.sort_values(["campaign", "arm_dir", "concurrency", "phase", "window"], na_position="first", ignore_index=True)
