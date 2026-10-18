#!/usr/bin/env python3
"""Idle power of one exclusive node: the baseline that splits the whole-node energy of notebook 13.

The node runs nothing after the job.sbatch checks (node check, environment manifest); this script only sleeps and
every `--interval` seconds records the job's consumed energy (sstat ConsumedEnergyRaw of the batch step, the same
counter run_online.py reads per phase), the node's latest power reading (scontrol CurrentWatts) and the busy share
of all CPUs since the previous sample (/proc/stat), which shows that the node was idle. Slurm reads the IPMI DCMI
sensor (whole node: both sockets, memory, platform) every 60 s, so energy differences carry steps of one update.

The record is rewritten after every sample, so a job cut short keeps what it measured:
$BC_DATA/raw_results/power/<campaign>/idle-<host>-<job>.json
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def cpu_times() -> tuple[int, int]:
    """(busy, total) jiffies of all CPUs; iowait and idle count as idle."""
    v = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = v[3] + v[4]
    total = sum(v[:8])   # guest time is already contained in user and nice
    return total - idle, total


def current_watts(host: str) -> float | None:
    m = re.search(r"CurrentWatts=(\d+)", bclib.run_capture(["scontrol", "show", "node", host], timeout=20))
    return float(m.group(1)) if m else None


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=1200, help="idle seconds to record")
    p.add_argument("--interval", type=float, default=30, help="seconds between samples (the sensor updates every 60 s)")
    p.add_argument("--campaign", default=f"idle-{time.strftime('%Y%m%d')}")
    args = p.parse_args()
    host = os.uname().nodename.split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    out = Path(os.environ["BC_DATA"]) / "raw_results" / "power" / args.campaign / f"idle-{host}-{job}.json"
    rec = {"timestamp": bclib.now_iso(), **bclib.host_facts(), "campaign": args.campaign, "duration_s": args.duration,
           "interval_s": args.interval, "energy_source": "sstat -j <job>.batch ConsumedEnergyRaw (J), IPMI DCMI, whole node",
           "power_source": "scontrol show node CurrentWatts (W), latest IPMI reading", "samples": []}
    bclib.log(f"idle power on {host}: {args.duration:.0f} s, a sample every {args.interval:.0f} s -> {out}")
    t0 = time.monotonic()
    busy0, total0 = None, None
    while True:
        busy, total = cpu_times()
        s = {"t_s": round(time.monotonic() - t0, 3), "time": bclib.now_iso(), "energy_j": bclib.slurm_energy_j(),
             "current_watts": current_watts(host),
             "cpu_busy_share": (busy - busy0) / (total - total0) if total0 is not None and total > total0 else None,
             "loadavg_1m": float(Path("/proc/loadavg").read_text().split()[0])}
        busy0, total0 = busy, total
        rec["samples"].append(s)
        bclib.write_json(out, rec)
        bclib.log(f"t={s['t_s']:7.1f} s  energy {s['energy_j']} J  current {s['current_watts']} W  "
                  f"cpu busy {s['cpu_busy_share'] if s['cpu_busy_share'] is None else round(s['cpu_busy_share'], 4)}")
        if s["t_s"] >= args.duration:
            break
        time.sleep(max(0.0, args.interval - (time.monotonic() - t0 - s["t_s"])))
    e = [x for x in rec["samples"] if x["energy_j"] is not None]
    if len(e) >= 2 and e[-1]["t_s"] > e[0]["t_s"]:
        bclib.log(f"average node power {(e[-1]['energy_j'] - e[0]['energy_j']) / (e[-1]['t_s'] - e[0]['t_s']):.0f} W "
                  f"over {e[-1]['t_s'] - e[0]['t_s']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
