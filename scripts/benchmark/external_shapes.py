#!/usr/bin/env python3
"""External-reference shapes (notebook 13): our server at exactly the shape and concurrencies of a published result.

Each cell of configs/experiment.toml [external_shapes] is one run_online.py sweep with a fixed concurrency list (no
adaptive extension or plateau stop), the primary layout and repetitions, in the campaign `external-shapes`. One Slurm
job per (arm, cell); `scripts/cluster/submit.sh external-shapes [arm ...]` submits them.

  python scripts/benchmark/external_shapes.py --jobs                 # arm, workload, time limit per job (for submit.sh)
  python scripts/benchmark/external_shapes.py --dry-run              # every server and client command line
  python scripts/benchmark/external_shapes.py --arm vllm-bf16 --workload ext1000   # run one cell (inside a Slurm job)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402


def plan(cfg: dict[str, Any], arms: list[str] | None, workloads: list[str] | None) -> list[dict[str, Any]]:
    sec = cfg["external_shapes"]
    allowed = sec["arms"] + sec.get("optional_arms", [])
    arms = arms or sec["arms"]
    unknown = set(arms) - set(allowed)
    if unknown:
        raise SystemExit(f"arm(s) {sorted(unknown)} not in [external_shapes]; allowed: {allowed}")
    cells = [c for c in sec["cells"] if not workloads or c["workload"] in workloads]
    runs = []
    for arm in arms:
        for c in cells:
            cmd = [sys.executable, str(bclib.SCRIPTS / "run_online.py"), "--model", sec["model"], "--arm", arm,
                   "--workload", c["workload"], "--campaign", sec["campaign"],
                   "--concurrency", ",".join(str(x) for x in c["concurrency"])]
            runs.append({"arm": arm, "workload": c["workload"], "time_limit": c["time_limit"], "cmd": cmd})
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", help="arm (repeatable); default: [external_shapes].arms")
    ap.add_argument("--workload", action="append", help="cell workload (repeatable); default: every cell")
    ap.add_argument("--jobs", action="store_true", help="print 'arm workload time_limit' per job and exit")
    ap.add_argument("--dry-run", action="store_true", help="print each run's command and its server/client commands")
    args = ap.parse_args()
    runs = plan(bclib.load_config(), args.arm, args.workload)
    if args.jobs:
        print("\n".join(f"{r['arm']} {r['workload']} {r['time_limit']}" for r in runs))
        return 0
    rc = 0
    for r in runs:
        print(f"# {r['arm']} {r['workload']}\n{' '.join(r['cmd'])}", flush=True)
        if args.dry_run:
            subprocess.run([*r["cmd"], "--print-commands"], check=True)
        else:
            rc |= subprocess.run(r["cmd"], check=False).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
