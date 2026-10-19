#!/usr/bin/env python3
"""Power of vLLM BF16 on the flat-mode node cresco8-hbm14: RAPL and IPMI, server memory on HBM vs DDR.

One job (configs/experiment.toml [power_flat]):
  1. checks: flat mode (memory-only NUMA nodes), passwordless sudo for turbostat and ipmitool, the NUMA shim;
  2. idle baseline: power_sampler.py for idle_start_s with nothing running;
  3. for each placement (HBM node 2, then DDR node 0): run_online.py --mem-node <n> --tag mem<n> --power, the fixed
     concurrency list, the primary repetitions; numastat must show >= 90% of the server's memory on that node;
  4. closing idle (idle_end_s), a drift check of the idle baseline.
Meters files: $BC_DATA/raw_results/power/<campaign>/meters-idle-{start,end}-<host>-<job>.jsonl (idle) and
$BC_DATA/telemetry/online/<campaign>/.../c<C>.power.jsonl (per point); run summary run-<host>-<job>.json.

  python scripts/benchmark/power_flat.py --dry-run     # every command line (any machine)
  python scripts/benchmark/power_flat.py               # inside the Slurm job on cresco8-hbm14
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402
from membind_sensitivity import LOCAL_SHARE_MIN, numastat_share  # noqa: E402


def plan(cfg: dict) -> list[dict]:
    sec = cfg["power_flat"]
    runs = []
    for node in sec["placements"]:
        cmd = [sys.executable, str(bclib.SCRIPTS / "run_online.py"), "--model", sec["model"], "--arm", sec["arm"],
               "--workload", sec["workload"], "--campaign", sec["campaign"],
               "--concurrency", ",".join(str(c) for c in sec["concurrency"]), "--mem-node", str(node), "--tag", f"mem{node}",
               "--power", "--power-cpus", sec["power_cpus"], "--memory-mode", "flat"]
        runs.append({"node": node, "tag": f"mem{node}", "cmd": cmd})
    return runs


def sampler(sec: dict, out: Path, duration: float) -> list[str]:
    return ["taskset", "-c", bclib.load_config()["topology"]["client_cpus"], sys.executable,
            str(bclib.SCRIPTS / "power_sampler.py"), "--out", str(out), "--duration", str(duration), "--cpus", sec["power_cpus"]]


def preflight() -> list[str]:
    problems = []
    if os.environ.get("BC_MEMORY_MODE") != "flat":
        problems.append(f"node is not in flat mode (BC_MEMORY_MODE={os.environ.get('BC_MEMORY_MODE')})")
    for cmd in (["sudo", "-n", "turbostat", "--version"], ["sudo", "-n", "ipmitool", "sensor"]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            problems.append(f"{' '.join(cmd)} failed: {(r.stderr or r.stdout).strip()[:200]}")
    shim = Path(bclib.load_config()["topology"]["numa_shim"])
    if not shim.is_file():
        problems.append(f"{shim} missing (submit.sh build-shim)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = bclib.load_config()
    sec = cfg["power_flat"]
    runs = plan(cfg)
    host = os.uname().nodename.split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    base = Path(os.environ.get("BC_DATA", "/tmp/bc-data"))
    pdir = base / "raw_results" / "power" / sec["campaign"]
    idle = {k: pdir / f"meters-idle-{k}-{host}-{job}.jsonl" for k in ("start", "end")}
    if args.dry_run:
        print("# idle start\n" + " ".join(sampler(sec, idle["start"], sec["idle_start_s"])))
        for r in runs:
            print(f"# placement memory node {r['node']}\n{' '.join(r['cmd'])}", flush=True)
            subprocess.run([*r["cmd"], "--print-commands"], check=True)
        print("# idle end\n" + " ".join(sampler(sec, idle["end"], sec["idle_end_s"])))
        return 0
    problems = preflight()
    summary = {"timestamp": bclib.now_iso(), **bclib.host_facts(), "campaign": sec["campaign"], "config": sec,
               "preflight_problems": problems, "phases": [], "runs": []}
    out = pdir / f"run-{host}-{job}.json"
    if problems:
        bclib.write_json(out, summary)
        for p in problems:
            bclib.log(f"PREFLIGHT: {p}")
        return 2

    def idle_phase(name: str, duration: float) -> None:
        t0 = bclib.mono()
        e0 = bclib.slurm_energy_j()
        bclib.log(f"idle {name}: {duration:.0f} s")
        rc = subprocess.run(sampler(sec, idle[name], duration)).returncode
        summary["phases"].append({"phase": f"idle-{name}", "t_start": t0, "t_end": bclib.mono(), "energy_j_start": e0,
                                  "energy_j_end": bclib.slurm_energy_j(), "sampler_rc": rc, "file": str(idle[name])})
        bclib.write_json(out, summary)

    idle_phase("start", sec["idle_start_s"])
    problems_n = 0
    for r in runs:
        bclib.log(f"placement: server memory on node {r['node']}")
        t0 = bclib.mono()
        rc = subprocess.run(r["cmd"]).returncode
        arm_dir = f"{sec['arm']}__{r['tag']}"
        points = []
        for c in sec["concurrency"]:
            pd = base / "raw_results" / "online" / sec["campaign"] / sec["model"] / arm_dir / sec["workload"] / f"c{c:03d}"
            meta = json.loads((pd / "metadata.json").read_text()) if (pd / "metadata.json").exists() else {}
            share = numastat_share(pd / "numastat.txt", r["node"])
            ok = str(meta.get("status", "")).startswith("ok") and share is not None and share >= LOCAL_SHARE_MIN
            problems_n += not ok
            points.append({"concurrency": c, "status": meta.get("status"), "share_on_node": share, "ok": ok})
            bclib.log(f"  C={c}: status {meta.get('status')}, server memory on node {r['node']}: "
                      f"{'n/a' if share is None else f'{share:.1%}'}")
        summary["runs"].append({"node": r["node"], "tag": r["tag"], "rc": rc, "t_start": t0, "t_end": bclib.mono(),
                                "command": r["cmd"], "points": points})
        bclib.write_json(out, summary)
    idle_phase("end", sec["idle_end_s"])
    return 1 if problems_n else 0


if __name__ == "__main__":
    sys.exit(main())
