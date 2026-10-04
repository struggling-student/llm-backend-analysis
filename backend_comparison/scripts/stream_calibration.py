#!/usr/bin/env python3
"""Experiment 0: STREAM triad under the inference telemetry.

Two purposes:
1. Re-measure sustainable memory bandwidth on exactly the layout the servers
   use (socket 0, 55 OpenMP threads on cores 0-54, membind node 0, cache mode),
   across working-set sizes that span "fits in HBM cache" to "spills to DDR".
2. Validate the core-PMU bandwidth proxy: STREAM's traffic is known, so the
   ratio (PMU-derived bytes) / (STREAM modeled bytes) shows whether the OCR
   events capture memory reads and writes, and with what factor.

Triad a[i] = b[i] + s*c[i] moves 3 x 8 B per element by STREAM's convention
(read b, read c, write a). Without non-temporal stores the write also causes a
read-for-ownership of a, so the hardware reads 3 x 8 B and writes 1 x 8 B.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    p = argparse.ArgumentParser()
    p.add_argument("--sizes-gib", default="4,8,16,32,48,64,96")
    p.add_argument("--threads", default="55,56")
    p.add_argument("--target-seconds", type=float, default=20)
    p.add_argument("--campaign", default="calibration")
    args = p.parse_args()
    cfg = bclib.load_config()
    topo = cfg["topology"]
    binary = os.environ["BC_STREAM_BIN"]
    out_dir = Path(os.environ["BC_DATA"]) / "raw_results" / "stream" / args.campaign
    tel_dir = Path(os.environ["BC_DATA"]) / "telemetry" / "stream" / args.campaign
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for threads in [int(t) for t in args.threads.split(",")]:
        cpus = f"0-{threads - 1}"
        for gib in [int(s) for s in args.sizes_gib.split(",")]:
            nbytes = gib * 2**30
            bw_guess = 500e9 if gib <= 32 else 200e9
            reps = max(5, math.ceil(args.target_seconds * bw_guess / nbytes))
            name = f"triad-t{threads}-{gib}gib"
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="close", OMP_PLACES="cores")
            cmd = ["taskset", "-c", cpus, "numactl", f"--membind={topo['server_mem_node']}", binary,
                   "--bytes", str(nbytes), "--warmups", "2", "--repetitions", str(reps)]
            tel = tel_dir / f"{name}.jsonl"
            t0 = time.monotonic()
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            telp = subprocess.Popen(["taskset", "-c", topo["client_cpus"], os.environ.get("BC_PYTHON", sys.executable),
                                     str(bclib.SCRIPTS / "telemetry.py"), "--root-pid", str(proc.pid), "--out", str(tel),
                                     "--interval", "0.5", "--cpus", topo["server_cpus"], "--uncore", "auto"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stdout, stderr = proc.communicate()
            t1 = time.monotonic()
            telp.send_signal(signal.SIGTERM)
            telp.wait(30)
            line = next((l for l in stdout.splitlines() if l.startswith("{")), None)
            res = json.loads(line) if line else None
            row = {"name": name, "threads": threads, "cpus": cpus, "working_set_gib": gib, "repetitions": reps,
                   "command": cmd, "returncode": proc.returncode, "t_start_mono": t0, "t_end_mono": t1,
                   "stream": res, "stderr": stderr[-2000:], "telemetry_file": str(tel)}
            if res:
                steady = res["steady_elapsed_seconds"]
                row["bandwidth_gbs_median"] = res["modeled_bytes_per_iteration"] / sorted(steady)[len(steady) // 2] / 1e9
                row["steady_seconds_total"] = sum(steady)
            bclib.log(f"{name}: {row.get('bandwidth_gbs_median', float('nan')):.1f} GB/s")
            results.append(row)
    bclib.write_json(out_dir / "stream_calibration.json", {"timestamp": bclib.now_iso(), **bclib.host_facts(),
                                                           "memory_mode": topo["memory_mode"], "binary": binary,
                                                           "runs": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
