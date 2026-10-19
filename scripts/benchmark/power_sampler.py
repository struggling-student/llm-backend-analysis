#!/usr/bin/env python3
"""Power meters of one node: RAPL through turbostat and the BMC through ipmitool, into one JSONL file.

Needs passwordless sudo for `turbostat` and `ipmitool sensor` (granted on cresco8-hbm14 only, 2026-10-19; `perf stat`
is granted without arguments only, so it cannot read the RAPL events).

  rapl  every --interval s: turbostat package rows (PkgWatt, RAMWatt, PkgTmp, UncMHz, Busy% of the one CPU read per
        package). PkgWatt is the RAPL package domain: cores, uncore and, on Xeon Max, the on-package HBM. RAMWatt is the
        DRAM domain (DDR5 DIMMs). Each value is the average over the interval that ends at the row's time.
  ipmi  back to back (one read takes ~6 s): every numeric "Watts" sensor of `ipmitool sensor` (Sys Power, CPU Power,
        Mem Power, PSU Power, ...). The BMC reports them in 5-10 W steps; the reading's time is not known more
        precisely than the read window [t_start, t_end].

turbostat reads its MSRs only on the CPUs of --cpus (one per package, outside the server's compute cores), so the
server cores are not interrupted. Times are CLOCK_MONOTONIC seconds, the clock of the harness phases and telemetry.

  power_sampler.py --out meters.jsonl [--duration 900]     # until --duration or SIGTERM
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

RAPL_COLUMNS = "Time_Of_Day_Seconds,Package,CPU,Busy%,PkgTmp,PkgWatt,RAMWatt,UncMHz"
STOP = threading.Event()
LOCK = threading.Lock()


def _terminate(signum, _frame):
    STOP.set()


def write(f, rec: dict) -> None:
    with LOCK:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()


def num(v: str) -> float | str | None:
    try:
        return float(v)
    except ValueError:
        return None if v in ("", "-") else v


def rapl_loop(f, cpus: str, interval: float, offset: float, errors: list) -> None:
    cmd = ["sudo", "-n", "turbostat", "--quiet", "--cpu", cpus, "--show", RAPL_COLUMNS, "--interval", str(interval)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    header, rows = None, []

    def flush_block():
        if not rows:
            return
        total = next((r for r in rows if r["Package"] == "-"), None)
        pkgs = {str(int(r["Package"])): r for r in rows if r["Package"] != "-"}
        t_wall = max(float(r["Time_Of_Day_Seconds"]) for r in rows)
        write(f, {"src": "rapl", "t_mono": round(t_wall - offset, 3), "t_wall": t_wall, "interval_s": interval,
                  "pkg_w": {p: num(r["PkgWatt"]) for p, r in pkgs.items()},
                  "ram_w": {p: num(r["RAMWatt"]) for p, r in pkgs.items()},
                  "pkg_tmp_c": {p: num(r["PkgTmp"]) for p, r in pkgs.items()},
                  "unc_mhz": {p: num(r["UncMHz"]) for p, r in pkgs.items()},
                  "read_cpu_busy_pct": {p: num(r["Busy%"]) for p, r in pkgs.items()},
                  "pkg_w_total": num(total["PkgWatt"]) if total else None,
                  "ram_w_total": num(total["RAMWatt"]) if total else None})
        rows.clear()

    try:
        for line in proc.stdout:
            if STOP.is_set():
                break
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0] == "Time_Of_Day_Seconds":
                flush_block()
                header = parts
            elif header and len(parts) == len(header):
                rows.append(dict(zip(header, parts)))
            elif line.strip():
                errors.append(f"turbostat: {line.strip()}")
        flush_block()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if proc.returncode not in (0, None, -15, 143) and not STOP.is_set():
        errors.append(f"turbostat exited with {proc.returncode}")


def ipmi_loop(f, errors: list) -> None:
    pat = re.compile(r"^([^|]+?)\s*\|\s*([-0-9.]+)\s*\|\s*Watts\s*\|")
    while not STOP.is_set():
        t0 = time.monotonic()
        try:
            out = subprocess.run(["sudo", "-n", "ipmitool", "sensor"], capture_output=True, text=True, timeout=60).stdout
        except subprocess.TimeoutExpired:
            errors.append("ipmitool sensor timed out")
            continue
        t1 = time.monotonic()
        watts = {m.group(1).strip(): float(m.group(2)) for m in map(pat.match, out.splitlines()) if m}
        if not watts:
            errors.append("ipmitool sensor returned no Watts sensors")
            STOP.wait(5)
            continue
        write(f, {"src": "ipmi", "t_mono_start": round(t0, 3), "t_mono_end": round(t1, 3), "watts": watts})


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, help="seconds to record (default: until SIGTERM)")
    p.add_argument("--cpus", default="55,111", help="one CPU per package for turbostat's MSR reads")
    p.add_argument("--interval", type=float, default=1.0, help="RAPL interval (s)")
    p.add_argument("--no-ipmi", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    offset = time.time() - time.monotonic()
    errors: list[str] = []
    with open(out, "a") as f:
        write(f, {"src": "start", "t_mono": round(time.monotonic(), 3), "wall_minus_mono_s": offset, "cpus": args.cpus,
                  "rapl_interval_s": args.interval, "ipmi": not args.no_ipmi, "rapl_columns": RAPL_COLUMNS})
        threads = [threading.Thread(target=rapl_loop, args=(f, args.cpus, args.interval, offset, errors), daemon=True)]
        if not args.no_ipmi:
            threads.append(threading.Thread(target=ipmi_loop, args=(f, errors), daemon=True))
        for t in threads:
            t.start()
        t_end = time.monotonic() + args.duration if args.duration else None
        while not STOP.is_set() and (t_end is None or time.monotonic() < t_end):
            STOP.wait(0.5)
        STOP.set()
        for t in threads:
            t.join(70)
        write(f, {"src": "stop", "t_mono": round(time.monotonic(), 3), "errors": errors[:50]})
    for e in errors[:5]:
        print(e, file=sys.stderr)
    return 1 if any("exited" in e or "no Watts" in e for e in errors) else 0


if __name__ == "__main__":
    sys.exit(main())
