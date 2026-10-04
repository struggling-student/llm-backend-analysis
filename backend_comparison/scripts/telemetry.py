#!/usr/bin/env python3
"""Timestamped hardware/software telemetry for one server process tree.

Every --interval seconds, one JSON line with deltas over the interval:

- Core PMU counters summed over every thread of the server process tree
  (perf_event_open per thread, user space only, which kernel.perf_event_paranoid=2
  allows without privileges). Threads are rediscovered every interval, so
  threads created after the start are picked up.
    cycles, instructions
    exe_amx_busy              EXE.AMX_BUSY                    cycles the AMX unit was busy
    ocr_rd_l3miss_local       OCR.READS_TO_CORE.L3_MISS_LOCAL  data/code reads + RFOs (demand and L1/L2
                                                              prefetch) served beyond the local L3
    ocr_wr_est_memory         OCR.WRITE_ESTIMATE.MEMORY       stores that likely resulted in a memory write
  Encodings are from Intel's perfmon tables for Sapphire Rapids
  (github.com/intel/perfmon, SPR/events/sapphirerapids_core.json). Every count
  is in 64-byte lines. In cache mode the memory behind the L3 is "HBM cache in
  front of DDR", so these counters measure traffic to the memory side as a
  whole: they cannot say which part hit in HBM and which went to DDR.
  The two OCR events use the two offcore-response MSRs, so no event is
  time-multiplexed; values are still scaled by time_enabled/time_running and
  the running fraction is recorded as a check.

- Uncore counters (DDR IMC CAS, HBM MCHBM CAS, M2M near-memory tag hit/miss) if
  the kernel allows system-wide events (perf_event_paranoid <= 0 or
  CAP_PERFMON). On cresco8 at paranoid=2 they are refused; the refusal is
  recorded and the run continues.

- /proc/stat per-CPU busy fraction for the given CPUs.
- Resident memory of the process tree, and per-NUMA-node used memory.
- The server's Prometheus /metrics (selected vllm:* / llamacpp:* series).

Stop with SIGTERM or SIGINT; the last partial interval is flushed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import signal
import struct
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

PERF_TYPE_HARDWARE = 0
PERF_TYPE_RAW = 4
FMT_ENABLED = 1 << 0
FMT_RUNNING = 1 << 1
FLAG_EXCLUDE_KERNEL = 1 << 5
FLAG_EXCLUDE_HV = 1 << 6
SYS_PERF_EVENT_OPEN = 298  # x86_64

# name -> (type, config, config1)
# Five events, two offcore-response MSRs (0x2A -> OFFCORE_RSP_0, 0x2B -> _1), so
# nothing is time-multiplexed. A first version also counted
# OCR.HWPF_L3.L3_MISS_LOCAL and LONGEST_LAT_CACHE.MISS: the former read exactly 0
# in STREAM and in every server run, the latter matched ocr_rd_l3miss_local within
# 0.2%, and the resulting multiplexing slowed vLLM decode by ~4.5% (llama.cpp
# ~1.3%). See report, "Telemetry overhead".
CORE_EVENTS: dict[str, tuple[int, int, int]] = {
    "cycles": (PERF_TYPE_HARDWARE, 0, 0),
    "instructions": (PERF_TYPE_HARDWARE, 1, 0),
    "exe_amx_busy": (PERF_TYPE_RAW, 0x02B7, 0),
    "ocr_rd_l3miss_local": (PERF_TYPE_RAW, 0x012A, 0x3F04C04477),
    "ocr_wr_est_memory": (PERF_TYPE_RAW, 0x012B, 0xFBFF80822),
}

# Uncore (system-wide, one CPU per socket). PMU type ids come from sysfs.
# uncore_type_14_* is how kernel 5.14 exposes the SPR HBM controllers (MCHBM);
# the mapping is inferred from the box count (32 per socket) and recorded as such.
UNCORE_EVENTS: dict[str, tuple[str, int]] = {
    "ddr_cas_rd": ("uncore_imc_", 0xCF05),
    "ddr_cas_wr": ("uncore_imc_", 0xF005),
    "hbm_cas_rd": ("uncore_type_14_", 0xCF05),
    "hbm_cas_wr": ("uncore_type_14_", 0xF005),
    "m2m_nm_rd_hit": ("uncore_m2m_", 0x031F),
    "m2m_tag_miss": ("uncore_m2m_", 0x034B),
}


class PerfEventAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32), ("size", ctypes.c_uint32), ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64), ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64), ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32), ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64), ("config2", ctypes.c_uint64),
    ]


_libc = ctypes.CDLL(None, use_errno=True)


def perf_open(etype: int, config: int, config1: int, pid: int, cpu: int, user_only: bool) -> int:
    attr = PerfEventAttr()
    attr.type, attr.size, attr.config, attr.config1 = etype, ctypes.sizeof(PerfEventAttr), config, config1
    attr.read_format = FMT_ENABLED | FMT_RUNNING
    attr.flags = (FLAG_EXCLUDE_KERNEL | FLAG_EXCLUDE_HV) if user_only else 0
    fd = _libc.syscall(SYS_PERF_EVENT_OPEN, ctypes.byref(attr), pid, cpu, -1, 0)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def perf_read(fd: int) -> tuple[int, int, int]:
    return struct.unpack("QQQ", os.read(fd, 24))


def process_tree(root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(d))
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(children.get(p, []))
    return out


def tids_of(pid: int) -> list[int]:
    try:
        return [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    except OSError:
        return []


def rss_bytes(pids: list[int]) -> int:
    total = 0
    for p in pids:
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except OSError:
            pass
    return total


def node_used_bytes() -> dict[str, int]:
    out = {}
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        vals = {}
        for line in (node / "meminfo").read_text().splitlines():
            parts = line.split()
            vals[parts[2].rstrip(":")] = int(parts[3]) * 1024
        out[node.name] = vals.get("MemTotal", 0) - vals.get("MemFree", 0)
    return out


def cpu_times(cpus: set[int]) -> dict[int, tuple[int, int]]:
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            parts = line.split()
            c = int(parts[0][3:])
            if c in cpus:
                v = list(map(int, parts[1:]))
                idle = v[3] + v[4]
                out[c] = (sum(v[:8]) - idle, sum(v[:8]))
    return out


METRIC_PREFIXES = (
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:iteration_tokens_total_sum", "vllm:iteration_tokens_total_count", "vllm:num_preemptions_total",
    "llamacpp:",
)


def scrape(url: str | None) -> dict[str, float] | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=0.5) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(METRIC_PREFIXES):
            continue
        name, _, value = line.rpartition(" ")
        name = name.split("{", 1)[0]
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return out


class CoreCounters:
    def __init__(self, events: dict[str, tuple[int, int, int]]):
        self.events = events
        self.fds: dict[int, dict[str, int]] = {}
        self.last: dict[tuple[int, str], tuple[int, int, int]] = {}
        self.errors: dict[str, str] = {}

    def attach(self, tids: list[int]) -> None:
        for tid in tids:
            if tid in self.fds:
                continue
            fds = {}
            for name, (t, c, c1) in self.events.items():
                try:
                    fds[name] = perf_open(t, c, c1, tid, -1, True)
                except OSError as exc:
                    self.errors.setdefault(name, str(exc))
            self.fds[tid] = fds

    def sample(self) -> tuple[dict[str, float], dict[str, float], int]:
        """Return (scaled deltas, min running fraction per event, live threads)."""
        delta = {n: 0.0 for n in self.events}
        run_frac: dict[str, float] = {}
        dead = []
        for tid, fds in self.fds.items():
            for name, fd in fds.items():
                try:
                    v, en, ru = perf_read(fd)
                except OSError:
                    dead.append(tid)
                    break
                pv, pen, pru = self.last.get((tid, name), (0, 0, 0))
                self.last[(tid, name)] = (v, en, ru)
                d_en, d_ru = en - pen, ru - pru
                if d_ru > 0:
                    delta[name] += (v - pv) * (d_en / d_ru)
                    run_frac[name] = min(run_frac.get(name, 1.0), d_ru / d_en)
        for tid in set(dead):
            for fd in self.fds.pop(tid, {}).values():
                os.close(fd)
        # A thread that exited keeps no fd entry; drop stale state.
        return delta, run_frac, len(self.fds)


class UncoreCounters:
    def __init__(self, cpu: int):
        self.fds: dict[str, list[int]] = {}
        self.last: dict[tuple[str, int], int] = {}
        self.status = "unavailable"
        self.error: str | None = None
        base = Path("/sys/bus/event_source/devices")
        for name, (prefix, config) in UNCORE_EVENTS.items():
            for dev in sorted(base.glob(prefix + "[0-9]*")):
                try:
                    t = int((dev / "type").read_text())
                    fd = perf_open(t, config, 0, -1, cpu, False)
                except (OSError, ValueError) as exc:
                    self.error = f"{dev.name}: {exc}"
                    continue
                self.fds.setdefault(name, []).append(fd)
        if self.fds:
            self.status = "available"

    def sample(self) -> dict[str, float]:
        out = {}
        for name, fds in self.fds.items():
            total = 0
            for fd in fds:
                v, _, _ = perf_read(fd)
                total += v - self.last.get((name, fd), 0)
                self.last[(name, fd)] = v
            out[name] = float(total)
        return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-pid", type=int, required=True, help="server process (its whole tree is measured)")
    p.add_argument("--out", required=True, help="JSONL output")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--cpus", default="0-55", help="CPUs whose utilization is sampled")
    p.add_argument("--metrics-url")
    p.add_argument("--no-pmu", action="store_true")
    p.add_argument("--uncore", choices=["auto", "off"], default="auto")
    p.add_argument("--uncore-cpu", type=int, default=0, help="a CPU of the measured socket")
    args = p.parse_args()

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    cpus: list[int] = []
    for part in args.cpus.split(","):
        a, _, b = part.partition("-")
        cpus.extend(range(int(a), int(b or a) + 1))
    cpuset = set(cpus)

    core = None if args.no_pmu else CoreCounters(CORE_EVENTS)
    uncore = UncoreCounters(args.uncore_cpu) if args.uncore == "auto" else None

    stopping = False

    def stop(*_: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    out = open(args.out, "a", buffering=1)
    header = {
        "type": "header", "t_mono": time.monotonic(), "t_epoch": time.time(), "root_pid": args.root_pid,
        "interval_s": args.interval, "cpus": args.cpus, "core_events": {k: [hex(v[1]), hex(v[2])] for k, v in CORE_EVENTS.items()},
        "uncore_status": uncore.status if uncore else "off", "uncore_error": uncore.error if uncore else None,
        "uncore_events": list(uncore.fds) if uncore else [],
        "perf_event_paranoid": Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip(),
        "line_bytes": 64,
    }
    out.write(json.dumps(header) + "\n")

    if core:
        core.attach([t for p_ in process_tree(args.root_pid) for t in tids_of(p_)])
        core.sample()
    if uncore and uncore.fds:
        uncore.sample()
    prev_cpu = cpu_times(cpuset)
    t_prev = time.monotonic()
    next_t = t_prev + args.interval
    while True:
        while not stopping and time.monotonic() < next_t:
            time.sleep(min(0.05, max(0.0, next_t - time.monotonic())))
        next_t += args.interval
        now = time.monotonic()
        alive = os.path.exists(f"/proc/{args.root_pid}")
        pids = process_tree(args.root_pid) if alive else []
        rec: dict[str, Any] = {"type": "sample", "t_mono": now, "t_epoch": time.time(), "dt": now - t_prev}
        if core:
            delta, frac, nthreads = core.sample()
            rec["pmu"] = delta
            rec["pmu_running_fraction"] = frac
            rec["threads"] = nthreads
            core.attach([t for p_ in pids for t in tids_of(p_)])
        if uncore and uncore.fds:
            rec["uncore"] = uncore.sample()
        cur_cpu = cpu_times(cpuset)
        busy = {}
        for c in cpus:
            if c in cur_cpu and c in prev_cpu:
                db, dt = cur_cpu[c][0] - prev_cpu[c][0], cur_cpu[c][1] - prev_cpu[c][1]
                busy[c] = round(db / dt, 4) if dt > 0 else 0.0
        rec["cpu_busy"] = busy
        prev_cpu = cur_cpu
        rec["rss_bytes"] = rss_bytes(pids)
        rec["node_used_bytes"] = node_used_bytes()
        m = scrape(args.metrics_url)
        if m is not None:
            rec["metrics"] = m
        out.write(json.dumps(rec, separators=(",", ":")) + "\n")
        t_prev = now
        if stopping or not alive:
            break
    out.write(json.dumps({"type": "footer", "t_mono": time.monotonic(), "core_errors": core.errors if core else None}) + "\n")
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
