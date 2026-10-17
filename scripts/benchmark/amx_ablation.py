#!/usr/bin/env python3
"""AMX ablations (review tasks T3.1 and T3.2). Settings: configs/experiment.toml [amx_ablation].

  vllm       one concurrency sweep of vLLM BF16 per (level, workload): `amx` (reference), `noamx_gemm` (oneDNN capped
             at AVX512_CORE_BF16), `noamx_all` (plus torch reporting no AMX, so vLLM's own attention and small-weight
             kernels take their AVX-512 paths). Campaign `amx-ablation`, arm directory `vllm-bf16__<level>`.
  verify     short points per level with ONEDNN_VERBOSE: what AMX_BUSY reads and which oneDNN implementations were
             created. Campaign `amx-ablation-verify`; writes amx_verify.json. Run before the sweeps.
  llamacpp   llama-bench prompt processing of Q8_0 with the AMX build and the AVX-512 build, order amx, avx512, amx,
             avx512 inside one job. Campaign `amx-build` (offline), arm directory `llamacpp-q8_0__<build>_p<pass>`.

  python scripts/benchmark/amx_ablation.py <command> --dry-run      # print every command line, start nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402

PY = sys.executable
SCRIPTS = bclib.SCRIPTS


def vllm_command(cfg: dict[str, Any], level: str, workload: str, *, verify: bool = False) -> list[str]:
    sec = cfg["amx_ablation"]["vllm"]
    env = dict(sec["levels"][level]["container_env"])
    cmd = [PY, str(SCRIPTS / "run_online.py"), "--model", sec["model"], "--arm", sec["arm"], "--workload", workload,
           "--tag", level]
    if verify:
        v = sec["verify"]
        env["ONEDNN_VERBOSE"] = v["onednn_verbose"]
        cmd += ["--campaign", v["campaign"], "--concurrency", ",".join(map(str, v["concurrency"])), "--reps", "1",
                "--num-requests", "8"]
    else:
        cmd += ["--campaign", sec["campaign"], "--concurrency", ",".join(map(str, sec["concurrency"]))]
    if env:
        cmd += ["--set", "container_env=" + json.dumps(env, sort_keys=True)]
    return cmd


def vllm_plan(cfg: dict[str, Any], levels: list[str] | None, workloads: list[str] | None) -> list[dict[str, Any]]:
    sec = cfg["amx_ablation"]["vllm"]
    levels = levels or list(sec["levels"])
    workloads = workloads or sec["workloads"]
    for lv in levels:
        if lv not in sec["levels"]:
            raise SystemExit(f"unknown level {lv!r}; available: {list(sec['levels'])}")
    return [{"level": lv, "workload": w, "cmd": vllm_command(cfg, lv, w)} for w in workloads for lv in levels]


def llamacpp_plan(cfg: dict[str, Any], passes: int | None = None) -> list[dict[str, Any]]:
    sec = cfg["amx_ablation"]["llamacpp"]
    runs = []
    for p in range(1, (passes or sec["passes"]) + 1):
        for build in sec["builds"]:
            tag = f"{build}_p{p}"
            cmd = [PY, str(SCRIPTS / "run_offline.py"), "--model", sec["model"], "--arm", sec["arm"],
                   "--campaign", sec["campaign"], "--workloads", ",".join(sec["workloads"]), "--build", build,
                   "--tag", tag, "--bench-tests", sec["tests"], "--batched-sizes", ""]
            runs.append({"build": build, "pass": p, "tag": tag, "cmd": cmd})
    return runs


# --------------------------------------------------------------------------- verification
ONEDNN_LINE = "onednn_verbose"


def onednn_histogram(log: Path) -> Counter:
    """Created primitives by (kind, implementation), from `onednn_verbose,v1,primitive,create:...,cpu,<kind>,<impl>,...`."""
    hist: Counter = Counter()
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith(ONEDNN_LINE) and ",create" in line:
            f = line.split(",")
            if len(f) > 6:
                hist[f"{f[5]} {f[6]}"] += 1
    return hist


def amx_busy(tel: Path, t0: float, t1: float) -> float | None:
    """EXE.AMX_BUSY ÷ cycles of the server's cores over the telemetry samples inside [t0, t1] (monotonic seconds)."""
    busy = cycles = 0.0
    for line in tel.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "sample" and t0 <= r["t_mono"] <= t1 and r.get("pmu"):
            busy += float(r["pmu"].get("exe_amx_busy", 0) or 0)
            cycles += float(r["pmu"].get("cycles", 0) or 0)
    return busy / cycles if cycles else None


def verify_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    sec = cfg["amx_ablation"]["vllm"]
    v = sec["verify"]
    base = Path(os.environ["BC_DATA"])
    out: dict[str, Any] = {"campaign": v["campaign"], "onednn_verbose": v["onednn_verbose"], "levels": {}}
    for level in sec["levels"]:
        arm_dir = f"{sec['arm']}__{level}"
        for c in v["concurrency"]:
            point = base / "raw_results" / "online" / v["campaign"] / sec["model"] / arm_dir / v["workload"] / f"c{c:03d}"
            meta_path = point / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            rep = next((p for p in meta.get("phases", []) if p["phase"] == "rep1"), None)
            tel = Path(meta.get("telemetry_file", ""))
            busy = amx_busy(tel, rep["t_start"], rep["t_end"]) if rep and tel.exists() else None
            log = point / "server.log"
            hist = onednn_histogram(log) if log.exists() else Counter()
            shim = "no_amx: torch reports no AMX" in log.read_text(errors="replace") if log.exists() else False
            out["levels"].setdefault(level, {})[f"c{c}"] = {
                "status": meta.get("status"), "amx_busy_frac": busy, "no_amx_shim_active": shim,
                "onednn_primitives": dict(hist.most_common()),
                "onednn_amx_primitives": sum(n for k, n in hist.items() if "amx" in k.lower())}
            if log.exists() and hist:   # keep the raw record small: the histogram replaces the verbose lines
                kept = [l for l in log.read_text(errors="replace").splitlines() if not l.startswith(ONEDNN_LINE)]
                log.write_text("\n".join(kept) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    for name in ("vllm", "verify", "llamacpp"):
        sp = sub.add_parser(name)
        sp.add_argument("--dry-run", action="store_true")
        if name != "llamacpp":
            sp.add_argument("--level", action="append", help="level (repeatable); default: all")
        if name == "vllm":
            sp.add_argument("--workload", action="append", help="workload (repeatable); default: the configured ones")
        if name == "llamacpp":
            sp.add_argument("--passes", type=int)
    args = ap.parse_args()
    cfg = bclib.load_config()

    if args.what == "vllm":
        runs = vllm_plan(cfg, args.level, args.workload)
    elif args.what == "verify":
        sec = cfg["amx_ablation"]["vllm"]
        runs = [{"level": lv, "workload": sec["verify"]["workload"], "cmd": vllm_command(cfg, lv, sec["verify"]["workload"], verify=True)}
                for lv in (args.level or list(sec["levels"]))]
    else:
        runs = llamacpp_plan(cfg, args.passes)

    for r in runs:
        label = " ".join(f"{k}={r[k]}" for k in ("level", "workload", "build", "pass") if k in r)
        print(f"# {args.what}: {label}\n{' '.join(r['cmd'])}", flush=True)
        if args.dry_run:
            subprocess.run([*r["cmd"], "--print-commands"], check=True)
            continue
        rc = subprocess.run(r["cmd"], check=False).returncode
        bclib.log(f"{args.what} {label}: exit {rc}")
    if args.dry_run:
        return 0
    if args.what == "verify":
        summary = verify_summary(cfg)
        dest = Path(os.environ["BC_DATA"]) / "raw_results" / "online" / summary["campaign"] / "amx_verify.json"
        bclib.write_json(dest, summary)
        for lv, pts in summary["levels"].items():
            for c, d in pts.items():
                busy = "n/a" if d["amx_busy_frac"] is None else f"{d['amx_busy_frac']:.4%}"
                bclib.log(f"{lv} {c}: AMX busy {busy}, oneDNN AMX primitives {d['onednn_amx_primitives']} of "
                          f"{sum(d['onednn_primitives'].values())}, no-AMX shim active {d['no_amx_shim_active']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
