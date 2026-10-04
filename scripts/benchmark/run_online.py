#!/usr/bin/env python3
"""Experiment B: online serving sweep for one (model, arm, workload).

For every concurrency C:
  1. start a fresh server (vllm serve or llama-server) on socket 0,
  2. start the telemetry sampler on the server's process tree,
  3. idle baseline, then warm-up requests (not measured),
  4. R measured repetitions of a closed-loop run with C workers (bench_client.py),
  5. stop telemetry and server; write metadata and summaries.

Sweep rule (declared in configs/experiment.toml [online], applied identically to
every arm): run the base concurrency list; stop early once two consecutive
doublings each add less than `plateau_gain` throughput; continue past the base
list only while the last doubling still added at least `extend_min_gain`. A
point whose projected duration exceeds `max_point_seconds` is skipped and the
skip is recorded. Results are resumable: finished points are not rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402

SCRIPTS = bclib.SCRIPTS


def server_command(cfg: dict[str, Any], model_key: str, arm_key: str, wl: dict[str, Any], c: int,
                   settings: dict[str, Any], port: int) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    topo, model, arm = cfg["topology"], cfg["models"][model_key], cfg["arms"][arm_key]
    margin = cfg["online"]["ctx_margin"]
    seq_len = wl["isl"] + wl["osl"] + margin
    prefix = ["taskset", "-c", topo["server_cpus"], "numactl", f"--membind={topo['server_mem_node']}"]
    env = dict(os.environ)
    info: dict[str, Any] = {"seq_len": seq_len}
    if arm["backend"] == "vllm":
        block = 128
        kv_bytes = c * bclib.round_up(seq_len, block) * model["kv_bytes_per_token"] * settings["kv_space_headroom"]
        kv_gib = max(1, math.ceil(kv_bytes / 2**30))
        cenv = {
            "VLLM_CPU_KVCACHE_SPACE": str(kv_gib),
            "VLLM_CPU_OMP_THREADS_BIND": topo["compute_cpus"],
            "VLLM_CPU_NUM_OF_RESERVED_CPU": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        cmd = prefix + ["apptainer", "exec", "--cleanenv"]
        for k, v in cenv.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += ["--bind", f"{os.environ['BC_STORE']}:{os.environ['BC_STORE']}", "--bind", f"{os.environ['BC_SCRATCH']}:{os.environ['BC_SCRATCH']}",
                *(["--bind", f"{os.environ['BC_LOCAL']}:{os.environ['BC_LOCAL']}"] if os.environ.get("BC_LOCAL") else []), settings["image"],
                "vllm", "serve", model["snapshot"],
                "--served-model-name", model["served_name"],
                "--host", "127.0.0.1", "--port", str(port),
                "--dtype", settings["dtype"],
                "--max-model-len", str(seq_len),
                "--max-num-seqs", str(max(settings["max_num_seqs"], c)),
                "--max-num-batched-tokens", str(settings["max_num_batched_tokens"]),
                "--no-enable-prefix-caching",
                "--generation-config", "vllm",
                "--seed", "0",
                "--disable-uvicorn-access-log",
                *settings.get("extra_args", [])]
        info.update({"kv_cache_gib": kv_gib, "container_env": cenv})
    else:
        ctx_slot = bclib.round_up(seq_len, 256)
        bin_dir = settings["bin_dir_amx"] if arm.get("build", "amx") == "amx" else settings["bin_dir_avx512"]
        gguf = str(Path(model["gguf_dir"]) / model["gguf"][arm["gguf_variant"]])
        kv_type = settings.get("cache_type", arm["kv_cache_dtype"])
        env.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores", "OMP_NUM_THREADS": str(topo["compute_threads"])})
        cmd = prefix + [f"{bin_dir}/llama-server", "-m", gguf, "-a", model["served_name"],
                        "--host", "127.0.0.1", "--port", str(port),
                        "-t", str(topo["compute_threads"]), "-tb", str(topo["compute_threads"]),
                        "-c", str(c * ctx_slot), "-np", str(c), "-cb", "-no-kvu",
                        "-b", str(settings["batch_size"]), "-ub", str(settings["ubatch_size"]),
                        "-fa", str(settings["flash_attn"]), "-ctk", kv_type, "-ctv", kv_type,
                        "--no-cache-prompt", "-cram", "0", "-lm", settings["load_mode"],
                        "--poll", str(settings["poll"]), "--metrics", "--slots", "--no-context-shift",
                        *settings.get("extra_args", [])]
        info.update({"ctx_per_slot": ctx_slot, "ctx_total": c * ctx_slot, "gguf": gguf, "kv_cache_dtype": kv_type,
                     "omp_env": {k: env[k] for k in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_NUM_THREADS")}})
    return cmd, env, info


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def stop_group(proc: subprocess.Popen, grace: float = 60) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(30)


def client_cmd(cfg: dict[str, Any], model_key: str, arm_key: str, wl: dict[str, Any], c: int, port: int,
               extra: list[str]) -> list[str]:
    topo, model, arm = cfg["topology"], cfg["models"][model_key], cfg["arms"][arm_key]
    return ["taskset", "-c", topo["client_cpus"], "numactl", f"--membind={topo['client_mem_node']}",
            os.environ.get("BC_PYTHON", sys.executable), str(SCRIPTS / "bench_client.py"),
            "--url", f"http://127.0.0.1:{port}", "--model", model["served_name"], "--backend", arm["backend"],
            "--isl", str(wl["isl"]), "--osl", str(wl["osl"]), "--concurrency", str(c),
            "--seed", str(cfg["online"]["seed"]), "--bos", str(model["bos_token_id"]),
            "--vocab", str(model["ordinary_vocab"]), *extra]


def run_json(cmd: list[str], log_path: Path) -> int:
    with open(log_path, "a") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def idle_core_busy(tel_path: Path, t0: float, t1: float, cpus: str) -> float | None:
    """Mean busy fraction of the compute cores over telemetry samples inside [t0, t1]."""
    want = {str(c) for c in bclib.cpu_list(cpus)}
    vals = []
    try:
        for line in tel_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("type") == "sample" and t0 + 1.5 <= rec["t_mono"] <= t1 + 0.1:
                b = [v for k, v in rec["cpu_busy"].items() if k in want]
                if b:
                    vals.append(sum(b) / len(b))
    except (OSError, ValueError):
        return None
    return sum(vals) / len(vals) if vals else None


def requests_for(cfg: dict[str, Any], c: int, override: int | None) -> tuple[int, int]:
    o = cfg["online"]
    waves = o["waves_small_c"] if c <= 16 else o["waves_large_c"]
    n = override or max(o["min_requests"], waves * c)
    return bclib.round_up(n, c), waves   # every worker owns the same number of requests


def run_point(cfg: dict[str, Any], args: argparse.Namespace, wl_key: str, c: int, settings: dict[str, Any],
              point_dir: Path, tel_path: Path, prev: dict[str, Any] | None) -> dict[str, Any]:
    o, topo, arm = cfg["online"], cfg["topology"], cfg["arms"][args.arm]
    wl = cfg["workloads"][wl_key]
    backend = arm["backend"]
    port = o["port_vllm"] if backend == "vllm" else o["port_llamacpp"]
    reps = args.reps or o["repetitions"]
    n_req, waves = requests_for(cfg, c, args.num_requests)
    point_dir.mkdir(parents=True, exist_ok=True)
    tel_path.parent.mkdir(parents=True, exist_ok=True)
    clog = point_dir / "client.log"

    for _ in range(90):
        if bclib.port_free(port):
            break
        time.sleep(1)
    cmd, env, sinfo = server_command(cfg, args.model, args.arm, wl, c, settings, port)
    meta: dict[str, Any] = {
        "experiment": cfg["experiment"], "campaign": args.campaign, "kind": "online",
        "timestamp": bclib.now_iso(), **bclib.host_facts(),
        "model": args.model, "model_id": cfg["models"][args.model]["hf_id"],
        "model_revision": cfg["models"][args.model]["revision"],
        "arm": args.arm, "backend": backend, "precision": arm["precision"], "weights": arm["weights"],
        "workload": wl_key, "isl": wl["isl"], "osl": wl["osl"], "concurrency": c,
        "num_requests_per_rep": n_req, "waves": waves, "repetitions": reps,
        "memory_mode": topo["memory_mode"], "sockets_used": 1, "socket": topo["socket"],
        "server_cpus": topo["server_cpus"], "compute_cpus": topo["compute_cpus"],
        "compute_threads": topo["compute_threads"], "numa_policy": f"membind={topo['server_mem_node']}",
        "client_cpus": topo["client_cpus"], "backend_settings": settings, "server_info": sinfo,
        "server_command": cmd, "server_command_str": shlex.join(cmd),
        "sampling": {"temperature": 0.0, "top_p": 1.0, "ignore_eos": True, "max_tokens": wl["osl"]},
        "prefix_caching": False, "telemetry_file": str(tel_path), "status": "running",
        "model_staging": args.staging,
    }
    bclib.write_json(point_dir / "metadata.json", meta)
    bclib.log(f"start server C={c}: {shlex.join(cmd)}")
    t_launch = time.monotonic()
    server = subprocess.Popen(cmd, env=env, stdout=open(point_dir / "server.log", "w"), stderr=subprocess.STDOUT,
                              start_new_session=True)
    tel = None
    try:
        health = f"http://127.0.0.1:{port}/health"
        while not http_ok(health):
            if server.poll() is not None:
                raise RuntimeError(f"server exited with {server.returncode} during startup (see server.log)")
            if time.monotonic() - t_launch > o["startup_timeout_s"]:
                raise RuntimeError("server startup timeout")
            time.sleep(2)
        meta["server_startup_s"] = time.monotonic() - t_launch
        bclib.log(f"server ready after {meta['server_startup_s']:.0f}s")
        # Where does the server's memory live? (weights + KV must be on node 0)
        with open(point_dir / "numastat.txt", "w") as f:
            for pid in sorted(set(subprocess.run(["pgrep", "-g", str(os.getpgid(server.pid))],
                                                 capture_output=True, text=True).stdout.split())):
                f.write(bclib.run_capture(f"numastat -p {pid}") + "\n\n")

        tel_cmd = ["taskset", "-c", topo["client_cpus"], os.environ.get("BC_PYTHON", sys.executable),
                   str(SCRIPTS / "telemetry.py"), "--root-pid", str(server.pid), "--out", str(tel_path),
                   "--interval", str(cfg["telemetry"]["interval_s"]), "--cpus", topo["server_cpus"],
                   "--metrics-url", f"http://127.0.0.1:{port}/metrics", "--uncore", cfg["telemetry"]["uncore"],
                   "--uncore-cpu", str(bclib.cpu_list(topo["compute_cpus"])[0])]
        if args.no_pmu or not cfg["telemetry"]["pmu"]:
            tel_cmd.append("--no-pmu")
        tel = subprocess.Popen(tel_cmd, stdout=open(point_dir / "telemetry.log", "w"), stderr=subprocess.STDOUT)
        meta["telemetry_command"] = tel_cmd
        phases: list[dict[str, Any]] = [{"phase": "idle", "t_start": time.monotonic()}]
        time.sleep(cfg["telemetry"]["idle_baseline_s"])
        phases[-1]["t_end"] = time.monotonic()
        idle_busy = idle_core_busy(tel_path, phases[-1]["t_start"], phases[-1]["t_end"], topo["compute_cpus"])
        meta["idle_compute_core_busy"] = idle_busy
        if idle_busy is not None and idle_busy > 0.10:
            raise RuntimeError(f"compute cores {idle_busy:.0%} busy while the server is idle: another process is "
                               "running on the server cores; the point would be contaminated")

        n_warm = max(2, min(c, 16))
        wsum = point_dir / "warmup.json"
        wcmd = client_cmd(cfg, args.model, args.arm, wl, c, port,
                          ["--warmup", str(n_warm), "--warmup-osl", str(o["warmup_osl"]), "--num-requests", "0",
                           "--summary", str(wsum)])
        phases.append({"phase": "warmup", "t_start": time.monotonic()})
        rc = run_json(wcmd, clog)
        phases[-1]["t_end"] = time.monotonic()
        warm = json.loads(wsum.read_text())
        if rc != 0 or warm["warmup"]["failed"]:
            raise RuntimeError(f"warm-up failed: {warm['warmup'].get('errors')}")
        est = warm["warmup"]["estimated_request_s"]
        if prev and prev.get("e2e_mean_s"):
            est = max(est, prev["e2e_mean_s"])
        projected = (est * prev["concurrency_scale"](c) if prev else est) * (waves + 1) * reps
        meta.update({"warmup_requests": n_warm, "estimated_request_s": est, "projected_point_s": projected,
                     "warmup_command": wcmd})
        if projected > o["max_point_seconds"] and not args.ignore_budget:
            meta["status"] = "skipped_budget"
            meta["skip_reason"] = f"projected {projected:.0f}s > max_point_seconds {o['max_point_seconds']}"
            bclib.log(meta["skip_reason"])
            return meta

        rep_summaries = []
        for r in range(1, reps + 1):
            out, summ = point_dir / f"rep{r}.jsonl.gz", point_dir / f"rep{r}.json"
            stagger = est if (o["stagger"] and c > 1) else 0.0
            rcmd = client_cmd(cfg, args.model, args.arm, wl, c, port,
                              ["--num-requests", str(n_req), "--stagger-s", f"{stagger:.3f}",
                               "--out", str(out), "--summary", str(summ)])
            phases.append({"phase": f"rep{r}", "t_start": time.monotonic()})
            bclib.log(f"C={c} rep {r}/{reps}: {n_req} requests, stagger {stagger:.1f}s")
            rc = run_json(rcmd, clog)
            phases[-1]["t_end"] = time.monotonic()
            s = json.loads(summ.read_text())
            s["command"] = rcmd
            s["returncode"] = rc
            rep_summaries.append(s)
            sm = s["summary"]
            fmt = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
            bclib.log(f"  steady total {fmt(sm['steady_total_tok_s'])} tok/s, output {fmt(sm['steady_output_tok_s'])} tok/s, "
                      f"window {fmt(sm['steady_window_s'])} s, steady requests {sm['steady_requests']}, "
                      f"failed {sm['failed']}, tpot p50 {sm['latency']['tpot_s']['p50']}")
            time.sleep(3)
        meta["phases"] = phases
        meta["reps"] = [{"rep": i + 1, "summary": s["summary"], "timing": s["timing"],
                         "client_resources": s["client_resources"], "prompt_manifest_sha256": s["prompt_manifest_sha256"],
                         "command": s["command"], "returncode": s["returncode"],
                         "raw_file": f"rep{i + 1}.jsonl.gz"} for i, s in enumerate(rep_summaries)]
        tot = [s["summary"]["steady_total_tok_s"] for s in rep_summaries if s["summary"]["steady_total_tok_s"]]
        e2e = [s["summary"]["latency"]["e2e_s"]["mean"] for s in rep_summaries if s["summary"]["latency"]["e2e_s"]["mean"]]
        meta["steady_total_tok_s_median"] = statistics.median(tot) if tot else None
        meta["e2e_mean_s"] = statistics.fmean(e2e) if e2e else None
        meta["failed_requests"] = sum(s["summary"]["failed"] for s in rep_summaries)
        meta["status"] = "ok" if meta["failed_requests"] == 0 else "ok_with_failures"
        return meta
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = str(exc)
        bclib.log(f"point failed: {exc}")
        return meta
    finally:
        if tel is not None:
            tel.send_signal(signal.SIGTERM)
            try:
                tel.wait(30)
            except subprocess.TimeoutExpired:
                tel.kill()
        stop_group(server)
        meta["server_returncode"] = server.returncode
        meta["finished"] = bclib.now_iso()
        bclib.write_json(point_dir / "metadata.json", meta)


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--campaign", required=True, help="results sub-directory, e.g. main, smoke, tuning")
    p.add_argument("--concurrency", help="comma list; overrides the config sweep and disables adaptivity")
    p.add_argument("--reps", type=int)
    p.add_argument("--num-requests", type=int, help="override requests per repetition")
    p.add_argument("--set", action="append", default=[], help="backend setting override key=value")
    p.add_argument("--tag", default="", help="suffix for the arm directory (tuning variants)")
    p.add_argument("--ignore-budget", action="store_true")
    p.add_argument("--no-pmu", action="store_true", help="telemetry without core PMU counters (overhead check)")
    p.add_argument("--config", default=str(bclib.CONFIG))
    args = p.parse_args()

    cfg = bclib.load_config(args.config)
    args.staging = bclib.stage_model(cfg, args.model, args.arm)
    arm = cfg["arms"][args.arm]
    settings = bclib.backend_settings(cfg, args.arm, bclib.parse_overrides(args.set))
    arm_dir = args.arm + (f"__{args.tag}" if args.tag else "")
    base = Path(os.environ["BC_DATA"])
    rel = Path("online") / args.campaign / args.model / arm_dir / args.workload
    o = cfg["online"]

    fixed = [int(x) for x in args.concurrency.split(",")] if args.concurrency else None
    plan = list(fixed or o["concurrency"])
    extend = [] if fixed else list(o["extend_concurrency"])
    thr: dict[int, float] = {}
    low_streak = 0
    prev: dict[str, Any] | None = None
    i = 0
    sweep_log = []
    while i < len(plan):
        c = plan[i]
        point_dir = base / "raw_results" / rel / f"c{c:03d}"
        tel_path = base / "telemetry" / rel / f"c{c:03d}.jsonl"
        meta_path = point_dir / "metadata.json"
        if meta_path.exists() and json.loads(meta_path.read_text()).get("status", "").startswith("ok"):
            meta = json.loads(meta_path.read_text())
            bclib.log(f"C={c}: already done, reusing")
        else:
            if point_dir.exists():
                stamp = time.strftime("%Y%m%dT%H%M%S")
                point_dir.rename(point_dir.with_name(point_dir.name + f".attempt-{stamp}"))
                if tel_path.exists():
                    tel_path.rename(tel_path.with_name(tel_path.name + f".attempt-{stamp}"))
            meta = run_point(cfg, args, args.workload, c, settings, point_dir, tel_path, prev)
        sweep_log.append({"concurrency": c, "status": meta.get("status"),
                          "steady_total_tok_s_median": meta.get("steady_total_tok_s_median")})
        if not meta.get("status", "").startswith("ok"):
            bclib.log(f"C={c}: {meta.get('status')} -> sweep stops here")
            break
        thr[c] = meta["steady_total_tok_s_median"]
        prev_c = c
        prev = {"e2e_mean_s": meta.get("e2e_mean_s"),
                "concurrency_scale": (lambda pc: (lambda nc: max(1.0, nc / pc)))(prev_c)}
        half = c // 2
        if not fixed and half in thr and thr[half]:
            gain = thr[c] / thr[half] - 1
            low_streak = low_streak + 1 if gain < o["plateau_gain"] else 0
            bclib.log(f"C={half}->{c}: throughput gain {gain:+.1%} (low-gain streak {low_streak})")
            if low_streak >= 2:
                bclib.log("plateau confirmed by two consecutive low-gain doublings -> stop")
                break
            if i == len(plan) - 1 and extend and gain >= o["extend_min_gain"]:
                plan.append(extend.pop(0))
        i += 1
    bclib.write_json(base / "raw_results" / rel / "sweep.json",
                     {"timestamp": bclib.now_iso(), "plan": plan, "points": sweep_log, "rule": {
                         k: o[k] for k in ("plateau_gain", "extend_min_gain", "max_point_seconds")}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
