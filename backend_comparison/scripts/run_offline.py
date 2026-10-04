#!/usr/bin/env python3
"""Experiment A: offline engine microbenchmarks for one (model, arm).

These are each engine's own benchmark tools, run under the same socket, cores,
memory binding and telemetry as the online experiment. They do NOT measure the
same thing end to end, so they are treated as engine microbenchmarks:

A1  batch 1, per workload shape (ISL, OSL):
    llama.cpp  llama-bench  pp  = -p ISL -n 0           prompt processing of ISL tokens
                            tg  = -p 0 -n OSL -d ISL    OSL single-token decode steps after an ISL-token context
                            pg  = -pg ISL,OSL           prompt + generation, one timed run
    vLLM       vllm bench latency --batch-size 1
                            prefill = --output-len 1    (prefill + 1 sampled token)
                            full    = --output-len OSL  (prefill + OSL tokens)
                            decode tok/s derived as (OSL-1) / (full - prefill)
A2  fixed batch B in [offline] batched_sizes:
    llama.cpp  llama-batched-bench -npp ISL -ntg OSL -npl B  (reports PP and TG time separately)
    vLLM       vllm bench latency --batch-size B, output-len 1 and OSL (same derivation as A1)
A3  vLLM only: vllm bench throughput, N prompts submitted at once (reference;
    there is no llama.cpp tool with the same scope, so no ratio is formed).

Scope differences are written into every result file (see SCOPE below).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402

SCOPE = {
    "llama-bench": "Forward passes only. Random prompt tokens, no tokenization, no sampling (tg feeds back a fixed "
                   "token), model load excluded, one sequence. pp time covers ISL tokens in ubatch-sized chunks; tg "
                   "with -d first fills ISL tokens of context, untimed. Reported value: mean of -r samples.",
    "llama-batched-bench": "B sequences of ISL random tokens prefilled together, then B tokens decoded per step for "
                           "OSL steps. No sampling, no tokenization, no HTTP. Reports PP time, TG time and total.",
    "vllm-bench-latency": "Wall time of LLM.generate() for B dummy token-id prompts: scheduler, input preparation, "
                          "forward passes and sampling included; no tokenization, detokenization disabled, no HTTP. "
                          "Same prompts every iteration, so prefix caching is disabled.",
    "vllm-bench-throughput": "Wall time of LLM.generate() for N random prompts submitted at once, scheduler free to "
                             "batch them (continuous batching, chunked prefill). vLLM-only reference.",
}


def with_telemetry(cfg: dict[str, Any], cmd: list[str], env: dict[str, str], out_dir: Path, name: str) -> dict[str, Any]:
    topo = cfg["topology"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{name}.log"
    tel = Path(os.environ["BC_DATA"]) / "telemetry" / out_dir.relative_to(Path(os.environ["BC_DATA"]) / "raw_results") / f"{name}.jsonl"
    tel.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    telp = subprocess.Popen(["taskset", "-c", topo["client_cpus"], os.environ.get("BC_PYTHON", sys.executable),
                             str(bclib.SCRIPTS / "telemetry.py"), "--root-pid", str(proc.pid), "--out", str(tel),
                             "--interval", str(cfg["telemetry"]["interval_s"]), "--cpus", topo["server_cpus"],
                             "--uncore", cfg["telemetry"]["uncore"]],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        telp.send_signal(signal.SIGTERM)
        telp.wait(30)
    return {"name": name, "command": cmd, "command_str": shlex.join(cmd), "returncode": rc,
            "elapsed_s": time.monotonic() - t0, "log": str(log), "telemetry_file": str(tel),
            "t_start_mono": t0}


def llamacpp_runs(cfg, args, settings, model, arm, wl_key, wl, out_dir):
    topo = cfg["topology"]
    bin_dir = settings["bin_dir_amx"] if arm.get("build", "amx") == "amx" else settings["bin_dir_avx512"]
    gguf = str(Path(model["gguf_dir"]) / model["gguf"][arm["gguf_variant"]])
    kv = settings.get("cache_type", arm["kv_cache_dtype"])
    env = dict(os.environ, OMP_PROC_BIND="close", OMP_PLACES="cores", OMP_NUM_THREADS=str(topo["compute_threads"]))
    pre = ["taskset", "-c", topo["server_cpus"], "numactl", f"--membind={topo['server_mem_node']}"]
    common = ["-m", gguf, "-t", str(topo["compute_threads"]), "-b", str(settings["batch_size"]),
              "-ub", str(settings["ubatch_size"]), "-fa", str(settings["flash_attn"]), "-ctk", kv, "-ctv", kv,
              "-lm", settings["load_mode"], "--poll", str(settings["poll"])]
    isl, osl, r = wl["isl"], wl["osl"], cfg["offline"]["batch1_repetitions"]
    results = []
    if not args.skip_batch1:
        bench = [f"{bin_dir}/llama-bench", *common, "-r", str(r), "-o", "jsonl"]
        for name, extra in [("pp", ["-p", str(isl), "-n", "0"]),
                            ("tg", ["-p", "0", "-n", str(osl), "-d", str(isl)]),
                            ("pg", ["-p", "0", "-n", "0", "-pg", f"{isl},{osl}"])]:
            res = with_telemetry(cfg, pre + bench + extra, env, out_dir, f"a1-{wl_key}-{name}")
            res.update({"tool": "llama-bench", "test": name, "batch": 1, "scope": SCOPE["llama-bench"],
                        "rows": [json.loads(l) for l in Path(res["log"]).read_text().splitlines() if l.startswith("{")]})
            results.append(res)
    for b in cfg["offline"]["batched_sizes"]:
        ctx = b * bclib.round_up(isl + osl + cfg["online"]["ctx_margin"], 256)
        for rep in range(1, cfg["offline"]["batched_repetitions"] + 1):
            cmd = pre + [f"{bin_dir}/llama-batched-bench", *common, "-tb", str(topo["compute_threads"]),
                         "-c", str(ctx), "-no-kvu", "-npp", str(isl), "-ntg", str(osl), "-npl", str(b),
                         "--output-format", "jsonl"]
            res = with_telemetry(cfg, cmd, env, out_dir, f"a2-{wl_key}-b{b}-r{rep}")
            res.update({"tool": "llama-batched-bench", "batch": b, "rep": rep, "scope": SCOPE["llama-batched-bench"],
                        "rows": [json.loads(l) for l in Path(res["log"]).read_text().splitlines() if l.startswith("{")]})
            results.append(res)
    return results


def vllm_runs(cfg, args, settings, model, arm, wl_key, wl, out_dir):
    topo = cfg["topology"]
    isl, osl = wl["isl"], wl["osl"]
    sizes = ([] if args.skip_batch1 else [1]) + list(cfg["offline"]["batched_sizes"])
    results = []

    def container(b: int, seq: int) -> tuple[list[str], dict[str, str]]:
        kv = max(1, math.ceil(b * bclib.round_up(seq, 128) * model["kv_bytes_per_token"] * settings["kv_space_headroom"] / 2**30))
        cenv = {"VLLM_CPU_KVCACHE_SPACE": str(kv), "VLLM_CPU_OMP_THREADS_BIND": topo["compute_cpus"],
                "VLLM_CPU_NUM_OF_RESERVED_CPU": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        cmd = ["taskset", "-c", topo["server_cpus"], "numactl", f"--membind={topo['server_mem_node']}",
               "apptainer", "exec", "--cleanenv"]
        for k, v in cenv.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += ["--bind", f"{os.environ['BC_STORE']}:{os.environ['BC_STORE']}", "--bind", f"{os.environ['BC_SCRATCH']}:{os.environ['BC_SCRATCH']}",
                *(["--bind", f"{os.environ['BC_LOCAL']}:{os.environ['BC_LOCAL']}"] if os.environ.get("BC_LOCAL") else []), settings["image"]]
        return cmd, cenv

    engine = ["--model", model["snapshot"], "--dtype", settings["dtype"], "--no-enable-prefix-caching",
              "--max-num-batched-tokens", str(settings["max_num_batched_tokens"]), "--seed", "0"]
    seq = isl + osl + cfg["online"]["ctx_margin"]
    for b in sizes:
        reps = cfg["offline"]["batch1_repetitions"] if b == 1 else cfg["offline"]["batched_repetitions"]
        for name, out_len in [("prefill", 1), ("full", osl)]:
            iters = max(reps, cfg["offline"]["vllm_prefill_iterations"]) if name == "prefill" else reps
            pre, _ = container(b, seq)
            jpath = out_dir / f"a{'1' if b == 1 else '2'}-{wl_key}-b{b}-{name}.json"
            cmd = pre + ["vllm", "bench", "latency", *engine, "--max-model-len", str(seq),
                         "--max-num-seqs", str(max(b, 1)), "--input-len", str(isl), "--output-len", str(out_len),
                         "--batch-size", str(b), "--num-iters-warmup", "2", "--num-iters", str(iters),
                         "--disable-detokenize", "--output-json", str(jpath)]
            res = with_telemetry(cfg, cmd, dict(os.environ), out_dir, jpath.stem)
            res.update({"tool": "vllm-bench-latency", "test": name, "batch": b, "output_len": out_len,
                        "scope": SCOPE["vllm-bench-latency"],
                        "result": json.loads(jpath.read_text()) if jpath.exists() else None})
            results.append(res)
    if not args.skip_throughput:
        n = cfg["offline"]["vllm_throughput_prompts"]
        pre, _ = container(n, seq)
        jpath = out_dir / f"a3-{wl_key}-throughput-n{n}.json"
        cmd = pre + ["vllm", "bench", "throughput", *engine, "--max-model-len", str(seq),
                     "--max-num-seqs", str(max(settings["max_num_seqs"], n)),
                     "--dataset-name", "random", "--random-input-len", str(isl), "--random-output-len", str(osl),
                     "--random-range-ratio", "0", "--num-prompts", str(n), "--disable-detokenize",
                     "--output-json", str(jpath)]
        res = with_telemetry(cfg, cmd, dict(os.environ), out_dir, jpath.stem)
        res.update({"tool": "vllm-bench-throughput", "batch": n, "scope": SCOPE["vllm-bench-throughput"],
                    "result": json.loads(jpath.read_text()) if jpath.exists() else None})
        results.append(res)
    return results


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--workloads", default="decode,balanced,prefill")
    p.add_argument("--campaign", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--skip-batch1", action="store_true")
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--batched-sizes", help="override [offline] batched_sizes, comma list (empty = none)")
    p.add_argument("--config", default=str(bclib.CONFIG))
    args = p.parse_args()

    cfg = bclib.load_config(args.config)
    if args.batched_sizes is not None:
        cfg["offline"]["batched_sizes"] = [int(x) for x in args.batched_sizes.split(",") if x]
    staging = bclib.stage_model(cfg, args.model, args.arm)
    arm, model = cfg["arms"][args.arm], cfg["models"][args.model]
    settings = bclib.backend_settings(cfg, args.arm, bclib.parse_overrides(args.set))
    for wl_key in args.workloads.split(","):
        wl = cfg["workloads"][wl_key]
        out_dir = Path(os.environ["BC_DATA"]) / "raw_results" / "offline" / args.campaign / args.model / args.arm / wl_key
        done = out_dir / "offline.json"
        if done.exists():
            bclib.log(f"{wl_key}: already done")
            continue
        bclib.log(f"offline {args.model} {args.arm} {wl_key}")
        runner = vllm_runs if arm["backend"] == "vllm" else llamacpp_runs
        results = runner(cfg, args, settings, model, arm, wl_key, wl, out_dir)
        bclib.write_json(done, {
            "experiment": cfg["experiment"], "campaign": args.campaign, "kind": "offline",
            "timestamp": bclib.now_iso(), **bclib.host_facts(),
            "model": args.model, "model_id": model["hf_id"], "model_revision": model["revision"],
            "arm": args.arm, "backend": arm["backend"], "precision": arm["precision"], "weights": arm["weights"],
            "workload": wl_key, "isl": wl["isl"], "osl": wl["osl"], "memory_mode": cfg["topology"]["memory_mode"],
            "sockets_used": 1, "server_cpus": cfg["topology"]["server_cpus"],
            "compute_cpus": cfg["topology"]["compute_cpus"], "compute_threads": cfg["topology"]["compute_threads"],
            "numa_policy": f"membind={cfg['topology']['server_mem_node']}", "backend_settings": settings,
            "model_staging": staging,
            "runs": results,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
