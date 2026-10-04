#!/usr/bin/env python3
"""Sanity check that the int8 representations still produce the model's text, and
that the int8 arms really execute AMX.

Not an accuracy benchmark: each arm serves the same 16 natural-text prompts with
greedy decoding (64 tokens). Its continuations are compared, word by word, with
the BF16 reference of the same engine (vllm-bf16 for vllm-w8a8, llamacpp-bf16
for llamacpp-q8_0) and with vllm-bf16. A broken quantization shows up as garbage
or an immediate divergence. Slightly different continuations after a few dozen
tokens are normal for any int8 scheme and are not a defect.

While each server generates, the PMU sampler records EXE.AMX_BUSY / cycles, so
the report can state which arms executed AMX instructions.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402
from run_online import http_ok, server_command, stop_group  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In computer architecture, a cache is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "The three laws of thermodynamics state that",
    "Once upon a time, in a small village by the sea,",
    "To compute the derivative of x^3 + 2x, we",
    "The main difference between TCP and UDP is",
    "High Bandwidth Memory (HBM) differs from DDR memory because",
    "Water boils at 100 degrees Celsius at sea level because",
    "A list of the planets of the solar system, in order from the Sun: Mercury,",
    "The French Revolution began in 1789 when",
    "SELECT name, COUNT(*) FROM orders",
    "The Pythagorean theorem states that",
    "Translate to Italian: 'The weather is beautiful today.' ->",
    "Large language models generate text by",
]


def complete(port: int, model: str, prompt: str, backend: str) -> str:
    body = {"model": model, "prompt": prompt, "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "seed": 0}
    if backend == "llamacpp":
        body["cache_prompt"] = False
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def common_words(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def amx_fraction(tel: Path) -> float | None:
    amx = cyc = 0.0
    for line in tel.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "sample" and r.get("pmu"):
            amx += r["pmu"].get("exe_amx_busy", 0)
            cyc += r["pmu"].get("cycles", 0)
    return amx / cyc if cyc else None


def main() -> int:
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(SystemExit(143)))
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama31_8b")
    p.add_argument("--arms", default="vllm-bf16,vllm-w8a8,llamacpp-bf16,llamacpp-q8_0")
    p.add_argument("--campaign", default="quality")
    args = p.parse_args()
    out_dir = Path(os.environ["BC_DATA"]) / "raw_results" / "quality" / args.campaign / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    wl = {"isl": 128, "osl": 64}
    results: dict[str, dict] = {}
    for arm in args.arms.split(","):
        cfg = bclib.load_config()
        staging = bclib.stage_model(cfg, args.model, arm)
        a = cfg["arms"][arm]
        settings = bclib.backend_settings(cfg, arm)
        port = cfg["online"]["port_vllm"] if a["backend"] == "vllm" else cfg["online"]["port_llamacpp"]
        cmd, env, info = server_command(cfg, args.model, arm, wl, 4, settings, port)
        log = out_dir / f"{arm}-server.log"
        server = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        tel = out_dir / f"{arm}-telemetry.jsonl"
        try:
            t0 = time.monotonic()
            while not http_ok(f"http://127.0.0.1:{port}/health"):
                if server.poll() is not None or time.monotonic() - t0 > cfg["online"]["startup_timeout_s"]:
                    raise RuntimeError(f"{arm}: server did not start (see {log})")
                time.sleep(2)
            telp = subprocess.Popen([sys.executable, str(bclib.SCRIPTS / "telemetry.py"), "--root-pid", str(server.pid),
                                     "--out", str(tel), "--uncore", "off", "--cpus", cfg["topology"]["server_cpus"]],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            texts = [complete(port, cfg["models"][args.model]["served_name"], pr, a["backend"]) for pr in PROMPTS]
            telp.send_signal(signal.SIGTERM)
            telp.wait(30)
            results[arm] = {"texts": texts, "amx_busy_fraction": amx_fraction(tel), "server_command": cmd,
                            "staging": staging, "precision": a["precision"]}
            bclib.log(f"{arm}: done, AMX busy fraction {results[arm]['amx_busy_fraction']}")
        finally:
            stop_group(server)
    refs = {"vllm-w8a8": "vllm-bf16", "llamacpp-q8_0": "llamacpp-bf16", "llamacpp-bf16": "vllm-bf16"}
    summary = {}
    for arm, r in results.items():
        row = {"amx_busy_fraction": r["amx_busy_fraction"]}
        for name, ref in (("same_engine_bf16", refs.get(arm)), ("vllm_bf16", "vllm-bf16")):
            if ref in results and ref != arm:
                pairs = list(zip(r["texts"], results[ref]["texts"]))
                row[f"vs_{name}"] = {
                    "reference": ref,
                    "identical_continuations": sum(x == y for x, y in pairs),
                    "mean_common_prefix_words": sum(common_words(x, y) for x, y in pairs) / len(pairs),
                    "min_common_prefix_words": min(common_words(x, y) for x, y in pairs),
                }
        summary[arm] = row
    bclib.write_json(out_dir / "quality.json", {"timestamp": bclib.now_iso(), **bclib.host_facts(),
                                                "prompts": PROMPTS, "max_tokens": 64, "summary": summary,
                                                "results": results})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
