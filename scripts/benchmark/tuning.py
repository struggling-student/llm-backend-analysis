#!/usr/bin/env python3
"""Small, documented tuning phase (8B, balanced 512->512, C = 1 and 16, 1 repetition).

Not a parameter search: a handful of settings each backend's documentation
names as the main serving knobs, so neither backend runs on an obviously
inappropriate default. Selection rule (applied by scripts/analysis/tuning_summary.py):
highest steady total tokens/s at C=16, unless C=1 TPOT gets more than 5% worse
than the default, in which case the default is kept.

llama.cpp (both arms): flash attention {on, off} x ubatch {512, 2048}, plus KV
cache type bf16 (instead of f16) with flash attention on and ubatch 512. vLLM: max_num_batched_tokens {2048, 4096, 8192}.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PY = os.environ.get("BC_PYTHON", sys.executable)
S = Path(__file__).resolve().parent

LLAMACPP = {
    "fa-on_ub512": ["flash_attn=on", "ubatch_size=512"],
    "fa-off_ub512": ["flash_attn=off", "ubatch_size=512"],
    "fa-on_ub2048": ["flash_attn=on", "ubatch_size=2048", "batch_size=2048"],
    "fa-off_ub2048": ["flash_attn=off", "ubatch_size=2048", "batch_size=2048"],
    "fa-on_ub512_kvbf16": ["flash_attn=on", "ubatch_size=512", "cache_type=bf16"],
}
VLLM = {
    "mnbt2048": ["max_num_batched_tokens=2048"],
    "mnbt4096": ["max_num_batched_tokens=4096"],
    "mnbt8192": ["max_num_batched_tokens=8192"],
}


def point(arm: str, tag: str, sets: list[str], model: str, workload: str) -> None:
    for c, n in ((1, 3), (16, 48)):
        cmd = [PY, str(S / "run_online.py"), "--campaign", "tuning", "--model", model, "--arm", arm,
               "--workload", workload, "--concurrency", str(c), "--num-requests", str(n), "--reps", "1",
               "--tag", tag, "--ignore-budget"]
        for s in sets:
            cmd += ["--set", s]
        print("==>", " ".join(cmd), flush=True)
        subprocess.run(cmd)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--model", default="llama31_8b")
    p.add_argument("--workload", default="balanced")
    p.add_argument("--stage", choices=["grid", "kv"], default="grid")
    p.add_argument("--kv-base", default="", help="for --stage kv: the winning grid tag, e.g. fa-on_ub512")
    args = p.parse_args()
    if args.arm.startswith("vllm"):
        for tag, sets in VLLM.items():
            point(args.arm, tag, sets, args.model, args.workload)
    elif args.stage == "grid":
        for tag, sets in LLAMACPP.items():
            point(args.arm, tag, sets, args.model, args.workload)
    else:
        base = LLAMACPP[args.kv_base]
        point(args.arm, f"{args.kv_base}_kvbf16", base + ["cache_type=bf16"], args.model, args.workload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
