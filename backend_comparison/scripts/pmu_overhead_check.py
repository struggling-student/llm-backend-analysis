#!/usr/bin/env python3
"""Does the per-thread PMU sampling slow the servers down? Same node, same
point, telemetry with and without core counters, alternating (A B A B)."""
import os, subprocess, sys
from pathlib import Path
PY = os.environ.get("BC_PYTHON", sys.executable)
S = Path(__file__).resolve().parent
for i, pmu in enumerate([True, False, True, False]):
    for arm in sys.argv[1:] or ["vllm-bf16"]:
        cmd = [PY, str(S / "run_online.py"), "--campaign", "pmu-overhead", "--model", "llama32_1b", "--arm", arm,
               "--workload", "balanced", "--concurrency", "1", "--num-requests", "4", "--reps", "1",
               "--tag", f"{'pmu' if pmu else 'nopmu'}{i}", "--ignore-budget"] + ([] if pmu else ["--no-pmu"])
        print("==>", " ".join(cmd), flush=True)
        subprocess.run(cmd)
