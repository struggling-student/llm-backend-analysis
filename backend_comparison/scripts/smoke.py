#!/usr/bin/env python3
"""Smoke-test matrix: exercise every code path on small inputs before the campaign.

Checks (reviewed from the outputs, see report/methodology notes):
- both servers start with the exact flags, accept token-id prompts, and return
  usage.prompt_tokens == ISL and usage.completion_tokens == OSL;
- streamed chunks == tokens (so chunk timestamps are token timestamps);
- telemetry: PMU counters attach to every server thread, metrics scrape works;
- offline tools run and their outputs parse.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PY = os.environ.get("BC_PYTHON", sys.executable)
S = Path(__file__).resolve().parent


def run(*args: str) -> None:
    print("==>", " ".join(args), flush=True)
    rc = subprocess.run([PY, *args]).returncode
    print(f"<== rc={rc}", flush=True)


def main() -> int:
    online = [str(S / "run_online.py"), "--campaign", "smoke", "--reps", "1"]
    for arm in ("vllm-bf16", "llamacpp-bf16", "llamacpp-q8_0"):
        run(*online, "--model", "llama32_1b", "--arm", arm, "--workload", "balanced",
            "--concurrency", "1,4", "--num-requests", "4")
    for arm in ("vllm-bf16", "llamacpp-bf16"):
        run(*online, "--model", "llama31_8b", "--arm", arm, "--workload", "prefill",
            "--concurrency", "2", "--num-requests", "2")
    offline = [str(S / "run_offline.py"), "--campaign", "smoke", "--model", "llama32_1b", "--workloads", "balanced",
               "--batched-sizes", "8"]
    for arm in ("vllm-bf16", "llamacpp-bf16"):
        run(*offline, "--arm", arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
