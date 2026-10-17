"""AMX ablations (review tasks T3.1 and T3.2): what the ablation levels really switch off, and what llama.cpp's AMX
build gains over its AVX-512 build.

T3.2: llama-bench prompt processing of Q8_0 with the AMX build and with the AVX-512 build (identical configuration
with GGML_AMX_* off), order amx, avx512, amx, avx512 inside one job. Arm directories are `llamacpp-q8_0__<build>_p<pass>`;
processing stores the suffix as `variant`.
T3.1: vLLM BF16 at three levels (`amx`, `noamx_gemm`, `noamx_all`); `amx_verify.json` of the verification campaign
holds what each level ran (EXE.AMX_BUSY and the oneDNN implementations created).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import roofline

VARIANT = re.compile(r"^(?P<build>amx|avx512)_p(?P<pass>\d+)$")
BUSY_CORES_MIN = 0.8   # telemetry samples in which the server's cores were at least this busy count as "running"


def parse_variant(variant: str) -> tuple[str, int]:
    m = VARIANT.match(variant)
    if not m:
        raise ValueError(f"not an AMX-build variant: {variant!r}")
    return m["build"], int(m["pass"])


def build_comparison(samples: pd.DataFrame, compute: pd.DataFrame, test: str = "prefill") -> pd.DataFrame:
    """Per workload: prompt-processing rate with each build (mean of all samples of both passes), the ratio AMX ÷
    AVX-512 formed inside each pass pair (same job, back to back), the arithmetic rate (modelled operations per prompt
    token × tokens/s) and its share of the AMX INT8 GEMM roof."""
    s = samples[samples["test"] == test].copy()
    parsed = s["variant"].map(parse_variant)
    s["build"] = [p[0] for p in parsed]
    s["pass_no"] = [p[1] for p in parsed]
    per = s.groupby(["model", "workload", "isl", "build", "pass_no"], as_index=False)["tok_s"].mean()
    rows = []
    for (model, wl, isl), g in per.groupby(["model", "workload", "isl"], sort=False):
        piv = g.pivot(index="pass_no", columns="build", values="tok_s")
        pair = piv["amx"] / piv["avx512"]
        arch = roofline.architecture(model)
        ops = roofline.ops_per_request(arch, int(isl), 1)["prefill_ops"] / isl
        roof = compute.loc["amx_int8_gemm", "tops"]
        amx, avx = g[g["build"] == "amx"]["tok_s"].mean(), g[g["build"] == "avx512"]["tok_s"].mean()
        rows.append({"model": model, "workload": wl, "isl": int(isl), "amx_tok_s": amx, "avx512_tok_s": avx,
                     "ratio": amx / avx, "ratio_min": pair.min(), "ratio_max": pair.max(),
                     "amx_tops": amx * ops / 1e12, "avx512_tops": avx * ops / 1e12,
                     "amx_share_of_roof": amx * ops / 1e12 / roof, "avx512_share_of_roof": avx * ops / 1e12 / roof,
                     "roof_tops": roof})
    return pd.DataFrame(rows).sort_values("isl", ignore_index=True)


def offline_amx_busy(telemetry_root: Path, campaign: str, test: str = "pp") -> pd.DataFrame:
    """EXE.AMX_BUSY ÷ cycles of the server cores over the samples in which the cores were busy (model load excluded)."""
    rows = []
    for f in sorted((telemetry_root / "offline" / campaign).glob(f"*/*/*/a1-*-{test}.jsonl")):
        model, arm_dir, workload = f.parts[-4], f.parts[-3], f.parts[-2]
        try:
            build, pass_no = parse_variant(arm_dir.partition("__")[2])
        except ValueError:
            continue
        busy = cycles = 0.0
        n = 0
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("type") != "sample" or not r.get("pmu"):
                continue
            cpu = [v for k, v in r["cpu_busy"].items() if int(k) <= 54]
            if cpu and float(np.mean(cpu)) >= BUSY_CORES_MIN:
                busy += float(r["pmu"].get("exe_amx_busy", 0) or 0)
                cycles += float(r["pmu"].get("cycles", 0) or 0)
                n += 1
        rows.append({"model": model, "workload": workload, "build": build, "pass_no": pass_no,
                     "amx_busy_frac": busy / cycles if cycles else np.nan, "samples": n})
    return pd.DataFrame(rows)


def verify_table(path: Path) -> pd.DataFrame:
    """The verification campaign's amx_verify.json as one row per (level, C)."""
    d = json.loads(Path(path).read_text())
    rows = []
    for level, pts in d["levels"].items():
        for c, v in pts.items():
            prims = v["onednn_primitives"]
            matmul = {k.split(" ", 1)[1]: n for k, n in prims.items() if k.startswith("matmul")}
            rows.append({"level": level, "concurrency": int(c.lstrip("c")), "status": v["status"],
                         "amx_busy_frac": v["amx_busy_frac"], "no_amx_shim_active": v["no_amx_shim_active"],
                         "matmul_implementations": ", ".join(f"{k} ×{n}" for k, n in sorted(matmul.items())),
                         "amx_primitives": v["onednn_amx_primitives"]})
    return pd.DataFrame(rows)
