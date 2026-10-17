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


# --------------------------------------------------------------------------- T3.1: vLLM sweeps without AMX
LEVELS = ["amx", "noamx_gemm", "noamx_all"]
LEVEL_LABELS = {"amx": "AMX allowed", "noamx_gemm": "oneDNN capped below AMX", "noamx_all": "every AMX path off"}


def level_frame(points: pd.DataFrame) -> pd.DataFrame:
    """The ablation campaign's points with `level` (= variant) and the step time (1 ÷ steps per second)."""
    out = points[points["variant"].isin(LEVELS)].copy()
    out["level"] = out["variant"]
    out["step_time_s"] = 1 / out["engine_steps_per_s_mean"]
    return out


def level_summary(points: pd.DataFrame, throughput: str = "total_tok_s_mean") -> pd.DataFrame:
    """Per (workload, level): throughput at C = 1 and at the highest C, peak ÷ C=1, gain of the last doubling, and
    each level's throughput relative to the AMX level at the same C (min and max over C, and at C = 1 and the highest C)."""
    d = level_frame(points)
    rows = []
    for wl, g in d.groupby("workload", sort=False):
        ref = g[g["level"] == "amx"].set_index("concurrency")[throughput]
        for lv in LEVELS:
            s = g[g["level"] == lv].set_index("concurrency")[throughput].sort_index()
            rel = s / ref.reindex(s.index)
            rows.append({"workload": wl, "level": lv, "c1": s.iloc[0], "c_max": int(s.index[-1]), "tput_c_max": s.iloc[-1],
                         "peak_over_c1": s.max() / s.iloc[0], "last_doubling_gain": s.iloc[-1] / s.iloc[-2] - 1,
                         "rel_c1": rel.iloc[0], "rel_c_max": rel.iloc[-1], "rel_min": rel.min(), "rel_max": rel.max()})
    return pd.DataFrame(rows)


def batching_check(points: pd.DataFrame) -> pd.DataFrame:
    """Do the levels form the same batches? Per (workload, C): largest relative difference of tokens per step to the AMX level,
    and the step rate of each level relative to the AMX level (the step cost is what AMX changes)."""
    d = level_frame(points)
    rows = []
    for (wl, c), g in d.groupby(["workload", "concurrency"]):
        g = g.set_index("level")
        base = g.loc["amx"]
        for lv in ("noamx_gemm", "noamx_all"):
            rows.append({"workload": wl, "concurrency": int(c), "level": lv,
                         "tokens_per_step_vs_amx": g.loc[lv, "tokens_per_step_mean"] / base["tokens_per_step_mean"] - 1,
                         "step_rate_vs_amx": g.loc[lv, "engine_steps_per_s_mean"] / base["engine_steps_per_s_mean"],
                         "tput_vs_amx": g.loc[lv, "total_tok_s_mean"] / base["total_tok_s_mean"]})
    return pd.DataFrame(rows)


def lead_decomposition(level_points: pd.DataFrame, llama: pd.DataFrame, throughput: str = "total_tok_s_mean") -> pd.DataFrame:
    """vLLM ÷ llama.cpp at equal C, split into the part AMX accounts for and the rest:
    vLLM(AMX) ÷ llama.cpp = [vLLM(AMX) ÷ vLLM(no AMX)] × [vLLM(no AMX) ÷ llama.cpp]; `amx_log_share` is the share of the
    logarithm of the total lead carried by the first factor. `llama` has configuration llama_cpp_bf16 on the same shapes."""
    d = level_frame(level_points)
    rows = []
    for wl, g in d.groupby("workload", sort=False):
        ll = llama[llama["workload"] == wl].set_index("concurrency")[throughput]
        piv = g.pivot(index="concurrency", columns="level", values=throughput)
        for c in piv.index.intersection(ll.index):
            total = piv.loc[c, "amx"] / ll[c]
            amx_part = piv.loc[c, "amx"] / piv.loc[c, "noamx_gemm"]
            rows.append({"workload": wl, "concurrency": int(c), "vllm_over_llama": total, "amx_factor": amx_part,
                         "rest_factor": piv.loc[c, "noamx_gemm"] / ll[c],
                         "amx_log_share": np.log(amx_part) / np.log(total) if total > 1 else np.nan})
    return pd.DataFrame(rows)
