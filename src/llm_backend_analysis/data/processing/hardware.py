"""Hardware compute ceilings of one Xeon Max socket (cache mode) -> roofline compute roofs.

Two sources, both byte-identical copies under data/raw/hardware/<campaign>/ (compute/, system/):
  * the hardware-only roofline study (hw-bench, 2026-09-21): full-socket AVX-512 FMA, AMX
    tile-instruction peaks and a oneDNN AMX BF16 GEMM, 56 threads on one socket;
  * scripts/benchmark/gemm_ceilings.py (this repository): BF16 and INT8 GEMMs on the server layout
    (55 threads, cores 0-54) inside the vLLM image, including the kernels vLLM runs for the Llama
    projections, at several shapes and two thread bindings. Same record schema.

hw-bench's own acceptance rules (its docs/VALIDATION.md) are applied here, never by deleting files:
  * the benchmark exited 0 and validated its result (checksum / finite output);
  * the record was taken on the node described by system/topology.json: a few exploratory
    submissions ran on the login host and are excluded;
  * oneDNN GEMM records must carry the corrected affinity note (an earlier series inherited
    OMP_PROC_BIND/OMP_PLACES and scaled inversely) and must name an AMX implementation in
    their DNNL_VERBOSE output.
"""

from __future__ import annotations

import json

import pandas as pd

from ... import paths
from .. import registry

ONEDNN_ACCEPTED_NOTE = "onednn_numactl_only_omp_affinity_unset"


def _reject_reason(rec: dict, node: str) -> str:
    res = rec.get("result") or {}
    if rec.get("returncode") != 0 or not res.get("valid"):
        return "benchmark failed or did not validate"
    if rec.get("hostname", "").split(".")[0] != node:
        return f"not run on the characterized node {node}"
    if res.get("benchmark") == "amx_onednn_gemm":
        if rec.get("notes") != ONEDNN_ACCEPTED_NOTE:
            return "oneDNN affinity anomaly (superseded series)"
        if "_amx" not in rec.get("stdout", ""):
            return "no AMX implementation in DNNL_VERBOSE"
    return ""


# Kernel -> (data type, how AMX use is established). "verbose": oneDNN must name an AMX implementation.
GEMM_KERNELS = {
    "amx_bf16_gemm": ("bf16", "verbose"),      # hw-bench torch.mm (DNNL_VERBOSE in the record)
    "torch_mm_bf16": ("bf16", "verbose"),
    "vllm_onednn_bf16": ("bf16", "verbose"),
    "torch_int_mm": ("int8", "verbose"),
    "vllm_onednn_int8": ("int8", "verbose"),
    "vllm_sgl_int8": ("int8", "by construction"),  # vLLM selects it only when AMX tiles are available
}


def _onednn_implementations(campaign_dir) -> dict[str, list[str]]:
    out = {}
    for f in (campaign_dir / "system").glob("onednn_verbose-*.json"):
        d = json.loads(f.read_text())
        out[d["kernel"]] = d.get("implementations", [])
    return out


def _amx_confirmed(rec: dict, kernel: str, impls: dict[str, list[str]]) -> bool | None:
    if kernel not in GEMM_KERNELS:
        return None
    how = GEMM_KERNELS[kernel][1]
    if how == "by construction":
        return True
    text = rec.get("stdout", "") + " ".join(impls.get(kernel, []))
    return "amx" in text


def process_hardware() -> pd.DataFrame:
    """One row per hw-bench compute record (accepted and rejected), with the reason for rejections."""
    rows = []
    for campaign_dir in sorted((paths.RAW / "hardware").glob("*")):
        node = json.loads((campaign_dir / "system" / "topology.json").read_text())["hostname"].split(".")[0]
        impls = _onednn_implementations(campaign_dir)
        for f in sorted((campaign_dir / "compute").glob("*.json")):
            rec = json.loads(f.read_text())
            res = rec.get("result") or {}
            reason = _reject_reason(rec, node)
            rows.append({
                "campaign": campaign_dir.name, "campaign_role": registry.campaign_role("hardware", campaign_dir.name),
                "host": rec.get("hostname", "").split(".")[0], "memory_mode": rec.get("memory_mode"),
                "socket_scope": rec.get("socket_scope"), "socket": rec.get("socket"), "threads": res.get("threads"),
                "kernel": res.get("kernel"), "benchmark": res.get("benchmark"),
                "dtype": res.get("dtype") or GEMM_KERNELS.get(res.get("kernel"), (None,))[0],
                "binding": (rec.get("notes") or "").removeprefix("binding=") if res.get("benchmark") == "gemm_ceiling" else "",
                "amx_confirmed": _amx_confirmed(rec, res.get("kernel"), impls),
                "gemm_m": res.get("m"), "gemm_n": res.get("n"), "gemm_k": res.get("k"),
                "notes": rec.get("notes") or "", "timestamp": rec.get("timestamp"),
                "iterations": res.get("iterations"), "gops": res.get("performance_gops"),
                "max_rel_error": res.get("max_rel_error_vs_reference"),
                "accepted": not reason, "reject_reason": reason,
                "source": paths.relative(f),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["campaign", "kernel", "socket", "source"], ignore_index=True)
