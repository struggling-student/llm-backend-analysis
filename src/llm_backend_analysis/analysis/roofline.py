"""Roofline placement of the serving points on one Xeon Max socket in HBM cache mode.

A point is placed by two numbers:

  arithmetic rate (TOP/s)   modelled operations of the tokens served in the steady window ÷ time.
                            Operations are counted as 2 × multiply-accumulates of the transformer
                            (FLOP for BF16, integer ops for the INT8 weight GEMMs); see ops_per_request.
  operational intensity     that rate ÷ the *measured* memory traffic (PMU reads + writes beyond L3,
  (OP/byte)                 i.e. HBM-cache hits plus DDR in cache mode).

The arithmetic is modelled because the core PMU has no FLOP/AMX-operation counter usable here;
the AMX busy-cycle counter cross-checks it (amx_cross_check). The bytes are measured.

Ceilings (all for the 55 compute cores of socket 0):
  memory    STREAM triad measured with the same PMU counters on the server layout
            (stream_calibration); per point at its modelled working set, so points whose
            working set exceeds the 64 GiB HBM cache get the lower DDR-dominated ceiling.
  compute   instruction peaks from hw-bench (AVX-512 FP32 FMA, AMX INT8 tile peak; 56 threads,
            scaled to 55 cores) and GEMM roofs = the best measured AMX GEMM of each data type
            (hw-bench's oneDNN BF16 GEMM and the server-layout GEMM job, which includes the kernels
            vLLM runs for the Llama projections). The AMX BF16 tile peak is derived (half the INT8
            tile rate: same tile instruction throughput, half the operations) and labelled so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import registry

# hw-bench instruction-throughput kernels -> how this module names the ceiling
MEASURED_CEILINGS = {
    "amx_int8_dot": ("amx_int8_peak", "AMX INT8 tile peak", "measured"),
    "avx512_fp32_fma": ("avx512_fma", "AVX-512 FP32 FMA", "measured"),
}
# GEMM roofs per data type: the best measured AMX GEMM of that type, whatever the kernel.
GEMM_ROOFS = {"bf16": ("amx_bf16_gemm", "AMX BF16 GEMM"), "int8": ("amx_int8_gemm", "AMX INT8 GEMM")}
GEMM_BENCHMARKS = ("amx_onednn_gemm", "gemm_ceiling")

# Compute path of each configuration's weight GEMMs -> the ceiling that bounds it.
# (Attention and the remaining element-wise work run on AVX-512 in every engine.)
KERNEL_CEILING = {
    "vllm_bf16": "amx_bf16_gemm",       # oneDNN onednn_mm (BF16 brgemm on AMX tiles)
    "llama_cpp_bf16": "avx512_fma",     # no BF16 AMX kernel in llama.cpp: AVX-512 (BF16 dot products)
    "llama_cpp_int8": "amx_int8_gemm",  # AMX INT8 tiles for >= 2 rows (AVX-512 VNNI at 1 row)
    "vllm_int8": "amx_int8_gemm",       # vLLM int8_scaled_mm_with_quant on AMX INT8 tiles
}
# Further roofs drawn next to a configuration's own kernel ceiling, for context.
CONTEXT_CEILINGS = {
    "vllm_bf16": ["vllm_bf16_kernels", "amx_bf16_peak"],
    "llama_cpp_bf16": ["amx_bf16_gemm", "amx_bf16_peak"],   # what an AMX BF16 kernel would allow
    "llama_cpp_int8": ["amx_int8_peak"],
    "vllm_int8": ["vllm_int8_kernels", "amx_int8_peak"],
}
# vLLM 0.30's kernel for the Llama projections, per precision (gemm_ceilings.py kernel names).
VLLM_PROJECTION_KERNELS = {"vllm_bf16_kernels": ("vllm_onednn_bf16", "vLLM BF16 projection kernels"),
                           "vllm_int8_kernels": ("vllm_sgl_int8", "vLLM INT8 projection kernels")}
HBM_CACHE_GIB = 64.0


# --------------------------------------------------------------------------- operation model
@dataclass(frozen=True)
class Architecture:
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool

    @property
    def linear_params(self) -> int:
        """Weights of the per-layer projections (q, k, v, o, gate, up, down); embeddings excluded."""
        q_o = 2 * self.hidden * self.heads * self.head_dim
        k_v = 2 * self.hidden * self.kv_heads * self.head_dim
        mlp = 3 * self.hidden * self.intermediate
        return self.layers * (q_o + k_v + mlp)

    @property
    def total_params(self) -> int:
        emb = self.vocab * self.hidden
        return self.linear_params + emb * (1 if self.tied else 2)

    @property
    def linear_ops(self) -> int:
        """Operations of one token through every projection (2 per weight)."""
        return 2 * self.linear_params

    @property
    def logits_ops(self) -> int:
        return 2 * self.hidden * self.vocab

    @property
    def attention_ops_per_position(self) -> int:
        """QKᵀ and PV for one query against one cached position, all layers and heads."""
        return 4 * self.layers * self.heads * self.head_dim


def architecture(model: str) -> Architecture:
    m = registry.experiment_config()["models"][model]
    return Architecture(layers=m["layers"], hidden=m["hidden_size"], intermediate=m["intermediate_size"],
                        heads=m["attention_heads"], kv_heads=m["kv_heads"], head_dim=m["head_dim"],
                        vocab=m["vocab_size"], tied=m["tied_embeddings"])


def ops_per_request(arch: Architecture, isl: int, osl: int) -> dict[str, float]:
    """Operations to serve one request (prompt caching off, logits only where a token is sampled).

    Prefill: every prompt token through the projections, causal attention (token i attends to i
    positions), logits for the last prompt token (it yields the first output token).
    Decode: the remaining osl-1 tokens, each through the projections and the logits, attending to
    its whole context isl + j."""
    a = arch.attention_ops_per_position
    prefill = isl * arch.linear_ops + a * isl * (isl + 1) / 2 + arch.logits_ops
    n = osl - 1
    decode = n * (arch.linear_ops + arch.logits_ops) + a * (n * isl + n * (n + 1) / 2)
    return {"prefill_ops": prefill, "decode_ops": decode, "request_ops": prefill + decode}


def projection_shapes(arch: Architecture) -> dict[str, tuple[int, int]]:
    """(N, K) of the fused projections a serving engine runs per layer: qkv, o, gate_up, down."""
    q, kv = arch.heads * arch.head_dim, arch.kv_heads * arch.head_dim
    return {"qkv": (q + 2 * kv, arch.hidden), "o": (arch.hidden, q), "gate_up": (2 * arch.intermediate, arch.hidden),
            "down": (arch.hidden, arch.intermediate)}


def projection_mix(gm: pd.DataFrame, arch: Architecture, kernel: str) -> pd.DataFrame:
    """Effective TOP/s of one kernel over a layer's projections: total operations ÷ the sum of each
    projection's time at its measured rate (operation-weighted harmonic mean), per binding and M."""
    shapes = projection_shapes(arch)
    g = gm[gm["kernel"] == kernel]
    rows = []
    for (binding, m), d in g.groupby(["binding", "gemm_m"]):
        rate = {name: d[(d["gemm_n"] == n) & (d["gemm_k"] == k)]["tops"] for name, (n, k) in shapes.items()}
        if any(r.empty for r in rate.values()):
            continue
        ops = {name: n * k for name, (n, k) in shapes.items()}
        rows.append({"kernel": kernel, "binding": binding, "gemm_m": int(m),
                     "tops": sum(ops.values()) / sum(ops[nm] / rate[nm].iloc[0] for nm in shapes),
                     **{f"tops_{nm}": rate[nm].iloc[0] for nm in shapes},
                     **{f"ops_share_{nm}": ops[nm] / sum(ops.values()) for nm in shapes}})
    return pd.DataFrame(rows)


def decode_step_intensity(arch: Architecture, weight_bytes: float, kv_bytes_per_token: float, batch: np.ndarray,
                          context: float) -> np.ndarray:
    """Minimum-traffic intensity of one decode step: batch tokens, weights read once, each token
    reads its own KV cache of `context` positions. Tends to (linear + logits + attention) ops ÷
    (context × KV bytes) as the batch grows: batching cannot lift long-context decode past it."""
    b = np.asarray(batch, dtype=float)
    ops = b * (arch.linear_ops + arch.logits_ops + arch.attention_ops_per_position * context)
    return ops / (weight_bytes + b * context * kv_bytes_per_token)


def decode_intensity_limit(arch: Architecture, kv_bytes_per_token: float, context: float) -> float:
    return (arch.linear_ops + arch.logits_ops + arch.attention_ops_per_position * context) / (context * kv_bytes_per_token)


# --------------------------------------------------------------------------- ceilings
def gemm_measurements(hardware: pd.DataFrame, socket: int | None = None, cores: int | None = None) -> pd.DataFrame:
    """Median TOP/s over repetitions per (campaign, kernel, binding, shape), scaled linearly to `cores`."""
    topo = registry.topology()
    socket = topo["socket"] if socket is None else socket
    cores = cores or topo["compute_threads"]
    h = hardware[(hardware["socket_scope"] == "single") & (hardware["socket"] == socket)
                 & hardware["benchmark"].isin(GEMM_BENCHMARKS)].copy()
    if h.empty:
        return pd.DataFrame()
    h["binding"] = h["binding"].fillna("").replace("", "mask")
    keys = ["campaign", "kernel", "dtype", "binding", "gemm_m", "gemm_n", "gemm_k", "threads", "amx_confirmed", "host"]
    g = (h.groupby(keys, dropna=False)["gops"]
         .agg(gops_median="median", gops_min="min", gops_max="max", n_records="count").reset_index())
    g["tops"] = g["gops_median"] / 1e3 * cores / g["threads"]
    g["spread_rel"] = (g["gops_max"] - g["gops_min"]) / g["gops_median"]
    g["shape"] = [f"{int(m)}×{int(n)}×{int(k)}" for m, n, k in zip(g["gemm_m"], g["gemm_n"], g["gemm_k"])]
    return g.sort_values(["dtype", "tops"], ascending=[True, False], ignore_index=True)


def compute_ceilings(hardware: pd.DataFrame, socket: int | None = None, cores: int | None = None,
                     model: str | None = None) -> pd.DataFrame:
    """Compute roofs in TOP/s for `cores` cores (default: the server's compute threads).

    Instruction peaks: median of hw-bench's accepted full-socket records of the server socket,
    scaled linearly by cores (the kernels are core-local). GEMM roofs: the best measured AMX GEMM
    of each data type (gemm_measurements, AMX use confirmed). Derived roofs, labelled as such: the
    AMX BF16 tile peak (½ × INT8 tile peak) and, only if no INT8 GEMM was measured, 2 × BF16 GEMM."""
    topo = registry.topology()
    socket = topo["socket"] if socket is None else socket
    cores = cores or topo["compute_threads"]
    h = hardware[(hardware["socket_scope"] == "single") & (hardware["socket"] == socket)]
    rows = []
    for kernel, (cid, label, basis) in MEASURED_CEILINGS.items():
        g = h[h["kernel"] == kernel]
        if g.empty:
            continue
        threads = int(g["threads"].iloc[0])
        med = g["gops"].median() / 1e3
        rows.append({"ceiling": cid, "label": label, "basis": basis, "tops": med * cores / threads,
                     "tops_measured_socket": med, "measured_threads": threads, "n_records": len(g),
                     "spread_rel": (g["gops"].max() - g["gops"].min()) / g["gops"].median(),
                     "host": ", ".join(sorted(g["host"].unique())), "kernel": kernel})
    gm = gemm_measurements(hardware, socket, cores)
    for dtype, (cid, label) in GEMM_ROOFS.items():
        ok = gm[(gm["dtype"] == dtype) & (gm["amx_confirmed"] == True)] if len(gm) else gm  # noqa: E712
        if len(ok):
            b = ok.loc[ok["tops"].idxmax()]
            rows.append({"ceiling": cid, "label": label, "tops": b["tops"],
                         "basis": f"measured: best of {len(ok)} GEMM configurations ({b['kernel']}, {b['shape']}, {b['binding']})",
                         "tops_measured_socket": b["gops_median"] / 1e3, "measured_threads": int(b["threads"]),
                         "n_records": int(b["n_records"]), "spread_rel": b["spread_rel"], "host": b["host"],
                         "kernel": b["kernel"]})
    if len(gm):
        arch = architecture(model or registry.primary_model())
        for cid, (kernel, label) in VLLM_PROJECTION_KERNELS.items():
            mix = projection_mix(gm, arch, kernel)
            if len(mix):
                b = mix.loc[mix["tops"].idxmax()]
                rows.append({"ceiling": cid, "label": label, "tops": b["tops"], "kernel": kernel,
                             "basis": f"measured: {kernel} over the 4 projection shapes, operation-weighted "
                                      f"(M = {b['gemm_m']}, {b['binding']})"})
    c = pd.DataFrame(rows).set_index("ceiling")
    derived = []
    if "amx_int8_peak" in c.index:
        derived.append(("amx_bf16_peak", "AMX BF16 tile peak", "derived: ½ × AMX INT8 tile peak",
                        c.loc["amx_int8_peak", "tops"] / 2))
    if "amx_int8_gemm" not in c.index and "amx_bf16_gemm" in c.index:
        derived.append(("amx_int8_gemm", "AMX INT8 GEMM", "derived: 2 × AMX BF16 GEMM", c.loc["amx_bf16_gemm", "tops"] * 2))
    for cid, label, basis, tops in derived:
        c.loc[cid, ["label", "basis", "tops"]] = [label, basis, tops]
    return c.sort_values("tops", ascending=False)


def memory_roofs(memory: pd.DataFrame) -> list[tuple[float, str]]:
    """The two memory roofs drawn on roofline plots: highest STREAM traffic (working set resident in
    the HBM cache) and the largest measured working set (well beyond it, DDR-dominated)."""
    top = memory.loc[memory["mem_total_gbs"].idxmax()]
    big = memory.loc[memory["working_set_gib"].idxmax()]
    return [(float(top["mem_total_gbs"]),
             f"STREAM, {top['working_set_gib']:g} GiB working set (in HBM cache), {top['mem_total_gbs']:.0f} GB/s"),
            (float(big["mem_total_gbs"]),
             f"STREAM, {big['working_set_gib']:g} GiB working set (beyond HBM cache), {big['mem_total_gbs']:.0f} GB/s")]


def memory_ceilings(stream: pd.DataFrame, threads: int | None = None) -> pd.DataFrame:
    """STREAM traffic seen by the PMU counters (GB/s) per working set, server thread count."""
    threads = threads or registry.topology()["compute_threads"]
    s = (stream[stream["threads"] == threads].groupby("working_set_gib")[["mem_total_gbs", "stream_triad_gbs"]]
         .median().reset_index())
    s["fits_hbm_cache"] = s["working_set_gib"] < HBM_CACHE_GIB
    return s


# --------------------------------------------------------------------------- placement
def add_roofline_metrics(points: pd.DataFrame, compute: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic rate, measured and minimum-traffic intensity, roofs and utilizations per point.

    `points` must already carry the traffic model (memory.add_traffic_model) for model_read_gbs."""
    out = points.copy()
    ops = pd.DataFrame([ops_per_request(architecture(r.model), r.isl, r.osl) for r in out.itertuples()],
                       index=out.index)
    out = out.join(ops)
    # Tokens processed in the steady window: prompt tokens at the prefill cost per token,
    # output tokens at the decode cost per output token.
    out["op_rate_tops"] = (out["input_tok_s_mean"] * out["prefill_ops"] / out["isl"]
                           + out["output_tok_s_mean"] * out["decode_ops"] / out["osl"]) / 1e12
    out["prefill_op_share"] = out["prefill_ops"] / out["request_ops"]
    out["intensity"] = out["op_rate_tops"] * 1e3 / out["mem_total_gbs_mean"]          # OP/byte
    out["intensity_min_traffic"] = out["op_rate_tops"] * 1e3 / out["model_read_gbs"]  # upper bound
    out["memory_roof_gbs"] = out["stream_pmu_total_gbs_at_ws"]
    out["kernel_ceiling"] = out["configuration_id"].map(KERNEL_CEILING)
    out["compute_roof_tops"] = out["kernel_ceiling"].map(compute["tops"])
    out["compute_roof_basis"] = out["kernel_ceiling"].map(compute["basis"])
    out["ridge_intensity"] = out["compute_roof_tops"] * 1e3 / out["memory_roof_gbs"]
    out["attainable_tops"] = np.minimum(out["compute_roof_tops"], out["intensity"] * out["memory_roof_gbs"] / 1e3)
    out["roof_efficiency"] = out["op_rate_tops"] / out["attainable_tops"]
    out["bw_utilization"] = out["mem_total_gbs_mean"] / out["memory_roof_gbs"]
    out["compute_utilization"] = out["op_rate_tops"] / out["compute_roof_tops"]
    out["ridge_ratio"] = out["intensity"] / out["ridge_intensity"]
    out["regime"] = np.where(out["intensity"] < out["ridge_intensity"], "memory-bound side", "compute-bound side")
    out["nearest_ceiling"] = np.where(out["bw_utilization"] >= out["compute_utilization"], "memory", "compute")
    out["beyond_hbm_cache"] = out["working_set_gib_model"] >= HBM_CACHE_GIB
    return out


def regime_crossings(points: pd.DataFrame) -> pd.DataFrame:
    """Per curve: intensity and regime at the lowest and highest C, and the first C on the compute side."""
    rows = []
    for key, g in points.groupby(["model", "configuration_id", "workload"], sort=False):
        g = g.sort_values("concurrency")
        first, last = g.iloc[0], g.iloc[-1]
        comp = g[g["regime"] == "compute-bound side"]
        rows.append({"model": key[0], "configuration_id": key[1], "workload": key[2],
                     "c_first": int(first["concurrency"]), "c_last": int(last["concurrency"]),
                     "intensity_first": first["intensity"], "intensity_last": last["intensity"],
                     "ridge_last": last["ridge_intensity"],
                     "op_rate_first": first["op_rate_tops"], "op_rate_last": last["op_rate_tops"],
                     "op_rate_max": g["op_rate_tops"].max(),
                     "bw_util_first": first["bw_utilization"], "bw_util_last": last["bw_utilization"],
                     "compute_util_max": g["compute_utilization"].max(),
                     "c_cross": int(comp["concurrency"].iloc[0]) if len(comp) else np.nan,
                     "regime_first": first["regime"], "regime_last": last["regime"]})
    return pd.DataFrame(rows)


def amx_cross_check(points: pd.DataFrame, tile_ops_per_cycle: dict[str, int] | None = None) -> pd.DataFrame:
    """Modelled arithmetic rate ÷ AMX tile peak at the measured clock, next to the measured AMX
    busy-cycle fraction. If AMX tile throughput bounds the GEMMs when busy, the two agree for
    configurations whose GEMMs run on AMX."""
    tile = tile_ops_per_cycle or {"bf16": 1024, "int8": 2048}   # per core: 16x16x32 BF16 / 16x16x64 INT8 MACs ÷ 16 cycles × 2
    cores = registry.topology()["compute_threads"]
    out = points.copy()
    peak = out["precision"].map(tile) * cores * out["ghz_effective_per_core_mean"] * 1e9 / 1e12
    out["amx_peak_at_clock_tops"] = peak
    out["modelled_amx_share"] = out["op_rate_tops"] / peak
    return out


def engine_ceilings(offline: pd.DataFrame) -> pd.DataFrame:
    """Highest offline prefill arithmetic rate of each configuration (engine benchmarks, any batch):
    the best the engine's own kernels reached on this socket without a server."""
    pre = offline[(offline["test"] == "prefill")]
    if pre.empty:
        return pd.DataFrame()
    med = pre.groupby(["model", "configuration_id", "workload", "isl", "tool", "benchmark", "batch"], as_index=False)["tok_s"].median()
    med["ops_per_prompt_token"] = [ops_per_request(architecture(r.model), int(r.isl), 1)["prefill_ops"] / r.isl
                                   for r in med.itertuples()]
    med["op_rate_tops"] = med["tok_s"] * med["ops_per_prompt_token"] / 1e12
    best = med.loc[med.groupby(["model", "configuration_id"])["op_rate_tops"].idxmax()]
    return best.reset_index(drop=True)
