#!/usr/bin/env python3
"""GEMM throughput of one kernel family, run INSIDE the vLLM CPU image (needs torch + vllm._C).

Called by gemm_ceilings.py; prints one JSON line per (shape) to stdout. Kernel families:

  torch_mm_bf16        torch.mm, BF16 x BF16 (oneDNN brgemm; hw-bench's reference GEMM)
  torch_int_mm         torch._int_mm, INT8 x INT8 -> INT32 (oneDNN)
  vllm_onednn_bf16     vLLM's BF16 dense-projection path: ops.create_onednn_mm + ops.onednn_mm
                       (prepacked weight; what vllm-bf16 runs for every Llama projection)
  vllm_sgl_int8        vLLM's W8A8 path for symmetric dynamic per-token activations:
                       convert_weight_packed + int8_scaled_mm_with_quant (AMX INT8; includes the
                       per-token activation quantization, as in serving)
  vllm_onednn_int8     vLLM's alternative oneDNN W8A8 path: create_onednn_scaled_mm +
                       onednn_scaled_int8_quant + onednn_scaled_mm

Operations = 2*M*N*K per call (activation quantization is not counted as work). Timing: warm-up
calls, then individually timed calls for at least --seconds; the record keeps every call time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch


def build(kernel: str, m: int, n: int, k: int):
    """Return (callable, check) for one shape. check() -> max relative error on a row slice."""
    g = torch.Generator().manual_seed(7)
    if kernel == "torch_mm_bf16":
        a = torch.randn((m, k), generator=g).to(torch.bfloat16)
        b = torch.randn((k, n), generator=g).to(torch.bfloat16)
        ref = lambda: a[:32].float() @ b.float()
        return (lambda: torch.mm(a, b)), (lambda out: _rel(out[:32].float(), ref()))
    if kernel == "torch_int_mm":
        a = torch.randint(-64, 64, (m, k), generator=g, dtype=torch.int8)
        b = torch.randint(-64, 64, (k, n), generator=g, dtype=torch.int8)
        ref = lambda: a[:32].double() @ b.double()
        return (lambda: torch._int_mm(a, b)), (lambda out: _rel(out[:32].double(), ref()))
    from vllm import _custom_ops as ops  # noqa: PLC0415 - only inside the image
    x = torch.randn((m, k), generator=g).to(torch.bfloat16)
    if kernel == "vllm_onednn_bf16":
        w = torch.randn((n, k), generator=g).to(torch.bfloat16)  # nn.Linear layout [N, K]
        handler = ops.create_onednn_mm(w.t(), 32)
        ref = lambda: x[:32].float() @ w.float().t()
        return (lambda: ops.onednn_mm(handler, x, None)), (lambda out: _rel(out[:32].float(), ref()))
    wq = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    ws = (torch.rand((n, 1), generator=g) * 0.01 + 0.001).float()
    ref = lambda: (_quant_rows(x[:32]) @ (wq.float() * ws).t())
    if kernel == "vllm_sgl_int8":
        packed = torch.ops._C.convert_weight_packed(wq)
        fn = lambda: torch.ops._C.int8_scaled_mm_with_quant(x, packed, ws.squeeze(1), None, x.dtype, True)
        return fn, (lambda out: _rel(out[:32].float(), ref()))
    if kernel == "vllm_onednn_int8":
        handler = ops.create_onednn_scaled_mm(wq.t(), ws, torch.bfloat16, True, False, 32)

        def fn():
            xq, xs, xzp = ops.onednn_scaled_int8_quant(x, None, None, True)
            out = torch.empty((m, n), dtype=x.dtype)
            ops.onednn_scaled_mm(handler, xq, out, xs, xzp, None, None)
            return out
        return fn, (lambda out: _rel(out[:32].float(), ref()))
    raise SystemExit(f"unknown kernel {kernel}")


def _quant_rows(x: torch.Tensor) -> torch.Tensor:
    """Symmetric dynamic per-token int8 quantization, dequantized (the reference for W8A8)."""
    xf = x.float()
    s = xf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
    return torch.round(xf / s).clamp(-127, 127) * s


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out - ref).abs().max() / ref.abs().max().clamp(min=1e-12))


def _omp_runtime() -> str:
    """OpenMP library loaded in this process (which affinity variables apply)."""
    with open("/proc/self/maps") as f:
        libs = {ln.rsplit("/", 1)[-1].strip() for ln in f if "omp" in ln.rsplit("/", 1)[-1]}
    return ", ".join(sorted(libs)) or "unknown"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True)
    p.add_argument("--shapes", required=True, help="m:n:k[,m:n:k...]")
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--seconds", type=float, default=1.5)
    args = p.parse_args()
    threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
    torch.set_num_threads(threads)
    bind_info = {k: os.environ[k] for k in ("KMP_AFFINITY", "GOMP_CPU_AFFINITY", "OMP_PROC_BIND", "OMP_PLACES")
                 if k in os.environ} or "affinity mask only"
    for spec in args.shapes.split(","):
        m, n, k = (int(v) for v in spec.split(":"))
        fn, check = build(args.kernel, m, n, k)
        for _ in range(args.warmups):
            out = fn()
        t0 = time.perf_counter()
        out = fn()
        est = time.perf_counter() - t0
        iters = max(5, math.ceil(args.seconds / max(est, 1e-6)))
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn()
            times.append(time.perf_counter() - t0)
        err = check(out)
        ops_per_call = 2 * m * n * k
        med = sorted(times)[len(times) // 2]
        print(json.dumps({
            "benchmark": "gemm_ceiling", "kernel": args.kernel, "m": m, "n": n, "k": k, "threads": threads,
            "thread_binding": bind_info, "warmup_iterations": args.warmups, "iterations": iters,
            "operations": ops_per_call, "seconds": times, "seconds_median": med,
            "performance_gops": ops_per_call / med / 1e9, "performance_gops_best": ops_per_call / min(times) / 1e9,
            "max_rel_error_vs_reference": err, "valid": bool(math.isfinite(err) and err < 0.05),
            "torch_version": torch.__version__, "openmp_runtime": _omp_runtime(),
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
