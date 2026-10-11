#!/usr/bin/env python3
"""Measured GEMM ceilings of the server layout (compute roofs for the roofline, notebook 10).

  scripts/cluster/submit.sh job 01:00:00 gemm scripts/benchmark/gemm_ceilings.py [--campaign NAME]

hw-bench measured a oneDNN AMX BF16 GEMM but no INT8 GEMM, so the roofline had to derive the
INT8 compute roof. This job measures, on the exact server layout (cores 0-54, numactl --membind=0,
55 threads, cache mode) and inside the vLLM image the servers run:

  * generic oneDNN square GEMMs: BF16 (torch.mm, hw-bench's reference) and INT8 (torch._int_mm);
  * the kernels vLLM 0.30.0 actually uses for the Llama projections: oneDNN `onednn_mm` (BF16)
    and `int8_scaled_mm_with_quant` (W8A8), plus vLLM's alternative oneDNN W8A8 path,
    at the four projection shapes of Llama-3.1-8B and several token counts M.

Each (kernel, binding) runs in its own container process, repeated --reps times. Records follow
hw-bench's schema (one JSON per kernel, shape and repetition) so that the processing reads both:
$BC_DATA/raw_results/hardware/<campaign>/compute/*.json and .../system/topology.json.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402

KERNELS = ["torch_mm_bf16", "torch_int_mm", "vllm_onednn_bf16", "vllm_sgl_int8", "vllm_onednn_int8"]
SQUARE = [(6144, 6144, 6144)]


def llama_shapes(cfg: dict, model: str, ms: list[int]) -> list[tuple[int, int, int]]:
    """(M, N, K) of the fused projections vLLM runs: qkv, o, gate_up, down."""
    m = cfg["models"][model]
    d, kvd, ff = m["hidden_size"], m["kv_heads"] * m["head_dim"], m["intermediate_size"]
    nk = [(m["attention_heads"] * m["head_dim"] + 2 * kvd, d), (d, m["attention_heads"] * m["head_dim"]),
          (2 * ff, d), (d, ff)]
    return [(mm, n, k) for mm in ms for n, k in nk]


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", default="gemm-ceilings")
    p.add_argument("--model", default="llama31_8b")
    p.add_argument("--ms", default="512,2048,8192", help="token counts M for the Llama projection shapes")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--kernels", default=",".join(KERNELS))
    p.add_argument("--seconds", type=float, default=1.5)
    args = p.parse_args()
    cfg = bclib.load_config()
    topo = cfg["topology"]
    image = cfg["backend"]["vllm"]["image"]  # the image the vLLM servers ran (load_config expands ${VAR})
    out = Path(os.environ["BC_DATA"]) / "raw_results" / "hardware" / args.campaign
    (out / "compute").mkdir(parents=True, exist_ok=True)
    (out / "system").mkdir(parents=True, exist_ok=True)
    host = bclib.host_facts()
    hostname = os.uname().nodename
    bclib.write_json(out / "system" / "topology.json",
                     {"hostname": hostname, "memory_mode": topo["memory_mode"], "socket": topo["socket"],
                      "compute_cpus": topo["compute_cpus"], "server_mem_node": topo["server_mem_node"],
                      "image": image, "host_facts": host, "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    shapes = SQUARE + llama_shapes(cfg, args.model, [int(v) for v in args.ms.split(",")])
    root = Path(__file__).resolve().parents[2]
    worker = root / "scripts" / "benchmark" / "gemm_worker.py"
    cpus = topo["compute_cpus"]
    threads = topo["compute_threads"]
    # Bindings: the affinity mask only (hw-bench's accepted setup), and one thread per core pinned the way
    # vLLM 0.30's CPU worker does it (vllm/utils/ompmultiprocessing.py). The image preloads Intel
    # OpenMP (libiomp5), for which vLLM sets the KMP_* variables below (and not GOMP_CPU_AFFINITY).
    ids = [str(c) for c in bclib.cpu_list(cpus)]
    bindings = {"mask": {}, "vllm_pinned": {"KMP_AFFINITY": f"granularity=fine,explicit,proclist=[{','.join(ids)}]",
                                            "KMP_BLOCKTIME": "5", "KMP_TPAUSE": "0"}}
    failures = 0
    for kernel in args.kernels.split(","):
        kshapes = SQUARE if kernel.startswith("torch_") else shapes[len(SQUARE):]
        for bname, bind in bindings.items():
            for rep in range(1, args.reps + 1):
                cmd = ["taskset", "-c", cpus, "numactl", f"--membind={topo['server_mem_node']}",
                       "apptainer", "exec", "--cleanenv", "--env", f"OMP_NUM_THREADS={threads}",
                       "--bind", f"{root}:{root}:ro", image, "python3", str(worker), "--kernel", kernel,
                       "--shapes", ",".join(f"{m}:{n}:{k}" for m, n, k in kshapes), "--seconds", str(args.seconds)]
                bclib.log(f"{kernel} {bname} rep {rep}")
                # APPTAINERENV_* rather than --env: --env splits values at commas (KMP_AFFINITY has many).
                env = dict(os.environ, ONEDNN_VERBOSE="0", **{f"APPTAINERENV_{k}": v for k, v in bind.items()})
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
                if proc.returncode != 0 or not lines:
                    failures += 1
                    bclib.log(f"  FAILED rc={proc.returncode}: {proc.stderr[-800:]}")
                for ln in lines or [json.dumps({"kernel": kernel, "valid": False})]:
                    res = json.loads(ln)
                    shape = f"m{res.get('m')}n{res.get('n')}k{res.get('k')}"
                    rec = {"schema_version": "1.0", "hostname": hostname, "memory_mode": topo["memory_mode"],
                           "socket_scope": "single", "socket": topo["socket"], "threads": threads,
                           "cpu_bind": cpus, "mem_bind": str(topo["server_mem_node"]), "notes": f"binding={bname}",
                           "container_env": {"OMP_NUM_THREADS": str(threads), **bind},
                           "campaign_repetition": rep, "timestamp": bclib.now_iso(), "command": cmd,
                           "returncode": proc.returncode, "stderr": proc.stderr[-4000:], "result": res}
                    bclib.write_json(out / "compute" / f"single-socket{topo['socket']}-{kernel}-{bname}-t{threads}-{shape}-r{rep}.json", rec)
                    if "performance_gops" in res:
                        bclib.log(f"  {shape}: {res['performance_gops'] / 1e3:.1f} TOP/s (err {res['max_rel_error_vs_reference']:.1e})")
    # One verbose pass per oneDNN kernel so the record names the implementation oneDNN chose (AMX or not).
    for kernel in [k for k in args.kernels.split(",") if k in ("torch_mm_bf16", "torch_int_mm", "vllm_onednn_bf16", "vllm_onednn_int8")]:
        kshape = SQUARE[0] if kernel.startswith("torch_") else shapes[len(SQUARE)]
        cmd = ["taskset", "-c", cpus, "numactl", f"--membind={topo['server_mem_node']}", "apptainer", "exec", "--cleanenv",
               "--env", f"OMP_NUM_THREADS={threads}", "--env", "ONEDNN_VERBOSE=1", "--env", "DNNL_VERBOSE=1",
               "--bind", f"{root}:{root}:ro", image, "python3", str(worker), "--kernel", kernel,
               "--shapes", "{}:{}:{}".format(*kshape), "--seconds", "0.05", "--warmups", "1"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        # v1 format: onednn_verbose,v1,primitive,exec,cpu,<primitive>,<implementation>,...
        impl = sorted({":".join(ln.split(",")[5:7]) for ln in proc.stdout.splitlines()
                       if ln.startswith(("onednn_verbose", "dnnl_verbose")) and ",exec," in ln and len(ln.split(",")) > 6})
        bclib.write_json(out / "system" / f"onednn_verbose-{kernel}.json",
                         {"kernel": kernel, "shape": kshape, "implementations": impl, "returncode": proc.returncode,
                          "verbose_lines": [ln for ln in proc.stdout.splitlines() if "verbose" in ln][:200]})
        bclib.log(f"oneDNN implementations for {kernel}: {impl}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
