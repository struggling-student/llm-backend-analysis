# Methodology

This document describes how the backend-selection experiment (`backend-selection-v2`, CRESCO8, 3–4 October
2026) was designed and run. Every parameter lives in [`configs/experiment.toml`](../configs/experiment.toml)
and is copied into each run's `metadata.json`; measured values are not repeated here — they are computed in the
notebooks.

## Why server-vs-server

An earlier selection study compared `vllm serve` with `llama-bench`. Those tools measure different things (a
batching server against single-sequence forward passes). This experiment compares **server against server**
(`vllm serve` vs `llama-server`) through **one shared client**, which is the primary criterion. Offline engine
microbenchmarks (Experiment A) are kept as a controlled reference and reported with their scope differences.

## Hardware and placement

- Nodes: `cresco8_hbm` partition, Intel Xeon CPU Max 9480, 2 sockets × 56 cores, no SMT, 64 GiB HBM2e per socket
  in **cache mode** in front of DDR5.
- Every run uses **one socket** with the same layout for both backends:

| Role | CPUs | Memory |
| --- | --- | --- |
| Server compute threads (55 OpenMP threads, one per core) | 0–54 | `numactl --membind=0` |
| Server HTTP front end (whole server process tree limited to 0–55) | 55 | node 0 |
| Client, telemetry sampler, orchestrator | 100–111 (socket 1) | node 1 |

- Every Slurm job checks that the node is in cache mode, kills processes a previous job may have left behind,
  records an environment manifest (`data/raw/environment/nodes/`) and runs a 10 s STREAM node check
  (`data/raw/environment/node_checks/`); jobs on nodes below 450 GB/s abort.

## Configurations

| Configuration | Backend | Weights | KV cache | AMX use |
| --- | --- | --- | --- | --- |
| `vllm_bf16` | vLLM 0.30.0 (official CPU image) | safetensors BF16 (HF snapshot) | BF16 | BF16 tiles: GEMMs and attention |
| `llama_cpp_bf16` | llama.cpp `9655061` (gcc 14, AMX build) | GGUF BF16, lossless conversion of the same snapshot | F16 | none (no BF16 AMX kernel) |
| `llama_cpp_int8` | same | GGUF Q8_0 from the BF16 GGUF | F16 | INT8 tiles for weight GEMMs with ≥ 2 rows |
| `vllm_int8` (pending) | vLLM 0.30.0 | compressed-tensors W8A8 RTN from the same snapshot | BF16 | INT8 tiles (`int8_scaled_mm`) |

`vllm_bf16` vs `llama_cpp_bf16` is the **precision-matched pair** and the basis of every backend ratio.
`llama_cpp_int8` is llama.cpp's fastest configuration; it is shown next to the others and compared with vLLM BF16
only with an explicit precision caveat. `vllm_int8` vs `llama_cpp_int8` will be the second precision-matched pair
(both on AMX-INT8). Its checkpoint was built with `scripts/benchmark/quantize_w8a8.py` (data-free RTN,
per-channel weights, dynamic per-token activations; versions pinned in
[`configs/requirements/llmcompressor.txt`](../configs/requirements/llmcompressor.txt)) and passed a
quality/AMX sanity check (`data/raw/quality/`, notebook 08).

Models: Llama-3.1-8B-Instruct (primary) and Llama-3.2-1B-Instruct (cross-check that a behaviour belongs to the
backend, not to one model size). Same Hugging Face revision for both backends.

## Fairness controls (online)

- **One client** (`scripts/benchmark/bench_client.py`, standard-library asyncio): same request body, same timing
  points, one fresh connection per request.
- **Prompts** are token-id lists (BOS + ISL−1 deterministic pseudo-random ordinary tokens): exactly ISL tokens
  reach both servers, with no tokenizer or chat-template difference and no shared prefixes. The prompt manifest
  hash is recorded per repetition.
- **Output length**: `max_tokens = OSL`, `ignore_eos`, temperature 0; prompt and completion token counts are
  checked for every request.
- **Prompt/prefix caching off** in both servers.
- **KV sizing**: each server is sized for exactly C full-length sequences (llama-server `-np C` with one slot per
  request and `--no-kv-unified`; vLLM `VLLM_CPU_KVCACHE_SPACE` from C × sequence length × KV bytes × 1.15).
- **Fresh server per concurrency point**: idle baseline, warm-up, then 3 measured repetitions.
- **Continuous batching** in both servers.

Differences that remain (and are stated in the notebooks): weight file format, KV-cache element type (BF16 vs
F16, both 2 bytes), thread binding mechanism, process structure of the HTTP front end, and — by design of the
engines — AMX coverage at BF16.

## Load model and sweep rule

Closed loop: C workers each send their next request as soon as the previous one finishes; start times are
staggered over one estimated request duration. Throughput is measured in the **steady window** in which exactly C
requests are in flight; latency statistics use requests whose whole life lies in that window.

The sweep rule was declared before any measurement and applied identically to every configuration: C = 1…64;
stop after two consecutive doublings that each add < 5% throughput; extend to 128 and 256 only while the last
doubling added ≥ 10%; skip a point projected to take more than 4 h (logged in `sweep.json`).

## Tuning

A small documented grid on the 8B balanced workload at C = 1 and 16 (one repetition): vLLM
`max_num_batched_tokens` ∈ {2048, 4096, 8192}; llama.cpp flash attention on/off × ubatch {512, 2048}, plus a BF16
KV cache. Rule: highest steady throughput at C = 16 unless C = 1 TPOT worsens by more than 5% versus the default.
Choices are in `configs/experiment.toml` `[tuned.*]`; the grid and the rule are reproduced in notebook 01 and by
`scripts/processing/summarize_tuning.py`.

## Telemetry

CRESCO8 runs `kernel.perf_event_paranoid = 2` without root, `perf` or `pcm`, so every **uncore** PMU (DDR IMC,
HBM controllers, near-memory-cache tags, RAPL) is unavailable. `scripts/benchmark/telemetry.py` samples, every
second, per-thread **core** counters of the server process tree:

| Quantity | Events |
| --- | --- |
| Memory reads | `OCR.READS_TO_CORE.L3_MISS_LOCAL` (64 B per count) |
| Memory writes | `OCR.WRITE_ESTIMATE.MEMORY` |
| AMX activity | `EXE.AMX_BUSY` ÷ cycles |

plus CPU busy time per core, RSS, NUMA memory use and the servers' Prometheus metrics. In cache mode the counters
see HBM-cache hits and DDR traffic **together**. Two steps make them usable:

1. **STREAM calibration (Experiment 0)** under the same sampler, 4–96 GiB working sets: gives the sustainable
   bandwidth of the server layout and validates the counters against known traffic. Server traffic is reported
   as a share of STREAM measured with the same instrument, not of a theoretical HBM peak.
2. **Working-set reasoning**: the modelled active working set (weights + live KV) of each point is compared with
   the 64 GiB HBM cache.

An earlier 7-event set (two multiplexed events) measurably slowed vLLM and was replaced before the campaign; its
data are kept as superseded campaigns. If an administrator lowers `perf_event_paranoid` on a node, the sampler
reads HBM/DDR CAS and near-memory-cache counters automatically.

## Deviations from the plan and incidents

All affected data are kept in `data/raw/` and labelled in `data/metadata/campaigns.toml`.

1. **Uncore telemetry unavailable** — HBM vs DDR cannot be separated (above).
2. **Two workloads added after the first results**, both run for every configuration under the same rule:
   8192→256 (the plan's optional long-context shape) and 4096→1024 (decode over a long KV cache, the regime in which
   KV traffic grows with concurrency).
3. **Client bug found during tuning** (shared request queue collapsed the steady window); fixed with per-worker
   request sets and a median-based duration estimate. Tuning re-run; first run kept as `tuning-v1-queue-bug`.
4. **Orphaned vLLM worker** from a cancelled smoke job slowed one node; found by the per-job node check. Fixed by
   SIGTERM handling, orphan reaping in `job.sbatch`, a hard node-check threshold and an idle-core guard per point.
5. **Lustre contention**: the first campaign attempt stalled while five jobs loaded the same weights
   (`main-v0-lustre-stall`); models are now staged to node-local storage. Load time is not measured.
6. **Storage quota** on the project store; outputs moved to scratch.
7. **Budget skip**: llama.cpp BF16 4096→1024 at C = 32 exceeded the 4 h budget (see its `sweep.json`).
8. **Maintenance**: HBM nodes were drained on 2026-10-04 before llama.cpp INT8 4096→1024 (8B) and the 1B INT8
   offline job ran. Neither affects the BF16 comparison.
9. `vllm bench throughput` (vLLM-only offline reference) ran once per shape.

## Processing corrections

Corrections are applied in code, never to raw files:

- **STREAM read model (2026-10-04).** Triad reads two arrays and writes one (non-temporal stores), so reads are 2/3
  of STREAM's modelled bytes. The earlier processing compared PMU reads with the full modelled traffic. Fixed in
  `src/llm_backend_analysis/data/processing/stream.py`.
- **HBM working set.** The retired report stated that every working set was ≤ 47 GiB; the processed data show one
  point above the 64 GiB HBM cache (vLLM 8192→256 at C = 64). Notebook 07 computes this explicitly.
