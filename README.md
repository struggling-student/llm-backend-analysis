# Backend selection v2: vLLM vs llama.cpp on Xeon Max (HBM, cache mode)

This experiment decides which inference backend the rest of the thesis uses. Later
experiments will push concurrent online serving until the memory system saturates.
The question is therefore **not** "which backend is faster" in general. It is:

> Which backend, vLLM or llama.cpp, gives the stronger experimental platform for
> concurrent, throughput-oriented LLM serving on the Xeon Max, and for turning more
> concurrent requests into more aggregate throughput and more HBM traffic?

The first backend-selection study compared `vllm serve` with `llama-bench`. Those tools
measure different things: batched serving versus single-sequence forward passes. This
version replaces that comparison with server-vs-server measurement through one shared
client. Offline engine microbenchmarks are reported separately, and their scope
differences are stated next to the results.

## Layout

| Path | Contents |
| --- | --- |
| `configs/experiment.toml` | Every parameter: models, arms, workloads, sweep rule, topology, backend settings |
| `configs/cresco8.env` | Cluster paths only (no hosts, accounts or credentials) |
| `scripts/benchmark/bench_client.py` | Common closed-loop OpenAI-compatible load generator (stdlib only) |
| `scripts/benchmark/telemetry.py` | 1 s telemetry: per-thread core PMU, CPU utilization, memory, server metrics, optional uncore |
| `scripts/benchmark/run_online.py` | Experiment B: server launch, warm-up, repetitions, telemetry, adaptive concurrency sweep |
| `scripts/benchmark/run_offline.py` | Experiment A: `llama-bench`, `llama-batched-bench`, `vllm bench latency/throughput` |
| `scripts/benchmark/stream_calibration.py` | Experiment 0: STREAM triad under the same telemetry (baseline + proxy validation) |
| `scripts/benchmark/prepare_models.py` | BF16 and Q8_0 GGUFs from the same HF snapshot vLLM serves |
| `scripts/benchmark/tuning.py` | Small documented tuning phase |
| `scripts/cluster/job.sbatch`, `scripts/cluster/submit.sh` | Slurm wrapper (cache-mode check, stale-server reaping, env capture, node health check) |
| `src/llm_backend_analysis/` | Local processing package (`scripts/processing/build_processed_data.py` runs it) |
| `data/raw/environment/` | Per-node environment manifests written by `scripts/benchmark/capture_env.py` |
| `data/raw/` | Mirror of the cluster outputs (`scripts/cluster/sync_results.sh`) |
| `data/processed/` | Normalized CSV and Parquet tables (generated) |

## Design summary

**Hardware.** CRESCO8 `cresco8_hbm` nodes run Xeon CPU Max 9480 in HBM **cache mode**:
2 sockets × 56 cores, no SMT, 64 GiB HBM2e per socket caching about 512 GB of DDR5.
Every run uses **one socket** with the same layout for both backends:

| Role | CPUs | Memory |
| --- | --- | --- |
| Server compute threads (55 OpenMP threads, one per core) | 0–54 | `numactl --membind=0` |
| Server HTTP front end (both servers' whole process tree is limited to 0–55) | 55 | node 0 |
| Client, telemetry sampler, orchestrator | 100–111 (socket 1) | node 1 |

**Arms.** Each arm uses the same Llama weights (same HF snapshot and revision):

| Arm | Backend | Weights | KV cache | AMX use |
| --- | --- | --- | --- | --- |
| `vllm-bf16` | vLLM 0.30.0 (official CPU image) | safetensors BF16 | BF16 | GEMMs and attention (BF16 tiles) |
| `llamacpp-bf16` | llama.cpp `9655061` (gcc 14, AMX build) | GGUF BF16, lossless conversion | F16 (or BF16 after tuning) | none: llama.cpp has no BF16 AMX kernel |
| `llamacpp-q8_0` | same | GGUF Q8_0, quantized from the BF16 GGUF | F16 | weight GEMMs with ≥ 2 rows (int8 tiles) |

`vllm-bf16` vs `llamacpp-bf16` is the **precision-matched** pair. `llamacpp-q8_0` is
llama.cpp's own best configuration on this CPU. It is plotted next to the others and
never divided into the BF16 vLLM numbers without an explicit precision caveat.

**Models.** The primary model is Llama-3.1-8B-Instruct. The secondary model is
Llama-3.2-1B-Instruct, used to check whether a behavior belongs to the backend or only
to the 8B model.

**Workloads (ISL→OSL):** decode-heavy 128→512, balanced 512→512, prefill-heavy
4096→128. Long context (8192→256) is optional.

**Fairness controls (online).**
- **One client.** `bench_client.py` drives both servers. Same request body, same timing points, one fresh connection per request.
- **Prompts.** Prompts are token-id lists (BOS + ISL−1 pseudo-random ordinary tokens, deterministic per request). Both servers get exactly ISL tokens, with no tokenizer or chat-template difference and no shared prefixes. Every repetition records its prompt manifest SHA-256.
- **Output length.** `max_tokens = OSL`, `ignore_eos`, `temperature 0`. The client checks `usage.prompt_tokens == ISL` and `usage.completion_tokens == OSL` for every request.
- **Prompt caching** is off in both: vLLM `--no-enable-prefix-caching`, llama-server `--no-cache-prompt -cram 0` plus per-request `cache_prompt:false`.
- **KV sizing.** Each server is sized for exactly C full-length sequences: llama-server `-np C -c C×ctx_slot --no-kv-unified`, vLLM `VLLM_CPU_KVCACHE_SPACE` = C × seq_len × KV bytes × 1.15.
- **Fresh state.** A fresh server is started for every concurrency point. Each point gets a 10 s idle baseline, warm-up, then 3 measured repetitions.
- **Continuous batching.** llama-server gets `-cb` and one slot per in-flight request. vLLM batches natively.

**Load model.** The load is a closed loop: C workers, each sending its next request as
soon as the previous one finishes. Worker start times are staggered over one estimated
request duration, so in-flight requests are spread over their life cycle. Throughput is
measured in a **steady-state window** in which exactly C requests are in flight. The
window starts when every worker has received its first token and ends when the first
worker runs out of requests. Latency percentiles use the requests that ran fully inside
that regime.

**Sweep rule.** The rule is declared before any measurement and applied identically to
every arm:
- The sweep runs C = 1, 2, 4, …, 64.
- It stops after two consecutive doublings that each add less than 5% throughput (a confirmed plateau).
- It extends to 128 and 256 only while the last doubling still added at least 10%.
- A point projected to take more than 3 h is skipped and the skip is logged.

## Telemetry: what can and cannot be measured

The node runs `kernel.perf_event_paranoid = 2`, with no root, `perf` or `pcm`. Every
**uncore** PMU is refused to unprivileged users:
- DDR IMC CAS counters (`uncore_imc_*`)
- the HBM controllers (`uncore_type_14_*`, i.e. MCHBM)
- the near-memory-cache tag counters (`uncore_m2m_*`)
- RAPL

Per-thread **core** counters are allowed. The sampler therefore measures the server's
memory traffic from the core side, using Intel's offcore-response events for Sapphire
Rapids:

| Quantity | Events |
| --- | --- |
| Memory reads | `OCR.READS_TO_CORE.L3_MISS_LOCAL` + `OCR.HWPF_L3.L3_MISS_LOCAL` (64 B per count) |
| Memory writes | `OCR.WRITE_ESTIMATE.MEMORY` (Intel labels it an estimate) |
| AMX activity | `EXE.AMX_BUSY` / cycles |

In cache mode these counters see **traffic to the memory side as a whole**, which is HBM
cache plus DDR. They cannot separate HBM hits from DDR misses. Two steps compensate:

1. **STREAM calibration (Experiment 0).** STREAM runs under the same sampler on the same
   cores at working sets from 4 to 96 GiB. Its traffic is known, so the proxy's capture
   ratio is measured. Application traffic is then normalized by STREAM traffic measured
   with the **same instrument** at a comparable working set. The headline utilization is
   *application memory traffic / sustainable STREAM traffic*, not a fraction of the
   theoretical HBM peak.
2. **Working-set reasoning.** In cache mode, a working set well below 64 GiB is served
   almost entirely by HBM after warm-up. The model reports its active working set
   (weights + live KV) for every point, so it is clear which regime applies.

If an administrator lowers `perf_event_paranoid` (≤ 0) on one node, the sampler starts
reading real HBM and DDR CAS counts and M2M tag hits/misses automatically.

## Reproducing

On the cluster login node (all compute runs through Slurm):

```bash
cd ~/dev/llm-backend-analysis
bash scripts/cluster/submit.sh prepare          # GGUFs into $BC_DATA/models (once)
bash scripts/cluster/submit.sh calibrate        # Experiment 0
bash scripts/cluster/submit.sh smoke            # validation matrix
bash scripts/cluster/submit.sh tuning vllm-bf16 # tuning phase (also llamacpp-bf16, llamacpp-q8_0)
bash scripts/cluster/submit.sh offline          # Experiment A (one job per model x arm)
bash scripts/cluster/submit.sh online           # Experiment B (one job per model x arm x workload)
```

Locally:

```bash
bash scripts/cluster/sync_results.sh            # cluster -> data/raw/
uv run python scripts/processing/build_processed_data.py
```

Each concurrency point writes `metadata.json`. It holds the exact server command, the
client commands, all settings, host facts, per-repetition summaries and the telemetry
file reference. Per-request records are in `rep*.jsonl.gz`, and server logs are kept.

## Runbook: INT8 pair (vLLM W8A8 vs llama.cpp Q8_0), after the 2026-10-05 maintenance

Already done:
- `vllm-w8a8` checkpoints (`$BC_DATA/models/<model>/w8a8-rtn/`, built by `scripts/benchmark/quantize_w8a8.py` with the
  llm-compressor venv `$BC_DATA/venvs/llmcompressor`; versions pinned in `configs/requirements/llmcompressor.txt`).
- The quality/AMX check (`scripts/benchmark/quality_check.py`, `data/raw/quality/`).

Still queued from the main campaign, starting on their own when the HBM nodes return:
- `llamacpp-q8_0` 4096→1024 (8B);
- 1B `llamacpp-q8_0` offline.

On the login node:

```bash
cd ~/dev/llm-backend-analysis
bash scripts/cluster/submit.sh int8-tuning                 # ~35 min on one HBM node
```

Locally, after it finishes:

```bash
bash scripts/cluster/sync_results.sh && uv run python scripts/analysis/tuning_summary.py
```

Copy the chosen `max_num_batched_tokens` into `[tuned.vllm-w8a8]` in `configs/experiment.toml`, sync the
repository to the cluster, then:

```bash
BC_CAMPAIGN=main bash scripts/cluster/submit.sh int8-campaign   # 5 online jobs + 1 offline job, 8B
```

When the jobs finish, rerun `sync_results.sh` and `scripts/processing/build_processed_data.py` (the analysis notebooks are being rebuilt).

## Status and deviations

Results and deviations are being moved from the retired report into notebooks and `docs/`.
