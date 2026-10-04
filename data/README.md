# Data

```
raw/        measurement source of truth — never edited
metadata/   what the raw data are: configurations, campaigns, integrity manifest
processed/  derived tables, rebuilt by `make data` (deterministic; committed so notebooks run without rebuilding)
```

## Raw measurements (`raw/`)

Byte-identical copies of the cluster outputs (`$BC_DATA` on CRESCO8 scratch), synced by
`scripts/cluster/sync_results.sh`. Paths mirror the cluster so that re-syncing never duplicates files.
`metadata/raw_checksums.sha256` records the SHA-256 of every file; `make validate` fails if one changes.

| Path | Produced by | Contents |
| --- | --- | --- |
| `online/<campaign>/<model>/<arm>[__<variant>]/<workload>/c<C>/` | `scripts/benchmark/run_online.py` | `metadata.json` (server/client commands, settings, host, per-repetition summaries), `rep<N>.jsonl.gz` (per-request token timestamps), `rep<N>.json`, `warmup.json`, `server.log`, `client.log`, `numastat.txt` |
| `online/…/<workload>/sweep.json` | idem | sweep plan, per-point status and why the sweep ended |
| `offline/<campaign>/<model>/<arm>/<workload>/` | `scripts/benchmark/run_offline.py` | `offline.json` (every tool run, command, parsed result) and tool logs |
| `stream/<campaign>/stream_calibration.json` | `scripts/benchmark/stream_calibration.py` | STREAM triad runs at 4–96 GiB, 55/56 threads |
| `quality/quality/<model>/` | `scripts/benchmark/quality_check.py` | generated texts, agreement summary, AMX activity, server logs |
| `telemetry/<kind>/<campaign>/…/<point>.jsonl` | `scripts/benchmark/telemetry.py` | 1 s samples: core PMU counters, CPU busy per core, RSS, NUMA usage, server metrics |
| `environment/nodes/<host>-<job>.json` | `scripts/benchmark/capture_env.py` | per-job environment manifest (kernel, CPU flags, versions, hashes) |
| `environment/node_checks/<host>-<job>.json` | `scripts/cluster/job.sbatch` | per-job 10 s STREAM node health check |
| `environment/models/<model>/manifest.json` | `scripts/benchmark/prepare_models.py`, `quantize_w8a8.py` | model files, sizes, SHA-256 |

Arm ids (`vllm-bf16`, `llamacpp-bf16`, `llamacpp-q8_0`, `vllm-w8a8`) and campaign names are the harness's
identifiers and stay as recorded; `metadata/configurations.toml` maps arms to canonical names and
`metadata/campaigns.toml` gives every campaign its role. Superseded and failed attempts are kept on purpose.

Raw files contain internal cluster host names, Slurm job ids and scratch paths as provenance; they contain no
credentials or tokens.

## Processed tables (`processed/`)

Built by `src/llm_backend_analysis/data/processing/`. Every table carries `campaign`, `campaign_role` and the
canonical `configuration_id`, `backend`, `precision`, `quantization` where applicable, plus a `source` path into
`raw/`. Contracts (required columns, unique keys, consistency with the metadata) are in
`src/llm_backend_analysis/data/schema.py`.

| File | One row per | Key columns |
| --- | --- | --- |
| `online_repetitions.csv` | repetition of an online point | `output_tok_s`, `total_tok_s` (steady window), latency `*_p50/_p95/_p99` per repetition, telemetry over the window (`mem_read_gbs`, `mem_write_gbs`, `mem_total_gbs`, `amx_busy_frac`, `cpu_busy_compute_mean`, `tokens_per_step`, `engine_steps_per_s`, …), STREAM baselines, `bw_util_vs_stream_max`, validity flags |
| `online_points.csv` | (campaign, model, arm, workload, C) | `<metric>_mean/_std/_min/_max/_ci95/_cv` over repetitions; `<latency>_pooled_p50/p95/p99` over all steady requests (`latency_n`, `p99_meaningful`); `n_reps`, `tokens_ok`, `manifest_consistent`, `failed_requests` |
| `online_requests.parquet` | request | `ttft_s`, `tpot_s`, `e2e_s`, `steady`, token counts |
| `online_sweeps.csv` | concurrency sweep | last point, its status and skip reason, the declared rule |
| `offline_samples.csv` | measured offline sample | `tool`, `benchmark` (`batch1`, `batched`, `throughput`), `batch`, `test` (`prefill`, `decode`, `total`), `tok_s` |
| `stream_calibration.csv` | STREAM run | `stream_triad_gbs`, PMU `mem_*_gbs`, `pmu_*_over_stream*` ratios |
| `node_checks.csv` | Slurm job | `triad_gbs` of the node health check |
| `tuning_runs.csv` | tuning point | `variant`, `is_default_variant`, `total_tok_s`, `tpot_s_p50`, server settings and command |
| `pmu_overhead.csv` | PMU on/off run | `pmu_enabled`, `run_order`, throughput and latency |
| `quality_check.csv` | (configuration, reference) | identical continuations, common-prefix words, AMX busy fraction |

Units: tokens/s, seconds (latency columns end in `_s`), GB/s (10⁹ B/s), GiB for working sets, fractions in
[0, 1] for utilizations.
