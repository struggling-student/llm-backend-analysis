# Scripts

| Folder | Runs on | Contents |
| --- | --- | --- |
| `benchmark/` | cluster compute nodes (via Slurm), stdlib Python | `run_online.py` (Experiment B), `run_offline.py` (Experiment A), `stream_calibration.py` (Experiment 0), `bench_client.py` (shared closed-loop client), `telemetry.py` (1 s PMU/CPU/memory sampler), `tuning.py`, `smoke.py`, `pmu_overhead_check.py`, `quality_check.py`, `prepare_models.py`, `quantize_w8a8.py`, `capture_env.py`, `bclib.py` (shared helpers, config loading) |
| `cluster/` | login node / local machine | `submit.sh` (job submission), `job.sbatch` (job wrapper: cache-mode check, orphan reaping, environment capture, node check), `sync_results.sh` (cluster → `data/raw/`, local) |
| `processing/` | local | `build_processed_data.py` (`make data`), `summarize_tuning.py` (tuning grid and selection rule) |
| `utilities/` | local | `execute_notebooks.py` (`make notebooks`), `validate_repository.py` (`make validate`), `raw_checksums.py` (`make checksums`) |

The harness reads every parameter from `configs/experiment.toml` and every path from `configs/cresco8.env`; see
[`../docs/cluster_runbook.md`](../docs/cluster_runbook.md).
