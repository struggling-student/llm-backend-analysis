# Cluster runbook (CRESCO8)

The benchmark harness runs on the cluster through Slurm; analysis runs locally. All compute goes through
`sbatch`; nothing heavy runs on the login node.

## Layout on the cluster

| What | Where |
| --- | --- |
| Code checkout | `~/dev/llm-backend-analysis` (AFS home; a clone of this repository) |
| Paths used by the harness | [`configs/cresco8.env`](../configs/cresco8.env) (paths only — no hosts or credentials) |
| Outputs (`$BC_DATA`) | `raw_results/`, `telemetry/`, `environment/`, `node_checks/`, `models/`, `slurm-logs/` on scratch |

The harness (`scripts/benchmark/`) uses only the Python standard library and the interpreter `$BC_PYTHON`
defined in `cresco8.env`; it never imports the local analysis package.

### Updating the checkout after the 2026-10-04 restructuring

Until 2026-10-04 the code lived in an untracked `backend_comparison/` folder of the checkout. The repository root
is now the project root, so the cluster checkout must be updated once before the next job:

```bash
cd ~/dev/llm-backend-analysis
git pull --ff-only
```

Then remove the old untracked `backend_comparison/` folder (it holds code only; results live in `$BC_DATA`). The
data directories of the repository (`data/raw/`, ~330 MB) are not needed on the cluster; to keep them out of the
AFS home use a sparse checkout:

```bash
git sparse-checkout set --no-cone '/*' '!/data/raw/'
```

`scripts/cluster/sync_results.sh` still finds `$BC_DATA` through an old-layout checkout, so syncing works before
and after the update.

## Submitting jobs (login node)

```bash
cd ~/dev/llm-backend-analysis
bash scripts/cluster/submit.sh prepare          # BF16 and Q8_0 GGUFs into $BC_DATA/models (once)
bash scripts/cluster/submit.sh calibrate        # Experiment 0: STREAM + PMU-proxy validation
bash scripts/cluster/submit.sh smoke            # validation matrix
bash scripts/cluster/submit.sh tuning vllm-bf16 # tuning phase (also llamacpp-bf16, llamacpp-q8_0)
bash scripts/cluster/submit.sh offline          # Experiment A (one job per model x arm)
bash scripts/cluster/submit.sh online           # Experiment B (one job per model x arm x workload)
```

`BC_CAMPAIGN` selects the campaign name (default `main`); `BC_DEPENDENCY` adds a Slurm dependency. Each
concurrency point writes `metadata.json` with the exact server and client commands, settings, host facts and
per-repetition summaries; per-request records are in `rep*.jsonl.gz`.

## Getting results into the repository (local machine)

```bash
make sync           # scripts/cluster/sync_results.sh: rsync $BC_DATA -> data/raw (never deletes)
make checksums      # record the new raw files; refuses if an existing raw file changed
make data notebooks validate
```

## INT8 pair: vLLM W8A8 vs llama.cpp Q8_0

Already done: W8A8 checkpoints in `$BC_DATA/models/<model>/w8a8-rtn/` (built by
`scripts/benchmark/quantize_w8a8.py` with the llm-compressor venv, versions in
[`configs/requirements/llmcompressor.txt`](../configs/requirements/llmcompressor.txt)) and the quality/AMX check
(`data/raw/quality/`). Still queued from the BF16 campaign: llama.cpp Q8_0 4096→1024 (8B) and the 1B Q8_0
offline job.

1. Tuning (login node, ~35 min on one HBM node):

   ```bash
   bash scripts/cluster/submit.sh int8-tuning
   ```

2. Locally, after it finishes:

   ```bash
   make sync && uv run python scripts/processing/summarize_tuning.py
   ```

   Copy the chosen `max_num_batched_tokens` into `[tuned.vllm-w8a8]` in `configs/experiment.toml`, commit, and
   update the cluster checkout.

3. Campaign (login node):

   ```bash
   BC_CAMPAIGN=main bash scripts/cluster/submit.sh int8-campaign   # 5 online jobs + 1 offline job, 8B
   ```

4. Locally: `make sync checksums`, set `status = "measured"` for `vllm_int8` in
   `data/metadata/configurations.toml`, then `make data notebooks validate`. The INT8 pair appears in notebooks
   04–09 automatically (blue dashed lines, open circles); revisit any interpretation the claim tables flag, and
   update the Notion pages from `results/tables/backend_selection/key_numbers.csv`.
