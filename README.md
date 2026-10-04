# LLM backend analysis: vLLM vs llama.cpp on Intel Xeon Max (HBM)

Backend selection for an MSc thesis on **HBM-aware LLM inference** on Intel Xeon CPU Max 9480 nodes
(CRESCO8, HBM in cache mode). The thesis studies concurrent online serving and how it loads the HBM/DDR
memory system; this repository holds the experiment that decides which inference backend those studies
run on.

> **Question.** Which backend, vLLM or llama.cpp, is the better platform for *concurrent online inference* on
> Xeon Max — one whose throughput responds to concurrency and that can load the memory subsystem enough to study
> HBM/DDR behaviour?
>
> **Current answer.** **vLLM**, on the precision-matched BF16 comparison. llama.cpp is the better single-request
> decoder but stops scaling after a few doublings of concurrency; vLLM keeps scaling and delivers several times
> more throughput under load. Concurrency does *not* by itself raise total memory traffic for either backend.
> The vLLM INT8 validation is pending. Evidence, numbers and caveats:
> [`notebooks/backend_selection/09_backend_selection_summary.ipynb`](notebooks/backend_selection/09_backend_selection_summary.ipynb).

## Source-of-truth hierarchy

| Layer | Where | Role |
| --- | --- | --- |
| Raw benchmark data | [`data/raw/`](data/raw) | **Measurement source of truth.** Byte-identical copies of the cluster outputs; never edited (checked by `data/metadata/raw_checksums.sha256`) |
| Experiment metadata | [`data/metadata/`](data/metadata), [`configs/`](configs) | What each configuration and campaign is; harness parameters |
| Processed datasets | [`data/processed/`](data/processed) | Reproducible derived tables, rebuilt deterministically by `make data` |
| Jupyter notebooks | [`notebooks/`](notebooks) | **Canonical analysis and interpretation**: data, transformations, plots, conclusions |
| Static figures and tables | [`results/`](results) | Presentation-only exports written by the notebooks themselves |
| Notion | thesis workspace | Research communication; quotes `results/tables/backend_selection/key_numbers.csv` |

Each layer is derived from the one above it by code in this repository; nothing is copied by hand.

```
data/raw  ──make data──▶  data/processed  ──make notebooks──▶  notebooks (executed)  ──▶  results/figures, results/tables  ──▶  Notion / thesis
```

## Repository structure

```
.
├── configs/                  harness parameters (experiment.toml), cluster paths (cresco8.env), pinned venv requirements
├── data/
│   ├── raw/                  immutable measurements, mirroring the cluster's $BC_DATA (online, offline, stream, quality,
│   │                         telemetry, environment)
│   ├── metadata/             configurations.toml, campaigns.toml, raw_checksums.sha256
│   └── processed/            CSV/Parquet tables built from raw data
├── notebooks/backend_selection/   01–09: one analytical question per notebook
├── results/
│   ├── figures/backend_selection/ selected SVG + PNG exports and their manifest (figures.json)
│   └── tables/backend_selection/  derived tables quoted by documentation (key_numbers.csv, …)
├── scripts/
│   ├── benchmark/            cluster-side harness: server runners, shared client, telemetry, model preparation
│   ├── cluster/              Slurm submission, job wrapper, result sync
│   ├── processing/           build_processed_data.py, summarize_tuning.py
│   └── utilities/            execute_notebooks.py, validate_repository.py, raw_checksums.py
├── src/llm_backend_analysis/ installable package: data (registry, telemetry, processing, loaders, schema),
│                             analysis (metrics, comparisons, memory, tuning), visualization (theme, plots, export)
├── docs/                     methodology, cluster runbook, how to extend the analysis
└── tests/
```

## Setup

Local analysis needs [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11 (the lock file pins every dependency):

```bash
make setup          # uv sync: creates .venv and installs llm_backend_analysis in editable mode
```

Static figure export uses Kaleido 0.2.1 (installed by `uv sync`, no browser needed). The cluster harness in
`scripts/benchmark/` uses only the Python standard library and runs with the cluster's own interpreter; see
[`docs/cluster_runbook.md`](docs/cluster_runbook.md).

## Reproducing the analysis

```bash
make data           # data/raw -> data/processed (deterministic)
make notebooks      # execute notebooks 01-09 top to bottom; rewrites results/figures and results/tables
make validate       # raw-data checksums, metadata coverage, freshness, notebook hygiene, figure manifest, links + tests
```

`make all` runs the three steps. Re-running on unchanged data leaves git clean: processed tables, executed
notebooks and SVG/PNG exports are byte-reproducible. Open the notebooks with `uv run jupyter lab`.

## Experiments and naming

| Dimension | Values | Defined in |
| --- | --- | --- |
| `configuration_id` | `vllm_bf16`, `llama_cpp_bf16`, `llama_cpp_int8`, `vllm_int8` (pending) | `data/metadata/configurations.toml` |
| `backend` | `vllm`, `llama_cpp` | idem |
| `precision` | `bf16`, `int8` | idem |
| `quantization` | `none`, `q8_0`, `w8a8_rtn` | idem |
| `model` | `llama31_8b` (primary), `llama32_1b` (cross-check) | `configs/experiment.toml` |
| `workload` | `decode` 128→512, `balanced` 512→512, `prefill` 4096→128, `longctx` 8192→256, `kvdecode` 4096→1024 | idem |
| `campaign`, `campaign_role` | e.g. `main` (primary), `tuning`, `smoke` (validation), `tuning-v1-queue-bug` (superseded) | `data/metadata/campaigns.toml` |
| `concurrency` | C = 1, 2, 4, … (declared sweep rule) | `configs/experiment.toml` |

The cluster harness identifies a configuration by its **arm** (`vllm-bf16`, `llamacpp-q8_0`, …). Arm ids are
frozen in the raw directory names and metadata, so raw data are never renamed; the processing layer maps every
arm to the canonical columns above. Conventions: `snake_case` for modules, files, columns and figure names;
notebooks are numbered `NN_question.ipynb`; every processed row carries its raw `source` path.

Raw layout (mirrors the cluster):

```
data/raw/online/<campaign>/<model>/<arm>[__<variant>]/<workload>/c<C>/   metadata.json, rep*.jsonl.gz, server.log, …
data/raw/offline/<campaign>/<model>/<arm>/<workload>/offline.json
data/raw/telemetry/<kind>/<campaign>/…/<point>.jsonl                      1 s telemetry samples
data/raw/stream/<campaign>/stream_calibration.json
data/raw/environment/{nodes,node_checks,models}/                          node manifests, per-job STREAM checks, model hashes
```

## Adding results (e.g. vLLM INT8)

1. Run the jobs on the cluster ([`docs/cluster_runbook.md`](docs/cluster_runbook.md)) and `make sync`.
2. `make checksums` to record the new raw files (existing files must be unchanged).
3. If a new arm or campaign appears, add one table to `data/metadata/configurations.toml` /
   `campaigns.toml` (`vllm_int8` is already declared; set its `status` to `"measured"`).
4. `make data && make notebooks && make validate`.

No analysis code changes are needed: configurations, colours (backend) and line styles (precision) come from
metadata, and the INT8 backend pair is formed automatically once both backends have INT8 data. Details:
[`docs/extending_the_analysis.md`](docs/extending_the_analysis.md).

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — hardware, configurations, fairness controls, load model,
  telemetry, deviations and incidents.
- [`docs/cluster_runbook.md`](docs/cluster_runbook.md) — running the harness on CRESCO8, including the INT8 pair.
- [`docs/extending_the_analysis.md`](docs/extending_the_analysis.md) — adding configurations, workloads,
  campaigns, notebooks and figure exports.
- [`data/README.md`](data/README.md) — data provenance and processed-table reference.
- [`notebooks/README.md`](notebooks/README.md) — the analytical question answered by each notebook.
- [`results/README.md`](results/README.md) — exported figures and tables, and where they are used.
- [`docs/notion.md`](docs/notion.md) — which Notion pages summarize this analysis and how to keep them in sync.
