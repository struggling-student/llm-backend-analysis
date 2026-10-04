# Extending the analysis

The pipeline is data-driven: configurations, campaigns, workloads and models are described by metadata, and
the analysis discovers them from the processed tables. Most additions need no code change.

## New results for an existing configuration (e.g. more concurrency points, another workload)

1. Run on the cluster and `make sync` (raw files land under `data/raw/` with the cluster's directory names).
2. `make checksums` — appends the new files to `data/metadata/raw_checksums.sha256`. It refuses if an already
   recorded raw file changed; raw files are never edited (if a re-synced file legitimately changed because a run
   was unfinished, inspect it and use `uv run python scripts/utilities/raw_checksums.py update --accept-changed`).
3. `make data notebooks validate`.

A new **workload** needs an entry under `[workloads.*]` in `configs/experiment.toml` (the harness reads it
anyway); labels and ordering in tables and figures come from there.

## A new configuration (arm)

1. Add the arm to `configs/experiment.toml` `[arms.*]` (the harness needs it to run).
2. Add one table to [`data/metadata/configurations.toml`](../data/metadata/configurations.toml):

   ```toml
   [configurations.<backend>_<precision>]
   arm = "<harness arm id>"
   backend = "vllm" | "llama_cpp"
   precision = "bf16" | "int8"
   quantization = "none" | "q8_0" | "w8a8_rtn" | …   # add [quantizations.<name>] with a label if new
   weight_format = "…"
   kv_cache_dtype = "…"
   amx_path = "…"
   tuning_default_variant = "…"
   status = "measured" | "pending"
   ```

3. `make data notebooks validate`. `make validate` fails if a raw arm or a harness arm has no configuration.

Styling follows from metadata: colour = backend, solid/filled vs dashed/open = precision, marker shape = backend
(`visualization/theme.py`). Backend comparisons are formed only within one precision
(`analysis.comparisons.backend_ratio`); `comparisons.backend_pairs` lists the precisions for which both backends
have data, so the INT8 pair appears as soon as `vllm_int8` has measurements.

### vLLM INT8 checklist

- [ ] `int8-tuning` synced, `summarize_tuning.py` choice copied into `[tuned.vllm-w8a8]`
- [ ] `int8-campaign` synced, `make checksums`
- [ ] `vllm_int8.status = "measured"` in `configurations.toml`
- [ ] `make data notebooks validate`
- [ ] notebook 08 section 3 shows the INT8 pair; notebook 09 claim "INT8 backend pair measured" holds
- [ ] interpretations flagged "✗ revisit" in any claims table reviewed and updated
- [ ] Notion pages updated from `results/tables/backend_selection/key_numbers.csv` and the regenerated figures

## A new campaign

Every directory `data/raw/<kind>/<campaign>/` must be declared in
[`data/metadata/campaigns.toml`](../data/metadata/campaigns.toml) with a `role` (`primary`, `tuning`,
`validation`, `superseded`, `failed`) and, for superseded or failed attempts, a `reason`. Processing reads every
campaign and stamps rows with the campaign and its role; loaders default to the primary campaign
(`registry.ANALYSIS_CAMPAIGN`).

## A new notebook

- One notebook answers one question; name it `NN_short_question.ipynb` in the topic folder.
- Start with title, objective, research question, data, method and assumptions; end with conclusions and
  limitations.
- Load data only through `llm_backend_analysis.data.loaders`; derive metrics with `analysis.*`; build figures with
  `visualization.plots` (add a builder there rather than a long plotting cell).
- Quote numbers in prose only through `reporting.show(...)` with computed values; back interpretive statements
  with `reporting.Claim`s where they could change with new data.
- `make notebooks` must run it top to bottom; `make validate` rejects errors, warnings, unexecuted cells and
  absolute paths in outputs.

## Exporting a figure or table for documentation

Export only figures that support a statement in the documentation:

```python
export_figure(fig, "snake_case_name", source=NOTEBOOK, caption="What the figure shows.")
```

This writes `results/figures/<topic>/<name>.svg` and `.png` from the same figure object the notebook displays and
records it in `figures.json`. Tables go through `reporting.export_table(df, name)` into `results/tables/<topic>/`.
Never edit exported files by hand; `make validate` flags files that no notebook produces any more.
