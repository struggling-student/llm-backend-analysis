# Results (presentation exports)

Everything here is **written by the notebooks** (`make notebooks`) from the same figure and table objects they
display; it is a presentation layer for Notion, the thesis, slides and this repository's documentation, not an
independent analysis. Do not edit these files by hand; regenerate them.

## `figures/backend_selection/`

Each figure exists as SVG (thesis, vector) and PNG (Notion, slides). [`figures.json`](figures/backend_selection/figures.json)
lists every figure with its source notebook and caption; `make validate` checks that the list and the files agree.

| Figure | Source notebook | Supports |
| --- | --- | --- |
| `throughput_vs_concurrency` | 04 | vLLM keeps scaling with concurrency, llama.cpp flattens early |
| `backend_ratio_vs_concurrency` | 04 | crossover and growing vLLM ÷ llama.cpp ratio at matched precision |
| `peak_throughput_by_workload` | 09 | best operating point of each configuration per workload |
| `single_request_latency` | 03 | llama.cpp's per-token advantage and vLLM's TTFT advantage at C = 1 |
| `latency_throughput_tradeoff` | 05 | throughput at equal per-token latency |
| `step_cost_vs_batch` | 06 | why: the cost of a batched forward step |
| `throughput_and_memory_traffic` | 07 | memory traffic falls with concurrency; neither backend nears the STREAM ceiling |
| `memory_traffic_components` | 07 | weight vs KV read traffic; KV share grows with C |
| `offline_decode_throughput` | 02 | offline crossover between batch 1 and batched decode |
| `llama_cpp_int8_over_bf16` | 08 | INT8 raises llama.cpp's level, not its scaling |
| `stream_calibration` | 01 | sustainable bandwidth and validation of the memory-traffic instrument |

## `tables/backend_selection/`

| Table | Content |
| --- | --- |
| `key_numbers.csv` | the computed values (ranges) quoted by the Notion pages, with definitions |
| `evidence_summary.csv` | criterion-by-criterion evidence used for the decision |
| `curve_summary.csv` | per configuration, model and workload: C = 1 values, peak, knee, highest C |
| `backend_ratio_bf16.csv` | vLLM BF16 ÷ llama.cpp BF16 total throughput at every common C |
