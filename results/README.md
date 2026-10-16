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
| `throughput_and_backend_ratio` | 09 | summary: total tokens/s vs C for the four configurations, and vLLM ÷ llama.cpp at equal C for both precision pairs |
| `traffic_and_bytes_per_token` | 09 | summary: memory traffic and memory bytes per output token vs C, with the STREAM ceiling |
| `single_request_latency` | 03 | at C = 1: llama.cpp's per-token advantage at BF16, vLLM's at INT8; vLLM's TTFT advantage |
| `latency_throughput_tradeoff` | 05 | throughput at equal per-token latency |
| `step_cost_vs_batch` | 06 | why: the cost of a batched forward step |
| `throughput_and_memory_traffic` | 07 | memory traffic falls with concurrency; neither backend nears the STREAM ceiling |
| `memory_traffic_components` | 07 | weight vs KV read traffic; KV share grows with C |
| `offline_decode_throughput` | 02 | offline crossover between batch 1 and batched decode |
| `llama_cpp_int8_over_bf16` | 08 | INT8 raises llama.cpp's level, not its scaling |
| `backend_ratio_vs_concurrency_int8` | 08 | INT8 pair: vLLM W8A8 ahead of llama.cpp Q8_0 at every C |
| `stream_calibration` | 01 | sustainable bandwidth and validation of the memory-traffic instrument |
| `roofline_by_configuration` | 10 | every serving point against the socket's memory and compute roofs, per configuration |
| `roofline_by_workload` | 10 | (appendix) both backends on one roofline per workload: placement and movement with concurrency |
| `ridge_ratio_vs_concurrency` | 10 | (appendix) when each configuration crosses from the memory-bound to the compute-bound side |
| `roofline_utilization` | 10 | (appendix) share of the memory roof vs share of the compute roof: which limit each point is near |
| `decode_intensity_limit` | 10 | why long-context decode stays memory-bound at any batch size (model) |
| `amx_model_cross_check` | 10 | (appendix) validation of the operation model against measured AMX busy cycles |

## `tables/backend_selection/`

| Table | Content |
| --- | --- |
| `key_numbers.csv` | the computed values (ranges) quoted by the Notion pages, with definitions |
| `evidence_summary.csv` | criterion-by-criterion evidence used for the decision |
| `curve_summary.csv` | per configuration, model and workload: C = 1 values, peak, knee, highest C |
| `backend_ratio_bf16.csv` | vLLM BF16 ÷ llama.cpp BF16 total throughput at every common C |
| `backend_ratio_int8.csv` | vLLM INT8 ÷ llama.cpp INT8 total throughput at every common C (8B) |
| `roofline_points.csv` | per online point: arithmetic rate, operational intensity, memory and compute roof, ridge, utilizations, regime |
| `roofline_compute_ceilings.csv` | compute roofs of the 55 server cores: instruction peaks, best measured GEMMs, vLLM projection-kernel ceilings |
