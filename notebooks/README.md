# Notebooks

The notebooks are the canonical analysis: each loads processed data through `llm_backend_analysis.data.loaders`,
computes its metrics with `llm_backend_analysis.analysis`, draws Plotly figures with
`llm_backend_analysis.visualization`, and states its interpretation, conclusions and limitations. Numbers quoted
in prose are computed in the notebook. Run them all with `make notebooks`; they are committed executed, with an
interactive Plotly output and a PNG fallback per figure (for viewers that do not run JavaScript, such as GitHub).

## `backend_selection/`

| Notebook | Question |
| --- | --- |
| [`01_measurement_validity`](backend_selection/01_measurement_validity.ipynb) | Is the comparison fair, and can its measurements (client, telemetry, nodes, tuning) be trusted? |
| [`02_offline_engine_benchmarks`](backend_selection/02_offline_engine_benchmarks.ipynb) | How fast does each engine prefill and decode at fixed batch sizes, without a server? |
| [`03_single_request_performance`](backend_selection/03_single_request_performance.ipynb) | With one request in flight, which backend gives the lower TTFT/TPOT and higher throughput? |
| [`04_concurrency_scaling`](backend_selection/04_concurrency_scaling.ipynb) | Which backend turns more concurrent requests into more aggregate throughput, and how far? |
| [`05_online_serving_latency`](backend_selection/05_online_serving_latency.ipynb) | What per-request latency comes with that throughput; how much throughput fits a latency budget? |
| [`06_batching_and_amx`](backend_selection/06_batching_and_amx.ipynb) | Why do they scale differently: batching, step cost, AMX use, core occupancy? |
| [`07_memory_traffic`](backend_selection/07_memory_traffic.ipynb) | Does concurrency raise memory-system pressure, and which backend keeps HBM busier? |
| [`08_precision_int8`](backend_selection/08_precision_int8.ipynb) | What does INT8 change in each engine, and does the backend ranking hold for the INT8 pair? |
| [`09_backend_selection_summary`](backend_selection/09_backend_selection_summary.ipynb) | Synthesis for the supervisors: three figures → about eight measured claims → labelled interpretation (inferred / hypothesis) → decision → table of what is measured, inferred, hypothesized or untested → thesis implications |
| [`10_roofline`](backend_selection/10_roofline.ipynb) | Roofline: which regimes are memory- or compute-bound, how close each backend gets to the socket's roofs, why long-context decode stays memory-bound; supporting figures in an appendix |

Primary model: Llama-3.1-8B-Instruct; Llama-3.2-1B-Instruct is the cross-check. Direct backend comparisons are formed
within one precision: the BF16 pair (both models) and the INT8 pair (vLLM W8A8 vs llama.cpp Q8_0, 8B).
