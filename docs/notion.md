# Notion pages derived from this repository

The thesis Notion workspace is the communication layer. It summarizes the notebooks; it is never a source of
numbers. Each page below quotes only values from
[`results/tables/backend_selection/key_numbers.csv`](../results/tables/backend_selection/key_numbers.csv) and the
tables exported next to it, and embeds only figures from
[`results/figures/backend_selection/`](../results/figures/backend_selection/figures.json).

| Page (language) | Role | Quotes | Figures |
| --- | --- | --- | --- |
| *Backend selection: vLLM vs llama.cpp on Xeon Max with AMX* (EN) | detailed technical report: objective, setup, fairness, baseline, single request, scaling, serving, offline, AMX, BF16/INT8, memory traffic, limitations, interpretation, decision, INT8 plan | `key_numbers.csv`, `curve_summary.csv`, `backend_ratio_bf16.csv`, `evidence_summary.csv`, the 8B offline table of notebook 02, the traffic table of notebook 07 | all eleven exports |
| *Perché vLLM — sintesi della selezione del backend* (IT) | concise reasoning: obiettivo, confronto, evidenze, concorrenza e memoria, ruolo di llama.cpp, limitazioni, decisione | `key_numbers.csv`, `evidence_summary.csv` | `backend_ratio_vs_concurrency`, `throughput_and_memory_traffic` |
| *MSc Thesis — HBM-Aware LLM Inference*, section "TL;DR — Perché vLLM" (IT) | short summary for the supervisors | `key_numbers.csv` | `throughput_vs_concurrency` |

Hierarchy: repository raw data → processing → notebooks → detailed page → Italian summary → thesis TL;DR. Each
level only summarizes the one below it.

The detailed page keeps the **earlier study** (v1, Llama-3.2-1B, `llm_bench` harness, AMX-vs-AVX-512 ablation) as
an appendix. Its numbers come from that harness, not from this repository; it is the only source of the AMX
ablation and of the attention/GEMM decomposition and is labelled as archived supporting evidence.

## Last synchronization

- 2026-10-04, analysis at commit `531abd8` (notebooks 01–09 and their exports), BF16 campaign plus llama.cpp INT8;
  vLLM INT8 pending.

## Updating after new results

1. `make data notebooks validate`.
2. Compare the new `key_numbers.csv` with the values on the three pages (start from the detailed page, then the
   Italian summary, then the TL;DR) and update every changed value; re-upload any figure whose PNG changed
   (`git diff --stat results/figures`).
3. Revisit statements flagged "✗ revisit" in the notebooks' claims tables before changing wording.
4. Add a line to "Last synchronization" above.
