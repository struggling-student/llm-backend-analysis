# Notion pages derived from this repository

The thesis Notion workspace is the communication layer. It summarizes the notebooks; it is never a source of
numbers. Each page below quotes only values from
[`results/tables/backend_selection/key_numbers.csv`](../results/tables/backend_selection/key_numbers.csv) and the
tables exported next to it, and embeds only figures from
[`results/figures/backend_selection/`](../results/figures/backend_selection/figures.json).

| Page (language) | Role | Quotes | Figures |
| --- | --- | --- | --- |
| *Backend selection: vLLM vs llama.cpp on Xeon Max with AMX* (EN) | detailed technical report: objective, setup, fairness, baseline, single request, scaling, serving, offline, AMX, BF16/INT8, memory traffic, limitations, interpretation, decision, INT8 plan | `key_numbers.csv`, `curve_summary.csv`, `backend_ratio_bf16.csv`, `backend_ratio_int8.csv`, `evidence_summary.csv`, the 8B offline table of notebook 02, the traffic table of notebook 07 | all twelve exports |
| *Perché vLLM — sintesi della selezione del backend* (IT) | concise reasoning: obiettivo, confronto, evidenze, concorrenza e memoria, ruolo di llama.cpp, limitazioni, decisione | `key_numbers.csv`, `evidence_summary.csv` | `backend_ratio_vs_concurrency`, `throughput_and_memory_traffic` |
| *MSc Thesis — HBM-Aware LLM Inference*, section "TL;DR — Perché vLLM" (IT) | short summary for the supervisors | `key_numbers.csv` | `throughput_vs_concurrency` |
| *Roofline analysis: vLLM vs llama.cpp on Xeon Max (HBM cache mode)* (EN) | roofline of the two backends: roofs of the socket (STREAM, measured AMX GEMMs, vLLM projection kernels), placement of every 8B point, movement with concurrency and context length, operation-model check, implications | `key_numbers.csv` (`roof_*`, `roofline_*`), `roofline_points.csv`, `roofline_compute_ceilings.csv` | the six exports of notebook 10 |

Hierarchy: repository raw data → processing → notebooks → detailed page → Italian summary → thesis TL;DR. Each
level only summarizes the one below it. The roofline page sits next to the detailed page (same level); the other
three pages carry a short roofline summary that links to it.

The detailed page keeps the **earlier study** (v1, Llama-3.2-1B, `llm_bench` harness, AMX-vs-AVX-512 ablation) as
an appendix. Its numbers come from that harness, not from this repository; it is the only source of the AMX
ablation and of the attention/GEMM decomposition and is labelled as archived supporting evidence.

## Last synchronization

- 2026-10-04, analysis at commit `531abd8` (notebooks 01–09 and their exports), BF16 campaign plus llama.cpp INT8;
  vLLM INT8 pending.
- 2026-10-11, analysis at commit `95a6a18`: vLLM INT8 pair added to all three pages (text, tables, evidence rows,
  new `backend_ratio_vs_concurrency_int8` figure). The Notion API cannot replace existing image blocks, so each
  regenerated figure was inserted below its predecessor with a "supersedes the figure above" note; the older images
  are to be deleted by hand (8 on the detailed page, 1 on the Italian summary, 1 on the TL;DR).
- 2026-10-11, analysis at commit `559197b`: new roofline page (notebook 10; six figures; the 8B regime table
  generated from `roofline_points.csv`), a roofline section (§17) and TL;DR bullet on the detailed page, a roofline
  section on the Italian summary and a roofline bullet on the thesis TL;DR.

## Updating after new results

1. `make data notebooks validate`.
2. Compare the new `key_numbers.csv` with the values on the three pages (start from the detailed page, then the
   Italian summary, then the TL;DR) and update every changed value; re-upload any figure whose PNG changed
   (`git diff --stat results/figures`).
3. Revisit statements flagged "✗ revisit" in the notebooks' claims tables before changing wording.
4. Add a line to "Last synchronization" above.
