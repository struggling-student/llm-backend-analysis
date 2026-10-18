# Notion pages derived from this repository

The thesis Notion workspace is the communication layer. It summarizes the notebooks; it is never a source of
numbers. Each page below quotes only values from
[`results/tables/backend_selection/key_numbers.csv`](../results/tables/backend_selection/key_numbers.csv) and the
tables exported next to it, and embeds only figures from
[`results/figures/backend_selection/`](../results/figures/backend_selection/figures.json).

| Page (language) | Role | Quotes | Figures |
| --- | --- | --- | --- |
| *Backend selection: vLLM vs llama.cpp on Xeon Max with AMX* (EN) | detailed technical report: objective, setup, fairness, baseline, single request, scaling, serving, offline, AMX, BF16/INT8, memory traffic, limitations, interpretation, decision, INT8 plan | `key_numbers.csv`, `curve_summary.csv`, `backend_ratio_bf16.csv`, `backend_ratio_int8.csv`, `evidence_summary.csv`, the 8B offline table of notebook 02, the traffic table of notebook 07 | all twelve exports |
| *Perché vLLM — sintesi della selezione del backend* (IT; **in the trash as of 2026-10-16**) | concise reasoning: obiettivo, confronto, evidenze, concorrenza e memoria, ruolo di llama.cpp, limitazioni, decisione | `key_numbers.csv`, `evidence_summary.csv` | `backend_ratio_vs_concurrency`, `throughput_and_memory_traffic` |
| *MSc Thesis — HBM-Aware LLM Inference*, section "TL;DR — Perché vLLM" (IT; **no longer present**, the page is blank) | short summary for the supervisors | `key_numbers.csv` | `throughput_vs_concurrency` |
| *Roofline analysis: vLLM vs llama.cpp on Xeon Max (HBM cache mode)* (EN; listed under "Deprecated" since 2026-10-16) | roofline of the two backends: roofs of the socket (STREAM, measured AMX GEMMs, vLLM projection kernels), placement of every 8B point, movement with concurrency and context length, operation-model check, implications | `key_numbers.csv` (`roof_*`, `roofline_*`), `roofline_points.csv`, `roofline_compute_ceilings.csv` | the six exports of notebook 10 |
| *External references: Llama-3.1-8B on other CPUs and GPUs vs our Xeon Max (BF16 and INT8)* (EN) | published results of other systems (Xeon 6, Arm, MLPerf CPU; L40S, H100, H200, B200) paired with our nearest points: throughput, TTFT, TPOT, energy per token (ours a 350 W placeholder), single-stream bandwidth use, numbered references | `external_key_numbers.csv`, `external_reference_comparison.csv`, `external_decode_bandwidth.csv`, `key_numbers.csv` (`roof_*`); sources in `data/external/sources.csv` | `external_reference_bf16` (notebook 13) |

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

- 2026-10-16, analysis at commit `4c5f3ff` (after the independent review of 2026-10-16, tracker task T1.1):
  - **Pages found.** Detailed page and roofline page exist (the roofline page now sits under "Deprecated" on the Thesis page).
    The Italian summary ("Perché vLLM") is in the trash and the thesis TL;DR no longer exists (the "MSc Thesis" page is blank), so
    neither was updated. The Italian page "Analisi dei backend LLM — documentazione completa dei notebook" (not one of the four) was
    not touched; it still quotes the 0.17% CV as a quality figure.
  - **Detailed page, text** (no number changed): TL;DR memory bullet and §12 reworded to bytes per output token (descriptive, not a
    criterion); roofline TL;DR and §17 ("below both roofs, limiting factor not identified", BF16 roof is a proxy); §9 mechanism
    split into inferred / hypothesis / INT8-pair caveat (AMX efficiency differs: 0.1%–4.0% busy, 10% of the INT8 roof, 2.06×);
    §3 validity (CV is within one server; cross-node 0.4%–1.7% / 1.2%–8.1%; Little's law, 54 of 189 points beyond 5%); §5 C=1 lead
    "comparable to slightly ahead"; §6 estimator-sensitive plateau positions; §13 limitations (tuning scope, uncertainty, HBM
    sensitivity not yet shown); §15 table rows (bytes per output token, measurement quality) and the compute-bound control wording;
    §11 figure list; review banner; analysis commit.
  - **Roofline page, text:** TL;DR (hypothesis for vLLM's gap, proxy roof), §4 and §5 wording (the rise of the memory-roof share
    at long context comes from the roof falling), §6 (inferred cause), §7 implications (vLLM is the only engine that reaches those
    regimes; response to the memory system not yet shown), §8 limitations, review banner.
  - **Figures inserted below their predecessors** (the Notion API did not replace image blocks): detailed page 8 (new
    `throughput_and_backend_ratio` and `traffic_and_bytes_per_token`; re-exported `throughput_vs_concurrency`,
    `latency_throughput_tradeoff`, `offline_decode_throughput`, `step_cost_vs_batch`, `throughput_and_memory_traffic`,
    `memory_traffic_components`); roofline page 4 (`roofline_by_configuration`, `roofline_by_workload`,
    `ridge_ratio_vs_concurrency`, `roofline_utilization`).
  - **Old images to delete by hand:** detailed page 15 (both `peak_throughput_by_workload`; the first `single_request_latency`;
    the first two `throughput_vs_concurrency`; the first two each of `latency_throughput_tradeoff`, `offline_decode_throughput`,
    `step_cost_vs_batch`, `throughput_and_memory_traffic`; the first `llama_cpp_int8_over_bf16`; the first `memory_traffic_components`);
    roofline page 4 (the first of each re-exported figure). The images still pending from 2026-10-11 are included in the count.

- 2026-10-18, analysis at commit `7b220f0` (tracker task T3.3 and the corrected memory-node run):
  - **Detailed page:** new section 18 (AMX ablation of vLLM BF16, notebook 12; llama.cpp Q8_0 AMX build vs AVX-512 build; memory-node sensitivity,
    notebook 11, corrected run `membind-v2`), each statement labelled measured / inferred / hypothesis with the caveats; the 'no AMX ablation' and
    'HBM sensitivity not yet shown' statements replaced (§9, §13); §14 and the §15 table extended (two rows); a TL;DR bullet; the update banner.
  - **Roofline page:** the 'response of vLLM to the memory system not yet shown' sentence replaced by the measured result; banner.
  - **Figures added** (not replacing anything): `amx_ablation_throughput`, `amx_build_prompt_processing`, `membind_ratio_and_traffic` on the detailed page.
    No image needs deleting for this sync; the 19 superseded images of the 2026-10-16 sync are still to be deleted by hand.
  - The Italian page "Analisi dei backend LLM — documentazione completa dei notebook" was not touched.

## Updating after new results

1. `make data notebooks validate`.
2. Compare the new `key_numbers.csv` with the values on the three pages (start from the detailed page, then the
   Italian summary, then the TL;DR) and update every changed value; re-upload any figure whose PNG changed
   (`git diff --stat results/figures`).
3. Revisit statements flagged "✗ revisit" in the notebooks' claims tables before changing wording.
4. Add a line to "Last synchronization" above.

- 2026-10-18, analysis at commit `f466009` (notebook 13): new page *External references: Llama-3.1-8B on other CPUs and GPUs vs
  our Xeon Max (BF16 and INT8)* under Thesis (Notion id `3fdb4f5c3f968132b82bf026f9bb23ef`). Its body is generated from the
  exported tables (no typed numbers). Contents: TL;DR, setup, sources with grades, overview figure `external_reference_bf16`,
  CPU (Lenovo LP2463, Arm, MLPerf CPU) and GPU (NIM, ML.ENERGY, Red Hat MLPerf) tables, energy (ours estimated), latency,
  decode bandwidth, next steps, limitations and 16 references. The other pages were not changed and do not link to it yet.
