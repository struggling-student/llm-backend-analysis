# Changes after the independent review (2026-10-16)

Branch `review-fixes-2026-10-16`. Scope: analysis and presentation only. No cluster job was run, `data/raw/` is
unchanged (`make validate`: 4715 recorded, 0 changed, 0 missing, 0 new), and no processed table changed except two
columns added by the Little's-law check (see item 5). Decision unchanged: **vLLM is the primary backend**; what changed is
how strongly each supporting statement is worded and what each one rests on.

Every number below is computed in a notebook and exported to
[`results/tables/backend_selection/key_numbers.csv`](results/tables/backend_selection/key_numbers.csv). Key numbers present before
the review are unchanged (0 removed, 0 values changed); 14 were added.

## FIX items

| # | Claim before | Claim after | Evidence (where computed) |
| --- | --- | --- | --- |
| 1 | nb01: "every server working set in this study is below 47 GiB"; methodology: "one point" above the HBM cache | Range computed from `online_points.working_set_gib_model`: 1.2–79.0 GiB, **2** of 189 points above 64 GiB (vLLM BF16 and INT8, 8192→256, C=64: 79 and 72 GiB) | nb01 §4; [methodology](docs/methodology.md) "Processing corrections" |
| 2 | nb02/06/08/09, README, methodology: INT8 pair has "both engines' weight GEMMs on AMX-INT8", so the result is "not an artefact of the missing BF16 AMX kernel" | The pair matches **bytes per step and precision class**; the gap is **not only** the missing BF16 AMX kernel, but **kernel/AMX efficiency still differs**. llama.cpp Q8_0: AMX busy 0.1%–4.0% online; best offline prefill = 10% of the AMX INT8 GEMM roof; AMX busy ÷ modelled arithmetic = 2.06× (median) | nb08 §3 (computed table); keys `llamacpp_int8_amx_busy`, `llamacpp_int8_offline_share_amx_int8_roof`, `llamacpp_int8_amx_busy_over_modelled` |
| 3a | nb06/07/09: vLLM at high C is "compute-bound" | "No longer bandwidth-bound and below both roofs": vLLM BF16 uses 17%–31% of the sustainable bandwidth at its highest C and the best point 42%–44% of the AMX GEMM roof. The limiting factor is **not identified** | nb09 claim 8; nb10 §2–3 |
| 3b | nb06 "what differs is how attention is executed"; nb10 "an AMX GEMM is limited by feeding the tiles" | Both labelled **hypothesis**, with the missing test named (no AMX ablation / per-operator timing; no tile-load counters) | nb06 §2; nb10 Appendix A; nb09 status table |
| 3c | nb10 §5: "concurrency brings vLLM back towards the memory roof" | **Hypothesis.** Measured: traffic *falls* on the long-context paths; the share of the roof rises only because the STREAM roof at the growing working set drops faster than the traffic | nb10 Appendix C (table of traffic, roof and share from C=1 and C=16) |
| 4 | nb09: "Memory traffic at equal C ≥ 8 → favours vLLM"; "vLLM keeps the memory system busier" | Removed as a criterion. Replaced with **bytes per output token** (vLLM ÷ llama.cpp 0.83×–1.00× at C=1, 0.54×–0.96× at C ≥ 8, BF16; 0.45×–0.87× INT8 pair) and the **C=1 share of the memory roof** (llama.cpp BF16 66%–67%, vLLM BF16 58%–59%), labelled *descriptive*. nb07 §3 reframed: traffic follows throughput | nb09 claim 7; nb07 §3; keys `bytes_per_output_token_ratio_*`, `c1_memory_roof_share_short_*` |
| 5 | nb04: "total and output tok/s differ by the constant (ISL+OSL)/OSL"; no estimator check | The constant claim is removed (observed ratio 0.71–1.12×). nb01 §3a compares the steady-window total tok/s to Little's law `C ÷ mean(E2E) × (ISL+OSL)`: **54 of 189 points differ by > 5%** (−18.3% … +14.8%; C=1 agrees within 0.23%). The backend ratio at the highest common C is unchanged by the choice of estimator (BF16 4.9×–7.0× both ways; INT8 5.3×–8.1× vs 5.3×–7.9×). C=1 uses a makespan window (`bench_client.py` `summarize`). Primary estimator kept | nb01 §3a; nb04 Method; keys `littles_law_*`, `ratio_at_highest_common_c_littles_law`; new processed columns come from `data/processed` code with tests (`tests/test_processed.py`, `tests/test_metrics.py`) |
| 6 | "median CV 0.17%" presented as the measurement uncertainty | Stated as **within-server repeatability on one node** (0.17%). Added a cross-node table (tuning campaign vs `main`, same settings, different node): **0.4%–1.7% at C=16, 1.2%–8.1% at C=1** (largest: llama.cpp BF16). Every curve (29 of 29) ran on one node | nb01 §6a; keys `cross_node_diff_c1`, `cross_node_diff_c16` |
| 7 | 4096→1024 used as evidence of KV-bound decode | Caveat wherever it is used: it is **78%** prefill by modelled operations and vLLM BF16 runs **255** tokens per step at C=64 (a prefill/decode mixture) | nb10 Appendix B/C and §4; nb09 interpretation; keys `kvdecode_prefill_op_share`, `vllm_bf16_kvdecode_tokens_per_step_at_highest_c` |
| 8 | llama.cpp BF16 compute roof shown as "AVX-512 FP32 FMA" | Labelled **proxy, unmeasured BF16 path** (its kernel uses VDPBF16PS, whose peak was not measured) in text, figure and key description | nb10 Method; `roofline_by_configuration`; key `roof_avx512_fma_tops` |

## SIMPLIFY items

- **nb09** ([09_backend_selection_summary](notebooks/backend_selection/09_backend_selection_summary.ipynb)) now reads: question → three
  figures → nine claims, each checked against the data → interpretation in which every statement is labelled *inferred* /
  *hypothesis* / *descriptive* → decision → a table of what is measured, inferred, hypothesized, untested or unmeasurable (HBM sensitivity,
  HBM/DDR split, AMX-vs-scheduler attribution) → thesis implications. The decision now says explicitly that it rests on claims 3–5
  (scaling and ≥ 4× lead, in both pairs, under a latency budget) and **not** on memory-traffic volume, the roofline or any hypothesis.
- **Displayed keys:** 16 of 69 (roofline keys displayed: 3). The full set is still exported to `key_numbers.csv`
  (`docs/notion.md` quotes it); the cell lists the keys that are exported but not displayed.
- **Cross-precision row** (vLLM BF16 vs llama.cpp INT8) removed from the summary's claims and evidence; it stays in nb08 §2.
- **Figures in nb09:** `peak_throughput_by_workload` retired (files and manifest entry removed). New: `throughput_and_backend_ratio`
  (total tok/s vs C plus vLLM ÷ llama.cpp for both pairs), `traffic_and_bytes_per_token` (traffic with the STREAM ceiling, and GB per
  output token), and the roofline placement.
- **Throughput metric:** stated once per notebook; nb04 says tables, ratios, peaks and knees use total tok/s and its throughput
  figures use output tok/s.
- **nb10** ([10_roofline](notebooks/backend_selection/10_roofline.ipynb)): sections 1–6 carry the argument (roofs → placement → what limits each point →
  decode-intensity limit → 1B cross-check → claims); the AMX cross-check, per-workload view, concurrency-to-ridge and share-of-roof
  scatter moved to Appendix A–D with their figures and exports. Section references in nb06 and nb09 updated.
- **Roofline figure:** only C=1 and the highest C are labelled, placed by a collision-avoiding routine with leader lines.
- **Captions and subtitles** of nb02, 04, 05, 07, 08, 09 figures say the error bars are ±1 SD over the repetitions of **one server instance**.

## Second pass (Phase 1 of the review tracker)

- **Tuning scope (T1.3):** nb01 §6 and `docs/methodology.md` state that the tuning used 512-token prompts and never varied `-b`/`-ub` or
  `max_num_batched_tokens` on the prefill-heavy shapes; the gap is argued, not shown, to be larger than any batching-knob effect (≤ 2.3%).
- **Estimator sensitivity (T1.7):** nb04 §5 marks each percent-level statement robust or sensitive under Little's law. The nb04 sentence
  that llama.cpp's gains "fall to about zero at the end of its sweeps" is now limited to the short-prompt shapes; on 8192→256 the last
  doubling is 1–2% (window) vs 8–12% (Little's law).
- **Repeated conclusion (T1.5):** the scaling result is stated in full in nb04 only.
- **1B cross-check (T1.6):** one sentence in nb09, key `ratio_at_highest_common_c_1b`.
- **Figures (T1.8):** at most two compute roofs per roofline panel; "(mostly prefill)" in the 4096→1024 panel title.
- **Harness (T2.1, nothing submitted):** memory-node override and the memory-node sensitivity driver; see `docs/cluster_runbook.md`.

## Experiment A (memory-node sensitivity, 2026-10-17)

- New notebook 11 and `analysis/membind.py`: vLLM BF16 keeps 0.19×–0.29× of its throughput with the server's memory on the other socket's node
  (llama.cpp control 0.28× at C=1); at high concurrency the loss exceeds what remote bandwidth alone explains (hypothesis: latency).
  nb09 gets claim 10, three status-table rows (measured / hypothesis / untested) and six key numbers; no existing number changed.
- Remote traffic is modelled because the core PMU read counter counts local reads only (nb01 §4).
- vLLM's CPU worker resets the memory policy, so the first attempt (`membind-v1-vllm-numa-override`) was not remote and is kept as superseded; the re-run uses an `LD_PRELOAD` shim.

## Validation

`make data notebooks validate` and pytest: all checks pass (environment, raw data integrity, metadata, processed data up to date,
schema, 10 notebooks executed and portable, figure manifest, Markdown links; 46 tests after the second pass). No claims-table row fails in any notebook.

## Notes for the reader

- The cross-node spread at **C=1 is larger (up to 8.1%, llama.cpp BF16) than at C=16** and is comparable to the BF16 single-request
  margin on 512→512 (vLLM ÷ llama.cpp 0.89×). The tuning runs used one repetition, so this overstates node-to-node variance somewhat, but
  the statement "llama.cpp leads single-request throughput at BF16" is now flagged in nb09 as holding by a margin that exceeds
  the measured spread **by little**. It does not enter the decision.
- 54 of 189 points deviate by more than 5% from Little's law, mainly prompt-heavy points at C=2–8 (+) and high-C points (−); the
  backend ratios used in the decision are insensitive to the estimator.
- Not done, by design (they need cluster jobs and your authorization): bandwidth-sensitivity test with remote `membind`, vLLM
  AMX-off ablation, llama.cpp Q8_0 AMX vs AVX-512 build comparison, HBM/DDR split (flat mode or uncore access). They are listed
  as *untested / unmeasurable* in the nb09 status table.
