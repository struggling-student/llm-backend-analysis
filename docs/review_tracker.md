# Independent review: vLLM vs llama.cpp backend selection on Xeon Max — tracker

Review date: 2026-10-16. Repository state when the review was written: `fbbcb67`. State this tracker was checked against:
`3ef844e` (local `main`, after the merge of `review-fixes-2026-10-16`; nothing pushed).

This file holds (1) the full content of the review with a status on every finding, (2) the open work organised in phases,
and (3) the **Done log**. When a task is finished, move its row from its phase (section 5) into the Done log (section 6) and update
the status tag where the finding appears in sections 2–4. The per-claim before/after record of the fixes already applied is in
[`CHANGES_REVIEW.md`](../CHANGES_REVIEW.md).

Status tags: **DONE** (applied and validated), **PARTIAL** (part done, remainder listed), **OPEN** (not started),
**NEEDS CLUSTER** (open, requires a Slurm job you authorise), **NEEDS ACCESS** (open, blocked on an administrator),
**DECISION** (open, needs your choice).

---

## 1. Verdict

Choosing **vLLM is justified, and the evidence for it is strong.** The review found that the summary claimed more than the data
showed on three points, none of which changes the decision:

1. **Mechanism.** The AMX/batching explanation and the "compute-bound" label were inferred, not demonstrated.
2. **INT8 pair.** It does not control for AMX the way the notebooks said (it controls bytes per step and precision class).
3. **HBM relevance.** No measurement yet shows that vLLM's throughput responds to memory bandwidth. Without it, "vLLM is the
   right platform for HBM research" is an argument, not a result.

Points 1 and 2 are fixed in the text (see Done). **Point 3 is still open** and is the main remaining scientific gap (Phase 2).

Two corrections to the original brief: the W8A8 vs Q8_0 study was already measured (campaign `main`, 2026-10-10), and there are
ten notebooks, not nine (`10_roofline` feeds notebook 09).

---

## 2. Scientific assessment and findings

### 2.1 What is valid and well supported (no action needed)

| ID | Area | Evidence |
| --- | --- | --- |
| V1 | Server-vs-server design | One shared stdlib client ([`bench_client.py`](../scripts/benchmark/bench_client.py)); token-id prompts, `ignore_eos`, fixed output length, prefix caching off, fresh server per point, same cores and `membind` for both backends. vLLM binds exactly cores 0–54 (confirmed in `server.log`). |
| V2 | Run validity | 0 failed of 39,948 requests; exact token counts; consistent prompt manifests; client CPU ≤ 0.30 of a core; idle cores ≤ 1.8% busy; node STREAM checks 490–539 GB/s. |
| V3 | Memory-traffic proxy | Reproduces STREAM within 0.97–1.00. Also validated on the real workloads: at C=1 the measured bytes per output token (~15.0 GB) match the 15.01 GB of weights streamed per step; vLLM decode reads match the weight + KV model (median 1.00). |
| V4 | Telemetry overhead | Within ±1% (A/B/A/B test); the multiplexed 7-event set was correctly retired. |
| V5 | Main effect | 4.9–7.0× (BF16) and 5.3–8.1× (INT8) at the highest common C, on both models and all five workloads; also holds within a latency budget (4.5–9.7×). No plausible measurement error reverses it. |
| V6 | Server does not hide engine speed | Online ISL/TTFT at C=1 reproduces offline prefill within 0.96–1.02 for three of four configurations (llama.cpp Q8_0 is the exception, 0.74–0.79). |
| V7 | "Traffic falls with C" | Correct and important; it is arithmetic (one weight read per step, fewer steps/s as batches grow). |
| V8 | Cross-node reproducibility | Tuning vs main for the chosen variants differ by 0.4–1.7% at C=16 (shown in notebook 01 since the fixes). |
| V9 | Provenance | Raw checksums, campaign roles, superseded campaigns kept, deviations documented, numbers computed in code. |

### 2.2 Problems and weaknesses

#### Confirmed defects

| ID | Finding | Status | Where / commit |
| --- | --- | --- | --- |
| P1 | Notebook 01 said every server working set is below 47 GiB; data give 1.2–79 GiB with 2 points above 64 GiB (vLLM BF16 and INT8, 8192→256, C=64); methodology said "one point". | **DONE** | `65d0e5f`; nb01 §4, [methodology](methodology.md) |
| P2 | INT8 pair described as an AMX control; llama.cpp Q8_0 AMX busy is 0.1–4% online, best offline prefill ≈ 10% of the AMX INT8 GEMM roof, AMX busy ÷ modelled arithmetic 2.06×. Pair controls bytes per step and precision class, not AMX efficiency. | **DONE** | `412daae`; nb08 §3, nb02, nb06, nb09, README, methodology |
| P3 | `steady_total_tok_s` estimator: nb04 claimed total/output differ by a constant (observed 0.71–1.12×); prompt tokens counted in a lump at first-token time inside a short window; C=1 uses a different window than C>1. Little's-law cross-check: decode-heavy within ±5%, prompt-heavy C=2–8 up to +15%, high-C 3–8% low. | **PARTIAL** | `ae3dca2`: check added (54 of 189 points differ > 5%), constant claim removed, primary estimator kept, backend ratios shown insensitive. **Remaining:** show that the plateau/knee/crossover statements that rest on a few percent (+2%/+5% doublings, knee, vLLM INT8 8192→256 drop at C=64, crossover C) survive the alternative estimator → task T1.7. |
| P4 | 0.17% median CV presented as measurement uncertainty; it is within-server repeatability on one node; every curve ran on one node. | **DONE** | `806f103`; nb01 §6a: cross-node 0.4–1.7% at C=16, 1.2–8.1% at C=1 |

#### Unsupported or overstated claims

| ID | Finding | Status | Where / commit |
| --- | --- | --- | --- |
| P5 | "vLLM is compute-bound at high C" — it is below both roofs (17–31% of memory roof, 42–44% of GEMM roof, AMX busy ≤ 18%). | **DONE** | `49886d0`; reworded to "no longer bandwidth-bound, below both roofs"; limiting factor labelled not identified |
| P6 | "Memory traffic at equal C ≥ 8 favours vLLM" / "vLLM keeps memory busier" — a consequence of more steps/s, not an advantage; llama.cpp BF16 uses more of the memory roof at C=1 (66–67% vs 58–59%). | **DONE** | `49886d0`, `625468d`; replaced by bytes per output token and C=1 roof share, labelled descriptive |
| P7 | nb10 §5 "concurrency brings vLLM back to the memory roof": measured traffic falls; the share rises only because the STREAM-at-working-set roof drops. | **PARTIAL** | Relabelled as hypothesis with a traffic/roof/share table (nb10 Appendix C). **Remaining:** the assumption that STREAM at a working-set size is the right roof for an LLM access pattern is untested → Phase 2/4 (Experiments A, D). |
| P8 | The 4096→1024 "long-KV decode" workload does not isolate KV-bound decode: 78% of operations are prefill, vLLM C=64 runs ~255 tokens/step. | **DONE** (caveat) | `49886d0`; caveat wherever it is used |
| P9 | Mechanism statements without measurement: "attention is what differs" (nb06), "AMX GEMM limited by feeding the tiles" (nb10). | **DONE** | `49886d0`; labelled hypothesis with the missing test named; measured/inferred/hypothesis table in nb09 |
| P10 | llama.cpp BF16 compute roof is the AVX-512 FP32 FMA peak, an unmeasured proxy for the VDPBF16PS path. | **DONE** | `49886d0`; labelled in text, figure and key description |

#### Fairness

| ID | Item | Status |
| --- | --- | --- |
| F1 | Legitimate, unavoidable differences (AMX coverage at BF16, GGUF vs safetensors, F16/BF16 KV cache, thread binding mechanism, front-end process structure) are disclosed in notebook 01. | OK, no action |
| F2 | Tuning scope: one workload (512→512), C=1 and 16, one repetition, 2–3 knobs. vLLM `max_num_batched_tokens` and llama.cpp `-b/-ub` matter most for prefill-heavy shapes that were never tuned. A 5× gap is unlikely to come from untuned knobs, but this should be stated. | **PARTIAL** — nb01 limitation states one workload / one repetition (DONE); it does not name the untuned prefill-heavy shapes or say the gap is unlikely to be a tuning artefact → T1.3 |
| F3 | Adaptive sweep: rule declared in advance and applied identically; early stops on llama.cpp long-prompt shapes reflect a real plateau. | OK, no action |

---

## 3. Missing work (only what materially raises confidence)

| ID | Experiment | Question it answers | Needed for the decision? | Status |
| --- | --- | --- | --- | --- |
| A | **Bandwidth sensitivity without admin access.** `numactl --membind=1` (memory over UPI) vs `membind=0`. Cells: vLLM BF16 decode C=1, decode C=128, 4096→1024 C=32; llama.cpp BF16 decode C=1 as a control; remote-memory STREAM for the roof. | Does vLLM's throughput respond to memory bandwidth in the regimes the thesis will study? Missing link between "best serving engine" and "suitable platform for HBM research". | No — **High for the thesis motivation**. Caveat: remote memory changes latency as well as bandwidth. | **NEEDS CLUSTER** (Phase 2) |
| B | **AMX ablation for vLLM.** `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`; verify `EXE.AMX_BUSY` ≈ 0 and `ONEDNN_VERBOSE`; balanced and decode-heavy, C=1…64. | Is AMX or the scheduler the reason vLLM scales? | No — only if the thesis wants a causal claim; otherwise the claim stays a labelled hypothesis. | **NEEDS CLUSTER** (Phase 3, optional) |
| C | **llama.cpp Q8_0, AMX build vs AVX-512 build** in `llama-bench` prefill (the AVX-512 build already exists). | Is the AMX path in llama.cpp effective at all? | No — low priority; the rewording in P2 was the required fix. | **NEEDS CLUSTER** (Phase 3, optional) |
| D | **HBM vs DDR split** (flat-mode node or lower `perf_event_paranoid`). | Direct HBM measurements; validates the STREAM-roof assumption (P7). | No — thesis work. Request access now. | **NEEDS ACCESS** (Phase 4) |

**Explicitly not needed:** more workloads, more models, more repetitions, open-loop load, another INT8 study. The INT8 pair is
done and useful (matches bytes per step; vLLM wins at every C) — keep it, correctly worded, do not expand it.

---

## 4. Unnecessary work, visualisation, objections

### 4.1 Unnecessary work and complexity

| ID | Finding | Status |
| --- | --- | --- |
| U1 | Notebook 09 overloaded (50 keys, 18 evidence rows, 16 claims). Target: ~8 claims, ~12–15 numbers, 2–3 figures. | **DONE** (`625468d`: nine claims, 16 of 69 keys displayed, three figures; full CSV still exported for Notion) |
| U2 | Notebook 10 heavy (26 cells, 7 exported figures, two roof families). Keep roofs table, placement figure, intensity limit; move the rest to an appendix. | **DONE** (`e989c4c`: sections 1–6 carry the argument, Appendix A–D) |
| U3 | Cross-precision comparison (vLLM BF16 vs llama.cpp INT8) superseded by the INT8 pair; keep in nb08, drop from the summary. | **DONE** (`625468d`) |
| U4 | Repeated conclusion "vLLM scales, llama.cpp plateaus" restated in notebooks 04, 05, 06, 08, 09, 10; each should state it once. | **OPEN** → T1.5 (only nb09/nb10 were restructured) |
| U5 | 1B cross-check: one sentence in the summary is enough. | **OPEN** → T1.6 (verify; nb09 currently mentions the 1B model only in a code comment, so the cross-check is not carried into the summary at all) |

### 4.2 Visualisation review

| ID | Finding | Status |
| --- | --- | --- |
| G1 | nb09's only figure was the peak bar chart (hides curve shape and crossover; ordering did not group precision pairs; total vs output tok/s inconsistent). Replace with throughput vs C + backend-ratio panel for both pairs, plus traffic and bytes per token vs C. | **DONE** (`a0654ec`, `625468d`: `throughput_and_backend_ratio`, `traffic_and_bytes_per_token`; `peak_throughput_by_workload` retired) |
| G2 | `roofline_by_configuration`: overlapping C labels, up to four roofs per panel. Label only C=1 and highest C. | **DONE** (`a0654ec`); roof count per panel not re-verified → covered by T1.8 |
| G3 | `throughput_and_memory_traffic`: add bytes per output token and the STREAM roof at each point's working set. | **DONE** via the new `traffic_and_bytes_per_token` figure |
| G4 | Captions: error bars are ±1 SD within one server instance. | **DONE** (`fdb01db`) |
| G5 | "Long-KV decode" panel title should carry the note "78% of operations are prefill". | **PARTIAL** — caveat is in the text (P8); the panel title/subtitle in the figures was not confirmed → T1.8 |
| G6 | Every summary figure follows question → evidence → finding → defensible interpretation, hypotheses labelled. | **DONE** for nb09; keep as a rule for new figures |

### 4.3 Objections a professor is likely to raise

| ID | Objection | Answer / action | Status |
| --- | --- | --- | --- |
| O1 | "The INT8 pair is supposed to control for AMX, but llama.cpp barely uses AMX." | Concede; reworded per P2. Decision does not depend on it. | **DONE** |
| O2 | "Concurrency does not raise memory pressure in your data. Why is vLLM still the HBM platform?" | vLLM is the only engine that reaches the bandwidth-relevant regimes (long context at high C). Back this with Experiment A, not with the "busier memory" ratio. | **PARTIAL** — wording fixed (P6); evidence needs Experiment A (Phase 2) |
| O3 | "Is vLLM compute-bound or not?" | Below both roofs; "not bandwidth-bound; limited by kernel efficiency (hypothesis)". | **DONE** |
| O4 | "Is the 0.17% CV your error bar?" | No — cross-node 0.4–1.7% and the Little's-law check are shown. | **DONE** |
| O5 | "Isn't this just AMX vs no AMX?" | Partly, at BF16: say so; either run Experiment B or present it as a confounded engine property (acceptable for backend selection). | **PARTIAL** — hypothesis labelled; causal claim needs Experiment B (optional) |
| O6 | "Cache mode: how do you know anything about HBM vs DDR?" | You don't yet; state it and point to Experiment D. | **PARTIAL** — stated in nb09 status table; access request not made → T4.1 |
| O7 | "BF16 or INT8 for the thesis?" | Review recommendation: **BF16 as primary** (twice the bytes, more memory-sensitive; avoids the W8A8 fidelity drift — 5 of 16 continuations identical vs 11 for Q8_0; quality check ran on non-Max hardware). INT8 as a secondary configuration. | **DECISION** → T1.2 (nb09 only says "a separate choice") |

---

## 5. Roadmap in phases (open work only)

Rule: when a task is complete, move its row to the Done log (section 6) with the commit hash, and update the tag on the finding it closes.

### Phase 1 — Close the write-up (no cluster, no new measurements)

| Task | What | Closes | Priority | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- |
| T1.1 | **Sync the three Notion pages and the roofline page to the post-review analysis** (`3ef844e`). Last sync was 2026-10-11 (`fbbcb67`), so Notion still carries the removed "memory busier → favours vLLM", "compute-bound" and "both on AMX-INT8" wording. Follow [`notion.md`](notion.md) "Updating after new results"; record the sync there. Delete the stale image blocks by hand (8 EN, 1 IT, 1 TL;DR; the Notion API cannot replace images). | P2, P5, P6, P7 downstream | **High** | Every value on the pages equals the current `key_numbers.csv`; none of the removed claims appears; "Last synchronization" has a new line; old images removed | OPEN |
| T1.2 | **Decide BF16 vs INT8 as the thesis primary configuration** and write the decision plus its three reasons in nb09 (Decision section). | O7 | **High** | One sentence stating the choice and the reasons in nb09 and README; no longer "a separate choice" | DECISION |
| T1.3 | **Complete the tuning-scope limitation:** name the untuned prefill-heavy shapes (`max_num_batched_tokens`, `-b/-ub`), say the 4.9–8.1× gap is unlikely to be a tuning artefact and why (e.g. vLLM spread 0.7%/0.3% across the grid; llama.cpp best variants chosen), in nb01 and methodology. | F2 | Medium | Limitation text present in nb01 §6 and [methodology](methodology.md) | OPEN |
| T1.4 | **Decide what to do with the C=1 cross-node spread** (up to 8.1%, llama.cpp BF16). It is comparable to the 512→512 BF16 single-request margin (0.89×). Either keep the existing flag in nb09 or drop "llama.cpp leads single-request throughput at BF16" to "comparable/slightly ahead, within node spread". Does not affect the decision. | P4 follow-up | Low | Wording matches the measured spread in nb03 and nb09 | OPEN |
| T1.5 | **Consolidate the repeated "vLLM scales, llama.cpp plateaus" conclusion**: state it fully once (nb04), reference it from nb05, 06, 08, 09, 10. | U4 | Medium | The conclusion text appears once; other notebooks link to nb04 | OPEN |
| T1.6 | **Add one sentence to nb09 for the 1B cross-check** with computed values (same ranking on 1B, BF16 pair), or confirm and document that it is deliberately left out. | U5 | Low | Sentence present with numbers from `key_numbers.csv`, or a recorded reason | OPEN |
| T1.7 | **Estimator sensitivity of the percent-level statements:** recompute plateau/knee/crossover statements (+2%/+5% doublings, vLLM INT8 8192→256 drop at C=64, first C with ratio ≥ 1) with the Little's-law estimator and state which are estimator-sensitive. Primary estimator unchanged. | P3 | Medium | A table in nb01 §3a (or nb04) marks each statement as robust/sensitive; sensitive ones are worded as "within estimator spread" | OPEN |
| T1.8 | **Figure polish check:** confirm `roofline_by_configuration` shows at most the bounding roof plus the measured GEMM roof per panel; add "78% of operations are prefill" to the 4096→1024 panel subtitle in the figures that show it. Re-export; check `git diff --stat results/figures`. | G2, G5 | Low | Subtitles/panels as described; manifest regenerated | OPEN |
| T1.9 | **Repository housekeeping:** decide when to push `main` (29 commits ahead of `origin/main`, nothing pushed), remove the merged branch/worktree `review-fixes-2026-10-16` and the failed child session branch `claude/jovial-blackburn-3b9142` once confirmed redundant, and commit or discard the unrelated local change in `configs/cresco8.env` (cluster paths moved to `store_0/projects/…`; the cluster checkout and `cluster_runbook.md` must match). | — | Low | `git status` clean; branches pruned; push decision recorded | DECISION |

### Phase 2 — Thesis motivation: does vLLM respond to memory bandwidth? (needs cluster authorisation)

| Task | What | Closes | Priority | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- |
| T2.1 | **Prepare Experiment A** (no submission): extend the harness to take a memory-node override (`--set` / config key for `server_mem_node`), define the cell list (vLLM BF16 decode C=1, decode C=128, 4096→1024 C=32; llama.cpp BF16 decode C=1 control), add a remote-memory STREAM calibration under the same PMU counters, register a new campaign in `data/metadata/campaigns.toml`. | O2, P7 | **High (thesis)** | Dry-run of the command lines for each cell; campaign role declared; existing cells still reproducible; nothing submitted | OPEN |
| T2.2 | **Run Experiment A** (explicit authorisation required: Slurm submission). | O2, P7 | **High (thesis)** | Per-cell throughput ratio `membind=1 ÷ membind=0` with repetitions; remote STREAM roof measured; raw data synced and checksummed | NEEDS CLUSTER |
| T2.3 | **Analyse and report Experiment A** as a short notebook section (or notebook 11) and one nb09 status-table row moved from "untested" to "measured". State the confound (remote memory changes latency as well as bandwidth). | O2, P7 | High (thesis) | Result tied to a specific regime; limitation stated; nb09 row updated; key numbers exported | OPEN (after T2.2) |

### Phase 3 — Mechanism (optional; only if the thesis states a causal mechanism)

| Task | What | Closes | Priority | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- |
| T3.1 | **Experiment B: vLLM AMX ablation** (`ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`; balanced and decode-heavy, C=1…64). | O5, P9 | Medium | `EXE.AMX_BUSY` ≈ 0 verified in telemetry and `ONEDNN_VERBOSE`; scaling curve with and without AMX | NEEDS CLUSTER |
| T3.2 | **Experiment C: llama.cpp Q8_0, AMX build vs AVX-512 build** in `llama-bench` prefill. | P2 | Low | Ratio between builds per shape, with AMX busy from telemetry | NEEDS CLUSTER |
| T3.3 | If Phase 3 is skipped, **record that choice**: keep the mechanism rows as "hypothesis" in the nb09 status table and say so in the thesis text. | O5 | Low | Written decision in nb09 or here | DECISION |

### Phase 4 — Instrumentation when access becomes available

| Task | What | Closes | Priority | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- |
| T4.1 | **Request** flat-mode node access or `perf_event_paranoid` ≤ 0 (uncore) on one HBM node from the cluster administrators. | O6 | Medium (long lead time) | Request sent; answer recorded here | OPEN |
| T4.2 | **Experiment D:** HBM vs DDR traffic split with uncore counters (the collector already switches to uncore automatically when permitted). | O6, P7 | Medium | HBM/DDR GB/s per point; STREAM-at-working-set roof assumption checked against it | NEEDS ACCESS |
| T4.3 | Re-read the roofline placement and the "HBM-cache + DDR combined" limitations in nb01, nb07, nb10 with the split in hand and update the nb09 status table. | P7 | Medium | Limitations rewritten from measurement, not from assumption | OPEN (after T4.2) |

---

## 6. Done log

Move finished tasks here, newest first. Evidence is the commit and the notebook section.

| Date | Item | Result | Commit / evidence |
| --- | --- | --- | --- |
| 2026-10-16 | Write-up of the review fixes | Per-claim before/after record | `3ef844e`; [`CHANGES_REVIEW.md`](../CHANGES_REVIEW.md) |
| 2026-10-16 | U1, G1, G3, P6, O3 — nb09 rebuilt | Claims → labelled interpretation → measured/inferred/hypothesis table; peak bar chart replaced; cross-precision row dropped; traffic framed as bytes per output token | `625468d` |
| 2026-10-16 | G4 — captions | Error bars stated as within-server repeatability of one instance | `fdb01db` |
| 2026-10-16 | U2 — nb10 simplification | Argument in §1–6, evidence in Appendix A–D | `e989c4c` |
| 2026-10-16 | G2 — roofline labels | Collision-free C labels (C=1 and highest C only); throughput-and-ratio figure | `a0654ec` |
| 2026-10-16 | P5, P6, P8, P9, P10 — claim wording | "Compute-bound" removed; mechanism statements labelled hypothesis; 4096→1024 caveat; proxy roof labelled | `49886d0` |
| 2026-10-16 | P2, O1 — INT8-pair AMX wording | Qualified with measured AMX efficiency (0.1–4% busy, 10% of the INT8 GEMM roof, 2.06× busy÷modelled) | `412daae` |
| 2026-10-16 | P4, V8 — uncertainty | Within-server repeatability vs cross-node table (0.4–1.7% at C=16, 1.2–8.1% at C=1) | `806f103` |
| 2026-10-16 | P3 (part) — estimator transparency | Little's-law check added (54 of 189 points > 5%); constant-ratio sentence removed; ratio at highest common C unchanged by estimator. Sensitivity of percent-level statements still open (T1.7) | `ae3dca2` |
| 2026-10-16 | P1 — working-set range | 1.2–79.0 GiB computed; 2 points above 64 GiB | `65d0e5f` |
| 2026-10-16 | Validation | `make data notebooks validate` and pytest pass (39 tests); `data/raw` unchanged (4715 recorded, 0 changed); 14 key numbers added, 0 removed or changed | `CHANGES_REVIEW.md` "Validation" |

---

## 7. Prioritised view of everything still open

| Priority | Task | Phase | Blocked on |
| --- | --- | --- | --- |
| **High** | T1.1 Notion sync | 1 | nothing |
| **High** | T1.2 BF16 vs INT8 primary | 1 | your decision |
| **High (thesis)** | T2.1 → T2.2 → T2.3 Experiment A | 2 | cluster authorisation for T2.2 |
| Medium | T1.3 tuning-scope limitation, T1.5 consolidate repeated conclusion, T1.7 estimator sensitivity | 1 | nothing |
| Medium | T3.1 AMX ablation (optional), T4.1 access request, T4.2/T4.3 HBM/DDR split | 3, 4 | cluster authorisation / administrators |
| Low | T1.4, T1.6, T1.8, T1.9, T3.2, T3.3 | 1, 3 | nothing / your decision |
