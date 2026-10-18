# Independent review: vLLM vs llama.cpp backend selection on Xeon Max — tracker

Review date: 2026-10-16. Repository state when the review was written: `fbbcb67`. State this tracker was checked against:
`07948ef` (`origin/main` after the push of 2026-10-17); later edits of this file are local until the next push.

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

Points 1 and 2 are fixed in the text (see Done). **Point 3 is measured for the regimes tested** (Experiment A, notebook 11, corrected run `membind-v2`): vLLM's throughput falls to 0.20×–0.30× with remote memory, also at high concurrency, so it responds to the memory system; the first run (remote vLLM points without the image preload) differed by only 1%–5%. HBM versus DDR remains open (Phase 4).

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
| P3 | `steady_total_tok_s` estimator: nb04 claimed total/output differ by a constant (observed 0.71–1.12×); prompt tokens counted in a lump at first-token time inside a short window; C=1 uses a different window than C>1. Little's-law cross-check: decode-heavy within ±5%, prompt-heavy C=2–8 up to +15%, high-C 3–8% low. | **DONE** | `ae3dca2` (check, constant claim removed); sensitivity of the percent-level statements: nb04 §5 (14 statements, 9 robust, 5 sensitive; ratios of several × robust, C-axis positions move by up to one doubling) |
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
| F2 | Tuning scope: one workload (512→512), C=1 and 16, one repetition, 2–3 knobs. vLLM `max_num_batched_tokens` and llama.cpp `-b/-ub` matter most for prefill-heavy shapes that were never tuned. A 5× gap is unlikely to come from untuned knobs, but this should be stated. | **DONE** — nb01 §6 (scope of the tuning, computed knob effects ≤ 2.3% vs flash attention +17–23% vs gap 4.9–8.1×), nb01 Limitations, [methodology](methodology.md). Stated as an argument, not a measurement; one prefill-heavy tuning run would close it |
| F3 | Adaptive sweep: rule declared in advance and applied identically; early stops on llama.cpp long-prompt shapes reflect a real plateau. | OK, no action |

---

## 3. Missing work (only what materially raises confidence)

| ID | Experiment | Question it answers | Needed for the decision? | Status |
| --- | --- | --- | --- | --- |
| A | **Bandwidth sensitivity without admin access.** (**DONE 2026-10-18** on the corrected run `membind-v2`, nb11) `numactl --membind=1` (memory over UPI) vs `membind=0`. Cells: vLLM BF16 decode C=1, decode C=128, 4096→1024 C=32; llama.cpp BF16 decode C=1 as a control; remote-memory STREAM for the roof. | Does vLLM's throughput respond to memory bandwidth in the regimes the thesis will study? Missing link between "best serving engine" and "suitable platform for HBM research". | No — **High for the thesis motivation**. Caveat: remote memory changes latency as well as bandwidth. | **DONE** (Phase 2; notebook 11) |
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
| U4 | Repeated conclusion "vLLM scales, llama.cpp plateaus" restated in notebooks 04, 05, 06, 08, 09, 10; each should state it once. | **DONE** — stated in full in nb04 Conclusions (which says so); nb06, nb08, nb09, nb10 point to nb04. nb05 never restated it (latency); the mechanism and roofline conclusions stay in their notebooks |
| U5 | 1B cross-check: one sentence in the summary is enough. | **DONE** — one computed sentence under the claims table in nb09; key `ratio_at_highest_common_c_1b` (6.5×–7.0×) |

### 4.2 Visualisation review

| ID | Finding | Status |
| --- | --- | --- |
| G1 | nb09's only figure was the peak bar chart (hides curve shape and crossover; ordering did not group precision pairs; total vs output tok/s inconsistent). Replace with throughput vs C + backend-ratio panel for both pairs, plus traffic and bytes per token vs C. | **DONE** (`a0654ec`, `625468d`: `throughput_and_backend_ratio`, `traffic_and_bytes_per_token`; `peak_throughput_by_workload` retired) |
| G2 | `roofline_by_configuration`: overlapping C labels, up to four roofs per panel. Label only C=1 and highest C. | **DONE** (`a0654ec`); roofs per panel now ≤ 2 compute roofs (own kernel roof + one reference; the tile peaks stay in the roofs table) + 2 memory roofs |
| G3 | `throughput_and_memory_traffic`: add bytes per output token and the STREAM roof at each point's working set. | **DONE** via the new `traffic_and_bytes_per_token` figure |
| G4 | Captions: error bars are ±1 SD within one server instance. | **DONE** (`fdb01db`) |
| G5 | "Long-KV decode" panel title should carry the note "78% of operations are prefill". | **DONE** — every workload-faceted figure titles the panel "4096→1024 (mostly prefill)" (`panel_note` in `configs/experiment.toml`); the 78% itself is computed in nb10 §4 and key `kvdecode_prefill_op_share` |
| G6 | Every summary figure follows question → evidence → finding → defensible interpretation, hypotheses labelled. | **DONE** for nb09; keep as a rule for new figures |

### 4.3 Objections a professor is likely to raise

| ID | Objection | Answer / action | Status |
| --- | --- | --- | --- |
| O1 | "The INT8 pair is supposed to control for AMX, but llama.cpp barely uses AMX." | Concede; reworded per P2. Decision does not depend on it. | **DONE** |
| O2 | "Concurrency does not raise memory pressure in your data. Why is vLLM still the HBM platform?" | vLLM is the only engine that reaches the bandwidth-relevant regimes (long context at high C). Back this with Experiment A, not with the "busier memory" ratio. | **DONE** — wording fixed (P6); Experiment A measured on the corrected run (nb11): vLLM's throughput falls to 0.20×–0.30× with remote memory in all three regimes, also at high concurrency; HBM vs DDR still open (O6) |
| O3 | "Is vLLM compute-bound or not?" | Below both roofs; "not bandwidth-bound; limited by kernel efficiency (hypothesis)". | **DONE** |
| O4 | "Is the 0.17% CV your error bar?" | No — cross-node 0.4–1.7% and the Little's-law check are shown. | **DONE** |
| O5 | "Isn't this just AMX vs no AMX?" | Partly, at BF16: say so; either run Experiment B or present it as a confounded engine property (acceptable for backend selection). | **DONE** — Experiment B measured (nb12, claims 11–12 in nb09): without AMX in oneDNN's GEMMs vLLM keeps 25% of its C=64 throughput; AMX carries 81%–92% of the lead at C = 16…64; the split of the remaining 1.3×–1.4× lead is still a hypothesis |
| O6 | "Cache mode: how do you know anything about HBM vs DDR?" | You don't yet; state it and point to Experiment D. | **PARTIAL** — stated in nb09 status table; access request not made → T4.1 |
| O7 | "BF16 or INT8 for the thesis?" | **Decided 2026-10-17: BF16 primary, INT8 secondary** (nb09 Decision, README). Reasons: **BF16 as primary** (twice the bytes, more memory-sensitive; avoids the W8A8 fidelity drift — 5 of 16 continuations identical vs 11 for Q8_0; quality check ran on non-Max hardware). INT8 as a secondary configuration. | **DONE** (T1.2) |

---

## 5. Roadmap in phases (open work only)

Rule: when a task is complete, move its row to the Done log (section 6) with the commit hash, and update the tag on the finding it closes.

### Phase 1 — Close the write-up (no cluster, no new measurements)

| Task | What | Closes | Priority | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- |

### Phase 2 — Thesis motivation: does vLLM respond to memory bandwidth? (done; see the Done log)

| Task | What | Closes | Priority | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- |

### Phase 3 — Mechanism (optional; only if the thesis states a causal mechanism)

| Task | What | Closes | Priority | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- |

### Phase 4 — Instrumentation when access becomes available

| Task | What | Closes | Priority | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- |
| T4.1 | **Request** flat-mode node access or `perf_event_paranoid` ≤ 0 (uncore) on one HBM node from the cluster administrators. | O6 | Medium (long lead time) | Request sent; answer recorded here | OPEN — a draft request already exists in the Notion page "CRESCO8 Xeon Max: missing instrumentation…" (§6, "Draft request (not sent)"; Italian twin "strumentazione mancante"); sending it is yours. Worth adding: the M2M tag counters would also check the cache-mode STREAM roof (P7) |
| T4.2 | **Experiment D:** HBM vs DDR traffic split with uncore counters (the collector already switches to uncore automatically when permitted). | O6, P7 | Medium | HBM/DDR GB/s per point; STREAM-at-working-set roof assumption checked against it | NEEDS ACCESS |
| T4.3 | Re-read the roofline placement and the "HBM-cache + DDR combined" limitations in nb01, nb07, nb10 with the split in hand and update the nb09 status table. | P7 | Medium | Limitations rewritten from measurement, not from assumption | OPEN (after T4.2) |

---

## 6. Done log

Move finished tasks here, newest first. Evidence is the commit and the notebook section.

| Date | Item | Result | Commit / evidence |
| --- | --- | --- | --- |
| 2026-10-18 | Notion sync 2 (AMX ablation, memory-node result) | Detailed page §18 and edits in §9, §13, §14, §15, TL;DR, banner; roofline page sentence and banner; 3 figures added; the "no AMX ablation" and "HBM sensitivity not yet shown" statements removed; nothing pushed | [`notion.md`](notion.md) "Last synchronization" |
| 2026-10-18 | T2.3b — Experiment A redone on `membind-v2` | Cause of the first run's confound: the NUMA shim replaced the image's `LD_PRELOAD` (tcmalloc, Intel OpenMP), so remote vLLM points ran without them (6 of 6 remote `server.log` warn; 0 of 16 in `membind-v2`). Corrected ratios remote ÷ local: vLLM decode C=1 **0.30**, decode C=128 **0.20**, 4096→1024 C=32 **0.25**; llama.cpp control **0.32** (its local rate was 12% lower than on the first run's node). The first run's vLLM ratios were 1%–5% off. At C=1 vLLM needs 77% and llama.cpp 94% of the remote roof; at C=128 the modelled traffic is 31% of it while bandwidth alone would cap the ratio at 0.67 (observed 0.20). All placement checks 100%. All 16 points have telemetry (synced in `879bd2e`): the 4096→1024 cell's step-rate ratio is 0.256 against a throughput ratio of 0.248 and its modelled remote traffic 48% of the remote roof (C=128: 31%) | `8ad698d`, `d9e5dd0`; [nb11](../notebooks/backend_selection/11_memory_node_sensitivity.ipynb) |
| 2026-10-17 | T3.3 — nb09 and nb06 AMX statements updated to the ablation | nb09: two status rows moved from "hypothesis" to measured (claim 11: 25% of the C=64 throughput, 6.3×–9.9× vs 20.5×–34.1× scaling, same batches, steps 4.0× longer) and inferred (claim 12: AMX carries 81%–92% of the lead at C = 16…64); the split of the remaining lead stays hypothesis; interpretation bullet and implications reworded; six key numbers; nb06 pointers to nb12. Checked beforehand that the ablation points are free of the LD_PRELOAD confound (0 of 30 server logs warn) | `5a4a576` |
| 2026-10-17 | T3.1 — vLLM AMX ablation (O5, P9) | Without AMX in oneDNN's GEMMs vLLM BF16 keeps **25%** of its throughput at C=64 (balanced 873 → 217 tok/s, decode 769 → 193), scales 6.3–9.9× instead of 20.5–34.1×, and a step takes **4.0–4.3×** as long while tokens per step stay the same (≤ 6%); AMX busy 0.3–11% → 0.05% → exactly 0. vLLM's own AMX kernels add ≤ 9% at C ≥ 8. AMX carries **81–92%** (log share) of vLLM's lead over llama.cpp BF16 at C=16…64; without AMX vLLM is still 1.26–1.39× ahead at C=64. Caveats: one node per curve, the balanced levels on three different nodes (AMX level vs primary campaign differs up to 14% at low C), so only effects ≥ 2× are interpreted; fallback kernels, not the same kernels slower. The nb09 status row "AMX coverage is the main reason vLLM scales" can move from hypothesis to measured | `data(raw)` 6 sweeps, jobs 5394489-94; notebook 12 §3-5 |
| 2026-10-17 | T3.2 — llama.cpp Q8_0, AMX build vs AVX-512 build | AMX build processes prompts **1.51×–2.44×** as fast (128–8192 tokens; pass pairs within ±0.8%), AMX busy 4.0–5.1% vs 0.0%, but only 5.8–7.7% of the AMX INT8 GEMM roof (AVX-512 build 3.1–3.8%). AMX is effective in llama.cpp but far from saturated: "available" ≠ "used efficiently", consistent with notebook 08 §3. Prompt processing only; decode and served runs not done | `778df44`, `446aba9`; notebook 12 §2, job 5394480 |
| 2026-10-17 | T1.9 — housekeeping (partly) | `configs/cresco8.env` committed (identical to the cluster checkout's file; paths verified to exist) and the runbook layout table matches it. The branches `review-fixes-2026-10-16` (3ef844e) and `claude/jovial-blackburn-3b9142` (fbbcb67) and the clean detached worktree `.claude/worktrees/jovial-blackburn-3b9142` (3ef844e) were removed (all ancestors of `main`). `main` was pushed to `origin` by you on 2026-10-17 (`926b450..07948ef`, 47 commits) | uncommitted |
| 2026-10-17 | T1.4 — C=1 spread and the llama.cpp lead | Reworded to "comparable to slightly ahead" in nb03 (computed sentence: lead 12%–17% on 8B against a cross-node spread of up to 8.1%, 1.5×–2.1× larger), nb09 (claim 1 now also checks that the lead exceeds the spread; new key `c1_llamacpp_lead_short_prompts_8b`; interpretation, status table, Decision) and the README. **Not** worded as "within the spread": the data say the lead exceeds it, by a small factor (the spread is measured on 8B 512→512 only, one repetition) | uncommitted |
| 2026-10-17 | T1.2 — BF16 vs INT8 decision | vLLM BF16 primary, vLLM INT8 (W8A8) secondary; three reasons in nb09 Decision and the README (twice the bytes per step and the memory experiments are BF16; W8A8 reproduces 5 of 16 BF16 continuations vs 11 for Q8_0; the BF16 pair covers both models and the roofline and memory-node analyses) | uncommitted |
| 2026-10-17 | T2.3 — Experiment A analysed (notebook 11), first run `membind` — superseded by the corrected run below (the first run's vLLM remote points lacked the image preload) | vLLM BF16 keeps **0.19×–0.29×** of its throughput with the server's memory on the other socket's node (decode C=1 0.29, decode C=128 0.19, 4096→1024 C=32 0.24; llama.cpp control 0.28 at C=1); remote STREAM 96–104 GB/s (about one fifth of local). At C=1 the loss follows the bandwidth (vLLM needs 76%, llama.cpp 93% of the remote roof); at high C the modelled remote traffic is 29–48% of the roof and the ratio is below the 0.53–0.67 that bandwidth alone would allow, so the extra loss is attributed to latency (hypothesis). Remote traffic is modelled (PMU is local-only). Closes the review's "HBM relevance" gap for this regime set; does not separate HBM from DDR. nb09 claim 10, status-table rows split into measured / hypothesis / untested; 6 key numbers; figure `membind_ratio_and_traffic` | uncommitted; [nb11](../notebooks/backend_selection/11_memory_node_sensitivity.ipynb) |
| 2026-10-16/17 | T2.2 — Experiment A run | STREAM roof 5384320; llama.cpp control 5384324; vLLM cells re-run with the NUMA shim (5384351-53) after the first attempt (5384321-23) was found not to be remote (5-10% of memory on node 1); numastat gate 99.9–100% on all 16 points of `membind`; first attempt kept as superseded `membind-v1-vllm-numa-override`; fix check `membind-shimcheck` | `e47c6db`, `a91924c`; [runbook](cluster_runbook.md) |
| 2026-10-16 | T1.1 — Notion sync | Detailed page and roofline page updated to the post-review wording (no number changed); 12 figures re-uploaded below their predecessors; the Italian summary is in the trash and the TL;DR no longer exists, so neither was touched; 19 superseded images to delete by hand; the Italian notebook-documentation page still quotes the 0.17% CV (not one of the four, not updated) | `09e4abc`; [`notion.md`](notion.md) "Last synchronization" |
| 2026-10-16 | T2.3 preparation (local, no job) | Remote-memory STREAM campaign `stream/calibration-membind1` synced and checksummed (96–104 GB/s, 20% of local). **New finding:** the PMU read counter is local-only; with memory on node 1 it reads zero reads (capture 0.33, writes 0.99–1.00), so remote-cell traffic must be modelled (nb01 §4 limit text + computed line). `analysis/membind.py` and 8 tests. Preliminary control (read on the cluster, not yet synced): llama.cpp BF16 decode C=1 28.3 → 7.95 tok/s (0.28×), TPOT 43 → 156 ms, implying about 96 GB/s, the remote STREAM roof | uncommitted |
| 2026-10-16 | T2.1 — Experiment A prepared (nothing submitted) | `run_online.py --mem-node` and `--print-commands`; `stream_calibration.py --mem-node`; driver `membind_sensitivity.py` (cells, A/B/A/B inside one job, numastat ≥ 90% check); `[membind_sensitivity]` in `experiment.toml`; campaigns `online/membind` and `stream/calibration-membind1` declared with `prepared = true`; `submit.sh membind` / `membind-stream`; runbook section; 7 new tests (46 pass) incl. recorded commands of the primary campaign still reproduced | uncommitted; [runbook](cluster_runbook.md) |
| 2026-10-16 | T1.8 — figure polish | Roofs per panel ≤ 2 compute roofs; "(mostly prefill)" in the 4096→1024 panel title of every workload-faceted figure; 10 figures re-exported, manifest consistent | uncommitted; `CONTEXT_CEILINGS`, `panel_note` |
| 2026-10-16 | T1.7 — estimator sensitivity | nb04 §5: 14 statements recomputed with Little's law; 9 robust (ratios ≥ 4×, BF16 crossover C, BF16 monotonicity, "scales further", peak C ordering), 5 sensitive (knee C 5/20 curves, peak C 4/20, plateau rule 3/20, sign of last doubling 3/20, INT8 monotonicity). Claim text in nb04 §2 and Limitations adjusted; pointer from nb01 §3a | uncommitted |
| 2026-10-16 | T1.6 — 1B cross-check sentence | nb09: same ranking, vLLM 6.5×–7.0× at the highest common C; new key `ratio_at_highest_common_c_1b` (one key added, none changed) | uncommitted |
| 2026-10-16 | T1.5 — repeated conclusion | Scaling result stated in full only in nb04 Conclusions; nb06, 08, 09, 10 refer to it | uncommitted |
| 2026-10-16 | T1.3 — tuning-scope limitation | nb01 §6 "scope of the tuning" with computed numbers, nb01 Limitations, [methodology](methodology.md) "Scope of the tuning" | uncommitted |
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
| Medium | T4.1 access request, T4.2/T4.3 HBM/DDR split | 4 | cluster authorisation / administrators |
