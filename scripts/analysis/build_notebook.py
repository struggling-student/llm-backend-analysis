#!/usr/bin/env python3
"""Build notebooks/analysis.ipynb (Plotly) from the processed tables.

Each major plot has Markdown before it (what is plotted, why it matters) and after
it (what can be observed, what cannot be concluded from this plot alone). The
"observed" text lives in report/observations.toml, written after looking at
the data, so the notebook and the report state the same observations.
Run:  uv run python scripts/analysis/build_notebook.py && uv run jupyter nbconvert --execute --inplace notebooks/analysis.ipynb
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


SECTIONS = [
    # (key, title, before, plot code, cannot-conclude)
    ("stream", "Experiment 0 — sustainable bandwidth and the PMU proxy",
     """**Plotted:** STREAM triad bandwidth on the server layout (socket 0, 55/56 threads, cache mode) for working
sets from 4 to 96 GiB (left), and the ratio between what the core PMU counters report and STREAM's modeled
traffic (right).
**Why it matters:** every bandwidth number for the inference servers comes from the same core counters. If the
counters reproduce STREAM's known traffic, application bandwidth can be normalized by STREAM measured *with the
same instrument*, and the normalization does not depend on the theoretical HBM peak.""",
     "F.stream_figure(stream, OUT); show('stream_calibration')",
     """The counters see memory-side traffic from the cores. In cache mode they cannot split HBM-cache hits from DDR
traffic, and they do not see the HBM cache's own fills and evictions toward DDR. The ratio is a consistency check
against STREAM's model, not a calibration against device-level CAS counts (not readable at
`perf_event_paranoid=2`)."""),
    ("offline_prefill", "Experiment A — offline prompt processing",
     """**Plotted:** prompt-processing tokens/s from each engine's own benchmark tool, at batch 1 (`llama-bench pp`
vs `vllm bench latency --output-len 1`) and at fixed batch 8 and 32 (`llama-batched-bench` vs `vllm bench latency`).
**Why it matters:** prefill is compute-bound; it shows how much of each engine's matrix work reaches AMX.""",
     "F.offline_bars(off, 'prefill', 'f01_offline_prefill', 'Offline prompt-processing throughput', OUT); show('f01_offline_prefill_llama31_8b')",
     """The tools differ in scope. vLLM's number includes scheduling, input preparation and sampling of one token;
llama-bench times forward passes only. Small differences are not engine differences."""),
    ("offline_decode", "Experiment A — offline generation",
     """**Plotted:** generation tokens/s: `llama-bench tg` with an ISL-token context vs vLLM's
(OSL−1)/(full − prefill) at batch 1, and both engines' batched decode at batch 8 and 32.
**Why it matters:** single-stream decode is limited by reading the weights once per token; batched decode shows
whether an engine amortizes that read over several sequences.""",
     "F.offline_bars(off, 'decode', 'f02_offline_decode', 'Offline generation throughput', OUT); show('f02_offline_decode_llama31_8b')",
     """vLLM's decode rate is derived from two separate runs (median prefill subtracted from each full run), so its
error bar understates the uncertainty. llama.cpp Q8_0 reads half the weight bytes of BF16; its batch-1 advantage
is partly precision, not engine."""),
    ("online_tput", "Experiment B — online throughput vs concurrency",
     """**Plotted:** steady-state output tokens/s (and total tokens/s) of each server under a closed loop with C
requests in flight, for each workload shape. Error bars: ±1 SD over 3 repetitions.
**Why it matters:** this is the primary selection criterion — how much aggregate work a server delivers as
concurrency rises.""",
     "for m in ('f03_output_throughput', 'f04_total_throughput'): show(f'{m}_llama31_8b')",
     """Throughput alone hides latency: a higher curve may be bought with unusable per-request latency (next plots).
Different precisions (Q8_0) are not engine comparisons."""),
    ("scaling", "Scaling efficiency",
     """**Plotted:** speedup over C=1 divided by C. 1.0 would be perfect linear scaling.
**Why it matters:** it shows how much of each additional concurrent request turns into extra throughput.""",
     "show('f12_scaling_efficiency_llama31_8b')",
     """Efficiency is relative to each backend's own C=1, so a backend that is slow at C=1 can look efficient. Read it
together with absolute throughput."""),
    ("latency", "Latency under load: TTFT, TPOT, end-to-end",
     """**Plotted:** median TTFT, TPOT and E2E latency of steady-state requests (log scale).
**Why it matters:** a serving backend must keep per-request latency usable while throughput grows.""",
     "for m in ('f05_ttft', 'f06_tpot', 'f07_e2e'): show(f'{m}_llama31_8b')",
     """Medians are per repetition and averaged; pooled p95/p99 are in `online_points.csv` with their sample counts.
p99 is reported only where at least 100 steady requests exist."""),
    ("batching", "Concurrency → batching",
     """**Plotted:** tokens processed per forward step, from each server's own counters (vLLM
`iteration_tokens_total`, llama.cpp `n_decode_total` and token counters), and AMX busy fraction.
**Why it matters:** it is the link between "more concurrent requests" and "more work per weight read".""",
     "for m in ('f14_tokens_per_step', 'f15_amx_busy', 'f16_cpu_util'): show(f'{m}_llama31_8b')",
     """Tokens per step mixes prefill chunks and decode tokens. It shows batching took place, not how efficiently each
step used the hardware."""),
    ("bandwidth", "Memory traffic vs concurrency",
     """**Plotted:** memory read and write traffic of the server process tree (core PMU, beyond L3), and the same
traffic divided by STREAM's sustainable traffic measured with the same counters.
**Why it matters:** this is the question the thesis needs answered: does more concurrency produce more memory
traffic, and how close does it get to the machine's sustainable bandwidth?""",
     "for m in ('f08_mem_read_bw', 'f09b_mem_write_bw', 'f10_bw_utilization'): show(f'{m}_llama31_8b')",
     """In cache mode the traffic is HBM-cache hits plus DDR, not separable here. A bandwidth plateau alone does not
prove bandwidth saturation; it must coincide with a throughput plateau and with compute not being the limit
(see the saturation table)."""),
    ("key", "The selection figure: throughput and memory traffic together",
     """**Plotted:** for each workload, aggregate output throughput (top) and memory traffic (bottom) against
concurrency, on shared x axes. The dotted line is STREAM's sustainable traffic.
**Why it matters:** it answers which backend turns additional concurrent requests into more throughput *and*
more memory-system load.""",
     "show('key_throughput_and_bandwidth_llama31_8b'); show('throughput_vs_bandwidth_llama31_8b')",
     """Separate panels, not a dual axis: the two quantities have different units and the eye must not compare their
heights."""),
    ("onebee", "Cross-check with the 1B model",
     """**Plotted:** the same key figure for Llama-3.2-1B.
**Why it matters:** it tests whether the 8B behaviour is a property of the backend or of the 8B model.""",
     "show('key_throughput_and_bandwidth_llama32_1b'); show('f06_tpot_llama32_1b')",
     """The 1B model has a 6.5× smaller weight read per step and 4× less KV per token, so its bandwidth demand per
token is much lower; absolute bandwidth numbers are not comparable across models."""),
]


def main() -> int:
    obs_path = ROOT / "report" / "observations.toml"
    obs = tomllib.loads(obs_path.read_text()) if obs_path.exists() else {}
    nb = nbf.v4.new_notebook()
    cells = [md("""
# Backend selection v2 — vLLM vs llama.cpp on Xeon Max (cache mode)

Analysis notebook for `backend_comparison/`. All numbers are read from `processed_results/` (built by
`scripts/analysis/process.py` from the raw results); all figures are produced by `scripts/analysis/figures.py`
and saved in `plots/` with their data. The written conclusions are in `report/backend_selection_report.md`.

Arms: **vLLM BF16** and **llama.cpp BF16** form the precision-matched pair (same weights, same precision).
**llama.cpp Q8_0** is llama.cpp's own best configuration on this CPU and differs in precision; it is shown, never
divided into the BF16 numbers.
"""), code("""
import sys
from pathlib import Path
import pandas as pd
import plotly.io as pio
from IPython.display import display, HTML
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
import figures as F
OUT = ROOT / 'plots'
P = ROOT / 'processed_results'
stream = pd.read_csv(P / 'stream_calibration.csv')
off = pd.read_csv(P / 'offline.csv')
pts = pd.read_csv(P / 'online_points.csv')
reps = pd.read_csv(P / 'online_reps.csv')
pts_main = pts[pts.campaign == 'main']
# (re)build every figure from the processed tables
import subprocess; subprocess.run([sys.executable, str(ROOT / 'scripts' / 'analysis' / 'figures.py')], check=True)

def show(name):
    p = OUT / f'{name}.html'
    if p.exists():
        display(HTML(p.read_text()))
    else:
        print(f'(figure {name} not available: no data)')
"""), md("## Validity checks"), code("""
cols = ['model','arm','workload','concurrency','n_reps','tokens_ok','manifest_consistent','failed_requests',
        'total_tok_s_cv','client_cpu_frac_mean','hosts']
display(pts_main[cols].sort_values(['model','workload','arm','concurrency']))
"""), md(obs.get("validity", {}).get("text", "_Observations to be written after the campaign._"))]
    for key, title, before, plot, cannot in SECTIONS:
        cells += [md(f"## {title}\n\n{before}"), code(plot),
                  md("**What can be observed.** " + obs.get(key, {}).get("text", "_to be written from the data_")
                     + f"\n\n**What cannot be concluded from this plot alone.** {cannot}")]
    cells += [md("""
## Saturation analysis

For every doubling of C: throughput gain, memory-traffic gain and median-TPOT increase. Saturation is claimed only
where throughput and memory traffic both plateau while latency keeps rising.
"""), code("""
sat = pts_main[['model','arm','workload','concurrency','total_tok_s_mean','gain_vs_prev','mem_total_gbs_mean',
                'bw_gain_vs_prev','tpot_increase_vs_prev','bw_util_vs_stream_max_mean','cpu_busy_compute_mean_mean',
                'amx_busy_frac_mean','tokens_per_step_mean']]
display(sat.sort_values(['model','workload','arm','concurrency']).round(3))
"""), md(obs.get("saturation", {}).get("text", "_to be written from the data_")),
              md("## Fact / observation / interpretation / conclusion\n\n" + obs.get("summary", {}).get(
                  "text", "_to be written from the data_"))]
    nb["cells"] = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / "analysis.ipynb"
    nbf.write(nb, out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
