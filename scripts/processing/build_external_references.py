"""Build data/external/reference_results.csv: published Llama-3.1-8B serving results of other CPUs and GPUs.

Machine-readable sources are downloaded and parsed (nothing typed): the ML.ENERGY leaderboard data file, the NVIDIA NIM
benchmarking page (Markdown) and the MLPerf Inference v5.1 result summaries of the two CPU-only submissions. Results that
exist only as text or figures (Lenovo LP2463, Arm, Red Hat) are transcribed in data/external/manual_results.csv, which
this script merges unchanged apart from derived columns (aggregate tok/s from per-user tok/s, J/token from tokens/kWh).

Downloads are cached in data/external/.cache/ (not committed; the sources are third-party documents). Every fetched
file's URL, size and SHA-256 go to data/external/retrieval.csv, so a later re-fetch shows whether a source changed.

    uv run python scripts/processing/build_external_references.py            # use the cache when present
    uv run python scripts/processing/build_external_references.py --refresh  # download again
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llm_backend_analysis import paths  # noqa: E402
from llm_backend_analysis.analysis.external import COLUMNS  # noqa: E402

EXTERNAL = paths.EXTERNAL
CACHE = EXTERNAL / ".cache"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MLPERF = "https://raw.githubusercontent.com/mlcommons/inference_results_v5.1/main/closed"
MLPERF_SYSTEMS = {  # source_id -> (submitter, system, hardware, cores per socket)
    "mlperf_v51_intel": ("Intel", "1-node-2S-GNR_128C", "Intel Xeon 6980P", 128),
    "mlperf_v51_lenovo": ("Lenovo", "ThinkSystem_SR650_V4_GNR_86C", "Intel Xeon 6787P", 86),
}
CNN_DAILYMAIL = (778, 73)  # mean input / output tokens of the MLPerf subset, as published by Red Hat (redhat_mlperf_v51)

DOWNLOADS = {
    "mlenergy_lm_arena_chat.json": "https://ml.energy/leaderboard/data/tasks/lm-arena-chat.json",
    "mlenergy_index.json": "https://ml.energy/leaderboard/data/index.json",
    "nim_performance_1.0.0.md": "https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.md",
}
for sid, (sub, system, _, _) in MLPERF_SYSTEMS.items():
    DOWNLOADS[f"{sid}_README.md"] = f"{MLPERF}/{sub}/results/{system}/README.md"
    DOWNLOADS[f"{sid}_system.json"] = f"{MLPERF}/{sub}/systems/{system}.json"
    DOWNLOADS[f"{sid}_measurement.json"] = f"{MLPERF}/{sub}/measurements/{system}/llama3_1-8b/Server/{system}.json"

NIM_HARDWARE = {"1 x H100 80G": ("NVIDIA H100 SXM 80GB", 1), "1 x H200 141GB": ("NVIDIA H200 SXM 141GB", 1),
                "1 x L40s 48GB": ("NVIDIA L40S 48GB (PCIe)", 1)}
MLENERGY_HARDWARE = {"H100": "NVIDIA H100 SXM 80GB", "B200": "NVIDIA B200 180GB"}


def fetch(name: str, refresh: bool) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if refresh or not path.exists():
        req = urllib.request.Request(DOWNLOADS[name], headers={"User-Agent": "llm-backend-analysis/0.2 (research)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
        record_retrieval(name, path)
    return path


def record_retrieval(name: str, path: Path) -> None:
    log = EXTERNAL / "retrieval.csv"
    rows = list(csv.DictReader(log.open())) if log.exists() else []
    rows = [r for r in rows if r["file"] != name]
    rows.append({"file": name, "url": DOWNLOADS[name], "retrieved": dt.date.today().isoformat(),
                 "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with log.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "url", "retrieved", "bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["file"]))


def mlenergy(refresh: bool) -> list[dict]:
    data = json.loads(fetch("mlenergy_lm_arena_chat.json", refresh).read_text())
    updated = json.loads(fetch("mlenergy_index.json", refresh).read_text())["last_updated"]
    rows = []
    for c in data["configurations"]:
        if c["model_id"] != MODEL:
            continue
        rows.append(dict(
            source_id="mlenergy_v3", hardware=MLENERGY_HARDWARE[c["gpu_model"]], hardware_class="gpu", devices=c["num_gpus"],
            device_unit="gpu", engine="vLLM", engine_version="0.11.1", model_id=MODEL, precision=c["weight_precision"].replace("bfloat16", "bf16"),
            scenario="saturated_batch", workload="LM Arena chat", isl=None, osl=round(c["avg_output_len"], 1),
            concurrency=c["max_num_seqs"], output_tok_s=c["output_throughput_tokens_per_sec"], tpot_s=c["median_itl_ms"] / 1000,
            tpot_stat="median ITL", power_w=c["avg_power_watts"], power_boundary="GPU only (Zeus), steady state",
            energy_j_per_output_token=c["energy_per_token_joules"], value_origin="json", location=f"lm-arena-chat.json (updated {updated})",
            notes=f"max_num_seqs {c['max_num_seqs']}, mean running batch {c['avg_batch_size']:.1f}; prompt lengths not published"))
    return rows


def nim(refresh: bool) -> list[dict]:
    """Tables of the 'Llama-3.1-8b-instruct Results' section, one row per concurrency."""
    lines = fetch("nim_performance_1.0.0.md", refresh).read_text().splitlines()
    rows, section, version, hw, prec, shape = [], False, None, None, None, None
    for line in lines:
        if line.startswith("## "):
            section = line.strip() == "## Llama-3.1-8b-instruct Results"
            continue
        if not section:
            continue
        if m := re.match(r"### Version: (\S+)", line):
            version = m[1]
        elif m := re.match(r"#### (\d+ x .+)$", line):
            hw = m[1].strip()
        elif m := re.match(r"#### (bf16|fp16|fp8) TP(\d+)", line):
            prec = m[1]
            if m[2] != "1":
                prec = None  # single-GPU results only
        elif m := re.match(r"#### (\d+) Input Tokens / (\d+) Output Tokens", line):
            shape = (int(m[1]), int(m[2]))
        elif prec and (m := re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)):
            hardware, n = NIM_HARDWARE[hw]
            rows.append(dict(
                source_id="nvidia_nim", hardware=hardware, hardware_class="gpu", devices=n, device_unit="gpu", engine="NVIDIA NIM",
                engine_version=version, model_id=MODEL, precision=prec, scenario="closed_loop", workload=f"{shape[0]}→{shape[1]}",
                isl=shape[0], osl=shape[1], concurrency=int(m[1]), output_tok_s=float(m[4]), ttft_s=float(m[2]) / 1000, ttft_stat="mean",
                tpot_s=float(m[3]) / 1000, tpot_stat="mean ITL", value_origin="table",
                location=f"NIM {version}, {hw}, {prec} TP1, {shape[0]} in / {shape[1]} out"))
    return rows


def mlperf(refresh: bool) -> list[dict]:
    rows = []
    for sid, (_, system, hardware, cores) in MLPERF_SYSTEMS.items():
        html = fetch(f"{sid}_README.md", refresh).read_text()
        sysinfo = json.loads(fetch(f"{sid}_system.json", refresh).read_text())
        meas = json.loads(fetch(f"{sid}_measurement.json", refresh).read_text())
        cells = re.search(r'<td class="model">llama3_1-8b</td>(.*?)</tr>|<td class="model">llama3_1-8b</td>(.*?)<tr>', html, re.S)
        perf = re.findall(r'<td class="perf">([\d.]+)</td>', cells.group(0))
        if len(perf) != 2:
            raise ValueError(f"{sid}: expected Server and Offline tokens/s, found {perf}")
        prec = {("INT4", "INT8"): "w4a8", ("INT8", "INT8"): "w8a8"}[(meas["weight_data_types"], meas["input_data_types"])]
        for scen, v in zip(("mlperf_server", "mlperf_offline"), perf):
            rows.append(dict(
                source_id=sid, hardware=hardware, hardware_class="cpu", devices=int(sysinfo["host_processors_per_node"]),
                device_unit="socket", engine=sysinfo.get("framework", ""), engine_version="MLPerf v5.1 submission",
                model_id=MODEL, precision=prec, scenario=scen, workload="CNN-DailyMail (MLPerf)", isl=CNN_DAILYMAIL[0],
                osl=CNN_DAILYMAIL[1], output_tok_s=float(v), value_origin="table", location=f"results/{system}/README.md, llama3_1-8b",
                notes=f"{sysinfo['host_processors_per_node']}x {cores} cores, {sysinfo['host_memory_configuration']}; "
                      f"weights {meas['weight_data_types']}, activations {meas['input_data_types']} (as declared)"))
    return rows


def manual() -> list[dict]:
    df = pd.read_csv(EXTERNAL / "manual_results.csv")
    derived = df["output_tok_s"].isna() & df["per_user_tok_s"].notna()
    df.loc[derived, "output_tok_s"] = df.loc[derived, "per_user_tok_s"] * df.loc[derived, "concurrency"]
    df.loc[derived, "value_origin"] = df.loc[derived, "value_origin"] + "; tok/s = per-user tok/s × users"
    e = df["energy_j_per_output_token"].isna() & df["tokens_per_kwh"].notna()
    df.loc[e, "energy_j_per_output_token"] = 3.6e6 / df.loc[e, "tokens_per_kwh"]
    return df.to_dict("records")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="download every source again")
    args = ap.parse_args()
    rows = mlenergy(args.refresh) + nim(args.refresh) + mlperf(args.refresh) + manual()
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df = df.sort_values(["hardware_class", "source_id", "hardware", "precision", "engine_version", "workload", "scenario", "concurrency"],
                        kind="mergesort", ignore_index=True)
    df.insert(0, "ref_id", [f"ext{i:03d}" for i in range(1, len(df) + 1)])
    df.to_csv(EXTERNAL / "reference_results.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(f"{len(df)} external rows from {df['source_id'].nunique()} sources -> {paths.relative(EXTERNAL / 'reference_results.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
