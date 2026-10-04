"""Raw measurements (data/raw/) -> processed tables (data/processed/).

Deterministic: rows are sorted, no timestamps are written, and every row carries the
campaign, its role (data/metadata/campaigns.toml), the canonical configuration
dimensions and the raw source path it came from. Run it with `make data`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ... import paths
from .offline import process_offline
from .online import aggregate_points, process_online, read_sweeps
from .stream import process_stream, read_node_checks, sustainable_lookup
from .validation_runs import process_pmu_overhead, process_quality, process_tuning

OUTPUTS = {
    "stream_calibration": "stream_calibration.csv",
    "node_checks": "node_checks.csv",
    "online_repetitions": "online_repetitions.csv",
    "online_points": "online_points.csv",
    "online_requests": "online_requests.parquet",
    "online_sweeps": "online_sweeps.csv",
    "offline_samples": "offline_samples.csv",
    "tuning_runs": "tuning_runs.csv",
    "pmu_overhead": "pmu_overhead.csv",
    "quality_check": "quality_check.csv",
}


def build_tables() -> dict[str, pd.DataFrame]:
    stream = process_stream()
    node_checks = read_node_checks()
    reps, requests = process_online(sustainable_lookup(stream), node_checks)
    return {
        "stream_calibration": stream,
        "node_checks": node_checks,
        "online_repetitions": reps,
        "online_points": aggregate_points(reps, requests),
        "online_requests": requests,
        "online_sweeps": read_sweeps(),
        "offline_samples": process_offline(),
        "tuning_runs": process_tuning(),
        "pmu_overhead": process_pmu_overhead(reps),
        "quality_check": process_quality(),
    }


def write_tables(tables: dict[str, pd.DataFrame], out: Path = paths.PROCESSED) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, filename in OUTPUTS.items():
        path = out / filename
        df = tables[name]
        if filename.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False, lineterminator="\n")
        written[name] = path
    return written
