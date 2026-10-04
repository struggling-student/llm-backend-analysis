"""Repository-relative locations. Nothing here depends on the current working directory,
so notebooks, scripts and tests resolve the same files wherever they are started.
Set LBA_ROOT to point the package at another checkout (e.g. a CI workspace)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LBA_ROOT") or Path(__file__).resolve().parents[2])

CONFIGS = ROOT / "configs"
EXPERIMENT_CONFIG = CONFIGS / "experiment.toml"

DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_TELEMETRY = RAW / "telemetry"
RAW_ENVIRONMENT = RAW / "environment"
METADATA = DATA / "metadata"
PROCESSED = DATA / "processed"

NOTEBOOKS = ROOT / "notebooks"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"


def local_telemetry_path(cluster_path: str | Path) -> Path:
    """Map a telemetry path recorded on the cluster ($BC_DATA/telemetry/...) to data/raw/telemetry/..."""
    parts = Path(cluster_path).parts
    if "telemetry" not in parts:
        raise ValueError(f"not a telemetry path: {cluster_path}")
    i = len(parts) - 1 - parts[::-1].index("telemetry")
    return RAW_TELEMETRY.joinpath(*parts[i + 1:])


def relative(path: Path) -> str:
    """Repository-relative string for provenance columns and messages."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
