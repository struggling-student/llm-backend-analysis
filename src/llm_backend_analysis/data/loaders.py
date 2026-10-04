"""Load processed tables for the notebooks.

Every loader selects the analysis campaign explicitly (default: the `primary` campaign of
each kind, see data/metadata/campaigns.toml) and adds human-readable labels plus a stable
sort order derived from the metadata, so new configurations (e.g. vLLM INT8) appear
without code changes.
"""

from __future__ import annotations

from functools import cache

import pandas as pd

from .. import paths
from . import registry
from .processing import OUTPUTS


@cache
def _read(name: str) -> pd.DataFrame:
    path = paths.PROCESSED / OUTPUTS[name]
    if not path.exists():
        raise FileNotFoundError(f"{paths.relative(path)} is missing: run `make data`")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def load_table(name: str) -> pd.DataFrame:
    """Raw access to one processed table (a copy; all campaigns)."""
    return _read(name).copy()


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add configuration/workload/model labels and order keys (does not drop or reorder rows)."""
    out = df.copy()
    if "configuration_id" in out:
        order = {c: i for i, c in enumerate(registry.configuration_order())}
        out["configuration_label"] = out["configuration_id"].map(registry.configuration_label)
        out["configuration_order"] = out["configuration_id"].map(order)
    if "workload" in out:
        w = registry.workloads().set_index("workload")
        out["workload_label"] = out["workload"].map(w["workload_label"])
        out["workload_shape"] = out["workload"].map(w["shape"])
        out["workload_order"] = out["workload"].map(w["order"])
    if "model" in out:
        out["model_label"] = out["model"].map(registry.models().set_index("model")["model_label"])
    return out


def _select(df: pd.DataFrame, campaign: str | None, model: str | None, include_variants: bool) -> pd.DataFrame:
    if campaign is not None:
        df = df[df["campaign"] == campaign]
    if model is not None:
        df = df[df["model"] == model]
    if not include_variants and "variant" in df:
        df = df[df["variant"].fillna("") == ""]
    return df


def _sorted(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    keys = [k for k in ["model", "workload_order", "configuration_order", *extra] if k in df]
    return df.sort_values(keys, ignore_index=True)


def online_points(campaign: str | None = registry.ANALYSIS_CAMPAIGN["online"], model: str | None = None,
                  include_variants: bool = False) -> pd.DataFrame:
    """Per-point statistics (mean/std/ci95 over repetitions, pooled latency percentiles)."""
    df = add_labels(_select(_read("online_points"), campaign, model, include_variants))
    return _sorted(df, ["concurrency"])


def online_repetitions(campaign: str | None = registry.ANALYSIS_CAMPAIGN["online"], model: str | None = None,
                       include_variants: bool = False) -> pd.DataFrame:
    df = add_labels(_select(_read("online_repetitions"), campaign, model, include_variants))
    return _sorted(df, ["concurrency", "rep"])


def online_requests(campaign: str | None = registry.ANALYSIS_CAMPAIGN["online"], model: str | None = None,
                    steady_only: bool = True) -> pd.DataFrame:
    df = _read("online_requests")
    df = df[~df["arm_dir"].str.contains("__", regex=False)]  # untagged arms only (no tuning/overhead variants)
    df = _select(df, campaign, model, include_variants=True)
    if steady_only:
        df = df[df["steady"]]
    return _sorted(add_labels(df), ["concurrency", "rep", "index"])


def online_sweeps(campaign: str | None = registry.ANALYSIS_CAMPAIGN["online"]) -> pd.DataFrame:
    return _sorted(add_labels(_select(_read("online_sweeps"), campaign, None, include_variants=False)), [])


def offline_samples(campaign: str | None = registry.ANALYSIS_CAMPAIGN["offline"], model: str | None = None) -> pd.DataFrame:
    df = add_labels(_select(_read("offline_samples"), campaign, model, include_variants=True))
    return _sorted(df, ["benchmark", "batch", "test", "sample"])


def stream_calibration(campaign: str | None = registry.ANALYSIS_CAMPAIGN["stream"]) -> pd.DataFrame:
    df = _read("stream_calibration")
    return (df[df["campaign"] == campaign] if campaign else df).reset_index(drop=True)


def node_checks() -> pd.DataFrame:
    return load_table("node_checks")


def tuning_runs(campaign: str | None = "tuning") -> pd.DataFrame:
    df = _read("tuning_runs")
    return _sorted(add_labels(df[df["campaign"] == campaign] if campaign else df), ["variant", "concurrency"])


def pmu_overhead(campaign: str | None = "pmu-overhead") -> pd.DataFrame:
    df = _read("pmu_overhead")
    return _sorted(add_labels(df[df["campaign"] == campaign] if campaign else df), ["run_order"])


def quality_check() -> pd.DataFrame:
    return _sorted(add_labels(load_table("quality_check")), ["comparison"])
