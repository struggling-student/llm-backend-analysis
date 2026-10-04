"""Experiment metadata as data: configurations, campaigns, workloads, models, topology.

Sources (never duplicated in code):
  configs/experiment.toml             harness parameters (models, arms, workloads, topology)
  data/metadata/configurations.toml   arm -> canonical backend / precision / quantization
  data/metadata/campaigns.toml        role of every raw campaign
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

import pandas as pd

from .. import paths

ANALYSIS_CAMPAIGN = {"online": "main", "offline": "main", "stream": "calibration"}
CONFIGURATION_DIMENSIONS = ["configuration_id", "backend", "precision", "quantization"]


@dataclass(frozen=True)
class Configuration:
    configuration_id: str
    arm: str
    backend: str
    precision: str
    quantization: str
    weight_format: str
    kv_cache_dtype: str
    amx_path: str
    tuning_default_variant: str
    status: str
    harness_precision: str  # precision key the harness uses (bf16 / q8_0 / w8a8), from experiment.toml
    label: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@cache
def experiment_config() -> dict[str, Any]:
    return tomllib.loads(paths.EXPERIMENT_CONFIG.read_text())


@cache
def _configuration_file() -> dict[str, Any]:
    return tomllib.loads((paths.METADATA / "configurations.toml").read_text())


@cache
def configurations() -> dict[str, Configuration]:
    """All configurations, in the declaration order of configurations.toml."""
    meta, arms = _configuration_file(), experiment_config()["arms"]
    out = {}
    for cid, c in meta["configurations"].items():
        arm = arms[c["arm"]]
        label = f"{meta['backends'][c['backend']]['label']} · {meta['precisions'][c['precision']]['label']}"
        if quant := meta["quantizations"][c["quantization"]]["label"]:
            label += f" ({quant})"
        out[cid] = Configuration(configuration_id=cid, arm=c["arm"], backend=c["backend"], precision=c["precision"],
                                 quantization=c["quantization"], weight_format=c["weight_format"],
                                 kv_cache_dtype=c["kv_cache_dtype"], amx_path=c["amx_path"],
                                 tuning_default_variant=c["tuning_default_variant"], status=c["status"],
                                 harness_precision=arm["precision"], label=label)
    return out


@cache
def _by_arm() -> dict[str, Configuration]:
    return {c.arm: c for c in configurations().values()}


def configuration_for_arm(arm: str) -> Configuration:
    try:
        return _by_arm()[arm]
    except KeyError:
        raise KeyError(f"arm {arm!r} has no entry in data/metadata/configurations.toml") from None


def split_arm_dir(arm_dir: str) -> tuple[str, str]:
    """`llamacpp-bf16__fa-on_ub512` -> (`llamacpp-bf16`, `fa-on_ub512`); untagged dirs have variant ''."""
    arm, _, variant = arm_dir.partition("__")
    return arm, variant


def configuration_columns(arm: str) -> dict[str, str]:
    c = configuration_for_arm(arm)
    return {"configuration_id": c.configuration_id, "backend": c.backend, "precision": c.precision,
            "quantization": c.quantization}


def configurations_frame() -> pd.DataFrame:
    return pd.DataFrame([c.as_dict() for c in configurations().values()])


def backend_label(backend: str) -> str:
    return _configuration_file()["backends"][backend]["label"]


def precision_label(precision: str) -> str:
    return _configuration_file()["precisions"][precision]["label"]


def configuration_label(configuration_id: str) -> str:
    return configurations()[configuration_id].label


def configuration_order() -> list[str]:
    return list(configurations())


# --------------------------------------------------------------------------- campaigns
@cache
def campaigns() -> pd.DataFrame:
    meta = tomllib.loads((paths.METADATA / "campaigns.toml").read_text())
    rows = [{"kind": kind, "campaign": name, **entry}
            for kind, block in meta.items() if isinstance(block, dict) for name, entry in block.items()]
    return pd.DataFrame(rows)


def campaign_role(kind: str, campaign: str) -> str:
    c = campaigns()
    hit = c[(c["kind"] == kind) & (c["campaign"] == campaign)]
    if hit.empty:
        raise KeyError(f"campaign {kind}/{campaign} is not declared in data/metadata/campaigns.toml")
    return str(hit["role"].iloc[0])


# --------------------------------------------------------------------------- workloads / models / topology
def workloads() -> pd.DataFrame:
    """Workload shapes in declaration order: key, isl, osl, label, short label (ISL→OSL)."""
    rows = []
    for i, (key, w) in enumerate(experiment_config()["workloads"].items()):
        rows.append({"workload": key, "isl": w["isl"], "osl": w["osl"], "order": i,
                     "workload_label": w["label"].replace("->", "→"), "shape": f"{w['isl']}→{w['osl']}"})
    return pd.DataFrame(rows)


def workload_label(workload: str, short: bool = False) -> str:
    w = workloads().set_index("workload").loc[workload]
    return w["shape"] if short else w["workload_label"]


def workload_order() -> list[str]:
    return workloads()["workload"].tolist()


def models() -> pd.DataFrame:
    rows = []
    for key, m in experiment_config()["models"].items():
        rows.append({"model": key, "hf_id": m["hf_id"], "revision": m["revision"], "params": m["params"],
                     "layers": m["layers"], "kv_bytes_per_token": m["kv_bytes_per_token"],
                     "weight_bytes_bf16": m["weight_bytes_bf16"], "primary": m.get("primary", False),
                     "model_label": m["hf_id"].split("/")[-1].replace("-Instruct", "")})
    return pd.DataFrame(rows)


def model_label(model: str) -> str:
    return models().set_index("model").loc[model, "model_label"]


def primary_model() -> str:
    m = models()
    return str(m.loc[m["primary"], "model"].iloc[0])


def step_weight_bytes(model: str, harness_precision: str) -> float:
    """Bytes one forward step streams (all tensors except the gathered input-embedding table)."""
    m = experiment_config()["models"][model]
    explicit = m.get("step_weight_bytes", {}).get(harness_precision)
    if explicit:
        return float(explicit)
    return m["weight_bytes_bf16"] * (1.0 if harness_precision == "bf16" else 34 / 64)


def parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def topology() -> dict[str, Any]:
    t = dict(experiment_config()["topology"])
    t["compute_cpu_list"] = parse_cpu_list(t["compute_cpus"])
    return t
