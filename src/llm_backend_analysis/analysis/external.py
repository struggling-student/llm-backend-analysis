"""Published results of other CPUs and GPUs for the same model, set against our points (notebook 13).

`data/external/reference_results.csv` (built by scripts/processing/build_external_references.py) holds one row per
external measurement. Each row is paired with our nearest point: same model, same precision group, nearest workload
shape (log distance of ISL and OSL) and nearest concurrency of that workload's sweep (log distance; our peak over the
sweep for throughput-only scenarios such as MLPerf Offline/Server). Ratios are "external advantage" factors, > 1
meaning the external system is better: throughput external ÷ ours, latency and energy ours ÷ external. Per-device
ratios divide the external value by its GPUs or sockets (ours is one socket); they are missing where the source does
not say how many sockets ran (Lenovo LP2463).

Shapes of published results that we re-measured exactly (campaign `external-shapes`, workloads with
`scope = "external"`) are candidates of the shape match; a row pairs only with a workload its configuration has data for,
so those rows reach grade A (same shape and concurrency) and the rest keep their nearest primary shape.

Energy. RAPL is not readable on CRESCO8, but Slurm records the node's energy through IPMI (DCMI sensor, whole node: both
sockets, memory, platform; 60 s updates). The external-shapes points store it per phase, so they carry a measured average
node power (`node_power_w`); a point with a window of at least `MIN_POWER_WINDOW_S` uses its own power. Every other point
uses one power: the median node power of the measured points (`measured_power_w`) or, without any, `POWER_W_ASSUMED`, the
TDP of the one socket the server runs on. The node boundary includes the idle second socket and the platform, so it is an
upper bound of the server's own power.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import paths
from ..data import registry

COLUMNS = ["source_id", "hardware", "hardware_class", "devices", "device_unit", "engine", "engine_version", "model_id",
           "precision", "scenario", "workload", "isl", "osl", "concurrency", "output_tok_s", "per_user_tok_s", "ttft_s",
           "ttft_stat", "tpot_s", "tpot_stat", "power_w", "power_boundary", "tokens_per_kwh", "energy_j_per_output_token",
           "value_origin", "location", "notes"]

PRECISION_GROUP = {"bf16": "bf16", "fp16": "bf16", "fp8": "8bit", "w8a8": "8bit", "w4a8": "4bit"}
# Our configurations per precision group: (selected backend, used for the ratios; the other backend, shown alongside)
OURS = {"bf16": ("vllm_bf16", "llama_cpp_bf16"), "8bit": ("vllm_int8", "llama_cpp_int8"), "4bit": ("vllm_int8", "llama_cpp_int8")}
# Our precision exactly (the other members of a group are near, not equal: FP8 vs INT8, FP16 vs BF16)
SAME_PRECISION = {"bf16": "bf16", "w8a8": "w8a8"}
SHORT_CONTEXT_ISL = 1024     # decode-bandwidth comparison: prompts short enough that weights dominate the traffic
PEAK_SCENARIOS = {"mlperf_offline", "mlperf_server"}
POWER_W_ASSUMED = 350.0      # Xeon CPU Max 9480 TDP (data/external/hardware_specs.csv); not a measurement
MEASURED_CAMPAIGN = "external-shapes"
MIN_POWER_WINDOW_S = 600.0   # 60 s IPMI updates shift the window by at most one interval: <= 10% error at 600 s
GRADE_A_DISTANCE = 0.35      # shape distance ≤ ln(1.4): ISL and OSL within about 1.2× each
GRADE_B_DISTANCE = 1.5
GRADE_TEXT = {"A": "same model, precision, shape and concurrency", "B": "same model and precision class, different shape",
              "C": "loosely comparable (dataset, scenario, 4-bit or different engine setting)"}


def references() -> pd.DataFrame:
    df = pd.read_csv(paths.EXTERNAL / "reference_results.csv")
    df["precision_group"] = df["precision"].map(PRECISION_GROUP)
    specs = hardware_specs()[["hardware", "peak_mem_bw_gbs", "power_rating_w"]]
    df = df.merge(specs, on="hardware", how="left")
    df["output_tok_s_per_device"] = df["output_tok_s"] / df["devices"]
    return df


def hardware_specs() -> pd.DataFrame:
    return pd.read_csv(paths.EXTERNAL / "hardware_specs.csv")


def sources() -> pd.DataFrame:
    return pd.read_csv(paths.EXTERNAL / "sources.csv")


def shape_distance(isl_ext: float, osl_ext: float, isl: float, osl: float) -> float:
    """|ln ISL ratio| + |ln OSL ratio|; only the OSL term when the external ISL is unknown."""
    d = abs(math.log(osl_ext / osl))
    if not (isl_ext is None or (isinstance(isl_ext, float) and math.isnan(isl_ext))):
        d += abs(math.log(isl_ext / isl))
    return d


def measured_power_w(points: pd.DataFrame) -> pd.Series:
    """Average node power (W) of the points with a measured energy window of at least MIN_POWER_WINDOW_S."""
    if "node_power_w" not in points:
        return pd.Series(dtype=float)
    ok = points["node_power_w"].notna() & (points["node_power_window_s"] >= MIN_POWER_WINDOW_S)
    return points.loc[ok, "node_power_w"]


def candidate_workloads(points: pd.DataFrame, configuration_id: str) -> pd.DataFrame:
    """Our workloads (primary and external-match shapes) for which `configuration_id` has points."""
    have = set(points.loc[points["configuration_id"] == configuration_id, "workload"])
    w = registry.workloads(include_external=True)
    return w[w["workload"].isin(have)]


def nearest_workload(isl_ext: float, osl_ext: float, workloads: pd.DataFrame) -> tuple[str, float]:
    """Our workload closest in shape; ties go to the one closest in total tokens, then to the workload order."""
    best = None
    for _, w in workloads.sort_values("order").iterrows():
        d = round(shape_distance(isl_ext, osl_ext, w["isl"], w["osl"]), 9)
        tot = ((0 if pd.isna(isl_ext) else isl_ext) + osl_ext) / (w["isl"] + w["osl"])
        key = (d, abs(math.log(tot)))
        if best is None or key < best[0]:
            best = (key, w["workload"], d)
    return best[1], best[2]


def nearest_concurrency(c: float, available) -> int:
    """Our sweep value nearest to `c` on a log scale (ties go to the lower value)."""
    av = sorted(int(v) for v in available)
    return min(av, key=lambda v: (round(abs(math.log(v / c)), 9), v))


def grade(row: pd.Series) -> str:
    fixed = row["scenario"] == "closed_loop"
    same_c = fixed and row["our_concurrency"] == row["concurrency"]
    if (row["precision"] in SAME_PRECISION and row["model_id"] == registry.models().set_index("model").loc[registry.primary_model(), "hf_id"]
            and fixed and same_c and row["shape_distance"] <= GRADE_A_DISTANCE):
        return "A"
    if row["precision_group"] in ("bf16", "8bit") and fixed and row["shape_distance"] <= GRADE_B_DISTANCE:
        return "B"
    return "C"


def _our_values(points: pd.DataFrame, configuration_id: str, workload: str, c, peak: bool) -> dict:
    g = points[(points["configuration_id"] == configuration_id) & (points["workload"] == workload)]
    if g.empty:
        return {}
    if peak:
        r = g.loc[g["output_tok_s_mean"].idxmax()]
    else:
        r = g[g["concurrency"] == nearest_concurrency(c, g["concurrency"])].iloc[0]
    return {"concurrency": int(r["concurrency"]), "max_concurrency": int(g["concurrency"].max()), "output_tok_s": r["output_tok_s_mean"], "total_tok_s": r["total_tok_s_mean"],
            "ttft_s": r["ttft_s_p50_mean"], "tpot_s": r["tpot_s_p50_mean"], "campaign": r.get("campaign"),
            "node_power_w": r.get("node_power_w", np.nan), "node_power_window_s": r.get("node_power_window_s", np.nan)}


def _basis(peak: bool, c_ext, ours: dict) -> str:
    if peak:
        return "peak over C"
    if ours["concurrency"] == c_ext:
        return "same C"
    return "our highest C" if c_ext > ours["max_concurrency"] else "nearest C"


def compare(refs: pd.DataFrame, points: pd.DataFrame, power_w: float = POWER_W_ASSUMED) -> pd.DataFrame:
    """One row per external measurement with our nearest point (vLLM and llama.cpp of the same precision group).

    `points` may hold the primary campaign and the external-match campaign; each row pairs with the nearest workload
    its selected configuration has data for. `power_w` applies to points without a measured node power."""
    rows = []
    for _, r in refs.iterrows():
        peak = r["scenario"] in PEAK_SCENARIOS or pd.isna(r["concurrency"])
        main_id, other_id = OURS[r["precision_group"]]
        wl, dist = nearest_workload(r["isl"], r["osl"], candidate_workloads(points, main_id))
        a = _our_values(points, main_id, wl, r["concurrency"], peak)
        b = _our_values(points, other_id, wl, r["concurrency"], peak)
        rows.append({**r.to_dict(), "our_workload": wl, "shape_distance": dist, "our_campaign": a["campaign"],
                     "our_basis": _basis(peak, r["concurrency"], a),
                     "ours_node_power_w": a["node_power_w"], "ours_node_power_window_s": a["node_power_window_s"],
                     "our_configuration": main_id, "our_concurrency": a["concurrency"],
                     "ours_output_tok_s": a["output_tok_s"], "ours_total_tok_s": a["total_tok_s"], "ours_ttft_s": a["ttft_s"],
                     "ours_tpot_s": a["tpot_s"],
                     "other_configuration": other_id, "other_concurrency": b.get("concurrency"),
                     "other_output_tok_s": b.get("output_tok_s"), "other_ttft_s": b.get("ttft_s"), "other_tpot_s": b.get("tpot_s")})
    df = pd.DataFrame(rows)
    df["grade"] = df.apply(grade, axis=1)
    df["power_w_assumed"] = power_w
    own = df["ours_node_power_w"].notna() & (df["ours_node_power_window_s"] >= MIN_POWER_WINDOW_S)
    df["ours_power_w"] = df["ours_node_power_w"].where(own, power_w)
    df["ours_power_basis"] = np.where(own, "measured at this point (node, IPMI)", "one power for all points")
    df["ours_energy_j_per_token_est"] = df["ours_power_w"] / df["ours_output_tok_s"]
    df["adv_throughput"] = df["output_tok_s"] / df["ours_output_tok_s"]
    df["adv_throughput_per_device"] = df["output_tok_s_per_device"] / df["ours_output_tok_s"]
    # Prompt plus output tokens per second: less sensitive to the shape mismatch than output tok/s alone (MLPerf's
    # 778 → 73 tokens has no counterpart among our shapes); only where the external ISL is known
    df["ext_total_tok_s"] = df["output_tok_s"] * (df["isl"] + df["osl"]) / df["osl"]
    df["adv_total_throughput_per_device"] = df["ext_total_tok_s"] / df["devices"] / df["ours_total_tok_s"]
    df["adv_ttft"] = df["ours_ttft_s"] / df["ttft_s"]
    df["adv_tpot"] = df["ours_tpot_s"] / df["tpot_s"]
    df["adv_energy"] = df["ours_energy_j_per_token_est"] / df["energy_j_per_output_token"]
    df["implied_power_w"] = df["output_tok_s"] * df["energy_j_per_output_token"]
    return df


def decode_bandwidth(refs: pd.DataFrame, points: pd.DataFrame, stream_gbs: float) -> pd.DataFrame:
    """Single-stream decode: weight bytes streamed per token ÷ time per token, against each system's memory roof.

    External: closed-loop C = 1 rows with prompts of at most SHORT_CONTEXT_ISL tokens (newest engine version of each
    source, hardware and precision) and a TPOT (Lenovo gives only per-user tok/s, whose inverse also contains the
    TTFT share, so its rate is a lower bound). Ours: C = 1 of every configuration on the decode-heavy and balanced shapes.
    The weight bytes are those of our checkpoints (FP8 counted as our W8A8 size); KV-cache and activation traffic are
    left out, so all rates are lower bounds of the true traffic."""
    model = registry.primary_model()
    confs = registry.configurations()
    wbytes = {"bf16": registry.step_weight_bytes(model, "bf16"), "8bit": registry.step_weight_bytes(model, "w8a8")}
    ext = refs[(refs["scenario"] == "closed_loop") & (refs["concurrency"] == 1) & refs["precision_group"].isin(wbytes)
               & (refs["isl"] <= SHORT_CONTEXT_ISL)].copy()
    newest = ext.groupby(["source_id", "hardware", "precision"])["engine_version"].transform("max")
    ext = ext[ext["engine_version"] == newest]
    ext["time_per_token_s"] = ext["tpot_s"].fillna(1 / ext["per_user_tok_s"])
    ext["time_basis"] = np.where(ext["tpot_s"].notna(), ext["tpot_stat"], "1 / per-user tok/s (includes TTFT share)")
    ext["weight_gb"] = ext["precision_group"].map(wbytes) / 1e9
    ext["roof_gbs"] = ext["peak_mem_bw_gbs"]
    ext["roof_kind"] = "vendor peak"
    ext["system"] = ext["hardware"]
    ours = points[(points["concurrency"] == 1) & points["workload"].isin(["decode", "balanced"])].copy()
    ours["time_per_token_s"] = ours["tpot_s_p50_mean"]
    ours["time_basis"] = "median TPOT"
    ours["weight_gb"] = ours["configuration_id"].map(lambda c: registry.step_weight_bytes(model, confs[c].harness_precision)) / 1e9
    ours["precision_group"] = ours["precision"].map({"bf16": "bf16", "int8": "8bit"})
    ours["roof_gbs"] = stream_gbs
    ours["roof_kind"] = "measured STREAM (HBM cache)"
    ours["system"] = "Xeon CPU Max 9480, 1 socket (ours, " + ours["configuration_id"].map(registry.configuration_label) + ")"
    ours["workload"] = ours["workload"].map(lambda w: registry.workload_label(w, short=True))
    cols = ["system", "precision_group", "workload", "time_per_token_s", "time_basis", "weight_gb", "roof_gbs", "roof_kind"]
    df = pd.concat([ext.assign(origin="external")[cols + ["origin", "source_id", "precision", "engine_version"]],
                    ours.assign(origin="ours", source_id="this repository", precision=ours["precision"],
                                engine_version="")[cols + ["origin", "source_id", "precision", "engine_version"]]],
                   ignore_index=True)
    df["effective_gbs"] = df["weight_gb"] / df["time_per_token_s"]
    df["roof_share"] = df["effective_gbs"] / df["roof_gbs"]
    return df


def energy_curves(comp: pd.DataFrame) -> pd.DataFrame:
    """Rows with a published J/token, next to our estimated J/token at the paired point."""
    e = comp[comp["energy_j_per_output_token"].notna()].copy()
    return e[["source_id", "hardware", "hardware_class", "precision", "workload", "concurrency", "output_tok_s",
              "energy_j_per_output_token", "implied_power_w", "power_w", "power_boundary", "our_workload", "our_concurrency",
              "ours_output_tok_s", "ours_power_w", "ours_power_basis", "ours_energy_j_per_token_est", "adv_energy", "grade"]]
