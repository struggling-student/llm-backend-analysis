import numpy as np
import pytest

from llm_backend_analysis.analysis import comparisons, metrics


def test_scaling_metrics(synthetic_points):
    p = metrics.add_scaling_metrics(synthetic_points)
    v = p[p["configuration_id"] == "vllm_bf16"].set_index("concurrency")
    assert v.loc[1, "speedup_vs_c1"] == 1
    assert v.loc[16, "speedup_vs_c1"] == pytest.approx(10)
    assert v.loc[16, "scaling_efficiency"] == pytest.approx(10 / 16)
    assert v.loc[2, "gain_vs_prev"] == pytest.approx(1.0)
    assert np.isnan(v.loc[1, "gain_vs_prev"])


def test_curve_summary_peak_and_plateau(synthetic_points):
    s = metrics.curve_summary(synthetic_points).set_index("configuration_id")
    assert s.loc["vllm_bf16", "peak_concurrency"] == 16
    assert s.loc["llama_cpp_bf16", "peak_total_tok_s"] == 23
    # llama.cpp BF16: +4.5% then 0% -> plateau after C=4
    assert s.loc["llama_cpp_bf16", "plateau_concurrency"] == 4
    assert np.isnan(s.loc["vllm_bf16", "plateau_concurrency"])


def test_backend_ratio_only_within_precision(synthetic_points):
    r = comparisons.backend_ratio(synthetic_points, precision="bf16")
    assert set(r["configuration_id_num"]) == {"vllm_bf16"}
    assert set(r["configuration_id_den"]) == {"llama_cpp_bf16"}
    assert r.set_index("concurrency").loc[1, "ratio"] == pytest.approx(10 / 12)


def test_int8_pair_is_discovered_from_metadata(synthetic_points):
    pairs = comparisons.backend_pairs(synthetic_points)
    assert set(pairs["precision"]) == {"bf16", "int8"}
    r8 = comparisons.backend_ratio(synthetic_points, precision="int8")
    assert r8.set_index("concurrency").loc[16, "ratio"] == pytest.approx(130 / 32)
    assert not comparisons.backend_pairs(synthetic_points[synthetic_points["configuration_id"] != "vllm_int8"]) \
        ["precision"].eq("int8").any()


def test_crossover(synthetic_points):
    c = comparisons.crossover_summary(comparisons.backend_ratio(synthetic_points, precision="bf16")).iloc[0]
    assert c["ratio_c1"] < 1 and c["first_c_ratio_ge_1"] == 2 and c["ratio_monotone_from_c2"]


def test_precision_ratio(synthetic_points):
    r = comparisons.precision_ratio(synthetic_points, "llama_cpp")
    assert r.set_index("concurrency").loc[1, "ratio"] == pytest.approx(18 / 12)


def test_latency_budget(synthetic_points):
    b = metrics.throughput_under_latency_budget(synthetic_points, 0.1).set_index("configuration_id")
    # tpot = 0.05 * (1 + C/8) <= 0.1  ->  C <= 8
    assert b.loc["vllm_bf16", "concurrency"] == 8
