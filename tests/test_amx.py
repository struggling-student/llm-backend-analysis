import json

import pandas as pd
import pytest

from llm_backend_analysis.analysis import amx


def samples(amx_rates=(400.0, 410.0), avx_rates=(200.0, 205.0)):
    rows = []
    for build, rates in (("amx", amx_rates), ("avx512", avx_rates)):
        for p, v in enumerate(rates, 1):
            for s in range(3):
                rows.append({"model": "llama31_8b", "workload": "balanced", "isl": 512, "test": "prefill",
                             "variant": f"{build}_p{p}", "sample": s, "tok_s": v})
    return pd.DataFrame(rows)


def test_variant_parsing():
    assert amx.parse_variant("avx512_p2") == ("avx512", 2)
    with pytest.raises(ValueError):
        amx.parse_variant("mem1_p1")


def test_ratio_is_formed_inside_each_pass_pair():
    compute = pd.DataFrame({"tops": [88.0]}, index=["amx_int8_gemm"])
    out = amx.build_comparison(samples(), compute)
    r = out.iloc[0]
    assert r["ratio"] == pytest.approx(405.0 / 202.5)
    assert r["ratio_min"] == pytest.approx(400 / 200) and r["ratio_max"] == pytest.approx(410 / 205)
    assert 0 < r["amx_share_of_roof"] < 1 and r["amx_share_of_roof"] > r["avx512_share_of_roof"]


def test_offline_amx_busy_ignores_the_idle_model_load(tmp_path):
    d = tmp_path / "offline" / "c" / "m" / "llamacpp-q8_0__amx_p1" / "balanced"
    d.mkdir(parents=True)
    busy_cpu = {str(i): 1.0 for i in range(55)}
    idle_cpu = {str(i): 0.05 for i in range(55)}
    lines = [{"type": "sample", "pmu": {"exe_amx_busy": 0, "cycles": 1000}, "cpu_busy": idle_cpu},
             {"type": "sample", "pmu": {"exe_amx_busy": 100, "cycles": 1000}, "cpu_busy": busy_cpu},
             {"type": "sample", "pmu": {"exe_amx_busy": 300, "cycles": 1000}, "cpu_busy": busy_cpu}]
    (d / "a1-balanced-pp.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    out = amx.offline_amx_busy(tmp_path, "c")
    assert out.iloc[0]["amx_busy_frac"] == pytest.approx(0.2) and out.iloc[0]["build"] == "amx"
