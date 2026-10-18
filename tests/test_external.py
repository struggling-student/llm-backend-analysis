import math

import pandas as pd
import pytest

from llm_backend_analysis.analysis import external
from llm_backend_analysis.data import registry


def test_shape_distance_uses_osl_only_when_isl_unknown():
    assert external.shape_distance(512, 512, 512, 512) == 0
    assert external.shape_distance(1000, 1000, 512, 512) == pytest.approx(2 * math.log(1000 / 512))
    assert external.shape_distance(float("nan"), 400, 128, 512) == pytest.approx(math.log(512 / 400))


def test_nearest_workload_breaks_ties_by_total_tokens():
    # 256→256 is equally far from 128→512 and 512→512 in shape; 128→512 has the closer token total (640 vs 1024)
    assert external.nearest_workload(256, 256, registry.workloads())[0] == "decode"
    assert external.nearest_workload(1000, 1000, registry.workloads())[0] == "balanced"
    assert external.nearest_workload(5000, 500, registry.workloads())[0] == "kvdecode"


def test_nearest_concurrency_is_logarithmic_and_prefers_lower_on_ties():
    assert external.nearest_concurrency(24, [8, 16, 32]) == 32
    assert external.nearest_concurrency(10, [8, 16]) == 8
    assert external.nearest_concurrency(2048, [1, 64, 128]) == 128
    assert external.nearest_concurrency(math.sqrt(8 * 16), [8, 16]) == 8


def _points():
    rows = []
    for c, out in ((1, 18.0), (64, 600.0), (128, 700.0)):
        rows.append({"configuration_id": "vllm_bf16", "workload": "balanced", "concurrency": c, "output_tok_s_mean": out,
                     "total_tok_s_mean": 2 * out, "ttft_s_p50_mean": 0.4, "tpot_s_p50_mean": c / out})
        rows.append({"configuration_id": "llama_cpp_bf16", "workload": "balanced", "concurrency": c, "output_tok_s_mean": out / 5,
                     "total_tok_s_mean": 2 * out / 5, "ttft_s_p50_mean": 2.0, "tpot_s_p50_mean": 5 * c / out})
    return pd.DataFrame(rows)


def _ref(**kw):
    base = dict(source_id="s", hardware="h", hardware_class="gpu", devices=1, device_unit="gpu", engine="e", engine_version="1",
                model_id=registry.models().set_index("model").loc[registry.primary_model(), "hf_id"], precision="bf16",
                precision_group="bf16", scenario="closed_loop", workload="512→512", isl=512, osl=512, concurrency=64,
                output_tok_s=6000.0, output_tok_s_per_device=6000.0, ttft_s=0.2, tpot_s=0.01, energy_j_per_output_token=0.1,
                peak_mem_bw_gbs=3350, power_rating_w=700)
    base.update(kw)
    return pd.DataFrame([base])


def test_compare_pairs_ratios_and_grade():
    c = external.compare(_ref(), _points(), power_w=350.0).iloc[0]
    assert c["our_concurrency"] == 64 and c["our_basis"] == "same C" and c["grade"] == "A"
    assert c["adv_throughput"] == pytest.approx(10.0)
    assert c["adv_ttft"] == pytest.approx(0.4 / 0.2)
    assert c["ours_energy_j_per_token_est"] == pytest.approx(350 / 600)
    assert c["adv_energy"] == pytest.approx((350 / 600) / 0.1)
    assert c["other_output_tok_s"] == pytest.approx(120.0)


def test_compare_grades_and_bases():
    far = external.compare(_ref(isl=1000, osl=1000, concurrency=250), _points()).iloc[0]
    assert far["grade"] == "B" and far["our_basis"] == "our highest C" and far["our_concurrency"] == 128
    fp8 = external.compare(_ref(precision="fp8", precision_group="8bit"), _points().replace(
        {"vllm_bf16": "vllm_int8", "llama_cpp_bf16": "llama_cpp_int8"})).iloc[0]
    assert fp8["grade"] == "B"   # FP8 is near our W8A8, not equal
    mlperf = external.compare(_ref(scenario="mlperf_offline", concurrency=float("nan")), _points()).iloc[0]
    assert mlperf["our_basis"] == "peak over C" and mlperf["grade"] == "C"


def test_compare_prefers_an_exact_external_shape_with_data_and_its_measured_power():
    ext = pd.DataFrame([{"configuration_id": "vllm_bf16", "workload": "ext1000", "concurrency": c, "output_tok_s_mean": out,
                         "total_tok_s_mean": 2 * out, "ttft_s_p50_mean": 0.5, "tpot_s_p50_mean": c / out, "campaign": external.MEASURED_CAMPAIGN,
                         "node_power_w": w, "node_power_window_s": win}
                        for c, out, w, win in ((1, 17.0, 600.0, 1200.0), (100, 650.0, 800.0, 300.0))])
    pts = pd.concat([_points().assign(campaign="main"), ext], ignore_index=True)
    c1 = external.compare(_ref(isl=1000, osl=1000, concurrency=1), pts, power_w=700.0).iloc[0]
    assert c1["our_workload"] == "ext1000" and c1["shape_distance"] == 0 and c1["grade"] == "A"
    assert c1["ours_power_w"] == 600.0 and c1["ours_energy_j_per_token_est"] == pytest.approx(600 / 17)
    assert pd.isna(c1["other_output_tok_s"])   # llama.cpp was not run at this shape
    c100 = external.compare(_ref(isl=1000, osl=1000, concurrency=100), pts, power_w=700.0).iloc[0]
    assert c100["grade"] == "A" and c100["ours_power_w"] == 700.0   # window too short for its own power
    fp8 = external.compare(_ref(isl=1000, osl=1000, precision="fp8", precision_group="8bit"),
                           pts.replace({"vllm_bf16": "vllm_int8", "llama_cpp_bf16": "llama_cpp_int8"})).iloc[0]
    assert fp8["our_workload"] == "ext1000"
    assert external.measured_power_w(pts).tolist() == [600.0]


def test_external_shapes_stay_out_of_the_primary_workload_list():
    assert "ext1000" not in registry.workload_order()
    assert {"ext1000", "ext256"} <= set(registry.workloads(include_external=True)["workload"])
    assert registry.workload_label("ext256", short=True) == "256→256"


def test_reference_table_is_consistent():
    refs = external.references()
    assert refs["precision_group"].notna().all()
    assert set(refs["source_id"]) <= set(external.sources()["source_id"])
    assert (refs["output_tok_s"] > 0).all()
    unknown = set(refs["hardware"]) - set(external.hardware_specs()["hardware"])
    assert all("Arm" in h or "6980P" in h or "6787P" in h for h in unknown)
