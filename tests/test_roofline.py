import numpy as np
import pandas as pd
import pytest

from llm_backend_analysis.analysis import roofline
from llm_backend_analysis.data import registry


@pytest.mark.parametrize("model", list(registry.experiment_config()["models"]))
def test_architecture_reproduces_parameter_count(model):
    arch = roofline.architecture(model)
    assert arch.total_params == pytest.approx(registry.experiment_config()["models"][model]["params"], rel=0.01)


def test_ops_per_request_counts_prefill_and_decode():
    arch = roofline.architecture(registry.primary_model())
    one = roofline.ops_per_request(arch, isl=1, osl=1)
    # One prompt token, attending to itself, plus the logits of the sampled token; no decode step.
    assert one["decode_ops"] == 0
    assert one["prefill_ops"] == arch.linear_ops + arch.attention_ops_per_position + arch.logits_ops
    a, b = roofline.ops_per_request(arch, 512, 512), roofline.ops_per_request(arch, 4096, 512)
    assert b["prefill_ops"] > 8 * a["prefill_ops"]  # causal attention grows faster than linearly
    assert b["decode_ops"] > a["decode_ops"]        # longer context, more attention per decoded token


def test_decode_intensity_tends_to_its_limit():
    arch = roofline.architecture(registry.primary_model())
    kv = registry.experiment_config()["models"][registry.primary_model()]["kv_bytes_per_token"]
    i = roofline.decode_step_intensity(arch, 15e9, kv, np.array([1, 64, 1e7]), 4096)
    assert i[0] < i[1] < i[2]
    assert i[2] == pytest.approx(roofline.decode_intensity_limit(arch, kv, 4096), rel=1e-3)


def _hardware(int8_gemm: bool = False):
    peaks = [("amx_int8_dot", "amx_peak", None, 218_000.0), ("avx512_fp32_fma", "avx512_peak", None, 8_200.0)]
    gemms = [("amx_bf16_gemm", "amx_onednn_gemm", "bf16", 27_000.0)]
    if int8_gemm:
        gemms += [("vllm_sgl_int8", "gemm_ceiling", "int8", 70_000.0), ("vllm_sgl_int8", "gemm_ceiling", "int8", 60_000.0),
                  ("torch_int_mm", "gemm_ceiling", "int8", 90_000.0)]  # fastest, but AMX not confirmed
    rows = [{"kernel": k, "benchmark": b, "dtype": d, "gops": g, "amx_confirmed": None if d is None else k != "torch_int_mm",
             "binding": "mask" if b == "gemm_ceiling" else "", "gemm_m": 6144, "gemm_n": 6144, "gemm_k": 6144}
            for k, b, d, g in peaks + gemms]
    return pd.DataFrame([{"socket_scope": "single", "socket": 0, "threads": 56, "host": "node", "campaign": "c", **r}
                         for r in rows])


def test_compute_ceilings_scale_cores_and_derive_roofs():
    c = roofline.compute_ceilings(_hardware(), socket=0, cores=28)
    assert c.loc["amx_bf16_gemm", "tops"] == pytest.approx(13.5)
    assert c.loc["amx_bf16_peak", "tops"] == pytest.approx(c.loc["amx_int8_peak", "tops"] / 2)
    assert c.loc["amx_int8_gemm", "tops"] == pytest.approx(2 * c.loc["amx_bf16_gemm", "tops"])
    assert c.loc["amx_int8_gemm", "basis"].startswith("derived")


def test_measured_int8_gemm_replaces_the_derived_roof():
    c = roofline.compute_ceilings(_hardware(int8_gemm=True), socket=0, cores=56)
    # median over the two vllm_sgl_int8 records; torch_int_mm is faster but not confirmed to use AMX
    assert c.loc["amx_int8_gemm", "tops"] == pytest.approx(65.0)
    assert c.loc["amx_int8_gemm", "basis"].startswith("measured")


def test_regime_follows_the_ridge():
    compute = roofline.compute_ceilings(_hardware(), socket=0, cores=56)
    base = {"model": registry.primary_model(), "configuration_id": "vllm_bf16", "workload": "balanced", "isl": 512,
            "osl": 512, "working_set_gib_model": 20.0, "stream_pmu_total_gbs_at_ws": 500.0}
    pts = pd.DataFrame([{**base, "concurrency": c, "input_tok_s_mean": t, "output_tok_s_mean": t,
                         "mem_total_gbs_mean": gbs, "model_read_gbs": gbs}
                        for c, t, gbs in [(1, 40.0, 300.0), (64, 900.0, 150.0)]])
    out = roofline.add_roofline_metrics(pts, compute)
    assert list(out["regime"]) == ["memory-bound side", "compute-bound side"]
    assert np.allclose(out["roof_efficiency"], np.maximum(out["bw_utilization"], out["compute_utilization"]))


def test_amx_efficiency_covers_only_amx_configurations():
    from llm_backend_analysis.analysis import memory
    from llm_backend_analysis.data import loaders
    model = registry.primary_model()
    compute = roofline.compute_ceilings(loaders.hardware_ceilings())
    pts = roofline.add_roofline_metrics(memory.add_traffic_model(loaders.online_points(model=model)), compute)
    eff = roofline.amx_efficiency(pts, compute, loaders.offline_samples(model=model))
    assert "llama_cpp_bf16" not in eff.index  # no AMX path: AVX-512 roof
    assert (eff["amx_busy_min"] <= eff["amx_busy_max"]).all()
    assert (eff["offline_share_of_roof"] > 0).all() and (eff["busy_over_modelled_points"] > 0).all()


def test_phase_mix_reports_prefill_share_and_batch():
    pts = pd.DataFrame([{"configuration_id": "vllm_bf16", "workload": "kvdecode", "concurrency": c,
                         "prefill_op_share": 0.78, "tokens_per_step_mean": t} for c, t in ((1, 5.0), (64, 255.0))])
    mix = roofline.phase_mix(pts).loc["vllm_bf16"]
    assert mix["c_max"] == 64 and mix["tokens_per_step_at_c_max"] == 255.0 and mix["prefill_op_share"] == 0.78
