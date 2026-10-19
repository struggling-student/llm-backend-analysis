import pandas as pd
import pytest

from llm_backend_analysis.analysis import power


def _phases():
    base = {"rapl_unc0_mhz": 800.0, "rapl_pkg0_tmp_c": 40.0, "ipmi_n": 10, "rapl_n": 60}
    rows = [{**base, "phase": power.IDLE_PHASE, "window": "idle", "arm_dir": None, "concurrency": None, "duration_s": 600.0,
             "rapl_pkg0_w": 135.0, "rapl_ram0_w": 21.0, "rapl_pkg1_w": 134.0, "rapl_ram1_w": 19.0, "ipmi_sys_w": 400.0,
             "ipmi_cpu_w": 269.0, "ipmi_mem_w": 40.0}]
    for arm, ram0, sys in (("vllm-bf16__mem2", 22.0, 670.0), ("vllm-bf16__mem0", 62.0, 730.0)):
        for rep, dur in (("rep1", 100.0), ("rep2", 300.0)):
            rows.append({**base, "phase": rep, "window": "steady", "arm_dir": arm, "concurrency": 1, "duration_s": dur,
                         "rapl_pkg0_w": 349.0 if rep == "rep1" else 350.0, "rapl_ram0_w": ram0, "rapl_pkg1_w": 170.0, "rapl_ram1_w": 19.0,
                         "ipmi_sys_w": sys, "ipmi_cpu_w": 519.0, "ipmi_mem_w": ram0 + 19.0})
    return pd.DataFrame(rows)


def _points():
    return pd.DataFrame([{"arm_dir": "vllm-bf16__mem2", "concurrency": 1, "output_tok_s_mean": 22.0, "ghz_effective_per_core_mean": 2.3},
                         {"arm_dir": "vllm-bf16__mem0", "concurrency": 1, "output_tok_s_mean": 14.0, "ghz_effective_per_core_mean": 2.5}])


def test_point_power_weights_by_duration_and_subtracts_idle():
    pp = power.point_power(_phases(), _points())
    h = pp[pp["arm_dir"] == "vllm-bf16__mem2"].iloc[0]
    assert h["pkg0"] == pytest.approx(349.75)                      # (349*100 + 350*300) / 400
    assert h["server_w"] == pytest.approx(349.75 + 22.0)
    assert h["server_above_idle_w"] == pytest.approx(349.75 + 22.0 - 156.0)
    assert h["platform_w"] == pytest.approx(670.0 - 519.0 - 41.0)
    assert h["j_per_token_server"] == pytest.approx((349.75 + 22.0) / 22.0)


def test_hbm_vs_ddr_ratios_and_deltas():
    hd = power.hbm_vs_ddr(power.point_power(_phases(), _points())).iloc[0]
    assert hd["throughput_ratio"] == pytest.approx(22.0 / 14.0)
    assert hd["delta_ram0"] == pytest.approx(-40.0) and hd["delta_sys"] == pytest.approx(-60.0)
