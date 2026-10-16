import numpy as np
import pandas as pd
import pytest

from llm_backend_analysis.analysis import membind


def points(local=(28.3, 28.4), remote=(7.95, 7.94), tpot=(0.043, 0.156)):
    rows = []
    for node, vals, t in ((0, local, tpot[0]), (1, remote, tpot[1])):
        for p, v in enumerate(vals, 1):
            rows.append({"model": "m", "configuration_id": "llama_cpp_bf16", "workload": "decode", "concurrency": 1,
                         "variant": f"mem{node}_p{p}", "total_tok_s_mean": v, "output_tok_s_mean": v * 0.8,
                         "tpot_s_p50_mean": t, "ttft_s_p50_mean": 0.5 * (1 + node)})
    return pd.DataFrame(rows)


def test_variant_parsing():
    assert membind.parse_variant("mem1_p2") == (1, 2)
    with pytest.raises(ValueError):
        membind.parse_variant("fa-on_ub512")


def test_pairs_are_formed_within_a_pass():
    pr = membind.paired_ratios(points(local=(30.0, 20.0), remote=(15.0, 10.0)))
    assert sorted(pr["throughput"].round(3)) == [0.5, 0.5]   # not 0.6: each pass is paired with its own reference
    assert (pr["tpot"] < 1).all() and (pr["ttft"] == 0.5).all()


def test_cell_summary_reports_ratio_range_and_repeatability():
    s = membind.cell_summary(points()).iloc[0]
    assert s["passes"] == 2 and s["throughput_ratio"] == pytest.approx(7.945 / 28.35, rel=1e-3)
    assert s["throughput_ratio_min"] <= s["throughput_ratio"] <= s["throughput_ratio_max"]
    assert s["pass_cv_reference"] < 0.01


def test_unpaired_passes_are_dropped():
    p = points()
    p = p[p["variant"] != "mem1_p2"]
    assert len(membind.paired_ratios(p)) == 1


def test_implied_bandwidth_tracks_the_remote_roof():
    # 15 GB of weights per token at 156 ms is 96 GB/s
    assert membind.implied_bandwidth_gbs(15.0e9, 0.156) == pytest.approx(96.2, rel=1e-2)


def test_remote_roof_interpolates_in_log_working_set():
    stream = pd.DataFrame({"campaign": "calibration-membind1", "threads": 55,
                           "working_set_gib": [4, 16, 64], "stream_triad_gbs": [104.0, 102.0, 98.0]})
    assert membind.remote_roof_gbs(stream, 8) == pytest.approx(103.0, abs=0.1)   # geometric midpoint of 4 and 16
    assert membind.remote_roof_gbs(stream, 16) == pytest.approx(102.0)
    with pytest.raises(ValueError):
        membind.remote_roof_gbs(stream, 8, campaign="calibration")


def test_placement_table(tmp_path):
    d = tmp_path / "membind" / "m" / "vllm-bf16__mem1_p1" / "decode" / "c001"
    d.mkdir(parents=True)
    (d / "numastat.txt").write_text("                Node 0   Node 1  Total\nTotal   1000.00   100.00 1100.00\n")
    t = membind.placement_table(tmp_path)
    assert t.iloc[0]["share_on_requested_node"] == pytest.approx(100 / 1100)
    assert not t.iloc[0]["valid"]
