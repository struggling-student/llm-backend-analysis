import math
import numpy as np
import pytest

from llm_backend_analysis.data import loaders, schema
from llm_backend_analysis.data.processing import online


@pytest.mark.parametrize("name", list(schema.TABLES))
def test_processed_table_contract(name):
    assert schema.check_table(name, loaders.load_table(name)) == []


def test_primary_campaign_is_complete():
    pts = loaders.online_points()
    assert (pts["failed_requests"] == 0).all()
    assert pts["tokens_ok"].all() and pts["manifest_consistent"].all()
    assert (pts["n_reps"] == 3).all()


def test_littles_law_estimate():
    # 4 requests in flight, 10 s each, 512 + 512 tokens per request -> 4 / 10 x 1024 tok/s
    assert online.littles_law_total_tok_s(4, 10.0, 512, 512) == pytest.approx(409.6)
    assert np.isnan(online.littles_law_total_tok_s(4, 0.0, 512, 512))
    assert np.isnan(online.littles_law_total_tok_s(4, float("nan"), 512, 512))


def test_littles_law_columns_follow_their_definition():
    reps = loaders.online_repetitions()
    expected = reps["concurrency"] / reps["e2e_s_mean"] * (reps["isl"] + reps["osl"])
    assert np.allclose(reps["littles_law_total_tok_s"], expected, rtol=1e-9)
    assert np.allclose(reps["steady_over_littles_law"], reps["total_tok_s"] / expected, rtol=1e-9)
    pts = loaders.online_points()
    mean = reps.groupby(["model", "arm_dir", "workload", "concurrency"])["steady_over_littles_law"].mean()
    got = pts.set_index(["model", "arm_dir", "workload", "concurrency"])["steady_over_littles_law_mean"]
    assert np.allclose(got.sort_index(), mean.reindex(got.index).sort_index(), rtol=1e-9)


def test_every_tuned_configuration_has_a_settings_match_in_the_primary_campaign():
    from llm_backend_analysis.analysis import tuning
    reps = loaders.online_repetitions()
    check = tuning.cross_node_check(loaders.tuning_runs(), reps)
    tuned = set(loaders.tuning_runs()["configuration_id"])
    assert set(check["configuration_id"]) == tuned
    assert reps["settings"].map(len).gt(2).all()  # every repetition records its server settings


def test_node_power_spans_the_repetitions_and_needs_energy_readings():
    from llm_backend_analysis.data.processing.online import node_power
    phases = [{"phase": "idle", "t_start": 0, "t_end": 10, "energy_j_start": 0, "energy_j_end": 0},
              {"phase": "rep1", "t_start": 100, "t_end": 400, "energy_j_start": 1000, "energy_j_end": 181000},
              {"phase": "rep2", "t_start": 403, "t_end": 700, "energy_j_start": 181000, "energy_j_end": 361000}]
    assert node_power(phases) == (600.0, 600)
    assert all(map(math.isnan, node_power([{"phase": "rep1", "t_start": 0, "t_end": 1}])))


def test_meter_means_keep_only_readings_inside_the_window():
    from llm_backend_analysis.data.processing.power import meter_means
    rapl = [{"t_mono": t, "interval_s": 1.0, "pkg_w": {"0": w, "1": 100.0}, "ram_w": {"0": 20.0, "1": 19.0},
             "unc_mhz": {"0": 1600.0}, "pkg_tmp_c": {"0": 50.0}} for t, w in ((10.0, 200.0), (11.0, 300.0), (12.0, 999.0))]
    ipmi = [{"t_mono_start": 9.0, "t_mono_end": 11.0, "watts": {"Sys Power": 500.0}},
            {"t_mono_start": 9.5, "t_mono_end": 13.0, "watts": {"Sys Power": 900.0}}]
    m = meter_means(rapl, ipmi, 9.0, 11.0)
    assert m["rapl_pkg0_w"] == 250.0 and m["rapl_n"] == 2 and m["ipmi_sys_w"] == 500.0 and m["ipmi_n"] == 1
    assert m["ipmi_cpu_w"] is None
