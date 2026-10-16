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
