import pytest

from llm_backend_analysis.data import loaders, schema


@pytest.mark.parametrize("name", list(schema.TABLES))
def test_processed_table_contract(name):
    assert schema.check_table(name, loaders.load_table(name)) == []


def test_primary_campaign_is_complete():
    pts = loaders.online_points()
    assert (pts["failed_requests"] == 0).all()
    assert pts["tokens_ok"].all() and pts["manifest_consistent"].all()
    assert (pts["n_reps"] == 3).all()
