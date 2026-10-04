import re

from llm_backend_analysis import paths
from llm_backend_analysis.data import registry


def test_every_harness_arm_has_a_configuration():
    arms = set(registry.experiment_config()["arms"])
    assert arms == {c.arm for c in registry.configurations().values()}


def test_configuration_naming_is_canonical():
    for cid, c in registry.configurations().items():
        assert c.backend in {"vllm", "llama_cpp"}
        assert c.precision in {"bf16", "int8"}
        assert cid == f"{c.backend}_{c.precision}"
        assert re.fullmatch(r"[a-z0-9_]+", c.quantization)


def test_every_raw_arm_directory_maps_to_a_configuration():
    for d in (paths.RAW / "online").glob("*/*/*"):
        if d.is_dir():
            arm, _ = registry.split_arm_dir(d.name)
            registry.configuration_for_arm(arm)


def test_every_raw_campaign_is_declared():
    declared = {(r.kind, r.campaign) for r in registry.campaigns().itertuples()}
    on_disk = {(k.name, c.name) for k in paths.RAW.iterdir() if k.is_dir() and k.name not in {"telemetry", "environment"}
               for c in k.iterdir() if c.is_dir()}
    assert on_disk <= declared, on_disk - declared


def test_telemetry_path_mapping():
    p = paths.local_telemetry_path("/lpor1/scratch_1/usr/x/backend-comparison/telemetry/online/main/m/a/w/c001.jsonl")
    assert p == paths.RAW_TELEMETRY / "online/main/m/a/w/c001.jsonl"
