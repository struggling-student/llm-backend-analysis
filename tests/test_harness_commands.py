"""Command lines of the benchmark harness: the memory-node override changes only `numactl --membind` of the
server, and the recorded commands of the primary campaign are still produced when no override is given."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
import bclib  # noqa: E402
import membind_sensitivity  # noqa: E402
import run_online  # noqa: E402


@pytest.fixture
def cfg(monkeypatch):
    for k, v in {"BC_STORE": "/store", "BC_SCRATCH": "/scratch", "BC_DATA": "/data", "BC_HF_CACHE": "/hf", "BC_IMAGES": "/img",
                 "BC_LLAMACPP": "/llama", "BC_STREAM_BIN": "/stream", "BC_STREAM_RESULTS": "/sr", "BC_PYTHON": "/py",
                 "HOME": "/home", "BC_LOCAL": "/local"}.items():
        monkeypatch.setenv(k, v)
    return bclib.load_config()


def command(cfg, model, arm, wl, c, mem_node=None):
    if mem_node is not None:
        cfg["topology"]["server_mem_node"] = mem_node
    settings = bclib.backend_settings(cfg, arm)
    port = cfg["online"]["port_vllm" if cfg["arms"][arm]["backend"] == "vllm" else "port_llamacpp"]
    return run_online.server_command(cfg, model, arm, cfg["workloads"][wl], c, settings, port)[0]


@pytest.mark.parametrize("arm", ["vllm-bf16", "llamacpp-bf16"])
def test_mem_node_changes_only_membind(cfg, arm):
    local = command(cfg, "llama31_8b", arm, "decode", 1, mem_node=0)
    remote = command(cfg, "llama31_8b", arm, "decode", 1, mem_node=1)
    assert "--membind=0" in local and "--membind=1" in remote
    def strip_shim(toks):  # the NUMA shim (vLLM only) is the one other allowed difference
        out = []
        for i, t in enumerate(toks):
            if t.startswith("LD_PRELOAD=") or (t == "--env" and toks[i + 1].startswith("LD_PRELOAD=")):
                continue
            out.append(t)
        return out
    assert [t for t in local if t != "--membind=0"] == [t for t in strip_shim(remote) if t != "--membind=1"]


@pytest.mark.parametrize("arm,directory", [("vllm-bf16", "vllm-bf16"), ("llamacpp-bf16", "llamacpp-bf16")])
def test_existing_cells_reproducible(cfg, arm, directory):
    """Without an override the rebuilt command matches the recorded one in every token that is not a cluster path."""
    recorded = json.loads((ROOT / "data/raw/online/main/llama31_8b" / directory / "decode/c001/metadata.json").read_text())
    rebuilt = command(cfg, "llama31_8b", arm, "decode", 1)
    keep = lambda toks: [t for t in toks if "/" not in t]  # noqa: E731
    assert keep(rebuilt) == keep(recorded["server_command"])


def test_vllm_gets_the_numa_shim_only_for_remote_memory(cfg):
    local = command(cfg, "llama31_8b", "vllm-bf16", "decode", 1, mem_node=0)
    remote = command(cfg, "llama31_8b", "vllm-bf16", "decode", 1, mem_node=1)
    shim = f"LD_PRELOAD={cfg['topology']['numa_shim']}"
    assert shim in "".join(remote) and shim not in "".join(local)
    # the image's own preload (tcmalloc, Intel OpenMP) must survive: `--env LD_PRELOAD` replaces it
    ld = next(t for t in remote if t.startswith("LD_PRELOAD="))
    assert ld.endswith(cfg["backend"]["vllm"]["image_ld_preload"]) and "libiomp5" in ld and "libtcmalloc" in ld
    # llama.cpp does not reset the memory policy, so it never needs the shim
    assert not any(t.startswith("LD_PRELOAD") for t in command(cfg, "llama31_8b", "llamacpp-bf16", "decode", 1, mem_node=1))


def test_plan_is_abab_within_each_cell(cfg):
    runs = membind_sensitivity.plan(cfg, ["vllm-decode-c1"], None)
    assert [(r["node"], r["pass"]) for r in runs] == [(0, 1), (1, 1), (0, 2), (1, 2)]
    assert all(r["campaign"] == "membind-v2" and "--mem-node" in r["cmd"] for r in runs)
    assert {r["tag"] for r in runs} == {"mem0_p1", "mem1_p1", "mem0_p2", "mem1_p2"}


def test_cells_are_the_ones_of_the_tracker(cfg):
    cells = {(c["arm"], c["workload"], c["concurrency"]) for c in cfg["membind_sensitivity"]["cells"]}
    assert cells == {("vllm-bf16", "decode", 1), ("vllm-bf16", "decode", 128), ("vllm-bf16", "kvdecode", 32),
                     ("llamacpp-bf16", "decode", 1)}


def test_numastat_share(tmp_path):
    f = tmp_path / "numastat.txt"
    f.write_text("Per-node process memory usage (in MBs) for PID 1 (x)\n                           Node 0          Node 1           Total\n"
                 "Total                     1000.00         3000.00         4000.00\n")
    assert membind_sensitivity.numastat_share(f, 1) == pytest.approx(0.75)
