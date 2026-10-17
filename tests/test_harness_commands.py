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
import amx_ablation  # noqa: E402,F401


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


# --------------------------------------------------------------------------- AMX ablations (T3.1, T3.2)
def test_amx_levels_cap_onednn_and_the_all_off_level_adds_the_torch_shim(cfg):
    import amx_ablation
    runs = {r["level"]: r["cmd"] for r in amx_ablation.vllm_plan(cfg, None, ["balanced"])}
    envs = {lv: json.loads(next(c for c in cmd if c.startswith("container_env=")).split("=", 1)[1]) if any(c.startswith("container_env=") for c in cmd) else {}
            for lv, cmd in runs.items()}
    assert envs["amx"] == {}
    assert envs["noamx_gemm"] == {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_BF16"}
    assert envs["noamx_all"]["BC_FORCE_NO_AMX"] == "1" and envs["noamx_all"]["ONEDNN_MAX_CPU_ISA"] == "AVX512_CORE_BF16"
    assert envs["noamx_all"]["PYTHONPATH"].endswith("shim/no_amx")
    assert all("--tag" in c and "--campaign" in c and "amx-ablation" in c for c in runs.values())


def test_container_env_reaches_the_vllm_server_command(cfg):
    settings = bclib.backend_settings(cfg, "vllm-bf16", {"container_env": {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_BF16"}})
    cmd = run_online.server_command(cfg, "llama31_8b", "vllm-bf16", cfg["workloads"]["balanced"], 1, settings, 8000)[0]
    assert "ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16" in cmd
    base = run_online.server_command(cfg, "llama31_8b", "vllm-bf16", cfg["workloads"]["balanced"], 1,
                                     bclib.backend_settings(cfg, "vllm-bf16"), 8000)[0]
    assert not any(t.startswith("ONEDNN_") for t in base)


def test_llamacpp_ablation_is_abab_over_both_builds(cfg):
    import amx_ablation
    runs = amx_ablation.llamacpp_plan(cfg)
    assert [(r["build"], r["pass"]) for r in runs] == [("amx", 1), ("avx512", 1), ("amx", 2), ("avx512", 2)]
    for r in runs:
        c = r["cmd"]
        assert c[c.index("--build") + 1] == r["build"] and c[c.index("--bench-tests") + 1] == "pp"
        assert c[c.index("--tag") + 1] == r["tag"] and c[c.index("--arm") + 1] == "llamacpp-q8_0"


def test_onednn_verbose_histogram_and_amx_busy(tmp_path):
    import amx_ablation
    log = tmp_path / "server.log"
    log.write_text("x\nonednn_verbose,v1,primitive,create:cache_miss,cpu,matmul,brg:avx10_1_512_amx,forward_inference,a,b,1.0\n"
                   "onednn_verbose,v1,primitive,create:cache_miss,cpu,matmul,brg:avx512_core_bf16,forward_inference,a,b,1.0\n")
    hist = amx_ablation.onednn_histogram(log)
    assert hist == {"matmul brg:avx10_1_512_amx": 1, "matmul brg:avx512_core_bf16": 1}
    tel = tmp_path / "t.jsonl"
    tel.write_text("\n".join(json.dumps({"type": "sample", "t_mono": t, "pmu": {"exe_amx_busy": 10, "cycles": 1000}}) for t in (1, 2, 3, 50)))
    assert amx_ablation.amx_busy(tel, 0, 10) == pytest.approx(0.01)
