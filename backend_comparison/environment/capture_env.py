#!/usr/bin/env python3
"""Record the software/hardware environment of a compute node for reproducibility.

Run on the compute node at the start of every job (scripts/*.sbatch do this).
Writes $BC_DATA/environment/<host>-<jobid>.json. Expensive facts (SIF and GGUF
SHA-256) are cached in $BC_DATA/environment/hashes.json and reused.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bclib  # noqa: E402


def cached_sha(path: str, cache: dict[str, dict]) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    st = p.stat()
    key = f"{p}:{st.st_size}:{int(st.st_mtime)}"
    if key not in cache:
        cache[key] = {"path": str(p), "sha256": bclib.sha256_file(p)}
    return cache[key]["sha256"]


def cmake_flags(build_dir: Path) -> dict[str, str]:
    out = {}
    cache = build_dir / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text().splitlines():
            m = re.match(r"^(GGML_[A-Z0-9_]+|CMAKE_BUILD_TYPE|CMAKE_C_COMPILER|CMAKE_CXX_FLAGS|LLAMA_[A-Z_]+):[A-Z]+=(.*)$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def main() -> int:
    cfg = bclib.load_config()
    data = Path(os.environ["BC_DATA"])
    env_dir = data / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    hashes_path = env_dir / "hashes.json"
    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}

    sh = bclib.run_capture
    lc = cfg["backend"]["llamacpp"]
    vl = cfg["backend"]["vllm"]
    m: dict = {"timestamp": bclib.now_iso(), **bclib.host_facts()}
    m["lscpu"] = sh("lscpu")
    m["numactl_H"] = sh("numactl -H")
    m["memory_mode_detected"] = ("flat" if re.search(r"node \d+ cpus:\s*$", m["numactl_H"], re.M) else "cache")
    m["meminfo"] = sh("head -5 /proc/meminfo")
    m["cmdline"] = sh("cat /proc/cmdline")
    m["amx_flags"] = sorted(set(re.findall(r"\bamx_\w+", Path("/proc/cpuinfo").read_text())))
    m["avx512_flags"] = sorted(set(re.findall(r"\bavx512\w*", Path("/proc/cpuinfo").read_text())))
    m["thp"] = sh("cat /sys/kernel/mm/transparent_hugepage/enabled")
    m["numa_balancing"] = sh("cat /proc/sys/kernel/numa_balancing")
    m["cpu_governor"] = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    m["uncore_pmus"] = sorted({re.sub(r"_\d+$", "", d) for d in os.listdir("/sys/bus/event_source/devices")})

    src = Path(lc["src_dir"])
    m["llamacpp"] = {
        "src_commit": sh(f"git -C {src} rev-parse HEAD"),
        "src_commit_date": sh(f"git -C {src} log -1 --format=%cI"),
        "builds": {},
    }
    for name in ("amx", "avx512"):
        bin_dir = Path(lc[f"bin_dir_{name}"])
        build_dir = Path(os.environ["BC_LLAMACPP"]) / f"build-{name}"
        lib = next(iter(sorted(build_dir.glob("bin/libggml-cpu.so*"))), None)
        m["llamacpp"]["builds"][name] = {
            "bin_dir": str(bin_dir),
            "version": sh(f"{bin_dir}/llama-server --version"),
            "cmake": cmake_flags(build_dir),
            "amx_instructions_libggml_cpu": sh(f"objdump -d {lib} | grep -cE 'tileloadd|tdpbf16ps|tdpbssd|ldtilecfg|tilestored'") if lib else None,
            "avx512bf16_instructions_libggml_cpu": sh(f"objdump -d {lib} | grep -c vdpbf16ps") if lib else None,
        }
    m["vllm"] = {
        "image": vl["image"],
        "image_sha256": cached_sha(vl["image"], hashes),
        "versions": sh(f"apptainer exec --cleanenv {vl['image']} python3 -c "
                       "\"import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__); "
                       "print(*[l.strip() for l in torch.__config__.show().splitlines() if 'oneDNN' in l or 'MKL' in l or 'CXX_FLAGS' in l], sep=chr(10))\"",
                       timeout=300),
    }
    m["apptainer"] = sh("apptainer --version")
    m["models"] = {}
    for key, mod in cfg["models"].items():
        ggufs = {v: str(Path(mod["gguf_dir"]) / f) for v, f in mod["gguf"].items()}
        m["models"][key] = {
            "hf_id": mod["hf_id"], "revision": mod["revision"], "snapshot": mod["snapshot"],
            "snapshot_exists": Path(mod["snapshot"]).exists(),
            "config": json.loads((Path(mod["snapshot"]) / "config.json").read_text()) if Path(mod["snapshot"]).exists() else None,
            "gguf": {v: {"path": p, "sha256": cached_sha(p, hashes), "bytes": Path(p).stat().st_size if Path(p).exists() else None}
                     for v, p in ggufs.items()},
        }
    bclib.write_json(hashes_path, hashes)
    out = env_dir / f"{socket.gethostname().split('.')[0]}-{os.environ.get('SLURM_JOB_ID', 'nojob')}.json"
    bclib.write_json(out, m)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
