"""Shared helpers for the backend-comparison scripts (stdlib only, Python >= 3.11)."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent  # scripts/benchmark
ROOT = SCRIPTS.parents[1]
CONFIG = ROOT / "configs" / "experiment.toml"

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand(value: Any) -> Any:
    """Expand ${VAR} in every string of a nested config structure."""
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise KeyError(f"environment variable {name} is not set (source configs/cresco8.env)")
            return os.environ[name]
        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def load_config(path: Path | str = CONFIG) -> dict[str, Any]:
    with open(path, "rb") as f:
        return expand(tomllib.load(f))


def backend_settings(cfg: dict[str, Any], arm: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Defaults from [backend.<backend>], then the tuning choices [tuned.<backend>]
    and [tuned.<arm>], then CLI overrides (tuning variants, smoke tests)."""
    backend = cfg["arms"][arm]["backend"]
    out = dict(cfg["backend"][backend])
    tuned = cfg.get("tuned", {})
    out.update({k: v for k, v in tuned.get(backend, {}).items() if not isinstance(v, dict)})
    out.update(tuned.get(arm, {}))
    out.update(overrides or {})
    return out


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """key=value pairs; values parsed as JSON when possible (ints, lists, booleans)."""
    out: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def mono() -> float:
    """CLOCK_MONOTONIC seconds; shared by every process on the node, so client
    and telemetry timestamps are directly comparable."""
    return time.monotonic()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = limit_bytes
        while True:
            chunk = f.read(1 << 24 if remaining is None else min(1 << 24, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return h.hexdigest()


def run_capture(cmd: list[str] | str, timeout: float = 120) -> str:
    try:
        res = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return (res.stdout + res.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return f"<failed: {exc}>"


def host_facts() -> dict[str, Any]:
    """Cheap per-run facts; the full manifest is scripts/benchmark/capture_env.py."""
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "kernel": os.uname().release,
        "python": sys.version.split()[0],
        "numa_nodes": run_capture("numactl -H | head -1"),
        "perf_event_paranoid": Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip(),
        "cpu_model": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                           if l.startswith("model name")), None),
    }


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_model(cfg: dict[str, Any], model_key: str, arm_key: str) -> dict[str, Any]:
    """Copy the files a run needs to node-local disk and point cfg at the copies.

    Five jobs loading the same 16 GB from the shared Lustre HDD pool at once made
    vLLM's weight loading stall for >25 min. Loading time is not measured, but it
    must not fail runs, so every job reads its model once from Lustre into
    $BC_LOCAL (node-local NVMe) and all server restarts load from there. The copy
    runs under numactl --membind=1 so its page cache stays on socket 1, away from
    the socket-0 memory the servers use. Weights end up in each server's own
    memory on node 0 (checked per point with numastat).
    """
    local = Path(os.environ.get("BC_LOCAL") or f"/tmp/bc-{os.environ.get('SLURM_JOB_ID', 'nojob')}")
    model, arm = cfg["models"][model_key], cfg["arms"][arm_key]
    info: dict[str, Any] = {"local_root": str(local)}
    if arm["backend"] == "vllm":
        src = Path(model[arm.get("checkpoint", "snapshot")])
        dst = local / model_key / ("hf" if "checkpoint" not in arm else arm["checkpoint"])
        if not (dst / ".complete").exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.name == "original":          # Meta's consolidated .pth duplicate, not used
                    continue
                subprocess.run(["numactl", "--membind=1", "cp", "-L", str(f), str(dst / f.name)], check=True)
            (dst / ".complete").touch()
        info.update({"source": str(src), "staged": str(dst)})
        model["snapshot"] = str(dst)
    else:
        name = model["gguf"][arm["gguf_variant"]]
        src = Path(model["gguf_dir"]) / name
        dst = local / model_key / name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["numactl", "--membind=1", "cp", str(src), str(dst) + ".partial"], check=True)
            os.replace(str(dst) + ".partial", dst)
        info.update({"source": str(src), "staged": str(dst)})
        model["gguf_dir"] = str(local / model_key)
    log(f"model staged to {info['staged']}")
    return info
