#!/usr/bin/env python3
"""Build the llama.cpp weight files from the same Hugging Face snapshots vLLM serves.

  snapshot (safetensors, BF16) --convert_hf_to_gguf.py --outtype bf16--> <model>-bf16.gguf   (lossless)
  <model>-bf16.gguf            --llama-quantize Q8_0-------------------> <model>-q8_0.gguf

The converter script comes from the same llama.cpp source tree (commit) as the
runtime binaries; it runs with the Python environment of the pinned llama.cpp
container image, which carries the converter's dependencies. Outputs go to a
new directory ($BC_DATA/models); earlier artifacts are never touched.
Run on a compute node (via scripts/job.sbatch).
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402


def run(cmd: list[str], log: Path, partial: Path, final: Path, attempts: int = 2) -> None:
    """Write to <final>.partial and rename only on success, so an interrupted or
    failed step can never leave a file that looks finished. Retries once
    (a transient Lustre write error was seen once while quantizing)."""
    for attempt in range(1, attempts + 1):
        partial.unlink(missing_ok=True)
        bclib.log(f"(attempt {attempt}) {shlex.join(cmd)}")
        with open(log, "a") as f:
            f.write(f"$ {shlex.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0 and partial.exists():
            os.replace(partial, final)
            return
        bclib.log(f"failed with rc={rc}")
    raise RuntimeError(f"{final.name}: all {attempts} attempts failed (see {log})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="llama32_1b,llama31_8b")
    p.add_argument("--converter-image", default=os.path.expandvars("${BC_IMAGES}/llamacpp-full-b-df03399b.sif"))
    p.add_argument("--seed-from", help="earlier data root whose models/<key>/ GGUFs are copied (SHA-256 verified) "
                                       "instead of being rebuilt")
    args = p.parse_args()
    cfg = bclib.load_config()
    lc = cfg["backend"]["llamacpp"]
    src = Path(lc["src_dir"])
    quant = Path(lc["bin_dir_amx"]) / "llama-quantize"
    for key in args.models.split(","):
        m = cfg["models"][key]
        out = Path(m["gguf_dir"])
        out.mkdir(parents=True, exist_ok=True)
        log = out / "preparation.log"
        bf16 = out / m["gguf"]["bf16"]
        q8 = out / m["gguf"]["q8_0"]
        steps = []
        if args.seed_from:
            for f in (bf16, q8):
                src_f = Path(args.seed_from) / "models" / key / f.name
                if src_f.exists() and not f.exists():
                    partial = Path(str(f) + ".partial")
                    shutil.copyfile(src_f, partial)
                    if bclib.sha256_file(src_f) != bclib.sha256_file(partial):
                        raise RuntimeError(f"copy of {src_f} does not match")
                    os.replace(partial, f)
                    steps.append({"output": str(f), "copied_from": str(src_f), "note": "SHA-256 verified copy"})
                    bclib.log(f"copied {src_f} -> {f}")
        if not bf16.exists():
            cmd = ["apptainer", "exec", "--cleanenv", "--bind", f"{os.environ['BC_STORE']}:{os.environ['BC_STORE']}", "--bind", f"{os.environ['BC_SCRATCH']}:{os.environ['BC_SCRATCH']}",
                   args.converter_image, "python3", str(src / "convert_hf_to_gguf.py"), m["snapshot"],
                   "--outtype", "bf16", "--outfile", str(bf16) + ".partial"]
            t = time.time()
            run(cmd, log, Path(str(bf16) + ".partial"), bf16)
            steps.append({"output": str(bf16), "command": cmd, "seconds": time.time() - t})
        if not q8.exists():
            # llama-quantize writing directly to this Lustre filesystem fails
            # reproducibly with "basic_ios::clear: iostream error" after ~3.4 GB
            # (8B model; the 1B model is unaffected), while the same command
            # writing to node-local /dev/shm succeeds. So quantize locally, copy,
            # and verify the copy byte-for-byte before renaming it into place.
            local = Path("/dev/shm") / (q8.name + ".partial")
            cmd = ["numactl", "--membind=0", str(quant), str(bf16), str(local), "Q8_0", "56"]
            t = time.time()
            run(cmd, log, local, Path("/dev/shm") / q8.name)
            done_local = Path("/dev/shm") / q8.name
            partial = Path(str(q8) + ".partial")
            shutil.copyfile(done_local, partial)
            if bclib.sha256_file(done_local) != bclib.sha256_file(partial):
                raise RuntimeError("copy of the quantized file does not match the local original")
            os.replace(partial, q8)
            done_local.unlink()
            steps.append({"output": str(q8), "command": cmd, "seconds": time.time() - t,
                          "note": "quantized to /dev/shm, copied to Lustre, SHA-256 verified"})
        manifest = out / "manifest.json"
        bclib.write_json(manifest, {
            "timestamp": bclib.now_iso(), "model": key, "hf_id": m["hf_id"], "revision": m["revision"],
            "snapshot": m["snapshot"], "converter": str(src / "convert_hf_to_gguf.py"),
            "converter_commit": bclib.run_capture(f"git -C {src} rev-parse HEAD"),
            "converter_python_image": args.converter_image, "quantizer": str(quant),
            "quantizer_version": bclib.run_capture(f"{quant} --version"),
            "files": {v: {"path": str(out / f), "bytes": (out / f).stat().st_size, "sha256": bclib.sha256_file(out / f)}
                      for v, f in m["gguf"].items()},
            "steps_this_run": steps,
        })
        bclib.log(f"{key}: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
