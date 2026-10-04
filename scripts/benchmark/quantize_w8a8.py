#!/usr/bin/env python3
"""Build vLLM's int8 counterpart of llama.cpp's Q8_0 from the same BF16 snapshot.

Scheme (compressed-tensors preset "W8A8"):
  weights      int8, symmetric, one scale per output channel, round-to-nearest (no calibration data)
  activations  int8, symmetric, one scale per token, computed at run time (dynamic)
  not quantized: lm_head and the token embedding (as in vLLM's usual W8A8 checkpoints)

Why this is the closest match to Q8_0: Q8_0 also stores int8 weights with
symmetric round-to-nearest scales, and llama.cpp quantizes activations to int8
on the fly (Q8_0/Q8_K) for its int8 dot products and AMX-INT8 tiles. The
remaining differences are the scale granularity (Q8_0: one scale per 32
weights, and llama.cpp also quantizes token_embd and output) and the kernels.
Both are recorded in the manifest.

Run with the llm-compressor venv on a compute node (any AMX node; no GPU):
  BC_JOB_PYTHON=$BC_DATA/venvs/llmcompressor/bin/python scripts/cluster/submit.sh job ... scripts/benchmark/quantize_w8a8.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402


def tensor_bytes(directory: Path) -> dict[str, int]:
    """Per-tensor byte sizes from the safetensors headers (no tensor data is read)."""
    sizes: dict[str, int] = {}
    for f in sorted(directory.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                a, b = meta["data_offsets"]
                sizes[name] = b - a
    return sizes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="llama31_8b,llama32_1b")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    args = p.parse_args()

    import torch
    import compressed_tensors
    import llmcompressor
    import transformers
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    cfg = bclib.load_config()
    for key in args.models.split(","):
        m = cfg["models"][key]
        out = Path(m["w8a8_dir"])
        if (out / "manifest.json").exists():
            bclib.log(f"{key}: {out} exists, skipping")
            continue
        partial = out.with_name(out.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        t0 = time.time()
        bclib.log(f"{key}: loading {m['snapshot']}")
        model = AutoModelForCausalLM.from_pretrained(m["snapshot"], dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(m["snapshot"])
        recipe = QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
        oneshot(model=model, recipe=recipe)          # no dataset: weights RTN, activations dynamic
        model.save_pretrained(partial, save_compressed=True)
        tok.save_pretrained(partial)
        for extra in ("generation_config.json", "LICENSE", "USE_POLICY.md"):
            src = Path(m["snapshot"]) / extra
            if src.exists() and not (partial / extra).exists():
                shutil.copy(src, partial / extra)
        qcfg = json.loads((partial / "config.json").read_text()).get("quantization_config")
        sizes = tensor_bytes(partial)
        embed = sum(v for k, v in sizes.items() if "embed_tokens" in k)
        lm_head = sum(v for k, v in sizes.items() if k.startswith("lm_head"))
        files = {}
        for f in sorted(partial.iterdir()):
            if f.is_file():
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                files[f.name] = {"bytes": f.stat().st_size, "sha256": h}
        manifest = {
            "timestamp": bclib.now_iso(), "model": key, "hf_id": m["hf_id"], "revision": m["revision"],
            "source_snapshot": m["snapshot"], "seconds": time.time() - t0,
            "recipe": {"modifier": "QuantizationModifier", "targets": "Linear", "scheme": "W8A8",
                       "ignore": ["lm_head"], "calibration_data": None},
            "quantization_config": qcfg,
            "versions": {"llmcompressor": llmcompressor.__version__, "compressed_tensors": compressed_tensors.__version__,
                         "transformers": transformers.__version__, "torch": torch.__version__},
            "tensor_bytes_total": sum(sizes.values()),
            "tensor_bytes_embed_tokens": embed,
            "tensor_bytes_lm_head": lm_head,
            # what a forward step streams: everything except the input-embedding table
            # (only C rows of it are gathered per step); tied embeddings are read via lm_head
            "step_weight_bytes": sum(sizes.values()) - (embed if lm_head else 0),
            "files": files,
        }
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(partial, out)
        bclib.log(f"{key}: wrote {out} ({manifest['tensor_bytes_total'] / 1e9:.2f} GB tensors, "
                  f"{manifest['seconds']:.0f}s)")
        del model
    return 0


if __name__ == "__main__":
    sys.exit(main())
