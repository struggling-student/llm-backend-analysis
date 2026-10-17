"""Makes PyTorch report that the CPU has no AMX (AMX ablation of vLLM, scripts/benchmark/amx_ablation.py).

Python imports `sitecustomize` at start-up of every interpreter whose sys.path contains this directory (PYTHONPATH,
inherited by vLLM's engine-core subprocesses). vLLM chooses its own AMX code with `torch.cpu._is_amx_tile_supported()`:
the attention kernel (`_get_attn_isa` returns "amx" or "vec") and its small-weight AMX linear kernels
(`check_cpu_sgl_kernel`). With the answer False it falls back to AVX-512 code. oneDNN's GEMMs ignore this answer and
are capped by `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` instead. Active only when BC_FORCE_NO_AMX=1.
"""

import os
import sys

if os.environ.get("BC_FORCE_NO_AMX") == "1":
    import torch

    _patched = []
    for _mod in (torch.cpu, getattr(torch._C, "_cpu", None)):
        for _name in ("_is_amx_tile_supported", "_is_amx_fp16_supported"):
            if _mod is not None and hasattr(_mod, _name):
                setattr(_mod, _name, lambda: False)
                _patched.append(f"{_mod.__name__}.{_name}")
    print(f"no_amx: torch reports no AMX ({', '.join(_patched)})", file=sys.stderr)
