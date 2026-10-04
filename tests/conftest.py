import pandas as pd
import pytest


@pytest.fixture
def synthetic_points() -> pd.DataFrame:
    """Small hand-made point table (not measurements) exercising the metric logic, including a
    vLLM INT8 configuration as it will appear once its campaign is processed."""
    rows = []
    curves = {
        ("vllm_bf16", "vllm", "bf16"): [10, 20, 40, 70, 100],
        ("llama_cpp_bf16", "llama_cpp", "bf16"): [12, 18, 22, 23, 23],
        ("llama_cpp_int8", "llama_cpp", "int8"): [18, 26, 31, 32, 32],
        ("vllm_int8", "vllm", "int8"): [16, 32, 60, 100, 130],
    }
    for (cid, backend, precision), tput in curves.items():
        for c, t in zip([1, 2, 4, 8, 16], tput):
            rows.append({"campaign": "main", "model": "m", "configuration_id": cid, "arm_dir": cid, "backend": backend,
                         "precision": precision, "quantization": "none", "workload": "balanced", "isl": 512, "osl": 512,
                         "concurrency": c, "total_tok_s_mean": float(t), "total_tok_s_std": 0.01 * t,
                         "output_tok_s_mean": t / 2, "tpot_s_p50_mean": 0.05 * (1 + c / 8), "ttft_s_p50_mean": 0.5,
                         "mem_total_gbs_mean": 300.0 / (1 + c / 4), "bw_util_vs_stream_max_mean": 0.6 / (1 + c / 4)})
    return pd.DataFrame(rows)
