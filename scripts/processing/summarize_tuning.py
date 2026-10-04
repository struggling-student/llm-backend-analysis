#!/usr/bin/env python3
"""Print the tuning grid and apply the declared selection rule (straight from data/raw/).

  uv run python scripts/processing/summarize_tuning.py [--campaign tuning]

Use it after syncing a tuning job (e.g. `submit.sh int8-tuning`), then copy the chosen
setting into configs/experiment.toml [tuned.<arm>] before launching the campaign.
The rule itself is llm_backend_analysis.analysis.tuning.select_variants.
"""

from __future__ import annotations

import argparse

import pandas as pd

from llm_backend_analysis.analysis.tuning import select_variants
from llm_backend_analysis.data.processing.validation_runs import process_tuning


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", default="tuning")
    args = p.parse_args()
    runs = process_tuning()
    runs = runs[runs["campaign"] == args.campaign]
    if runs.empty:
        print(f"no tuning runs in campaign {args.campaign!r}")
        return 1
    pd.set_option("display.width", 200)
    cols = ["model", "configuration_id", "variant", "concurrency", "status", "total_tok_s", "tpot_s_p50", "host"]
    print(runs[cols].to_string(index=False))
    print()
    print(select_variants(runs).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
