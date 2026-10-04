#!/usr/bin/env python3
"""Rebuild data/processed/ from data/raw/ (the `make data` step).

  uv run python scripts/processing/build_processed_data.py [--out DIR] [--check]

--check rebuilds into a temporary directory and fails if any committed processed
table differs, i.e. if data/processed/ is stale with respect to raw data and code.
"""

from __future__ import annotations

import argparse
import filecmp
import sys
import tempfile
from pathlib import Path

import pandas as pd

from llm_backend_analysis import paths
from llm_backend_analysis.data.processing import OUTPUTS, build_tables, write_tables


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=paths.PROCESSED)
    p.add_argument("--check", action="store_true", help="compare a fresh build with data/processed/")
    args = p.parse_args()
    tables = build_tables()
    if not args.check:
        for name, path in write_tables(tables, args.out).items():
            print(f"{len(tables[name]):>7} rows  {paths.relative(path)}")
        return 0
    stale = []
    with tempfile.TemporaryDirectory() as tmp:
        write_tables(tables, Path(tmp))
        for filename in OUTPUTS.values():
            fresh, committed = Path(tmp) / filename, paths.PROCESSED / filename
            if not committed.exists():
                stale.append(f"{filename} (missing)")
            elif filename.endswith(".parquet"):
                if not pd.read_parquet(fresh).equals(pd.read_parquet(committed)):
                    stale.append(filename)
            elif not filecmp.cmp(fresh, committed, shallow=False):
                stale.append(filename)
    for s in stale:
        print(f"STALE  data/processed/{s}")
    print("processed data up to date" if not stale else "run `make data` to regenerate")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
