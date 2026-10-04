#!/usr/bin/env python3
"""Execute notebooks top to bottom in a fresh kernel, in place (the `make notebooks` step).

  uv run python scripts/utilities/execute_notebooks.py notebooks/backend_selection/*.ipynb

Each notebook runs with its own directory as working directory, as Jupyter would. Exports
(results/figures, results/tables) are rewritten by the notebooks themselves. Execution
metadata (timestamps) is stripped so unchanged results leave git clean.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def strip_execution_metadata(nb: nbformat.NotebookNode) -> None:
    for cell in nb.cells:
        cell.metadata.pop("execution", None)


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 1
    failed = []
    for p in map(Path, paths):
        t0 = time.time()
        nb = nbformat.read(p, as_version=4)
        try:
            ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(p.parent)}})
        except Exception as exc:  # noqa: BLE001 - report and continue with the other notebooks
            failed.append(p)
            print(f"FAILED  {p}: {type(exc).__name__}: {str(exc).splitlines()[-1] if str(exc) else ''}")
            continue
        strip_execution_metadata(nb)
        nbformat.write(nb, p)
        print(f"ok      {p}  ({time.time() - t0:.0f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
