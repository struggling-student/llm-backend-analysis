#!/usr/bin/env python3
"""Check the repository against its reproducibility contract (the `make validate` step).

  uv run python scripts/utilities/validate_repository.py [--skip-rebuild]

Checks, in order: analysis environment; raw-data integrity (checksum manifest); campaign
and configuration metadata cover every raw directory; processed tables are up to date and
satisfy their schema; notebooks are executed, error-free and free of machine-specific paths;
exported figures match their manifest and no standalone HTML plots remain; relative links
in Markdown files resolve. Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def run(*a, **kw):
            try:
                ok, detail = fn(*a, **kw)
            except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            RESULTS.append((name, ok, detail))
            return ok
        return run
    return deco


@check("environment")
def environment():
    mods = ["pandas", "pyarrow", "numpy", "scipy", "plotly", "kaleido", "nbformat", "nbconvert", "llm_backend_analysis"]
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"missing modules: {missing} (run `make setup`)"
    import plotly
    return True, f"python {sys.version.split()[0]}, plotly {plotly.__version__}"


@check("raw data integrity")
def raw_integrity():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/utilities/raw_checksums.py"), "verify"],
                         capture_output=True, text=True)
    return out.returncode == 0, out.stdout.strip().splitlines()[-1] if out.stdout else out.stderr.strip()


@check("campaign and configuration metadata")
def metadata():
    from llm_backend_analysis import paths
    from llm_backend_analysis.data import registry
    declared = {(r.kind, r.campaign) for r in registry.campaigns().itertuples()}
    on_disk = {(k.name, c.name) for k in paths.RAW.iterdir() if k.is_dir() and k.name not in {"telemetry", "environment"}
               for c in k.iterdir() if c.is_dir()}
    problems = [f"undeclared campaign {k}/{c}" for k, c in sorted(on_disk - declared)]
    problems += [f"declared campaign {k}/{c} has no data" for k, c in sorted(declared - on_disk)]
    arms = {registry.split_arm_dir(d.name)[0] for d in (paths.RAW / "online").glob("*/*/*") if d.is_dir()}
    arms |= {d.name for d in (paths.RAW / "offline").glob("*/*/*") if d.is_dir()}
    known = {c.arm for c in registry.configurations().values()}
    problems += [f"arm {a} has no configuration" for a in sorted(arms - known)]
    problems += [f"harness arm {a} missing from configurations.toml"
                 for a in sorted(set(registry.experiment_config()["arms"]) - known)]
    return not problems, "; ".join(problems) or f"{len(declared)} campaigns, {len(known)} configurations"


@check("processed data up to date")
def processed_fresh(skip: bool):
    if skip:
        return True, "skipped (--skip-rebuild)"
    out = subprocess.run([sys.executable, str(ROOT / "scripts/processing/build_processed_data.py"), "--check"],
                         capture_output=True, text=True)
    return out.returncode == 0, out.stdout.strip().replace("\n", "; ")


@check("processed schema")
def processed_schema():
    from llm_backend_analysis.data import loaders, schema
    problems = [p for name in schema.TABLES for p in schema.check_table(name, loaders.load_table(name))]
    return not problems, "; ".join(problems) or f"{len(schema.TABLES)} tables"


ABS_PATH = re.compile(r"(/Users/|/home/|/var/folders/|/private/tmp/|[A-Z]:\\\\Users)")


@check("notebooks executed and portable")
def notebooks():
    problems, n = [], 0
    for nb_path in sorted((ROOT / "notebooks").glob("*/[0-9][0-9]_*.ipynb")):
        n += 1
        nb = json.loads(nb_path.read_text())
        rel = nb_path.relative_to(ROOT)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if cell.get("execution_count") is None and src.strip():
                problems.append(f"{rel} cell {i} not executed")
            if ABS_PATH.search(src):
                problems.append(f"{rel} cell {i} source contains an absolute path")
            for o in cell.get("outputs", []):
                if o["output_type"] == "error":
                    problems.append(f"{rel} cell {i} has an error output")
                if o["output_type"] == "stream" and o.get("name") == "stderr":
                    problems.append(f"{rel} cell {i} writes to stderr (warnings)")
                if ABS_PATH.search(json.dumps(o)):
                    problems.append(f"{rel} cell {i} output contains a machine-specific path")
        counts = [c["execution_count"] for c in nb["cells"] if c["cell_type"] == "code" and c.get("execution_count")]
        if counts != sorted(counts):
            problems.append(f"{rel} cells were not executed top to bottom")
    return not problems and n > 0, "; ".join(problems[:8]) or f"{n} notebooks"


@check("figure exports")
def figures():
    problems = []
    for manifest in (ROOT / "results/figures").glob("*/figures.json"):
        entries = json.loads(manifest.read_text())
        listed = {f for e in entries.values() for f in e["files"]}
        on_disk = {p.name for p in manifest.parent.iterdir() if p.name != "figures.json"}
        problems += [f"{manifest.parent.name}/{f} listed but missing" for f in sorted(listed - on_disk)]
        problems += [f"{manifest.parent.name}/{f} not produced by any notebook (stale?)" for f in sorted(on_disk - listed)]
        for name, e in entries.items():
            if not (ROOT / e["source_notebook"]).exists():
                problems.append(f"{name}: source notebook {e['source_notebook']} missing")
    html = [str(p.relative_to(ROOT)) for d in ("results", "notebooks", "data", "docs") for p in (ROOT / d).rglob("*.html")]
    problems += [f"standalone HTML artifact {h}" for h in html]
    return not problems, "; ".join(problems[:8]) or "manifest consistent, no HTML artifacts"


LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")


@check("markdown links")
def links():
    tracked = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    md_files = sorted(set(tracked) | {str(p.relative_to(ROOT)) for p in ROOT.glob("*.md")}
                      | {str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*.md")}
                      | {str(p.relative_to(ROOT)) for d in ("data", "notebooks", "results", "scripts")
                         for p in (ROOT / d).glob("README.md")})
    broken = []
    for rel in md_files:
        md = ROOT / rel
        for target in LINK.findall(md.read_text()):
            if re.match(r"^[a-z]+://|^mailto:", target):
                continue
            if not (md.parent / target).exists():
                broken.append(f"{rel} -> {target}")
    return not broken, "; ".join(broken[:8]) or f"{len(md_files)} Markdown files"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-rebuild", action="store_true", help="do not rebuild processed data to compare")
    args = p.parse_args()
    if environment():
        raw_integrity()
        metadata()
        processed_fresh(args.skip_rebuild)
        processed_schema()
        notebooks()
        figures()
    links()
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    return 0 if all(ok for _, ok, _ in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
