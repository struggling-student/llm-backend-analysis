#!/usr/bin/env python3
"""Integrity manifest of the raw measurements (data/raw/ is append-only).

  python scripts/utilities/raw_checksums.py verify   # default; exit 1 if a recorded file changed or vanished
  python scripts/utilities/raw_checksums.py update   # record files added by a new sync

`update` only *adds* entries for new files. It refuses to overwrite the checksum of
a file that changed: raw data is never edited in place. If a changed file is a
genuine re-sync of an unfinished run, inspect it and use `update --accept-changed`.
The manifest is data/metadata/raw_checksums.sha256 (sha256sum format, sorted).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "metadata" / "raw_checksums.sha256"
IGNORED = {".DS_Store"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_files() -> list[Path]:
    return sorted(p for p in RAW.rglob("*") if p.is_file() and p.name not in IGNORED)


def read_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        entries[rel] = digest
    return entries


def write_manifest(entries: dict[str, str]) -> None:
    MANIFEST.write_text("".join(f"{entries[k]}  {k}\n" for k in sorted(entries)))


def compare() -> tuple[dict[str, str], list[str], list[str], list[str]]:
    recorded = read_manifest()
    actual = {str(p.relative_to(ROOT)): sha256(p) for p in current_files()}
    changed = sorted(k for k in recorded if k in actual and actual[k] != recorded[k])
    missing = sorted(k for k in recorded if k not in actual)
    new = sorted(k for k in actual if k not in recorded)
    return actual, changed, missing, new


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", choices=["verify", "update"], default="verify")
    p.add_argument("--accept-changed", action="store_true", help="with update: also re-record changed files")
    args = p.parse_args()
    actual, changed, missing, new = compare()
    recorded = read_manifest()
    for label, items in (("CHANGED", changed), ("MISSING", missing), ("NEW", new)):
        for k in items[:20]:
            print(f"{label:8s} {k}")
        if len(items) > 20:
            print(f"{label:8s} ... and {len(items) - 20} more")
    if args.action == "verify":
        ok = not changed and not missing
        print(f"raw data: {len(recorded)} recorded, {len(changed)} changed, {len(missing)} missing, {len(new)} new"
              + (" (run `make checksums` after a sync)" if new else ""))
        return 0 if ok else 1
    if (changed and not args.accept_changed) or missing:
        print("refusing to update: recorded raw files changed or disappeared", file=sys.stderr)
        return 1
    recorded.update({k: actual[k] for k in new + (changed if args.accept_changed else [])})
    write_manifest(recorded)
    print(f"manifest: {len(recorded)} files ({len(new)} added)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
