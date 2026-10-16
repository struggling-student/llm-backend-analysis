#!/usr/bin/env python3
"""Memory-node sensitivity (the review's Experiment A, not the offline Experiment A): throughput with the server's memory on the local node vs the remote node.

For each cell of configs/experiment.toml [membind_sensitivity] the cores stay on socket 0 and only
`numactl --membind` of the server changes: node 0 (reference) or node 1 (remote memory over the socket
interconnect). Within one job the order is node 0, node 1, node 0, node 1 (`passes` = 2); every pass is a
separate arm directory `<arm>__mem<node>_p<pass>` of the campaign `membind`, with the harness's own
repetitions per point. The ratio is therefore formed on one node and one day.

Remote memory lowers the bandwidth AND raises the latency, so a throughput drop shows sensitivity to the memory
system, not to bandwidth alone (docs/review_tracker.md, T2.1-T2.3).

  python scripts/benchmark/membind_sensitivity.py --dry-run             # print the commands of every cell
  python scripts/benchmark/membind_sensitivity.py --cell vllm-decode-c1 # run one cell (inside a Slurm job)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bclib  # noqa: E402

LOCAL_SHARE_MIN = 0.90   # fraction of the server's resident memory that must sit on the requested node


def plan(cfg: dict[str, Any], names: list[str] | None, passes: int | None) -> list[dict[str, Any]]:
    """One entry per run_online.py invocation, in execution order (grouped by cell, A/B/A/B inside a cell)."""
    sec = cfg["membind_sensitivity"]
    cells = {c["name"]: c for c in sec["cells"]}
    unknown = set(names or []) - set(cells)
    if unknown:
        raise SystemExit(f"unknown cell(s) {sorted(unknown)}; available: {sorted(cells)}")
    runs = []
    for name in names or list(cells):
        c = cells[name]
        for p in range(1, (passes or sec["passes"]) + 1):
            for node in (sec["reference_node"], sec["remote_node"]):
                tag = f"mem{node}_p{p}"
                cmd = [sys.executable, str(bclib.SCRIPTS / "run_online.py"), "--model", c["model"], "--arm", c["arm"],
                       "--workload", c["workload"], "--campaign", sec["campaign"], "--concurrency", str(c["concurrency"]),
                       "--mem-node", str(node), "--tag", tag]
                runs.append({"cell": name, "node": node, "pass": p, "tag": tag, "cmd": cmd, "spec": c,
                             "campaign": sec["campaign"]})
    return runs


def numastat_share(path: Path, node: int) -> float | None:
    """Share of the server's memory on `node`, from the `Total` rows of the `numastat -p` tables."""
    if not path.exists():
        return None
    per_node: list[float] = []
    for line in path.read_text().splitlines():
        if line.startswith("Total"):
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
            if len(nums) >= 3:
                per_node = [a + b for a, b in zip(per_node or [0.0] * (len(nums) - 1), nums[:-1])]
    total = sum(per_node)
    return per_node[node] / total if total and node < len(per_node) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", help="cell name (repeatable); default: every cell")
    ap.add_argument("--passes", type=int, help="override [membind_sensitivity].passes")
    ap.add_argument("--list", action="store_true", help="print the cell names and exit")
    ap.add_argument("--dry-run", action="store_true", help="print each run's command (and its server/client commands)")
    args = ap.parse_args()
    cfg = bclib.load_config()
    if args.list:
        print("\n".join(c["name"] for c in cfg["membind_sensitivity"]["cells"]))
        return 0
    runs = plan(cfg, args.cell, args.passes)
    for r in runs:
        print(f"# {r['cell']}  pass {r['pass']}  memory node {r['node']}\n{' '.join(r['cmd'])}", flush=True)
        if args.dry_run:
            subprocess.run([*r["cmd"], "--print-commands"], check=True)
    if args.dry_run:
        return 0
    import os
    base = Path(os.environ["BC_DATA"]) / "raw_results" / "online" / cfg["membind_sensitivity"]["campaign"]
    problems = 0
    for r in runs:
        bclib.log(f"{r['cell']} pass {r['pass']} node {r['node']}")
        subprocess.run(r["cmd"], check=False)
        c = r["spec"]
        point = base / c["model"] / f"{c['arm']}__{r['tag']}" / c["workload"] / f"c{c['concurrency']:03d}"
        meta = json.loads((point / "metadata.json").read_text()) if (point / "metadata.json").exists() else {}
        share = numastat_share(point / "numastat.txt", r["node"])
        bclib.log(f"  status {meta.get('status')}, server memory on node {r['node']}: "
                  f"{'n/a' if share is None else f'{share:.0%}'}")
        if not str(meta.get("status", "")).startswith("ok") or (share is not None and share < LOCAL_SHARE_MIN):
            problems += 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
