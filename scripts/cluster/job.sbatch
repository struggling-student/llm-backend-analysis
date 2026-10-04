#!/bin/bash
# Generic job wrapper: sbatch [opts] scripts/cluster/job.sbatch <script.py> [args...]
# Submit through scripts/cluster/submit.sh, which exports BC_ROOT and sets the log path.
#SBATCH --job-name=bc
#SBATCH --partition=cresco8_hbm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --time=24:00:00
source /etc/profile >/dev/null 2>&1 || true   # before set -e: site profile scripts return non-zero
set -euo pipefail

BC_ROOT="${BC_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$BC_ROOT"
source configs/cresco8.env
export PYTHONUNBUFFERED=1

echo "job ${SLURM_JOB_ID} on $(hostname) at $(date -Is): $*"

# 1. The experiment is defined for cache mode: two NUMA nodes, both with CPUs.
if numactl -H | grep -qE '^node [0-9]+ cpus:\s*$'; then
  echo "ERROR: memory-only NUMA nodes present -> flat mode node; this experiment needs cache mode" >&2
  exit 3
fi

# 2. Exclusive node: remove servers a previous job on this node may have left behind.
#    vLLM renames its worker processes (VLLM::Worker, VLLM::EngineCore), so match those too.
ORPHANS='llama-server|llama-bench|llama-batched-bench|llama-quantize|vllm serve|vllm bench|VLLM::|EngineCore|stream_triad'
pkill -9 -u "$USER" -f "$ORPHANS" 2>/dev/null || true
sleep 2
echo "processes of $USER using CPU at job start (should be only this job):"
ps -u "$USER" -o pid,ppid,pcpu,etimes,cmd --sort=-pcpu | head -8 | cut -c1-150
export BC_LOCAL="/tmp/bc-${SLURM_JOB_ID}"   # node-local model staging (see bclib.stage_model)
cleanup() { pkill -9 -u "$USER" -f "$ORPHANS" 2>/dev/null || true; rm -rf "$BC_LOCAL"; }
trap cleanup EXIT

# 3. Environment manifest (versions, build flags, topology, PMU permissions, hashes).
"$BC_PYTHON" scripts/benchmark/capture_env.py || echo "WARNING: capture_env failed" >&2

# 4. Node health check: 10 s STREAM triad on the server layout, so results
#    from different (nominally identical) nodes can be checked for equivalence.
mkdir -p "$BC_DATA/node_checks"
OMP_NUM_THREADS=55 OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-54 numactl --membind=0 \
  "$BC_STREAM_BIN" --bytes $((8 * 1024**3)) --warmups 2 --repetitions 400 \
  > "$BC_DATA/node_checks/$(hostname -s)-${SLURM_JOB_ID}.json" 2>&1 || true
NODE_GBS=$("$BC_PYTHON" - "$BC_DATA/node_checks/$(hostname -s)-${SLURM_JOB_ID}.json" <<'PY'
import json, statistics, sys
lines = [l for l in open(sys.argv[1]) if l.startswith("{")]
o = json.loads(lines[-1]) if lines else None
print(f"{o['modeled_bytes_per_iteration'] / statistics.median(o['steady_elapsed_seconds']) / 1e9:.1f}" if o else "0")
PY
)
echo "node check: STREAM triad 8 GiB, 55 threads, socket 0: ${NODE_GBS} GB/s"
# Healthy cache-mode nodes give ~495-510 GB/s. A clearly lower value means the
# server cores or memory are not in the state the experiment assumes.
if [[ "${BC_SKIP_NODE_CHECK:-0}" != 1 ]] && awk "BEGIN{exit !(${NODE_GBS} < 450)}"; then
  echo "ERROR: node check ${NODE_GBS} GB/s < 450 GB/s; refusing to run on $(hostname -s)" >&2
  exit 4
fi

# 5. The actual work (BC_JOB_PYTHON selects another interpreter, e.g. the llm-compressor venv).
exec "${BC_JOB_PYTHON:-$BC_PYTHON}" "$@"
