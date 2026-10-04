#!/bin/bash
# Copy cluster results into this repository (run locally). Read-only on the cluster.
#   scripts/sync_results.sh [ssh-alias]
# Mirrors $BC_DATA/{raw_results,telemetry,environment,node_checks} and the GGUF
# manifests; model weights and Slurm logs stay on the cluster. Never uses --delete:
# scratch may be purged, and these local copies are then the only ones left.
set -euo pipefail
HOST="${1:-cresco8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# BC_DATA as defined by the cluster checkout's configs/cresco8.env.
CHECKOUT="${BC_CLUSTER_CHECKOUT:-dev/llm-backend-analysis/backend_comparison}"
REMOTE="$(ssh "$HOST" "source $CHECKOUT/configs/cresco8.env >/dev/null && echo \$BC_DATA")"
[[ -n "$REMOTE" ]] || { echo "could not resolve BC_DATA on $HOST" >&2; exit 1; }
echo "remote data root: $REMOTE"
for d in raw_results telemetry; do
  rsync -a --exclude '*.attempt-*' "$HOST:$REMOTE/$d/" "$HERE/$d/"
done
mkdir -p "$HERE/environment/nodes" "$HERE/environment/node_checks" "$HERE/environment/models"
rsync -a "$HOST:$REMOTE/environment/" "$HERE/environment/nodes/"
rsync -a "$HOST:$REMOTE/node_checks/" "$HERE/environment/node_checks/" 2>/dev/null || true
rsync -a --include '*/' --include 'manifest.json' --include 'preparation.log' --exclude '*' \
  "$HOST:$REMOTE/models/" "$HERE/environment/models/"
echo "synced into $HERE"
