#!/bin/bash
# Copy cluster results into this repository (run locally). Read-only on the cluster.
#   scripts/cluster/sync_results.sh [ssh-alias]
# Mirrors $BC_DATA/{raw_results,telemetry,environment,node_checks} and the GGUF
# manifests into data/raw/ (paths below it mirror $BC_DATA, so re-syncing never
# duplicates data); model weights and Slurm logs stay on the cluster. Never uses --delete:
# scratch may be purged, and these local copies are then the only ones left.
set -euo pipefail
HOST="${1:-cresco8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW="$HERE/data/raw"
# BC_DATA as defined by the cluster checkout's configs/cresco8.env.
# Falls back to the pre-2026-10-04 checkout layout (code under backend_comparison/).
CHECKOUT="${BC_CLUSTER_CHECKOUT:-dev/llm-backend-analysis}"
REMOTE="$(ssh "$HOST" "for c in $CHECKOUT $CHECKOUT/backend_comparison; do [ -f \$c/configs/cresco8.env ] && source \$c/configs/cresco8.env >/dev/null && echo \$BC_DATA && break; done")"
[[ -n "$REMOTE" ]] || { echo "could not resolve BC_DATA on $HOST" >&2; exit 1; }
echo "remote data root: $REMOTE"
# $BC_DATA/raw_results/<kind>/... -> data/raw/<kind>/...; telemetry keeps its own subtree.
mkdir -p "$RAW/telemetry" "$RAW/environment/nodes" "$RAW/environment/node_checks" "$RAW/environment/models"
rsync -a --exclude '*.attempt-*' "$HOST:$REMOTE/raw_results/" "$RAW/"
rsync -a --exclude '*.attempt-*' "$HOST:$REMOTE/telemetry/" "$RAW/telemetry/"
rsync -a "$HOST:$REMOTE/environment/" "$RAW/environment/nodes/"
rsync -a "$HOST:$REMOTE/node_checks/" "$RAW/environment/node_checks/" 2>/dev/null || true
rsync -a --include '*/' --include 'manifest.json' --include 'preparation.log' --exclude '*' \
  "$HOST:$REMOTE/models/" "$RAW/environment/models/"
echo "synced into $RAW"
