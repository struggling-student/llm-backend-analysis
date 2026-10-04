#!/bin/bash
# Submit backend-comparison jobs. Run from anywhere on the login node.
#
#   scripts/submit.sh prepare                       # build GGUFs (BF16, Q8_0)
#   scripts/submit.sh calibrate                     # STREAM + PMU-proxy validation
#   scripts/submit.sh smoke                         # small end-to-end validation matrix
#   scripts/submit.sh tuning                        # small backend tuning phase
#   scripts/submit.sh offline  [models] [arms]      # Experiment A
#   scripts/submit.sh online   [models] [arms] [workloads]   # Experiment B, one job per cell
#   scripts/submit.sh int8-tuning                   # tuning for vllm-w8a8 (then set [tuned.vllm-w8a8])
#   scripts/submit.sh int8-campaign                 # vllm-w8a8, 8B: 5 online workloads + offline
#   scripts/submit.sh job <time> <name> <script.py> [args]   # anything else
#
# Defaults: models=llama31_8b,llama32_1b  arms=vllm-bf16,llamacpp-bf16,llamacpp-q8_0
#           workloads=decode,balanced,prefill  campaign=$BC_CAMPAIGN (default: main)
set -euo pipefail
BC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BC_ROOT"
source configs/cresco8.env
CAMPAIGN="${BC_CAMPAIGN:-main}"
PARTITION="${BC_PARTITION:-cresco8_hbm}"
mkdir -p "$BC_DATA/slurm-logs"

sub() {  # time name script args...
  local time=$1 name=$2; shift 2
  sbatch --parsable --partition="$PARTITION" --account="$SLURM_ACCOUNT" --time="$time" --job-name="bc-$name" \
    --output="$BC_DATA/slurm-logs/%x-%j.out" --export="ALL,BC_ROOT=$BC_ROOT" \
    ${BC_DEPENDENCY:+--dependency=$BC_DEPENDENCY} scripts/job.sbatch "$@"
}

what="${1:-}"; shift || true
case "$what" in
  prepare)   sub 03:00:00 prepare scripts/prepare_models.py ;;
  calibrate) sub 02:00:00 stream scripts/stream_calibration.py ;;
  smoke)     sub 04:00:00 smoke scripts/smoke.py ;;
  tuning)    # submit.sh tuning <arm> [tuning.py args]
    a=$1; shift; echo "tuning $a: $(sub 24:00:00 "tune-$a" scripts/tuning.py --arm "$a" "$@")" ;;
  offline)
    IFS=, read -ra MODELS <<< "${1:-llama31_8b,llama32_1b}"
    IFS=, read -ra ARMS <<< "${2:-vllm-bf16,llamacpp-bf16,llamacpp-q8_0}"
    for m in "${MODELS[@]}"; do for a in "${ARMS[@]}"; do
      echo "offline $m $a: $(sub 24:00:00 "off-$m-$a" scripts/run_offline.py --model "$m" --arm "$a" --campaign "$CAMPAIGN")"
    done; done ;;
  online)
    IFS=, read -ra MODELS <<< "${1:-llama31_8b,llama32_1b}"
    IFS=, read -ra ARMS <<< "${2:-vllm-bf16,llamacpp-bf16,llamacpp-q8_0}"
    IFS=, read -ra WLS <<< "${3:-decode,balanced,prefill}"
    for m in "${MODELS[@]}"; do for a in "${ARMS[@]}"; do for w in "${WLS[@]}"; do
      echo "online $m $a $w: $(sub 24:00:00 "on-$m-$a-$w" scripts/run_online.py --model "$m" --arm "$a" --workload "$w" --campaign "$CAMPAIGN")"
    done; done; done ;;
  int8-tuning) echo "tuning vllm-w8a8: $(sub 24:00:00 tune-vllm-w8a8 scripts/tuning.py --arm vllm-w8a8)" ;;
  int8-campaign)
    for w in decode balanced prefill longctx kvdecode; do
      echo "online llama31_8b vllm-w8a8 $w: $(sub 24:00:00 "on-llama31_8b-vllm-w8a8-$w" scripts/run_online.py --model llama31_8b --arm vllm-w8a8 --workload "$w" --campaign "$CAMPAIGN")"
    done
    echo "offline llama31_8b vllm-w8a8: $(sub 24:00:00 off-llama31_8b-vllm-w8a8 scripts/run_offline.py --model llama31_8b --arm vllm-w8a8 --campaign "$CAMPAIGN")" ;;
  job) t=$1 n=$2; shift 2; sub "$t" "$n" "$@" ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
