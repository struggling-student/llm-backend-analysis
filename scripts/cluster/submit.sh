#!/bin/bash
# Submit backend-comparison jobs. Run from anywhere on the login node.
#
#   scripts/cluster/submit.sh prepare                       # build GGUFs (BF16, Q8_0)
#   scripts/cluster/submit.sh calibrate                     # STREAM + PMU-proxy validation
#   scripts/cluster/submit.sh smoke                         # small end-to-end validation matrix
#   scripts/cluster/submit.sh tuning                        # small backend tuning phase
#   scripts/cluster/submit.sh offline  [models] [arms]      # Experiment A
#   scripts/cluster/submit.sh online   [models] [arms] [workloads]   # Experiment B, one job per cell
#   scripts/cluster/submit.sh int8-tuning                   # tuning for vllm-w8a8 (then set [tuned.vllm-w8a8])
#   scripts/cluster/submit.sh int8-campaign                 # vllm-w8a8, 8B: 5 online workloads + offline
#   scripts/cluster/submit.sh membind [cell ...]            # memory-node sensitivity: remote vs local server memory, one job per cell
#   scripts/cluster/submit.sh build-shim                    # compile the vLLM NUMA shim inside the image (login node, seconds)
#   scripts/cluster/submit.sh amx-verify                    # T3.1: AMX_BUSY and oneDNN implementations per ablation level (short)
#   scripts/cluster/submit.sh amx-vllm [level ...]          # T3.1: vLLM BF16 AMX ablation sweeps, one job per (level, workload)
#   scripts/cluster/submit.sh amx-llamacpp                  # T3.2: llama.cpp Q8_0 prompt processing, AMX vs AVX-512 build
#   scripts/cluster/submit.sh membind-shimcheck             # one short C=1 vLLM run with memory on node 1; check numastat after
#   scripts/cluster/submit.sh membind-stream                # memory-node sensitivity: STREAM roof with remote memory (node 1)
#   scripts/cluster/submit.sh external-shapes [arm ...]     # notebook 13: published shapes (1000->1000, 256->256), one job per (arm, shape)
#   scripts/cluster/submit.sh idle-power [seconds]          # notebook 13: idle node power (sleep after the node check), default 1200 s
#   scripts/cluster/submit.sh job-energy <jobid ...>        # sacct ConsumedEnergy of finished jobs -> $BC_DATA/slurm_energy (login node, no job)
#   scripts/cluster/submit.sh job <time> <name> <script.py> [args]   # anything else (path from repo root)
#
# Defaults: models=llama31_8b,llama32_1b  arms=vllm-bf16,llamacpp-bf16,llamacpp-q8_0
#           workloads=decode,balanced,prefill  campaign=$BC_CAMPAIGN (default: main)
set -euo pipefail
BC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BC_ROOT"
source configs/cresco8.env
CAMPAIGN="${BC_CAMPAIGN:-main}"
PARTITION="${BC_PARTITION:-cresco8_hbm}"
mkdir -p "$BC_DATA/slurm-logs"

sub() {  # time name script args...
  local time=$1 name=$2; shift 2
  sbatch --parsable --partition="$PARTITION" --account="$SLURM_ACCOUNT" --time="$time" --job-name="bc-$name" \
    --output="$BC_DATA/slurm-logs/%x-%j.out" --export="ALL,BC_ROOT=$BC_ROOT" \
    ${BC_DEPENDENCY:+--dependency=$BC_DEPENDENCY} scripts/cluster/job.sbatch "$@"
}

what="${1:-}"; shift || true
case "$what" in
  prepare)   sub 03:00:00 prepare scripts/benchmark/prepare_models.py ;;
  calibrate) sub 02:00:00 stream scripts/benchmark/stream_calibration.py ;;
  smoke)     sub 04:00:00 smoke scripts/benchmark/smoke.py ;;
  tuning)    # submit.sh tuning <arm> [tuning.py args]
    a=$1; shift; echo "tuning $a: $(sub 24:00:00 "tune-$a" scripts/benchmark/tuning.py --arm "$a" "$@")" ;;
  offline)
    IFS=, read -ra MODELS <<< "${1:-llama31_8b,llama32_1b}"
    IFS=, read -ra ARMS <<< "${2:-vllm-bf16,llamacpp-bf16,llamacpp-q8_0}"
    for m in "${MODELS[@]}"; do for a in "${ARMS[@]}"; do
      echo "offline $m $a: $(sub 24:00:00 "off-$m-$a" scripts/benchmark/run_offline.py --model "$m" --arm "$a" --campaign "$CAMPAIGN")"
    done; done ;;
  online)
    IFS=, read -ra MODELS <<< "${1:-llama31_8b,llama32_1b}"
    IFS=, read -ra ARMS <<< "${2:-vllm-bf16,llamacpp-bf16,llamacpp-q8_0}"
    IFS=, read -ra WLS <<< "${3:-decode,balanced,prefill}"
    for m in "${MODELS[@]}"; do for a in "${ARMS[@]}"; do for w in "${WLS[@]}"; do
      echo "online $m $a $w: $(sub 24:00:00 "on-$m-$a-$w" scripts/benchmark/run_online.py --model "$m" --arm "$a" --workload "$w" --campaign "$CAMPAIGN")"
    done; done; done ;;
  int8-tuning) echo "tuning vllm-w8a8: $(sub 24:00:00 tune-vllm-w8a8 scripts/benchmark/tuning.py --arm vllm-w8a8)" ;;
  int8-campaign)
    for w in decode balanced prefill longctx kvdecode; do
      echo "online llama31_8b vllm-w8a8 $w: $(sub 24:00:00 "on-llama31_8b-vllm-w8a8-$w" scripts/benchmark/run_online.py --model llama31_8b --arm vllm-w8a8 --workload "$w" --campaign "$CAMPAIGN")"
    done
    echo "offline llama31_8b vllm-w8a8: $(sub 24:00:00 off-llama31_8b-vllm-w8a8 scripts/benchmark/run_offline.py --model llama31_8b --arm vllm-w8a8 --campaign "$CAMPAIGN")" ;;
  membind)   # one job per cell: node 0 / node 1 / node 0 / node 1 inside one job, so the ratio is on one node
    CELLS=("$@"); [[ ${#CELLS[@]} -gt 0 ]] || mapfile -t CELLS < <("$BC_PYTHON" scripts/benchmark/membind_sensitivity.py --list)
    for c in "${CELLS[@]}"; do
      echo "membind $c: $(sub 12:00:00 "mb-$c" scripts/benchmark/membind_sensitivity.py --cell "$c")"
    done ;;
  build-shim)
    mkdir -p "$BC_DATA/shim/no_amx" && cp "$BC_ROOT/scripts/benchmark/no_amx/sitecustomize.py" "$BC_DATA/shim/no_amx/"
    apptainer exec --cleanenv --bind "$BC_DATA:$BC_DATA" --bind "$BC_ROOT:$BC_ROOT" "$BC_IMAGES/vllm-cpu-v0.30.0.sif" gcc -shared -fPIC -O2 \
      -o "$BC_DATA/shim/numa_noop.so" "$BC_ROOT/scripts/benchmark/numa_noop.c" && ls -l "$BC_DATA/shim/numa_noop.so" ;;
  amx-verify)   echo "amx verify: $(sub 03:00:00 amx-verify scripts/benchmark/amx_ablation.py verify)" ;;
  amx-vllm)     # levels default to all; one job per (level, workload) so the sweeps run side by side
    LEVELS=("$@"); [[ ${#LEVELS[@]} -gt 0 ]] || LEVELS=(amx noamx_gemm noamx_all)
    for w in balanced decode; do for lv in "${LEVELS[@]}"; do
      echo "amx vllm $lv $w: $(sub 24:00:00 "amx-$lv-$w" scripts/benchmark/amx_ablation.py vllm --level "$lv" --workload "$w")"
    done; done ;;
  amx-llamacpp) echo "amx llamacpp: $(sub 03:00:00 amx-llamacpp scripts/benchmark/amx_ablation.py llamacpp)" ;;
  membind-shimcheck)
    echo "membind shimcheck: $(sub 00:45:00 mb-shimcheck scripts/benchmark/run_online.py --model llama31_8b --arm vllm-bf16 --workload decode --campaign membind-shimcheck --concurrency 1 --reps 1 --num-requests 2 --mem-node 1)" ;;
  membind-stream)
    echo "membind stream: $(sub 02:00:00 stream-membind1 scripts/benchmark/stream_calibration.py --mem-node 1 --threads 55 --campaign calibration-membind1)" ;;
  external-shapes)   # arms default to [external_shapes].arms (vllm-bf16); vllm-w8a8 is optional
    ARMS=(); for a in "$@"; do ARMS+=(--arm "$a"); done
    while read -r a w t; do
      echo "external-shapes $a $w: $(sub "$t" "ext-$a-$w" scripts/benchmark/external_shapes.py --arm "$a" --workload "$w")"
    done < <("$BC_PYTHON" scripts/benchmark/external_shapes.py --jobs ${ARMS[@]+"${ARMS[@]}"}) ;;
  idle-power)   # whole-node power with nothing running: the baseline that splits notebook 13's node energy
    d="${1:-1200}"
    echo "idle power ${d} s: $(sub "$(( d / 60 + 15 ))" idle-power scripts/benchmark/idle_power.py --duration "$d")"   # --time in minutes
    echo "afterwards: scripts/cluster/submit.sh job-energy <jobid>" ;;
  job-energy)   # whole-node IPMI energy (Slurm acct_gather_energy/ipmi, DCMI sensor, 60 s updates) of finished jobs
    mkdir -p "$BC_DATA/slurm_energy"
    for j in "$@"; do
      sacct -j "$j" --parsable2 --format=JobID,JobName,NodeList,State,Start,End,Elapsed,ElapsedRaw,ConsumedEnergy,ConsumedEnergyRaw \
        > "$BC_DATA/slurm_energy/$j.psv" && echo "$BC_DATA/slurm_energy/$j.psv" && cat "$BC_DATA/slurm_energy/$j.psv"
    done ;;
  job) t=$1 n=$2; shift 2; sub "$t" "$n" "$@" ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac
