#!/usr/bin/env bash
# Full experimental protocol (see detection/PROTOCOL.md). Run from the repo root:
#
#   GPUS="0 1 2 3" WORKERS=32 bash detection/run_protocol.sh data
#   GPUS="0 1 2 3" bash detection/run_protocol.sh sanity
#   GPUS="0 1 2 3" bash detection/run_protocol.sh main      # P2: 4 archs x 3 seeds
#   GPUS="0 1 2 3" bash detection/run_protocol.sh scaling   # P3: pool size / symmetry ablation
#   GPUS="0 1 2 3" bash detection/run_protocol.sh sandbox   # P4: physics-shift augmentation
#   bash detection/run_protocol.sh eval
#
# Jobs are spread round-robin over $GPUS; each GPU runs its queue sequentially in the
# background (nohup), so the SSH session can be closed. Logs: runs/<name>/stdout.log.
# Re-running a phase resumes unfinished runs (train.py restarts from last.pt) and
# skips finished ones (runs/<name>/DONE).
set -euo pipefail

PHASE=${1:?usage: run_protocol.sh data|sanity|main|scaling|sandbox|eval}
GPUS=(${GPUS:-0})
WORKERS=${WORKERS:-16}
SEEDS=(${SEEDS:-0 1 2})
STEPS=${STEPS:-40000}
POOL_N=${POOL_N:-30000}
ARCHS=(unet transunet segformer vit)
PY="uv run --extra train python"

declare -a JOBS=()   # "name|extra args"
add() { JOBS+=("$1|$2"); }

launch() {
  local n=${#GPUS[@]}
  for ((g = 0; g < n; g++)); do
    local gpu=${GPUS[$g]} queue=""
    for ((j = g; j < ${#JOBS[@]}; j += n)); do
      local name=${JOBS[$j]%%|*} extra=${JOBS[$j]#*|}
      mkdir -p "runs/$name"
      queue+="[ -f runs/$name/DONE ] || { CUDA_VISIBLE_DEVICES=$gpu $PY -m detection.train --out runs/$name $extra > runs/$name/stdout.log 2>&1 && touch runs/$name/DONE; }; "
    done
    [ -n "$queue" ] && nohup bash -c "$queue" > "runs/queue_gpu${gpu}_${PHASE}.log" 2>&1 &
    echo "GPU $gpu: queued"
  done
  echo "${#JOBS[@]} jobs launched. Follow with: tail -f runs/<name>/stdout.log  |  nvidia-smi"
}

mkdir -p runs
case $PHASE in
  data)
    $PY -m detection.data_gen pool --n "$POOL_N" --out data/pool --workers "$WORKERS"
    $PY -m detection.data_gen sets --out data/eval --workers "$WORKERS"
    $PY -m detection.data_gen verify --pool data/pool --eval data/eval
    ;;
  sanity)
    # P1: each model must overfit one fixed batch (loss -> ~0). Also a throughput probe.
    for a in "${ARCHS[@]}"; do add "sanity_$a" "--arch $a --sanity --steps 500 --warmup 50 --log-every 50"; done
    launch
    ;;
  main)
    # P2: baseline protocol, identical budget for every arch.
    for a in "${ARCHS[@]}"; do for s in "${SEEDS[@]}"; do
      add "${a}_s$s" "--arch $a --seed $s --steps $STEPS"
    done; done
    launch
    ;;
  scaling)
    # P3: does the model overfit *geometries*? Shrink the pool; drop the exact symmetries.
    for a in "${ARCHS[@]}"; do
      for k in 1000 10000; do add "${a}_pool${k}_s0" "--arch $a --seed 0 --steps $STEPS --pool-size $k"; done
      add "${a}_nosym_pool10000_s0" "--arch $a --seed 0 --steps $STEPS --pool-size 10000 --no-symmetry"
    done
    launch
    ;;
  sandbox)
    # P4: physics-shift augmentation (scale/shear, polarity, noise level) -> OOD robustness.
    # P4b: low-SNR emphasis (challenge-2 frames sit at |i| ~ 3), one factor at a time.
    for a in "${ARCHS[@]}"; do for s in "${SEEDS[@]}"; do
      add "${a}_shift_s$s" "--arch $a --seed $s --steps $STEPS --affine --polarity --noise-jitter 0.3"
      add "${a}_lowsnr_s$s" "--arch $a --seed $s --steps $STEPS --intensity-law loguniform"
    done; done
    launch
    ;;
  eval)
    $PY -m detection.evaluate --runs "runs/*_s[0-9]*" --out challenge1/results/transformers
    ;;
  *) echo "unknown phase $PHASE"; exit 1 ;;
esac
