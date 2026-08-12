#!/usr/bin/env bash
# Evaluate the independent training-seed confirmations after they release the GPU.
set -euo pipefail

WAIT_FOR_SESSION="${UMI_WAIT_FOR_TMUX:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_SEEDS=(2000 3000)

if [[ -n "$WAIT_FOR_SESSION" ]]; then
  echo "[$(date '+%F %T')] waiting for tmux session $WAIT_FOR_SESSION to release the GPU"
  while tmux has-session -t "$WAIT_FOR_SESSION" 2>/dev/null; do
    sleep 60
  done
fi

for seed in "${TRAINING_SEEDS[@]}"; do
  UMI_WAIT_FOR_TMUX= "$SCRIPT_DIR/supervise_evaluations.sh" "$seed" 5
done
