#!/usr/bin/env bash
# Safely insert strict seed-1000 controls before the still-idle evaluation chain.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TRAINING_SESSION=umi_arch_supervisor_20260812
CAPACITY_SESSION=umi_arch_capacity_control_20260812
EVALUATION_SESSION=umi_arch_eval_supervisor_20260812
CONFIRMATION_TRAIN_SESSION=umi_arch_confirmation_train_20260812
CONFIRMATION_EVAL_SESSION=umi_arch_confirmation_eval_20260812
MONITOR_SESSION=umi_arch_chain_monitor_20260812

timestamp() {
  date '+%F %T'
}

require_session() {
  local session="$1"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    echo "[$(timestamp)] required session is missing: $session" >&2
    exit 2
  fi
}

require_waiting_for() {
  local session="$1" dependency="$2" pane
  require_session "$session"
  pane="$(tmux capture-pane -pt "$session:0" -S -40)"
  if [[ "$pane" != *"waiting for tmux session $dependency to release the GPU"* ]]; then
    echo "[$(timestamp)] refusing to replace non-idle session $session" >&2
    exit 2
  fi
}

require_session "$TRAINING_SESSION"
if tmux has-session -t "$CAPACITY_SESSION" 2>/dev/null; then
  echo "[$(timestamp)] capacity-control session already exists: $CAPACITY_SESSION" >&2
  exit 2
fi
require_waiting_for "$EVALUATION_SESSION" "$TRAINING_SESSION"
require_waiting_for "$CONFIRMATION_TRAIN_SESSION" "$EVALUATION_SESSION"
require_waiting_for "$CONFIRMATION_EVAL_SESSION" "$CONFIRMATION_TRAIN_SESSION"

# Remove successors in reverse dependency order so no released waiter can start.
tmux kill-session -t "$CONFIRMATION_EVAL_SESSION"
tmux kill-session -t "$CONFIRMATION_TRAIN_SESSION"
tmux kill-session -t "$EVALUATION_SESSION"

tmux new-session -d -s "$CAPACITY_SESSION" \
  "cd '$REPO' && UMI_WAIT_FOR_TMUX=$TRAINING_SESSION bash examples/umi_relative_ee/act_flow_ablation/supervise_capacity_control.sh 1000"
tmux new-session -d -s "$EVALUATION_SESSION" \
  "cd '$REPO' && UMI_WAIT_FOR_TMUX=$CAPACITY_SESSION bash examples/umi_relative_ee/act_flow_ablation/supervise_evaluations.sh 1000 5"
tmux new-session -d -s "$CONFIRMATION_TRAIN_SESSION" \
  "cd '$REPO' && UMI_WAIT_FOR_TMUX=$EVALUATION_SESSION bash examples/umi_relative_ee/act_flow_ablation/supervise_confirmation_training.sh 100000 2000 3000"
tmux new-session -d -s "$CONFIRMATION_EVAL_SESSION" \
  "cd '$REPO' && UMI_WAIT_FOR_TMUX=$CONFIRMATION_TRAIN_SESSION bash examples/umi_relative_ee/act_flow_ablation/supervise_confirmation_evaluations.sh"

if tmux has-session -t "$MONITOR_SESSION" 2>/dev/null; then
  tmux kill-session -t "$MONITOR_SESSION"
fi
tmux new-session -d -s "$MONITOR_SESSION" \
  "sleep 2; cd '$REPO'; exec examples/umi_relative_ee/act_flow_ablation/monitor_experiment_chain.sh"

sleep 2
for session in \
  "$TRAINING_SESSION" \
  "$CAPACITY_SESSION" \
  "$EVALUATION_SESSION" \
  "$CONFIRMATION_TRAIN_SESSION" \
  "$CONFIRMATION_EVAL_SESSION" \
  "$MONITOR_SESSION"; do
  require_session "$session"
done
echo "[$(timestamp)] inserted strict controls and verified the six-session chain"
