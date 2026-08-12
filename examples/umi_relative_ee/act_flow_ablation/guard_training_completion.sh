#!/usr/bin/env bash
# Bridge durable completion into a supervisor that was started before the
# shared recovery helper existed. It never marks partial checkpoints complete.
set -uo pipefail

RUN_NAME="${1:?usage: guard_training_completion.sh RUN_NAME STEPS [OWNER_TMUX]}"
STEPS="${2:?usage: guard_training_completion.sh RUN_NAME STEPS [OWNER_TMUX]}"
OWNER_TMUX="${3:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
. "$SCRIPT_DIR/training_completion.sh"

while true; do
  if recover_training_completion "$ARTIFACT_ROOT" "$RUN_NAME" "$STEPS"; then
    echo "[$(date '+%F %T')] completion guard finished: $RUN_NAME"
    exit 0
  fi
  if [[ -n "$OWNER_TMUX" ]] && ! tmux has-session -t "$OWNER_TMUX" 2>/dev/null; then
    echo "[$(date '+%F %T')] completion guard stopped because owner exited before durable completion: $RUN_NAME"
    exit 1
  fi
  # Poll cheaply until the final checkpoint begins, then close the short race
  # between the trainer's terminal message and the legacy supervisor retry.
  if [[ -d "$ARTIFACT_ROOT/train/$RUN_NAME/checkpoints/$STEPS" ]]; then
    sleep 0.01
  else
    sleep 2
  fi
done
