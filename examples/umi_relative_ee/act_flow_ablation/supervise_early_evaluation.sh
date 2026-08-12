#!/usr/bin/env bash
# Evaluate a completed companion/candidate promptly while the long dependency
# chain continues. Canonical output paths make the later matrix evaluator skip
# work already completed here.
set -uo pipefail

RUN_NAME="${1:?usage: supervise_early_evaluation.sh RUN_NAME STEPS WAIT_TMUX [INFERENCE_SEED ...]}"
STEPS="${2:?usage: supervise_early_evaluation.sh RUN_NAME STEPS WAIT_TMUX [INFERENCE_SEED ...]}"
WAIT_TMUX="${3:?usage: supervise_early_evaluation.sh RUN_NAME STEPS WAIT_TMUX [INFERENCE_SEED ...]}"
shift 3
if [[ "$#" -gt 0 ]]; then INFERENCE_SEEDS=("$@"); else INFERENCE_SEEDS=(1000); fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
MAX_ATTEMPTS="${UMI_MAX_ATTEMPTS:-3}"
SAMPLES_PER_EPISODE="${UMI_EVAL_SAMPLES_PER_EPISODE:-5}"
. "$SCRIPT_DIR/training_completion.sh"

timestamp() { date '+%F %T'; }
eval_log() { printf '%s/logs/eval_common_h32_%s_seed%s.log' "$ARTIFACT_ROOT" "$RUN_NAME" "$1"; }
eval_dir() { printf '%s/eval_common_h32/%s/seed%s' "$ARTIFACT_ROOT" "$RUN_NAME" "$1"; }

evaluation_complete() {
  local seed="$1" out log reports
  out="$(eval_dir "$seed")"
  log="$(eval_log "$seed")"
  reports="$(find "$out" -maxdepth 1 -type f -name '*_open_loop_metrics.json' -size +0c 2>/dev/null | wc -l)"
  [[ "$reports" -eq 1 ]] && grep -Fq "] completed evaluation $RUN_NAME seed $seed" "$log"
}

echo "[$(timestamp)] early evaluator waiting for $WAIT_TMUX"
while tmux has-session -t "$WAIT_TMUX" 2>/dev/null; do sleep 10; done

if ! training_is_complete "$ARTIFACT_ROOT" "$RUN_NAME" "$STEPS"; then
  echo "[$(timestamp)] training did not reach durable completion; not evaluating $RUN_NAME"
  exit 1
fi

for seed in "${INFERENCE_SEEDS[@]}"; do
  if evaluation_complete "$seed"; then
    echo "[$(timestamp)] already evaluated: $RUN_NAME seed=$seed"
    continue
  fi
  for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    out="$(eval_dir "$seed")"
    log="$(eval_log "$seed")"
    if [[ -e "$out" || -e "$log" ]]; then
      archive="$ARTIFACT_ROOT/interrupted_evaluations/${RUN_NAME}_seed${seed}_$(date '+%Y%m%d_%H%M%S')"
      mkdir -p "$archive"
      [[ ! -e "$out" ]] || mv "$out" "$archive/output"
      [[ ! -e "$log" ]] || mv "$log" "$archive/evaluation.log"
    fi
    echo "[$(timestamp)] early evaluation attempt $attempt/$MAX_ATTEMPTS: $RUN_NAME seed=$seed"
    "$SCRIPT_DIR/evaluate_one.sh" "$RUN_NAME" "$seed" "$SAMPLES_PER_EPISODE" || true
    evaluation_complete "$seed" && break
  done
  evaluation_complete "$seed" || echo "[$(timestamp)] early evaluation exhausted retries: $RUN_NAME seed=$seed"
done

echo "[$(timestamp)] early evaluator finished: $RUN_NAME"

