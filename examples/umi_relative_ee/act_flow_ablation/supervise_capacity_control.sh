#!/usr/bin/env bash
# Train the strict seed-1000 architecture/objective controls before evaluation.
# This supplements the already-running queue without contending for its GPU.
set -uo pipefail

TRAINING_SEED="${1:-1000}"
MAX_ATTEMPTS="${UMI_MAX_ATTEMPTS:-3}"
WAIT_FOR_SESSION="${UMI_WAIT_FOR_TMUX:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/training_completion.sh"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
SUPERVISOR_LOG="$ARTIFACT_ROOT/logs/capacity_control_supervisor_$(date '+%Y%m%d_%H%M%S').log"
VARIANTS=(act_r50_v1_vae act_r18_diffusion_lr1e5)
BUDGETS=(30000 100000)

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/interrupted"
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

timestamp() {
  date '+%F %T'
}

if [[ -n "$WAIT_FOR_SESSION" ]]; then
  echo "[$(timestamp)] waiting for tmux session $WAIT_FOR_SESSION to release the GPU"
  while tmux has-session -t "$WAIT_FOR_SESSION" 2>/dev/null; do
    sleep 60
  done
fi

run_name() {
  printf '%s_seed%s_%ssteps' "$1" "$TRAINING_SEED" "$2"
}

is_complete() {
  local variant="$1" steps="$2" name
  name="$(run_name "$variant" "$steps")"
  training_is_complete "$ARTIFACT_ROOT" "$name" "$steps"
}

archive_incomplete() {
  local variant="$1" steps="$2" name out log archive_stamp archive_dir
  name="$(run_name "$variant" "$steps")"
  out="$ARTIFACT_ROOT/train/$name"
  log="$ARTIFACT_ROOT/logs/$name.log"
  if [[ ! -e "$out" && ! -e "$log" ]]; then
    return
  fi
  archive_stamp="$(date '+%Y%m%d_%H%M%S')"
  archive_dir="$ARTIFACT_ROOT/interrupted/${name}_${archive_stamp}"
  mkdir -p "$archive_dir"
  [[ ! -e "$out" ]] || mv "$out" "$archive_dir/train"
  [[ ! -e "$log" ]] || mv "$log" "$archive_dir/train.log"
  echo "[$(timestamp)] archived incomplete $name at $archive_dir"
}

run_with_retries() {
  local variant="$1" steps="$2" attempt workers status
  if is_complete "$variant" "$steps"; then
    echo "[$(timestamp)] already complete: $(run_name "$variant" "$steps")"
    return 0
  fi
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    workers=4
    if [[ "$attempt" -eq 2 ]]; then
      workers=2
    elif [[ "$attempt" -gt 2 ]]; then
      workers=0
    fi
    archive_incomplete "$variant" "$steps"
    echo "[$(timestamp)] attempt $attempt/$MAX_ATTEMPTS: $(run_name "$variant" "$steps") workers=$workers"
    if [[ "$workers" -eq 0 ]]; then
      UMI_NUM_WORKERS=0 UMI_PERSISTENT_WORKERS=false \
        "$SCRIPT_DIR/run_one.sh" "$variant" "$steps" "$TRAINING_SEED"
    else
      UMI_NUM_WORKERS="$workers" "$SCRIPT_DIR/run_one.sh" "$variant" "$steps" "$TRAINING_SEED"
    fi
    status=$?
    recover_training_completion "$ARTIFACT_ROOT" "$(run_name "$variant" "$steps")" "$steps" || true
    if is_complete "$variant" "$steps"; then
      echo "[$(timestamp)] verified complete: $(run_name "$variant" "$steps")"
      return 0
    fi
    echo "[$(timestamp)] failed: $(run_name "$variant" "$steps") status=$status"
  done
  echo "[$(timestamp)] exhausted retries: $(run_name "$variant" "$steps"); advancing queue"
  return 1
}

echo "[$(timestamp)] additional-control supervisor started; log=$SUPERVISOR_LOG"
for variant in "${VARIANTS[@]}"; do
  for steps in "${BUDGETS[@]}"; do
    run_with_retries "$variant" "$steps" || true
  done
done
echo "[$(timestamp)] additional-control supervisor finished all queued entries"
