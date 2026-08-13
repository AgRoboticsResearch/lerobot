#!/usr/bin/env bash
# Train independent seed replications of the promoted 100k controlled matrix.
# Optionally waits for an earlier tmux session, preserving strict single-GPU use.
set -uo pipefail

STEPS="${1:-100000}"
shift || true
if [[ "$#" -gt 0 ]]; then
  TRAINING_SEEDS=("$@")
else
  TRAINING_SEEDS=(2000 3000)
fi
MAX_ATTEMPTS="${UMI_MAX_ATTEMPTS:-3}"
WAIT_FOR_SESSION="${UMI_WAIT_FOR_TMUX:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/training_completion.sh"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
SUPERVISOR_LOG="$ARTIFACT_ROOT/logs/confirmation_train_supervisor_$(date '+%Y%m%d_%H%M%S').log"
VARIANTS=(
  act_r18_vae
  act_r50_vae
  act_r50_v1_vae
  act_r18_l1
  act_r18_flow_u_lr1e5
  act_r18_diffusion_lr1e5
  diffusion_r18
)

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
  printf '%s_seed%s_%ssteps' "$1" "$2" "$STEPS"
}

is_complete() {
  local variant="$1" seed="$2" name
  name="$(run_name "$variant" "$seed")"
  training_is_complete "$ARTIFACT_ROOT" "$name" "$STEPS"
}

is_running() {
  # True if any process (main or its PyAV dataloader workers, which share the
  # command line) is currently training this exact run. Lets the supervisor
  # defer to an in-flight companion queue instead of archiving its in-progress
  # directory and starting a conflicting run. Fixed-string match is safe because
  # run names are unique (they encode the seed).
  local name="$1"
  pgrep -fa train_umi_relative_ee.py 2>/dev/null | grep -qF -- "job_name=$name"
}

wait_for_companion() {
  # If a companion queue is already training this run, wait for it to finish so
  # we inherit its result rather than overwriting it. A 12h valve prevents an
  # indefinitely stuck companion from deadlocking the whole evaluation chain.
  local name="$1" waited=0
  if is_running "$name"; then
    echo "[$(timestamp)] in-flight elsewhere; waiting for companion: $name"
    while is_running "$name"; do
      sleep 60
      waited=$((waited + 60))
      if (( waited >= 43200 )); then
        echo "[$(timestamp)] companion wait exceeded 12h; taking over: $name"
        return 0
      fi
    done
    echo "[$(timestamp)] companion finished; re-checking: $name"
  fi
}

archive_incomplete() {
  local variant="$1" seed="$2" name out log archive_stamp archive_dir
  name="$(run_name "$variant" "$seed")"
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
  local variant="$1" seed="$2" attempt workers status name
  name="$(run_name "$variant" "$seed")"
  wait_for_companion "$name"
  if is_complete "$variant" "$seed"; then
    echo "[$(timestamp)] already complete: $(run_name "$variant" "$seed")"
    return 0
  fi
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    workers=4
    if [[ "$attempt" -eq 2 ]]; then
      workers=2
    elif [[ "$attempt" -gt 2 ]]; then
      workers=0
    fi
    archive_incomplete "$variant" "$seed"
    echo "[$(timestamp)] attempt $attempt/$MAX_ATTEMPTS: $(run_name "$variant" "$seed") workers=$workers"
    if [[ "$workers" -eq 0 ]]; then
      UMI_NUM_WORKERS=0 UMI_PERSISTENT_WORKERS=false \
        "$SCRIPT_DIR/run_one.sh" "$variant" "$STEPS" "$seed"
    else
      UMI_NUM_WORKERS="$workers" "$SCRIPT_DIR/run_one.sh" "$variant" "$STEPS" "$seed"
    fi
    status=$?
    recover_training_completion "$ARTIFACT_ROOT" "$(run_name "$variant" "$seed")" "$STEPS" || true
    if is_complete "$variant" "$seed"; then
      echo "[$(timestamp)] verified complete: $(run_name "$variant" "$seed")"
      return 0
    fi
    echo "[$(timestamp)] failed: $(run_name "$variant" "$seed") status=$status"
  done
  echo "[$(timestamp)] exhausted retries: $(run_name "$variant" "$seed"); advancing queue"
  return 1
}

echo "[$(timestamp)] confirmation supervisor started; log=$SUPERVISOR_LOG"
for seed in "${TRAINING_SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    run_with_retries "$variant" "$seed" || true
  done
done
echo "[$(timestamp)] confirmation supervisor finished all queued entries"
