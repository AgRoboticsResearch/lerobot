#!/usr/bin/env bash
# Front-run the multi-seed confirmation phase using otherwise-idle GPU memory.
#
# The confirmation supervisor (supervise_confirmation_training.sh) is gated behind
# capacity-control + evaluation and will not start for hours. Its completion
# predicate skips any run whose durable checkpoint already exists, so a companion
# run that finishes first is pure head-start and never causes a duplicate. This
# launcher mirrors supervise_remaining.sh's recovery contract: bounded retry with
# a 4 -> 2 -> 0 worker ladder, interrupted-attempt preservation under
# ARTIFACT_ROOT/interrupted, and the conservative durable-completion check that
# also recovers a run whose shell wrapper returned nonzero after the trainer
# already wrote its final checkpoint.
#
# Usage:
#   run_companion.sh <seed> <variant1> [variant2 ...]
# Example:
#   run_companion.sh 2000 act_r50_vae act_r50_v1_vae
set -uo pipefail

SEED="${1:?usage: run_companion.sh <seed> <variant...>}"
shift
VARIANTS=("$@")
[[ ${#VARIANTS[@]} -gt 0 ]] || { echo "no variants given" >&2; exit 2; }

MAX_ATTEMPTS="${UMI_MAX_ATTEMPTS:-3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/training_completion.sh"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
SUPERVISOR_LOG="$ARTIFACT_ROOT/logs/companion_seed${SEED}_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/interrupted"
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

timestamp() { date '+%F %T'; }
run_name() { printf '%s_seed%s_100000steps' "$1" "$SEED"; }

is_complete() {
  training_is_complete "$ARTIFACT_ROOT" "$(run_name "$1")" 100000
}

checkpoint_resumable() {
  # True iff the run has a non-corrupt `last` checkpoint: the symlink resolves,
  # and both train_config.json and a non-empty model.safetensors exist beneath it.
  # This guards against resuming from a half-written checkpoint (e.g. a 0-byte
  # model.safetensors produced when a write was interrupted by a disk fault or
  # OOM kill). Used on EVERY retry attempt so a transient attempt-1 failure
  # continues from the intact checkpoint instead of archiving it.
  local out="$1" last step cfg weights
  last="$out/checkpoints/last"
  [[ -L "$last" ]] || return 1
  step="$(readlink "$last")"
  cfg="$out/checkpoints/$step/pretrained_model/train_config.json"
  weights="$out/checkpoints/$step/pretrained_model/model.safetensors"
  [[ -f "$cfg" && -s "$weights" ]]
}

archive_incomplete() {
  local name out log archive_dir
  name="$(run_name "$1")"
  out="$ARTIFACT_ROOT/train/$name"
  log="$ARTIFACT_ROOT/logs/$name.log"
  [[ -e "$out" || -e "$log" ]] || return 0
  archive_dir="$ARTIFACT_ROOT/interrupted/${name}_$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "$archive_dir"
  [[ ! -e "$out" ]] || mv "$out" "$archive_dir/train"
  [[ ! -e "$log" ]] || mv "$log" "$archive_dir/train.log"
  echo "[$(timestamp)] archived incomplete $name at $archive_dir"
}

# ACT runs are stable with four PyAV workers on this dataset; the temporal-U-Net
# Diffusion Policy segfaulted at four workers, so it gets two. The ladder still
# degrades 4 -> 2 -> 0 across retry attempts as insurance.
preferred_workers() {
  case "$1" in
    diffusion_r18|umi_official*) echo 2 ;;
    *) echo 4 ;;
  esac
}

MAX_CONCURRENT="${UMI_MAX_CONCURRENT:-4}"
MIN_FREE_VRAM="${UMI_MIN_FREE_VRAM:-4000}"

wait_for_slot() {
  # Bound total concurrent training jobs and preserve a VRAM margin so several
  # waiting companion queues cannot oversubscribe the single GPU. As soon as any
  # in-flight job finishes, the next waiter starts, closing idle gaps without
  # manual intervention. We count DISTINCT job_name values because PyAV dataloader
  # workers inherit the parent's full command line and would otherwise be
  # over-counted (one job appears as ~1 + num_workers processes).
  local n free
  while :; do
    n="$(pgrep -fa train_umi_relative_ee.py 2>/dev/null | grep -oE 'job_name=[a-z][a-z0-9_]*_seed[0-9]+_[0-9]+steps' | sort -u | wc -l | tr -d ' ')"
    n="${n:-0}"
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1)"
    free="${free:-0}"
    if (( n < MAX_CONCURRENT )) && (( free >= MIN_FREE_VRAM )); then
      return 0
    fi
    sleep 30
  done
}

is_running() {
  # True if any process is currently training this exact run (main or its
  # dataloader workers, which share the command line). Fixed-string match is
  # safe because run names are unique (they encode the seed).
  local name="$1"
  pgrep -fa train_umi_relative_ee.py 2>/dev/null | grep -qF -- "job_name=$name"
}

wait_for_companion() {
  # Defer to any other queue already training this run so we inherit its result
  # instead of archiving its in-progress directory and starting a conflicting
  # run. A 10h valve prevents a stuck peer from deadlocking us.
  local name="$1" waited=0
  if is_running "$name"; then
    echo "[$(timestamp)] in-flight elsewhere; waiting for peer: $name"
    while is_running "$name"; do
      sleep 60
      waited=$((waited + 60))
      if (( waited >= 36000 )); then
        echo "[$(timestamp)] peer wait exceeded 10h; taking over: $name"
        return 0
      fi
    done
    echo "[$(timestamp)] peer finished; re-checking: $name"
  fi
}

run_with_retries() {
  local variant="$1" workers attempt attempt_workers status
  workers="$(preferred_workers "$variant")"
  if is_complete "$variant"; then
    echo "[$(timestamp)] already complete: $(run_name "$variant"); skipping"
    return 0
  fi
  wait_for_companion "$(run_name "$variant")"
  if is_complete "$variant"; then
    echo "[$(timestamp)] peer completed; skipping: $(run_name "$variant")"
    return 0
  fi
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    attempt_workers="$workers"
    if [[ "$attempt" -eq 2 && "$workers" -gt 2 ]]; then
      attempt_workers=2
    elif [[ "$attempt" -gt 2 && "$workers" -gt 0 ]]; then
      attempt_workers=0
    fi
    wait_for_slot
    local out="$ARTIFACT_ROOT/train/$(run_name "$variant")"
    local resume_env=()
    if checkpoint_resumable "$out"; then
      echo "[$(timestamp)] attempt $attempt: RESUME from intact checkpoint: $(run_name "$variant") workers=$attempt_workers"
      resume_env=(UMI_RESUME=true)
    else
      archive_incomplete "$variant"
      echo "[$(timestamp)] attempt $attempt/$MAX_ATTEMPTS: $(run_name "$variant") workers=$attempt_workers"
    fi
    env "${resume_env[@]}" UMI_NUM_WORKERS="$attempt_workers" UMI_SAVE_FREQ=10000 \
      "$SCRIPT_DIR/run_one.sh" "$variant" 100000 "$SEED"
    status=$?
    recover_training_completion "$ARTIFACT_ROOT" "$(run_name "$variant")" 100000 || true
    if is_complete "$variant"; then
      echo "[$(timestamp)] verified complete: $(run_name "$variant")"
      return 0
    fi
    echo "[$(timestamp)] failed: $(run_name "$variant") status=$status"
  done
  echo "[$(timestamp)] exhausted retries: $(run_name "$variant"); advancing queue"
  return 1
}

echo "[$(timestamp)] companion supervisor started seed=$SEED variants=${VARIANTS[*]} log=$SUPERVISOR_LOG"
for variant in "${VARIANTS[@]}"; do
  run_with_retries "$variant" || true
done
echo "[$(timestamp)] companion supervisor finished seed=$SEED variants=${VARIANTS[*]}"
