#!/usr/bin/env bash
# Reliably finish the official-DP candidates and remaining 100k stage-two runs.
# Incomplete attempts are preserved under ARTIFACT_ROOT/interrupted, and a failed
# child cannot terminate the rest of the queue.
set -uo pipefail

STAGE2_STEPS="${1:-100000}"
OFFICIAL_STEPS="${2:-30000}"
TRAINING_SEED="${3:-1000}"
MAX_ATTEMPTS="${UMI_MAX_ATTEMPTS:-3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/training_completion.sh"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
SUPERVISOR_LOG="$ARTIFACT_ROOT/logs/supervisor_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/interrupted"
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

timestamp() {
  date '+%F %T'
}

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
  local variant="$1" steps="$2" workers="$3" batch="${4:-}" attempt attempt_workers status
  if is_complete "$variant" "$steps"; then
    echo "[$(timestamp)] already complete: $(run_name "$variant" "$steps")"
    return 0
  fi
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    attempt_workers="$workers"
    if [[ "$attempt" -eq 2 && "$workers" -gt 2 ]]; then
      attempt_workers=2
    elif [[ "$attempt" -gt 2 && "$workers" -gt 0 ]]; then
      attempt_workers=0
    fi
    archive_incomplete "$variant" "$steps"
    echo "[$(timestamp)] attempt $attempt/$MAX_ATTEMPTS: $(run_name "$variant" "$steps") workers=$attempt_workers batch=${batch:-default}"
    if [[ -n "$batch" ]]; then
      UMI_NUM_WORKERS="$attempt_workers" UMI_PERSISTENT_WORKERS=false UMI_OFFICIAL_BATCH_SIZE="$batch" \
        "$SCRIPT_DIR/run_one.sh" "$variant" "$steps" "$TRAINING_SEED"
    elif [[ "$attempt_workers" -eq 0 ]]; then
      UMI_NUM_WORKERS=0 UMI_PERSISTENT_WORKERS=false \
        "$SCRIPT_DIR/run_one.sh" "$variant" "$steps" "$TRAINING_SEED"
    else
      UMI_NUM_WORKERS="$attempt_workers" \
        "$SCRIPT_DIR/run_one.sh" "$variant" "$steps" "$TRAINING_SEED"
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

select_official_batch() {
  local variant="$1" batch smoke_root status
  SELECTED_BATCH=""
  for batch in 64 32 16 8; do
    smoke_root="$ARTIFACT_ROOT/smoke_supervised/$(date '+%Y%m%d_%H%M%S')_${variant}_batch${batch}"
    echo "[$(timestamp)] smoke: $variant batch=$batch workers=0"
    UMI_ABLATION_ROOT="$smoke_root" UMI_NUM_WORKERS=0 UMI_PERSISTENT_WORKERS=false \
      UMI_OFFICIAL_BATCH_SIZE="$batch" UMI_SAVE_CHECKPOINT=false UMI_VAL_FREQ=1000 \
      "$SCRIPT_DIR/run_one.sh" "$variant" 2 "$TRAINING_SEED"
    status=$?
    if [[ "$status" -eq 0 ]]; then
      SELECTED_BATCH="$batch"
      return 0
    fi
    echo "[$(timestamp)] smoke failed: $variant batch=$batch status=$status"
  done
  return 1
}

echo "[$(timestamp)] supervisor started; log=$SUPERVISOR_LOG"

for variant in umi_official_dp umi_official_transformer_dp; do
  select_official_batch "$variant"
  if [[ -z "$SELECTED_BATCH" ]]; then
    echo "[$(timestamp)] no safe batch for $variant; advancing queue"
    continue
  fi
  # Two PyAV loader processes overlap decoding with GPU work. Four workers
  # previously segfaulted on this dataset, while zero workers spent ~0.69 s/step
  # decoding synchronously and left the GPU idle between ~0.19 s updates.
  UMI_SAVE_FREQ=10000 run_with_retries "$variant" "$OFFICIAL_STEPS" 2 "$SELECTED_BATCH" || true
done

for variant in act_r18_l1 act_r18_flow_u_lr1e5 diffusion_r18; do
  run_with_retries "$variant" "$STAGE2_STEPS" 4 || true
done

echo "[$(timestamp)] supervisor finished all queued entries"
