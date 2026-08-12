#!/usr/bin/env bash
# Run the SmolVLA notation ablation and the LingBot-VA domain-adaptation candidate
# only after the existing confirmation/evaluation chain releases the host GPU.
set -uo pipefail

WAIT_FOR_SESSION="${UMI_WAIT_FOR_TMUX:-umi_arch_confirmation_eval_20260812}"
MAX_ATTEMPTS="${UMI_MAX_ATTEMPTS:-3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/training_completion.sh"
. "$SCRIPT_DIR/evaluation_completion.sh"
. "$SCRIPT_DIR/lingbot_asset_validation.sh"
REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
SUPERVISOR_LOG="$ARTIFACT_ROOT/logs/extended_candidates_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/interrupted" "$ARTIFACT_ROOT/interrupted_evaluations"
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

timestamp() { date '+%F %T'; }

smoke_lingbot() {
  local smoke_root status trainable frozen
  while tmux has-session -t umi_lingbot_prefetch_20260812 2>/dev/null; do
    echo "[$(timestamp)] waiting for verified LingBot assets"
    sleep 60
  done
  trainable="${UMI_LINGBOT_CHECKPOINT:-$ARTIFACT_ROOT/pretrained/lingbot_va_libero_long}"
  frozen="${UMI_LINGBOT_FROZEN:-$ARTIFACT_ROOT/pretrained/lingbot_va_frozen_libero_long}"
  if ! lingbot_assets_complete "$trainable" "$frozen"; then
    echo "[$(timestamp)] LingBot assets failed structural validation; skipping GPU smoke and long run"
    return 1
  fi
  smoke_root="$ARTIFACT_ROOT/smoke_supervised/$(date '+%Y%m%d_%H%M%S')_lingbot_va_axis_angle"
  echo "[$(timestamp)] LingBot host-GPU preflight: two updates, batch=1, workers=0"
  UMI_ABLATION_ROOT="$smoke_root" UMI_NUM_WORKERS=0 UMI_PERSISTENT_WORKERS=false \
    UMI_LINGBOT_CHECKPOINT="$trainable" UMI_LINGBOT_FROZEN="$frozen" \
    UMI_LINGBOT_BATCH_SIZE=1 UMI_SAVE_CHECKPOINT=false UMI_VAL_FREQ=1000 \
    "$SCRIPT_DIR/run_one.sh" lingbot_va_axis_angle 2 1000
  status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "[$(timestamp)] LingBot host-GPU preflight failed status=$status; preserving smoke artifacts"
    return 1
  fi
  echo "[$(timestamp)] LingBot host-GPU preflight passed"
}

if [[ -n "$WAIT_FOR_SESSION" ]]; then
  echo "[$(timestamp)] waiting for $WAIT_FOR_SESSION to release the host GPU"
  while tmux has-session -t "$WAIT_FOR_SESSION" 2>/dev/null; do sleep 60; done
fi

run_name() { printf '%s_seed%s_%ssteps' "$1" "$3" "$2"; }

is_complete() {
  local name
  name="$(run_name "$1" "$2" "$3")"
  training_is_complete "$ARTIFACT_ROOT" "$name" "$2"
}

archive_incomplete() {
  local name stamp archive
  name="$(run_name "$1" "$2" "$3")"
  if [[ ! -e "$ARTIFACT_ROOT/train/$name" && ! -e "$ARTIFACT_ROOT/logs/$name.log" ]]; then return; fi
  stamp="$(date '+%Y%m%d_%H%M%S')"
  archive="$ARTIFACT_ROOT/interrupted/${name}_${stamp}"
  mkdir -p "$archive"
  [[ ! -e "$ARTIFACT_ROOT/train/$name" ]] || mv "$ARTIFACT_ROOT/train/$name" "$archive/train"
  [[ ! -e "$ARTIFACT_ROOT/logs/$name.log" ]] || mv "$ARTIFACT_ROOT/logs/$name.log" "$archive/train.log"
}

train_one() {
  local variant="$1" steps="$2" seed="$3" attempt workers status
  if is_complete "$variant" "$steps" "$seed"; then
    echo "[$(timestamp)] already complete: $(run_name "$variant" "$steps" "$seed")"
    return 0
  fi
  for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    workers=4
    [[ "$attempt" -ne 2 ]] || workers=2
    [[ "$attempt" -lt 3 ]] || workers=0
    [[ "$variant" != lingbot_va_axis_angle ]] || workers=2
    archive_incomplete "$variant" "$steps" "$seed"
    echo "[$(timestamp)] train attempt $attempt/$MAX_ATTEMPTS: $(run_name "$variant" "$steps" "$seed") workers=$workers"
    UMI_NUM_WORKERS="$workers" UMI_PERSISTENT_WORKERS=false UMI_SAVE_FREQ=10000 \
      "$SCRIPT_DIR/run_one.sh" "$variant" "$steps" "$seed"
    status=$?
    recover_training_completion "$ARTIFACT_ROOT" "$(run_name "$variant" "$steps" "$seed")" "$steps" || true
    if is_complete "$variant" "$steps" "$seed"; then return 0; fi
  done
  echo "[$(timestamp)] exhausted training retries: $(run_name "$variant" "$steps" "$seed")"
  return 1
}

evaluate_one_run() {
  local variant="$1" steps="$2" seed="$3" inference_seed run_name out log attempt status reports archive
  run_name="$(run_name "$variant" "$steps" "$seed")"
  for inference_seed in 1000 2000 3000; do
    out="$ARTIFACT_ROOT/eval_common_h32/$run_name/seed$inference_seed"
    log="$ARTIFACT_ROOT/logs/eval_common_h32_${run_name}_seed${inference_seed}.log"
    if canonical_evaluation_complete "$out" "$log" "$run_name" "$inference_seed" "$steps"; then
      echo "[$(timestamp)] already evaluated: $run_name inference_seed=$inference_seed"
      continue
    fi
    for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
      if [[ -e "$out" || -e "$log" ]]; then
        archive="$ARTIFACT_ROOT/interrupted_evaluations/${run_name}_seed${inference_seed}_$(date '+%Y%m%d_%H%M%S')"
        mkdir -p "$archive"
        [[ ! -e "$out" ]] || mv "$out" "$archive/output"
        [[ ! -e "$log" ]] || mv "$log" "$archive/evaluation.log"
      fi
      "$SCRIPT_DIR/evaluate_one.sh" "$run_name" "$inference_seed" 5
      status=$?
      [[ "$status" -ne 0 ]] || break
    done
  done
}

echo "[$(timestamp)] extended candidate supervisor started; log=$SUPERVISOR_LOG"
# LingBot is a 5B pretrained domain-adaptation candidate, not an architecture-
# matched causal control. One seed is sufficient to establish feasibility and
# comparison; repeated seeds are promoted only if the first run is viable.
LINGBOT_VIABLE=false
if smoke_lingbot; then
  train_one lingbot_va_axis_angle 30000 1000 || true
  if is_complete lingbot_va_axis_angle 30000 1000; then LINGBOT_VIABLE=true; fi
fi

for variant in smolvla_rot6d smolvla_axis_angle; do
  train_one "$variant" 30000 1000 || true
  for seed in 1000 2000 3000; do train_one "$variant" 100000 "$seed" || true; done
done

for variant in smolvla_rot6d smolvla_axis_angle; do
  evaluate_one_run "$variant" 30000 1000
  for seed in 1000 2000 3000; do evaluate_one_run "$variant" 100000 "$seed"; done
done
if [[ "$LINGBOT_VIABLE" == true ]]; then
  evaluate_one_run lingbot_va_axis_angle 30000 1000
fi

cd "$REPO"
UV_CACHE_DIR=/tmp/uv-cache-umi-ablation uv run python "$SCRIPT_DIR/collect_results.py" \
  --artifact_root "$ARTIFACT_ROOT" || true
MPLCONFIGDIR=/tmp/lerobot-matplotlib UV_CACHE_DIR=/tmp/uv-cache-umi-ablation uv run python \
  "$SCRIPT_DIR/plot_results.py" --artifact_root "$ARTIFACT_ROOT" || true
echo "[$(timestamp)] extended candidate supervisor finished"
