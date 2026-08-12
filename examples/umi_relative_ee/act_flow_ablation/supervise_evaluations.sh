#!/usr/bin/env bash
# Evaluate all promoted checkpoints after the training supervisor releases the GPU.
# Failed evaluations are archived and retried without terminating the matrix.
set -uo pipefail

TRAINING_SEED="${1:-1000}"
SAMPLES_PER_EPISODE="${2:-5}"
MAX_ATTEMPTS="${UMI_MAX_ATTEMPTS:-3}"
WAIT_FOR_SESSION="${UMI_WAIT_FOR_TMUX:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/evaluation_completion.sh"
REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
SUPERVISOR_LOG="$ARTIFACT_ROOT/logs/eval_supervisor_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/interrupted_evaluations"
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

evaluation_log() {
  printf '%s/logs/eval_common_h32_%s_seed%s.log' "$ARTIFACT_ROOT" "$1" "$2"
}

evaluation_dir() {
  printf '%s/eval_common_h32/%s/seed%s' "$ARTIFACT_ROOT" "$1" "$2"
}

is_complete() {
  local run_name="$1" inference_seed="$2" out log steps
  out="$(evaluation_dir "$run_name" "$inference_seed")"
  log="$(evaluation_log "$run_name" "$inference_seed")"
  steps="${run_name##*_}"
  steps="${steps%steps}"
  canonical_evaluation_complete "$out" "$log" "$run_name" "$inference_seed" "$steps"
}

archive_incomplete() {
  local run_name="$1" inference_seed="$2" out log archive_stamp archive_dir
  out="$(evaluation_dir "$run_name" "$inference_seed")"
  log="$(evaluation_log "$run_name" "$inference_seed")"
  if [[ ! -e "$out" && ! -e "$log" ]]; then
    return
  fi
  archive_stamp="$(date '+%Y%m%d_%H%M%S')"
  archive_dir="$ARTIFACT_ROOT/interrupted_evaluations/${run_name}_seed${inference_seed}_${archive_stamp}"
  mkdir -p "$archive_dir"
  [[ ! -e "$out" ]] || mv "$out" "$archive_dir/output"
  [[ ! -e "$log" ]] || mv "$log" "$archive_dir/evaluation.log"
  echo "[$(timestamp)] archived incomplete evaluation at $archive_dir"
}

checkpoint_exists() {
  local run_name="$1"
  find "$ARTIFACT_ROOT/train/$run_name/checkpoints" -mindepth 3 -maxdepth 3 \
    -type f -name model.safetensors -print -quit 2>/dev/null | grep -q .
}

run_with_retries() {
  local run_name="$1" inference_seed="$2" attempt status
  if ! checkpoint_exists "$run_name"; then
    echo "[$(timestamp)] missing checkpoint, skipping: $run_name"
    return 1
  fi
  if is_complete "$run_name" "$inference_seed"; then
    echo "[$(timestamp)] already complete: $run_name inference_seed=$inference_seed"
    return 0
  fi
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    archive_incomplete "$run_name" "$inference_seed"
    echo "[$(timestamp)] evaluation attempt $attempt/$MAX_ATTEMPTS: $run_name inference_seed=$inference_seed"
    "$SCRIPT_DIR/evaluate_one.sh" "$run_name" "$inference_seed" "$SAMPLES_PER_EPISODE"
    status=$?
    if [[ "$status" -eq 0 ]] && is_complete "$run_name" "$inference_seed"; then
      echo "[$(timestamp)] verified evaluation: $run_name inference_seed=$inference_seed"
      return 0
    fi
    echo "[$(timestamp)] evaluation failed: $run_name inference_seed=$inference_seed status=$status"
  done
  echo "[$(timestamp)] exhausted evaluation retries: $run_name inference_seed=$inference_seed"
  return 1
}

evaluate_deterministic() {
  local variant="$1" steps="$2"
  run_with_retries "${variant}_seed${TRAINING_SEED}_${steps}steps" 1000 || true
}

evaluate_generative() {
  local variant="$1" steps="$2" inference_seed
  for inference_seed in 1000 2000 3000; do
    run_with_retries "${variant}_seed${TRAINING_SEED}_${steps}steps" "$inference_seed" || true
  done
}

echo "[$(timestamp)] evaluation supervisor started; log=$SUPERVISOR_LOG"

evaluate_deterministic act_r18_vae 100000
evaluate_deterministic act_r50_vae 100000
evaluate_deterministic act_r50_v1_vae 100000
if [[ "$TRAINING_SEED" -eq 1000 ]]; then
  evaluate_deterministic act_r50_v1_vae 30000
fi
evaluate_deterministic act_r18_l1 100000
evaluate_generative act_r18_flow_u_lr1e5 100000
evaluate_generative act_r18_diffusion_lr1e5 100000
if [[ "$TRAINING_SEED" -eq 1000 ]]; then
  evaluate_generative act_r18_diffusion_lr1e5 30000
fi
evaluate_generative diffusion_r18 100000
evaluate_generative umi_official_dp 30000
evaluate_generative umi_official_transformer_dp 30000

echo "[$(timestamp)] collecting completed results"
cd "$REPO"
UV_CACHE_DIR=/tmp/uv-cache-umi-ablation uv run python \
  "$SCRIPT_DIR/collect_results.py" --artifact_root "$ARTIFACT_ROOT"
collect_status=$?
if [[ "$collect_status" -eq 0 ]]; then
  MPLCONFIGDIR=/tmp/lerobot-matplotlib UV_CACHE_DIR=/tmp/uv-cache-umi-ablation uv run python \
    "$SCRIPT_DIR/plot_results.py" --artifact_root "$ARTIFACT_ROOT"
  plot_status=$?
else
  plot_status=1
fi
echo "[$(timestamp)] evaluation supervisor finished collect_status=$collect_status plot_status=$plot_status"
