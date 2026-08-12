#!/usr/bin/env bash
# Observe the non-contending training/evaluation chain without using the GPU.
# This monitor never terminates jobs: it records telemetry and emits warnings
# for a human/agent to investigate while the bounded-retry supervisors retain
# ownership of recovery and queue advancement.
set -uo pipefail

ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
POLL_SECONDS="${UMI_MONITOR_POLL_SECONDS:-300}"
STALE_SECONDS="${UMI_MONITOR_STALE_SECONDS:-1200}"
MIN_FREE_KB="${UMI_MONITOR_MIN_FREE_KB:-52428800}"
SESSIONS=(
  umi_arch_supervisor_20260812
  umi_arch_capacity_control_20260812
  umi_arch_eval_supervisor_20260812
  umi_arch_confirmation_train_20260812
  umi_arch_confirmation_eval_20260812
  umi_arch_extended_candidates_20260812
  umi_act_l1_100k_companion_20260812
  umi_official_dp_completion_guard_20260812
  umi_act_l1_100k_completion_guard_20260812
  umi_official_transformer_dp_completion_guard_20260812
  umi_official_dp_early_eval_20260812
  umi_act_l1_100k_early_eval_20260812
  umi_official_transformer_dp_early_eval_20260812
  umi_lingbot_prefetch_20260812
)
MONITOR_LOG="$ARTIFACT_ROOT/logs/chain_monitor_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$ARTIFACT_ROOT/logs"
exec > >(tee -a "$MONITOR_LOG") 2>&1

timestamp() {
  date '+%F %T'
}

echo "[$(timestamp)] chain monitor started; poll=${POLL_SECONDS}s stale=${STALE_SECONDS}s"

while true; do
  live_sessions=()
  for session in "${SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      live_sessions+=("$session")
    fi
  done

  if [[ "${#live_sessions[@]}" -eq 0 ]]; then
    echo "[$(timestamp)] all experiment-chain sessions have exited; monitor finished"
    break
  fi

  latest_record="$(
    find "$ARTIFACT_ROOT/logs" -maxdepth 1 -type f \
      \( -name '*steps.log' -o -name 'eval_common_h32_*.log' \) \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1
  )"
  latest_log="none"
  log_age="unknown"
  log_age_label="unknown"
  if [[ -n "$latest_record" ]]; then
    latest_epoch="${latest_record%%.*}"
    latest_log="${latest_record#* }"
    log_age="$(( $(date '+%s') - latest_epoch ))"
    log_age_label="${log_age}s"
  fi

  free_kb="$(df -Pk "$ARTIFACT_ROOT" | awk 'NR == 2 {print $4}')"
  gpu="$(
    nvidia-smi \
      --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader 2>&1 | tr '\n' ';'
  )"
  active_processes="$(
    pgrep -af 'examples/umi_relative_ee/(train_umi_relative_ee.py|eval_open_loop_dataset.py)' \
      2>/dev/null | wc -l
  )"

  echo "[$(timestamp)] live=${live_sessions[*]} processes=$active_processes latest_log=$latest_log age=$log_age_label free_kb=$free_kb gpu=$gpu"
  if [[ "$log_age" != "unknown" && "$log_age" -gt "$STALE_SECONDS" ]]; then
    echo "[$(timestamp)] WARNING latest experiment log is stale by ${log_age}s"
  fi
  if [[ "$free_kb" -lt "$MIN_FREE_KB" ]]; then
    echo "[$(timestamp)] WARNING artifact filesystem free space is below ${MIN_FREE_KB} KiB"
  fi

  sleep "$POLL_SECONDS"
done
