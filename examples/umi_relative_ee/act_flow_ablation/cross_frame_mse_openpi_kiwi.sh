#!/usr/bin/env bash
# Native-horizon adjacent-frame MSE for the five official OpenPI/JAX rows.
set -uo pipefail

OPENPI=/mnt/data/zfei/openpi
PY=$OPENPI/.venv/bin/python
EVAL=/mnt/data/zfei/openpi_eval_open_loop.py
CKPTS=/mnt/data/zfei/openpi_checkpoints
ARCHIVE=/mnt/data/zfei/lerobot-act-flow-ablation/archive/report_ckpts
DATA=/mnt/data/zfei/sroiv2_strawberry_validation_rotvec
ROOT=/mnt/data/zfei/lerobot-act-flow-ablation/eval/cross_frame_mse
OUT_ROOT=$ROOT/openpi_tree

mkdir -p "$OUT_ROOT" "$ROOT/openpi_logs" "$ROOT/openpi_status"

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
}

run_one() {
  local run=$1 config=$2 checkpoint=$3 action_horizon=$4 horizons=$5
  local output=$OUT_ROOT/$run/seed1000/${run}_cross_frame_mse_metrics.json
  local log=$ROOT/openpi_logs/${run}.log
  if [ -f "$output" ]; then
    printf 'done\t%s\n' "$run" >"$ROOT/openpi_status/${run}.status"
    return 0
  fi
  while [ "$(du -sk "$checkpoint" 2>/dev/null | awk '{print $1 + 0}')" -lt 5500000 ]; do
    sleep 60
  done
  while [ "$(free_mib)" -lt 10000 ]; do sleep 30; done
  mkdir -p "$(dirname "$output")"
  printf '[%s] start %s free=%s MiB\n' "$(date '+%F %T')" "$run" "$(free_mib)" >>"$ROOT/openpi_driver.log"
  if (cd "$OPENPI" && \
      HF_LEROBOT_HOME=/mnt/data/zfei HF_HUB_OFFLINE=1 \
      PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      timeout 21600 "$PY" "$EVAL" \
        --config-name "$config" --checkpoint "$checkpoint" \
        --dataset_root "$DATA" --samples_per_episode 5 \
        --action_horizon "$action_horizon" \
        --query_min_action_offset -1 --query_max_action_offset 31 \
        --cross_frame_mse_eval --cross_frame_mse_horizons "$horizons" \
        --seed 1000 --output "$output") >"$log" 2>&1; then
    printf 'done\t%s\n' "$run" >"$ROOT/openpi_status/${run}.status"
    printf '[%s] done %s\n' "$(date '+%F %T')" "$run" >>"$ROOT/openpi_driver.log"
  else
    rc=$?
    printf 'failed:%s\t%s\n' "$rc" "$run" >"$ROOT/openpi_status/${run}.status"
    printf '[%s] FAILED rc=%s %s\n' "$(date '+%F %T')" "$rc" "$run" >>"$ROOT/openpi_driver.log"
  fi
}

# Do not overlap a JAX model with the torch sweep. Derive the expected count
# from the manifest so newly added canonical families cannot be skipped.
expected_torch=$(awk 'END { print NR - 1 }' "$ROOT/manifest.tsv")
while [ "$(grep -l '^done' "$ROOT"/status/*.status 2>/dev/null | wc -l)" -lt "$expected_torch" ]; do
  sleep 60
done
while [ ! -x "$PY" ] || [ ! -f "$OPENPI/src/openpi/policies/policy.py" ]; do
  sleep 60
done

run_one pi05_lora_sroi_rot6d_seed1000_0020000steps \
  pi05_lora_sroi_rot6d "$ARCHIVE/pi05_lora_sroi_rot6d/run1/19999" 10 10
run_one pi05_lora_sroi_rotvec_seed1000_0020000steps \
  pi05_lora_sroi_rotvec "$ARCHIVE/pi05_lora_sroi_rotvec/run1/19999" 10 10
run_one pi05_lora_sroi_rot6d_h30_seed1000_0020000steps \
  pi05_lora_sroi_rot6d_h30 "$ARCHIVE/pi05_lora_sroi_rot6d_h30/run1/19999" 30 10,30
run_one pi05_openpi1m_seed1000_0100001steps \
  pi05_lora_sroi_rot6d_1m "$CKPTS/pi05_lora_sroi_rot6d_1m/run1/100000" 10 10
run_one pi05_lora_sroi_rot6d_h30_bs4_1m_seed1000_0100000steps \
  pi05_lora_sroi_rot6d_h30_bs4_1m \
  "$CKPTS/pi05_lora_sroi_rot6d_h30_bs4_1m/run1/100000" 30 10,30
