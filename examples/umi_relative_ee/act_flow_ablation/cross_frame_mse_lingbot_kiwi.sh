#!/usr/bin/env bash
# Canonical h10 adjacent-frame direct-MSE sweep for the three LingBot-VA
# checkpoints. LingBot emits 16 actions, so h30 is structurally unsupported.
set -euo pipefail

REPO=/home/zfei/code/lerobot-fei-v5.0-umi-unified
PY=$REPO/.venv/bin/python
EVAL=$REPO/examples/umi_relative_ee/eval_open_loop_dataset.py
ROOT=/mnt/data/zfei/lerobot-act-flow-ablation/eval/lingbot_cross_frame
CHECKPOINT_ROOT=$ROOT/checkpoints
VAL_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
VAL_REPO=local/sroiv2_strawberry_picking_lab_validation

mkdir -p "$ROOT/results" "$ROOT/logs" "$ROOT/status"
cd "$REPO"

for step in 050000 100000 200000; do
  numeric_step=$((10#$step))
  run="lingbot_va_axis_angle_seed1000_${numeric_step}steps"
  checkpoint="$CHECKPOINT_ROOT/$step"
  output="$ROOT/results/$run/seed1000"
  log="$ROOT/logs/$run.log"
  status="$ROOT/status/$run.status"
  if compgen -G "$output/*_cross_frame_mse_metrics.json" >/dev/null; then
    printf 'done\t%s\n' "$run" >"$status"
    continue
  fi
  mkdir -p "$output"
  printf '[%s] start %s\n' "$(date '+%F %T')" "$run"
  if HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH=src timeout 21600 "$PY" "$EVAL" \
      --pretrained_path="$checkpoint" \
      --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
      --samples_per_episode=5 \
      --query_min_action_offset=-1 --query_max_action_offset=31 \
      --cross_frame_mse_eval --cross_frame_mse_horizons=10 \
      --seed=1000 --device=cuda --video_backend=pyav --output_dir="$output" \
      >"$log" 2>&1; then
    printf 'done\t%s\n' "$run" >"$status"
    printf '[%s] done %s\n' "$(date '+%F %T')" "$run"
  else
    rc=$?
    printf 'failed:%s\t%s\n' "$rc" "$run" >"$status"
    printf '[%s] FAILED rc=%s %s\n' "$(date '+%F %T')" "$rc" "$run"
    exit "$rc"
  fi
done
