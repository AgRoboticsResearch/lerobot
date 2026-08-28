#!/usr/bin/env bash
# Canonical §9.2.9 h10 eval sweep for the Q4 state-window W=5 arm (host).
# Idempotent: skips checkpoints whose metrics JSON already exists.
# NOTE: explicit --query_*_offset bounds are MANDATORY — the eval defaults
# shift with the state window (W=5 -> 4 leading negative deltas).
set -uo pipefail

ROOT=/mnt/data1/projects/lerobot-arch-exp
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
DATASET=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
TRAIN_RUN=act_r50_v1_vae_state5_seed1000_500000steps
EVAL_RUN=act_r50_v1_vae_state5_seed1000_0500000steps

cd "$REPO" || exit 1

for STEP in 100000 200000 300000 400000 500000; do
  CKPT="$ROOT/train/$TRAIN_RUN/checkpoints/$STEP/pretrained_model"
  OUT="$ROOT/reeval_v2metrics/eval_unified_h10/$EVAL_RUN/seed1000"
  if [[ ! -d "$CKPT" ]]; then
    echo "SKIP step=$STEP (checkpoint missing)"
    continue
  fi
  if compgen -G "$OUT/${TRAIN_RUN}_${STEP}_open_loop_metrics.json" > /dev/null; then
    echo "DONE step=$STEP (metrics exist)"
    continue
  fi
  echo "[$(date '+%F %T')] eval step=$STEP"
  timeout 7200 uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
    --pretrained_path="$CKPT" \
    --dataset_root="$DATASET" \
    --repo_id=local/sroiv2_strawberry_picking_lab_validation \
    --samples_per_episode=5 \
    --query_min_action_offset=-1 \
    --query_max_action_offset=31 \
    --eval_horizon=10 \
    --seed=1000 \
    --device=cuda \
    --video_backend=pyav \
    --output_dir="$OUT" || { echo "FAILED step=$STEP"; exit 1; }
  # Mirror into the §9.2.13 jerk tree so compile_physical_jerk picks the row up.
  JERK="$ROOT/reeval_v2metrics/eval_unified_h10_jerk/$EVAL_RUN/seed1000"
  mkdir -p "$JERK"
  cp -r "$ROOT/reeval_v2metrics/eval_unified_h10/$EVAL_RUN/seed1000"/. "$JERK"/
done
echo "[$(date '+%F %T')] sweep complete: $EVAL_RUN"
