#!/usr/bin/env bash
# Native-h30 (§9.2.11 full-chunk) eval sweep for the Q4 state-window W=4 arm
# (host) — feeds Fig. 15 (plot_unified_h30.py budget curves).
# Idempotent: skips checkpoints whose metrics JSON already exists.
# NO --eval_horizon flag: the compiler requires eval_horizon=None (full
# 30-step chunk scoring). Explicit --query_*_offset bounds are MANDATORY —
# the eval defaults shift with the state window (W=4 -> 3 leading negative
# deltas).
set -uo pipefail

ROOT=/mnt/data1/projects/lerobot-arch-exp
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
DATASET=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
TRAIN_RUN=act_r50_v1_vae_state4_seed1000_500000steps

cd "$REPO" || exit 1

for STEP in 100000 200000 300000 400000 500000; do
  CKPT="$ROOT/train/$TRAIN_RUN/checkpoints/$STEP/pretrained_model"
  # Per-checkpoint run dir (7-digit zero-padded) — one dir per step, else the
  # compilers collapse the sweep into a single summary row.
  RUN_DIR="$ROOT/reeval_v2metrics/eval_common_h32/act_r50_v1_vae_state4_seed1000_$(printf '%07d' "$STEP")steps"
  OUT="$RUN_DIR/seed1000"
  if [[ ! -d "$CKPT" ]]; then
    echo "SKIP step=$STEP (checkpoint missing)"
    continue
  fi
  if compgen -G "$OUT/${TRAIN_RUN}_${STEP}_open_loop_metrics.json" > /dev/null; then
    echo "DONE step=$STEP (metrics exist)"
    continue
  fi
  echo "[$(date '+%F %T')] eval step=$STEP (native h30)"
  timeout 7200 uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
    --pretrained_path="$CKPT" \
    --dataset_root="$DATASET" \
    --repo_id=local/sroiv2_strawberry_picking_lab_validation \
    --samples_per_episode=5 \
    --query_min_action_offset=-1 \
    --query_max_action_offset=31 \
    --seed=1000 \
    --device=cuda \
    --video_backend=pyav \
    --output_dir="$OUT" || { echo "FAILED step=$STEP"; exit 1; }
done
echo "[$(date '+%F %T')] native-h30 sweep complete: $TRAIN_RUN"
