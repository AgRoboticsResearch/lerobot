#!/usr/bin/env bash
# Canonical §9.2.9 h10 eval sweep for the Q4 state-window W=10 arm (kiwi).
# Idempotent: skips checkpoints whose metrics JSON already exists.
# NOTE: explicit --query_*_offset bounds are MANDATORY — the eval defaults
# shift with the state window (W=10 -> 9 leading negative deltas).
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
ROOT=/mnt/data/zfei/lerobot-act-flow-ablation
REPO=/home/zfei/code/lerobot-fei-v5.0-umi-unified
DATASET=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
TRAIN_RUN=act_r50_v1_vae_state10_seed1000_500000steps
EVAL_RUN=act_r50_v1_vae_state10_seed1000_0500000steps

cd "$REPO" || exit 1

for STEP in 100000 200000 300000 400000 500000; do
  CKPT="$ROOT/train/$TRAIN_RUN/checkpoints/$STEP/pretrained_model"
  OUT="$ROOT/eval/state10_h10/$EVAL_RUN/seed1000"
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
done
echo "[$(date '+%F %T')] sweep complete: $EVAL_RUN (rsync back to host reeval_v2metrics happens on the host side)"
