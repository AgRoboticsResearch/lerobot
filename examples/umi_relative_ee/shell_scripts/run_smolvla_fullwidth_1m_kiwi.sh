#!/usr/bin/env bash
# SmolVLA OpenPI full-width flow-matching, 1M steps, 1302_occlusion + validation.
# Intended for the "kiwi" machine (RTX 5080, 16GB).
# Run on kiwi:  bash examples/umi_relative_ee/shell_scripts/run_smolvla_fullwidth_1m_kiwi.sh
set -euo pipefail

PY=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
REPO=/home/zfei/code/lerobot-fei-v5.0-umi-unified
TRAIN_REPO=sroi/sroiv2_strawberry_picking_lab_1302_occlusion
TRAIN_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_1302_occlusion
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
VAL_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
OUT=outputs/train/smolvla_openpi_fullwidth_1302_1M
LOG="$REPO/examples/umi_relative_ee/logs/smolvla_openpi_fullwidth_1M.log"
mkdir -p "$(dirname "$LOG")"

cd "$REPO"

env HF_HUB_OFFLINE=1 "$PY" examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id="$TRAIN_REPO" --dataset.root="$TRAIN_ROOT" \
  --validation_dataset.repo_id="$VAL_REPO" --validation_dataset.root="$VAL_ROOT" \
  --val_freq=50000 \
  --policy.path=lerobot/smolvla_base --policy.input_features=null \
  --policy.use_umi_relative_ee=true \
  --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=1000000 --policy.scheduler_decay_lr=0.0000025 \
  --policy.repo_id=zfff/smolvla_openpi_fullwidth_1302_1M --policy.push_to_hub=false \
  --seed=1000 --steps=1000000 --save_freq=100000 --log_freq=200 --eval_freq=0 \
  --batch_size=8 --num_workers=4 \
  --output_dir="$OUT" --job_name=smolvla_openpi_fullwidth_1302_1M \
  --wandb.enable=true --wandb.project=lerobot \
  > "$LOG" 2>&1
echo "[$(date '+%F %T')] SmolVLA 1M training finished: $OUT (log $LOG)"
