#!/usr/bin/env bash
# pi0.5 LoRA OpenPI full-width flow-matching, 1M steps, 1302_occlusion + validation.
# Intended for this host (RTX 4090, 24GB).
set -euo pipefail

PY=/home/zfei/anaconda3/envs/py312/bin/python
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
TRAIN_REPO=sroi/sroiv2_strawberry_picking_lab_1302_occlusion
TRAIN_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
OUT=outputs/train/pi05_lora_openpi_fullwidth_1302_1M
LOG="$REPO/examples/umi_relative_ee/logs/pi05_openpi_fullwidth_1M.log"
mkdir -p "$(dirname "$LOG")"

cd "$REPO"

env HF_HUB_OFFLINE=1 "$PY" examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id="$TRAIN_REPO" --dataset.root="$TRAIN_ROOT" \
  --validation_dataset.repo_id="$VAL_REPO" --validation_dataset.root="$VAL_ROOT" \
  --val_freq=50000 \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.0001 --policy.scheduler_decay_lr=0.00001 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=1000000 \
  --policy.repo_id=zfff/pi05_lora_openpi_fullwidth_1302_1M --policy.push_to_hub=false \
  --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=16 \
  --batch_size=2 --num_workers=8 --prefetch_factor=2 \
  --seed=1000 --steps=1000000 --save_freq=100000 --log_freq=50 --eval_freq=0 \
  --output_dir="$OUT" --job_name=pi05_lora_openpi_fullwidth_1302_1M \
  --wandb.enable=true --wandb.project=lerobot \
  > "$LOG" 2>&1
echo "[$(date '+%F %T')] pi0.5 1M training finished: $OUT (log $LOG)"
