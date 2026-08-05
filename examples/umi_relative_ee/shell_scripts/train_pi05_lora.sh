#!/usr/bin/env bash
set -euo pipefail

# Override these from the environment when the dataset/model output moves.
PYTHON_BIN="${PYTHON_BIN:-/home/zfei/anaconda3/envs/py312/bin/python}"
DATASET_REPO_ID="${DATASET_REPO_ID:-sroi/sroiv2_strawberry_picking_lab_1000onesb}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb}"
VALIDATION_DATASET_REPO_ID="${VALIDATION_DATASET_REPO_ID:-sroi/sroiv2_strawberry_picking_lab_validation}"
VALIDATION_DATASET_ROOT="${VALIDATION_DATASET_ROOT:-/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation}"
VAL_FREQ="${VAL_FREQ:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/pi05_lora_umi_relative_ee}"
POLICY_REPO_ID="${POLICY_REPO_ID:-zfff/pi05_lora_umi_relative_ee}"

"${PYTHON_BIN}" examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --validation_dataset.repo_id="${VALIDATION_DATASET_REPO_ID}" \
  --validation_dataset.root="${VALIDATION_DATASET_ROOT}" \
  --val_freq="${VAL_FREQ}" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_decay_lr=0.00001 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=50000 \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=16 \
  --batch_size=2 \
  --num_workers=8 \
  --prefetch_factor=2 \
  --steps=50000 \
  --save_freq=5000 \
  --log_freq=50 \
  --eval_freq=0 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=pi05_lora_umi_relative_ee \
  --wandb.enable=true
