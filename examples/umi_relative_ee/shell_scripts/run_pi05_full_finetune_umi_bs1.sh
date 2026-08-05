#!/usr/bin/env bash
set -euo pipefail

# Full-parameter π0.5 fine-tuning memory trial for the 10-D UMI task.
# This is expected to exceed a single 24 GB RTX 4090 once gradients and AdamW
# states are materialized. Environment overrides make a no-checkpoint smoke
# test possible before committing disk space to a full run.

TRAIN_STEPS="${TRAIN_STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-100000}"
VAL_FREQ="${VAL_FREQ:-5000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
RUN_SUFFIX="${RUN_SUFFIX:-full}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEFAULT_PYTHON_BIN=".venv/bin/python"
if [[ -x /home/zfei/anaconda3/envs/py312/bin/python ]]; then
  DEFAULT_PYTHON_BIN="/home/zfei/anaconda3/envs/py312/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

DATA_ROOT="${DATA_ROOT:-/home/zfei/data/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion}"
VALIDATION_ROOT="${VALIDATION_ROOT:-/home/zfei/data/lerobot/sroiv2_strawberry_picking_lab_validation}"
RUN_NAME="pi05_full_finetune_masked_1302_bs1_${RUN_SUFFIX}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${RUN_NAME}}"
POLICY_REPO_ID="zfff/${RUN_NAME}"

exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" HF_HUB_OFFLINE=1 \
  "${PYTHON_BIN}" examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root="${DATA_ROOT}" \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root="${VALIDATION_ROOT}" \
  --val_freq="${VAL_FREQ}" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true \
  --policy.flow_matching_padding_mode=masked_subspace \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.00005 \
  --policy.scheduler_decay_lr=0.000005 \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${TRAIN_STEPS}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --num_workers=4 \
  --prefetch_factor=2 \
  --seed=1000 \
  --steps="${TRAIN_STEPS}" \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq=1 \
  --eval_freq=0 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project=lerobot
