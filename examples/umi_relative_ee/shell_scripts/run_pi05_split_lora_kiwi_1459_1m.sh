#!/usr/bin/env bash
set -euo pipefail

# π0.5 38M split-rank LoRA (global r/alpha 16, action-expert r/alpha 32),
# masked-subspace flow, trained FROM SCRATCH on
# sroiv2_strawberry_picking_lab_1459_occlusion for 1M steps.
#
# Intended for the "kiwi" host (RTX 5080, 16GB). Same module scope and PEFT
# recipe as run_pi05_openpi_split_lora_umi.sh (the 1302 local 38M run); only
# the host paths, dataset, and step count differ. The validation set is the
# shared sroiv2_strawberry_picking_lab_validation.
#
# Usage: bash examples/umi_relative_ee/shell_scripts/run_pi05_split_lora_kiwi_1459_1m.sh [4]

BATCH_SIZE="${1:-4}"
TRAIN_STEPS=1000000
WARMUP_STEPS=1000
SAVE_FREQ=50000
VAL_FREQ=10000

PY="${PYTHON_BIN:-.venv/bin/python}"
REPO="${REPO:-/home/zfei/code/lerobot-fei-v5.0-umi-unified}"
TRAIN_REPO=sroi/sroiv2_strawberry_picking_lab_1459_occlusion
TRAIN_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_1459_occlusion
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
VAL_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_validation

RUN_NAME="pi05_openpi_split_lora_masked_1459_bs${BATCH_SIZE}_1m"
OUTPUT_DIR="outputs/train/${RUN_NAME}"
POLICY_REPO_ID="zfff/${RUN_NAME}"

LORA_TARGET='.*\.(paligemma|gemma_expert)\..*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
EXPERT_PATTERN='.*\.gemma_expert\..*'

cd "$REPO"

exec env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  "${PY}" examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id="${TRAIN_REPO}" \
  --dataset.root="${TRAIN_ROOT}" \
  --validation_dataset.repo_id="${VAL_REPO}" \
  --validation_dataset.root="${VAL_ROOT}" \
  --val_freq="${VAL_FREQ}" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true \
  --policy.flow_matching_padding_mode=masked_subspace \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.00005 \
  --policy.scheduler_decay_lr=0.000005 \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${TRAIN_STEPS}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=16 \
  --peft.rank_pattern="{'${EXPERT_PATTERN}': 32}" \
  --peft.alpha_pattern="{'${EXPERT_PATTERN}': 32}" \
  --peft.target_modules="${LORA_TARGET}" \
  --peft.full_training_modules='["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]' \
  --batch_size="${BATCH_SIZE}" \
  --num_workers=8 \
  --prefetch_factor=2 \
  --seed=1000 \
  --steps="${TRAIN_STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq=50 \
  --eval_freq=0 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --wandb.enable=true \
  --wandb.project=lerobot
