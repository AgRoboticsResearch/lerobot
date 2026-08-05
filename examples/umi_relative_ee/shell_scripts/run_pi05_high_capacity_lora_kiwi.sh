#!/usr/bin/env bash
set -euo pipefail

# High-capacity mixed-rank π0.5 LoRA calibrated for kiwi's 16 GB RTX 5080.
# The default 96/192 ranks contain 220,916,768 trainable parameters. Always
# pass the two-step smoke gate before launching the default 100K-step run.

BATCH_SIZE="${BATCH_SIZE:-4}"
VLM_RANK="${VLM_RANK:-96}"
EXPERT_RANK="${EXPERT_RANK:-192}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-12500}"
VAL_FREQ="${VAL_FREQ:-5000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
RUN_SUFFIX="${RUN_SUFFIX:-full}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/zfei/data/sroiv2_strawberry_picking_lab_1302_occlusion}"
VALIDATION_ROOT="${VALIDATION_ROOT:-/home/zfei/data/sroiv2_strawberry_picking_lab_validation}"

RUN_NAME="pi05_high_capacity_lora_r${VLM_RANK}_expert_r${EXPERT_RANK}_masked_1302_bs${BATCH_SIZE}_${RUN_SUFFIX}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${RUN_NAME}}"
POLICY_REPO_ID="zfff/${RUN_NAME}"

LORA_TARGET='.*\.(paligemma|gemma_expert)\..*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
EXPERT_PATTERN='.*\.gemma_expert\..*'

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
  --peft.r="${VLM_RANK}" \
  --peft.lora_alpha="${VLM_RANK}" \
  --peft.rank_pattern="{'${EXPERT_PATTERN}': ${EXPERT_RANK}}" \
  --peft.alpha_pattern="{'${EXPERT_PATTERN}': ${EXPERT_RANK}}" \
  --peft.target_modules="${LORA_TARGET}" \
  --peft.full_training_modules='["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]' \
  --batch_size="${BATCH_SIZE}" \
  --num_workers=8 \
  --prefetch_factor=2 \
  --seed=1000 \
  --steps="${TRAIN_STEPS}" \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq=50 \
  --eval_freq=0 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project=lerobot
