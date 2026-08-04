#!/usr/bin/env bash
set -euo pipefail

# Broad π0.5 LoRA for the 10-D UMI relative-EE task.
# Usage: bash examples/umi_relative_ee/run_pi05_broad_lora_umi.sh [2|4]
# Batch 4 uses half as many optimizer steps as batch 2 so both runs see the
# same number of training examples.

BATCH_SIZE="${1:-2}"
case "${BATCH_SIZE}" in
  2)
    TRAIN_STEPS=150000
    WARMUP_STEPS=2000
    SAVE_FREQ=25000
    VAL_FREQ=10000
    ;;
  4)
    TRAIN_STEPS=75000
    WARMUP_STEPS=1000
    SAVE_FREQ=12500
    VAL_FREQ=5000
    ;;
  *)
    echo "batch size must be 2 or 4" >&2
    exit 2
    ;;
esac

RUN_NAME="pi05_broad_lora_masked_1302_bs${BATCH_SIZE}"
OUTPUT_DIR="outputs/train/${RUN_NAME}"
POLICY_REPO_ID="zfff/${RUN_NAME}"

# Adapt q/k/v/o and the FFN in both the PaliGemma language model and the Gemma
# action expert. The four action/time projections are fully trained and saved
# with the adapter instead of receiving another low-rank approximation.
LORA_TARGET='.*\.(paligemma|gemma_expert)\..*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'

exec env HF_HUB_OFFLINE=1 /home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
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
