#!/usr/bin/env bash
set -euo pipefail

# JAX-vs-PyTorch stack A/B: lerobot-port π0.5 LoRA with the OFFICIAL-openpi
# recipe of pi05_lora_sroi_rot6d_h30, changing only the training stack.
#
# Matched to official openpi (host, /home/zfei/codes/openpi):
#   - rot6d 10D UMI relative actions, chunk/execute 30 (= openpi action_horizon 30)
#   - split-rank LoRA: PaliGemma r/α 16 + gemma_expert r/α 32
#     (== openpi gemma_2b_lora + gemma_300m_lora), same module scope
#   - pi05_base init, frozen vision tower, EMA off (openpi LoRA: ema_decay=None)
#   - bs16, 20 000 steps (= 320k samples), save every 5k, keep 10k+20k
#   - optimizer: lr 2.5e-5 peak, betas (0.9, 0.95), eps 1e-8, wd 1e-10,
#     grad-clip 1.0  (openpi AdamW defaults; wd overridden from the port's 0.01)
#   - schedule: 1k warmup, cosine over 30 000 steps to 2.5e-6, stopping at
#     20k steps MID-COSINE exactly like openpi (scheduler_auto_scale=false;
#     the port default would squeeze the decay into 20k and end at the floor)
#   - flow loss FULL-WIDTH over the padded action dim
#     (flow_matching_padding_mode=openpi_full_width == openpi behavior;
#     the SmolVLA A/B showed masked/full-width tie, matched here anyway)
#   - Beta(1.5, 1) flow-time sampling, 224px, quantile (q01/q99) norm —
#     already pi05-port defaults identical to openpi
#
# Remaining stack-native differences (recorded in RESEARCH_REPORT.md §9.2.5):
#   state construction (port: derived 20D two-pose state; openpi: current-frame
#   10D state), norm-stats sample (full 140k train frames vs 30k sampled),
#   bf16 numerics JAX vs PyTorch, v3.0 vs resharded-v2.1 dataset layout.
#
# Usage: bash examples/umi_relative_ee/act_flow_ablation/run_pi05_port_openpi_matched_h30.sh
# Artifacts land under /mnt/data1/projects/lerobot-arch-exp (§8 incident 12).

BATCH_SIZE=16
TRAIN_STEPS=20000
WARMUP_STEPS=1000
DECAY_STEPS=30000
SAVE_FREQ=5000
VAL_FREQ=10000

PY="${PYTHON_BIN:-uv run python}"
REPO="${REPO:-/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified}"
TRAIN_REPO=sroi/sroiv2_strawberry_picking_lab_1459_occlusion
TRAIN_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1459_occlusion
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation

RUN_NAME="pi05_port_openpi_args_rot6d_h30_bs${BATCH_SIZE}_20k"
OUTPUT_DIR="/mnt/data1/projects/lerobot-arch-exp/outputs/train/${RUN_NAME}"

LORA_TARGET='.*\.(paligemma|gemma_expert)\..*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
EXPERT_PATTERN='.*\.gemma_expert\..*'

cd "$REPO"

exec env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ${PY} examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id="${TRAIN_REPO}" \
  --dataset.root="${TRAIN_ROOT}" \
  --validation_dataset.repo_id="${VAL_REPO}" \
  --validation_dataset.root="${VAL_ROOT}" \
  --val_freq="${VAL_FREQ}" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true \
  --policy.flow_matching_padding_mode=openpi_full_width \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.000025 \
  --policy.optimizer_weight_decay=0.0000000001 \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${DECAY_STEPS}" \
  --policy.scheduler_decay_lr=0.0000025 \
  --policy.scheduler_auto_scale=false \
  --policy.repo_id="zfff/${RUN_NAME}" \
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
  --wandb.enable=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}"
