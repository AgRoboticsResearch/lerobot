#!/usr/bin/env bash
# Identity-rot6d A/B (Arm B). All three policies train on sroiv2_strawberry_picking
# _lab_1302_occlusion and validate on the 100-episode validation set, with
# --policy.umi_rot6d_identity_norm=true and val_freq=10000. Baselines (Arm A) are the
# operator's existing runs; only the identity arms are trained here.
#
# Usage:  bash run_identity_ab.sh {act|smolvla|pi05}
set -euo pipefail

POLICY="${1:?policy required: act|smolvla|pi05}"
PY=/home/zfei/anaconda3/envs/py312/bin/python
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
TRAIN_REPO=sroi/sroiv2_strawberry_picking_lab_1302_occlusion
TRAIN_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
DATA="--dataset.repo_id=$TRAIN_REPO --dataset.root=$TRAIN_ROOT \
  --validation_dataset.repo_id=$VAL_REPO --validation_dataset.root=$VAL_ROOT --val_freq=10000"
LOG="$REPO/examples/umi_relative_ee/logs/${POLICY}_identity_1302.log"
mkdir -p "$(dirname "$LOG")"

cd "$REPO"; export PYTHONPATH=src

case "$POLICY" in
  act)
    OUT=outputs/train/act_umi_identity_rot6d_1302
    nohup "$PY" examples/umi_relative_ee/train_relative_ee_processor.py $DATA \
      --policy.type=act --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
      --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 \
      --policy.repo_id=zfff/act_umi_identity_rot6d_1302 --policy.push_to_hub=false \
      --seed=1000 --save_freq=100000 --steps=2500000 --batch_size=8 --num_workers=4 \
      --log_freq=200 --eval_freq=0 \
      --output_dir=$OUT --job_name=act_umi_identity_rot6d_1302 \
      --wandb.enable=true --wandb.project=lerobot \
      > "$LOG" 2>&1 &
    ;;
  smolvla)
    OUT=outputs/train/smolvla_umi_identity_rot6d_1302
    nohup env HF_HUB_OFFLINE=1 "$PY" examples/umi_relative_ee/train_relative_ee_processor.py $DATA \
      --policy.path=lerobot/smolvla_base --policy.input_features=null \
      --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
      --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 --policy.train_state_proj=true \
      --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
      --policy.scheduler_decay_steps=30000 --policy.scheduler_decay_lr=0.0000025 \
      --policy.repo_id=zfff/smolvla_umi_identity_rot6d_1302 --policy.push_to_hub=false \
      --seed=1000 --batch_size=8 --num_workers=4 --steps=2500000 --save_freq=100000 \
      --log_freq=200 --eval_freq=0 \
      --output_dir=$OUT --job_name=smolvla_umi_identity_rot6d_1302 \
      --wandb.enable=true --wandb.project=lerobot \
      > "$LOG" 2>&1 &
    ;;
  pi05)
    OUT=outputs/train/pi05_lora_r16_umi_identity_rot6d_1302
    nohup env HF_HUB_OFFLINE=1 "$PY" examples/umi_relative_ee/train_pi05_lora.py $DATA \
      --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
      --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
      --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=false \
      --policy.chunk_size=30 --policy.n_action_steps=30 \
      --policy.optimizer_lr=0.0001 --policy.scheduler_decay_lr=0.00001 \
      --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=500000 \
      --policy.repo_id=zfff/pi05_lora_r16_umi_identity_rot6d_1302 --policy.push_to_hub=false \
      --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=16 \
      --batch_size=2 --num_workers=8 --prefetch_factor=2 \
      --seed=1000 --steps=500000 --save_freq=100000 --log_freq=50 --eval_freq=0 \
      --output_dir=$OUT --job_name=pi05_lora_r16_umi_identity_rot6d_1302 \
      --wandb.enable=true --wandb.project=lerobot \
      > "$LOG" 2>&1 &
    ;;
  *) echo "unknown policy $POLICY (use act|smolvla|pi05)"; exit 1 ;;
esac
echo "launched $POLICY identity-rot6d (1302): pid $!, log $LOG, out $OUT"
