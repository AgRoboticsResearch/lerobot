#!/usr/bin/env bash
# SmolVLA rotation-notation A/B (rot6d vs axis_angle) on kiwi (RTX 5080, PyTorch).
# Single-seed, 100k each, mirroring run_one.sh's smolvla_* configs but using
# kiwi's main-repo .venv + the synced ablation src (PYTHONPATH) since kiwi has no
# uv-managed ablation env. Base lerobot/smolvla_base is cached on kiwi.
set -uo pipefail
REPO=/home/zfei/kiwi-ablation
VENV=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
ART=/home/zfei/kiwi-ablation/artifacts
TRAIN_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_1459_occlusion
VAL_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
STEPS=100000
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$ART/train" "$ART/logs"

run() {
  local rep="$1"
  local name="smolvla_${rep}_seed1000_${STEPS}steps"
  local out="$ART/train/$name"
  local log="$ART/logs/$name.log"
  [ -e "$out" ] && { echo "[$(date +%T)] skip existing $name"; return; }
  echo "[$(date +%T)] START $name"
  cd "$REPO" && PYTHONPATH=src "$VENV" examples/umi_relative_ee/train_umi_relative_ee.py \
    --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1459_occlusion --dataset.root="$TRAIN_ROOT" \
    --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation --validation_dataset.root="$VAL_ROOT" \
    --dataset.use_imagenet_stats=true --validation_dataset.use_imagenet_stats=true \
    --dataset.video_backend=pyav --validation_dataset.video_backend=pyav \
    --policy.device=cuda --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
    --policy.push_to_hub=false --seed=1000 --steps="$STEPS" \
    --num_workers=4 --prefetch_factor=4 --persistent_workers=true \
    --log_freq=200 --val_freq=10000 --eval_freq=0 --save_checkpoint=true --save_freq=20000 \
    --output_dir="$out" --job_name="$name" --wandb.enable=false --batch_size=8 \
    --policy.path=lerobot/smolvla_base --policy.input_features=null \
    --policy.chunk_size=30 --policy.n_action_steps=30 \
    --policy.umi_rotation_representation="$rep" \
    --policy.flow_matching_padding_mode=openpi_full_width --policy.train_state_proj=true \
    --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps="$STEPS" --policy.scheduler_decay_lr=0.0000025 \
    > "$log" 2>&1
  echo "[$(date +%T)] DONE $name"
}

run rot6d
run axis_angle
echo "[$(date +%T)] === SmolVLA notation A/B finished ==="
