#!/usr/bin/env bash
# Train ACT-R18 (VAE — the act_r18_vae recipe from
# examples/umi_relative_ee/act_flow_ablation/, matching the historical 3M
# production ACT and the 1M R50-VAE experiment apart from the backbone) on the
# target-conditioned strawberry dataset (observation.images.camera +
# observation.images.target_ref).
#
# The extra target_ref image is picked up automatically: the dataset factory
# maps every observation.images.* video feature to a VISUAL policy input and
# ACT treats each one as a camera on the shared ResNet backbone.
#
# Datasets are read from a local copy on /mnt/data1 (the GLOWAY USB drive is
# the archive copy; training reads video for days and must not depend on it).
set -euo pipefail

STEPS="${1:-1000000}"
SEED="${2:-1000}"
SAVE_FREQ="${3:-100000}"

REPO="${UMI_REPO:-/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified}"
ARTIFACT_ROOT="${TC_ROOT:-/mnt/data1/projects/target_condition_sb_picking}"
DATASET_ROOT="$ARTIFACT_ROOT/datasets"
TRAIN_REPO=sroi/sroiv2_strawberry_picking_lab_1459_occlusion_target_ref
TRAIN_ROOT="$DATASET_ROOT/sroiv2_strawberry_picking_lab_1459_occlusion_target_ref"
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation_target_ref
VAL_ROOT="$DATASET_ROOT/sroiv2_strawberry_picking_lab_validation_target_ref"
VAL_FREQ=10000
BATCH_SIZE=8
NUM_WORKERS=4
RUN_NAME="act_r18_vae_target_ref_seed${SEED}_${STEPS}steps"
OUT="$ARTIFACT_ROOT/train/$RUN_NAME"
LOG="$ARTIFACT_ROOT/logs/$RUN_NAME.log"

if [[ -e "$OUT" || -e "$LOG" ]] && [[ "${UMI_RESUME:-false}" != "true" ]]; then
  echo "Refusing to overwrite existing run: $OUT or $LOG" >&2
  exit 2
fi
mkdir -p "$ARTIFACT_ROOT/train" "$ARTIFACT_ROOT/logs"
cd "$REPO"

record_exit() {
  status=$?
  echo "[$(date '+%F %T')] exited $RUN_NAME status=$status" | tee -a "$LOG"
}
trap record_exit EXIT

COMMON=(
  examples/umi_relative_ee/train_umi_relative_ee.py
  --dataset.repo_id="$TRAIN_REPO"
  --dataset.root="$TRAIN_ROOT"
  --validation_dataset.repo_id="$VAL_REPO"
  --validation_dataset.root="$VAL_ROOT"
  --dataset.use_imagenet_stats=true
  --validation_dataset.use_imagenet_stats=true
  --dataset.video_backend=pyav
  --validation_dataset.video_backend=pyav
  --policy.device=cuda
  --policy.use_umi_relative_ee=true
  --policy.umi_rot6d_identity_norm=true
  --policy.push_to_hub=false
  --seed="$SEED"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --prefetch_factor=4
  --persistent_workers=true
  --log_freq=200
  --val_freq="$VAL_FREQ"
  --eval_freq=0
  --save_checkpoint=true
  --save_freq="$SAVE_FREQ"
  --output_dir="$OUT"
  --job_name="$RUN_NAME"
  --wandb.enable=false
)

# act_r18_vae: ResNet-18 (ImageNet-V1), VAE objective — same recipe as the
# historical 3M production ACT and the 1M R50-VAE run (backbone aside).
POLICY=(
  --policy.type=act
  --policy.chunk_size=30
  --policy.n_action_steps=30
  --policy.vision_backbone=resnet18
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
  --policy.use_vae=true
  --policy.optimizer_lr=0.00001
  --policy.optimizer_lr_backbone=0.00001
)

echo "[$(date '+%F %T')] starting $RUN_NAME on host GPU" | tee "$LOG"
if [[ "${UMI_RESUME:-false}" == "true" ]]; then
  # Canonical LeRobot resume: reload the FULL config from the checkpoint's
  # train_config.json, then --resume=true restores optimizer / scheduler /
  # global-step and continues to --steps. See act_flow_ablation/run_one.sh.
  RESUME_CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"
  if [[ ! -f "$RESUME_CFG" ]]; then
    echo "[run] resume requested but $RESUME_CFG missing; aborting (checkpoint left intact)" >&2
    exit 3
  fi
  echo "[run] resume=true for $RUN_NAME from $RESUME_CFG" | tee -a "$LOG"
  HF_HUB_OFFLINE=0 PYTHONPATH=src uv run python \
    examples/umi_relative_ee/train_umi_relative_ee.py \
    --config_path="$RESUME_CFG" \
    --resume=true \
    --output_dir="$OUT" \
    --job_name="$RUN_NAME" \
    --num_workers="$NUM_WORKERS" \
    --prefetch_factor=4 \
    --persistent_workers=true \
    --log_freq=200 \
    --val_freq="$VAL_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint=true \
    --policy.device=cuda \
    --wandb.enable=false \
    2>&1 | tee -a "$LOG"
  echo "[$(date '+%F %T')] completed $RUN_NAME" | tee -a "$LOG"
  exit 0
fi
HF_HUB_OFFLINE=0 PYTHONPATH=src uv run python \
  "${COMMON[@]}" "${POLICY[@]}" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] completed $RUN_NAME" | tee -a "$LOG"
