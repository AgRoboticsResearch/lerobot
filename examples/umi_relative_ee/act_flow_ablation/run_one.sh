#!/usr/bin/env bash
# Run one matched ACT / flow-matching / Diffusion Policy experiment.
# Training is intentionally foregrounded; launch/parallelization belongs to the caller.
set -euo pipefail

VARIANT="${1:?usage: run_one.sh VARIANT STEPS [SEED]}"
STEPS="${2:?usage: run_one.sh VARIANT STEPS [SEED]}"
SEED="${3:-1000}"

REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
TRAIN_REPO=sroi/sroiv2_strawberry_picking_lab_1459_occlusion
TRAIN_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1459_occlusion
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_FREQ="${UMI_VAL_FREQ:-10000}"
BATCH_SIZE=8
NUM_WORKERS="${UMI_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${UMI_PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${UMI_PERSISTENT_WORKERS:-true}"
HF_HUB_OFFLINE_VALUE=0
SAVE_CHECKPOINT="${UMI_SAVE_CHECKPOINT:-true}"
RUN_NAME="${VARIANT}_seed${SEED}_${STEPS}steps"
OUT="$ARTIFACT_ROOT/train/$RUN_NAME"
LOG="$ARTIFACT_ROOT/logs/$RUN_NAME.log"

if [[ -e "$OUT" || -e "$LOG" ]]; then
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
  --num_workers="$NUM_WORKERS"
  --prefetch_factor="$PREFETCH_FACTOR"
  --persistent_workers="$PERSISTENT_WORKERS"
  --log_freq=200
  --val_freq="$VAL_FREQ"
  --eval_freq=0
  --save_checkpoint="$SAVE_CHECKPOINT"
  --save_freq="$STEPS"
  --output_dir="$OUT"
  --job_name="$RUN_NAME"
  --wandb.enable=false
)

POLICY=(
  --policy.type=act
  --policy.chunk_size=30
  --policy.n_action_steps=30
  --policy.vision_backbone=resnet18
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
  --policy.optimizer_lr=0.00001
  --policy.optimizer_lr_backbone=0.00001
)

case "$VARIANT" in
  act_r18_vae)
    POLICY+=(--policy.use_vae=true)
    ;;
  act_r34_vae)
    POLICY+=(
      --policy.use_vae=true
      --policy.vision_backbone=resnet34
      --policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1
    )
    ;;
  act_r50_vae)
    POLICY+=(
      --policy.use_vae=true
      --policy.vision_backbone=resnet50
      --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V2
    )
    ;;
  act_r50_v1_vae)
    POLICY+=(
      --policy.use_vae=true
      --policy.vision_backbone=resnet50
      --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1
    )
    ;;
  act_r50_large)
    POLICY+=(
      --policy.use_vae=true
      --policy.vision_backbone=resnet50
      --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V2
      --policy.dim_model=768
      --policy.n_heads=12
      --policy.dim_feedforward=4096
      --policy.n_encoder_layers=6
      --policy.n_decoder_layers=3
      --policy.latent_dim=64
    )
    ;;
  act_r18_l1)
    POLICY+=(--policy.use_vae=false --policy.action_objective=l1)
    ;;
  act_r18_flow_u_lr1e5)
    POLICY+=(
      --policy.use_vae=false
      --policy.action_objective=flow_matching
      --policy.flow_time_sampling_beta_alpha=1.0
      --policy.flow_time_sampling_beta_beta=1.0
    )
    ;;
  act_r18_flow_u_lr1e4)
    POLICY+=(
      --policy.use_vae=false
      --policy.action_objective=flow_matching
      --policy.flow_time_sampling_beta_alpha=1.0
      --policy.flow_time_sampling_beta_beta=1.0
      --policy.optimizer_lr=0.0001
      --policy.optimizer_lr_backbone=0.0001
    )
    ;;
  act_r18_flow_beta_lr1e4)
    POLICY+=(
      --policy.use_vae=false
      --policy.action_objective=flow_matching
      --policy.flow_time_sampling_beta_alpha=1.5
      --policy.flow_time_sampling_beta_beta=1.0
      --policy.optimizer_lr=0.0001
      --policy.optimizer_lr_backbone=0.0001
    )
    ;;
  act_r18_diffusion_lr1e5)
    POLICY+=(
      --policy.use_vae=false
      --policy.action_objective=diffusion
      --policy.diffusion_num_train_timesteps=100
      --policy.diffusion_num_inference_steps=10
      --policy.diffusion_beta_schedule=squaredcos_cap_v2
      --policy.diffusion_clip_sample=true
    )
    ;;
  diffusion_r18)
    POLICY=(
      --policy.type=diffusion
      --policy.n_obs_steps=1
      --policy.horizon=32
      --policy.chunk_size=30
      --policy.n_action_steps=30
      --policy.drop_n_last_frames=0
      --policy.vision_backbone=resnet18
      --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
      --policy.down_dims=[256,512,1024]
      --policy.noise_scheduler_type=DDIM
      --policy.num_train_timesteps=100
      --policy.num_inference_steps=10
      --policy.do_mask_loss_for_padding=true
      --policy.optimizer_lr=0.0001
    )
    ;;
  umi_official_dp)
    BATCH_SIZE="${UMI_OFFICIAL_BATCH_SIZE:-64}"
    HF_HUB_OFFLINE_VALUE=1
    POLICY=(
      --policy.type=umi_official_dp
      --policy.n_obs_steps=1
      --policy.horizon=32
      --policy.chunk_size=30
      --policy.n_action_steps=30
      --policy.drop_n_last_frames=0
      --policy.vision_model_name=vit_base_patch16_clip_224.openai
      --policy.vision_pretrained=true
      --policy.image_size=[224,224]
      --policy.crop_ratio=0.95
      --policy.down_dims=[256,512,1024]
      --policy.noise_scheduler_type=DDIM
      --policy.num_train_timesteps=50
      --policy.num_inference_steps=16
      --policy.input_perturbation=0.1
      --policy.use_ema=true
      --policy.optimizer_lr=0.0003
      --policy.scheduler_warmup_steps=2000
    )
    ;;
  umi_official_transformer_dp)
    BATCH_SIZE="${UMI_OFFICIAL_BATCH_SIZE:-64}"
    HF_HUB_OFFLINE_VALUE=1
    POLICY=(
      --policy.type=umi_official_transformer_dp
      --policy.n_obs_steps=1
      --policy.horizon=32
      --policy.chunk_size=30
      --policy.n_action_steps=30
      --policy.drop_n_last_frames=0
      --policy.vision_model_name=vit_base_patch16_clip_224.openai
      --policy.vision_pretrained=true
      --policy.image_size=[224,224]
      --policy.crop_ratio=0.95
      --policy.random_rotation_degrees=5.0
      --policy.transformer_num_layers=7
      --policy.transformer_num_heads=8
      --policy.transformer_dim=768
      --policy.transformer_dropout=0.1
      --policy.noise_scheduler_type=DDIM
      --policy.num_train_timesteps=50
      --policy.num_inference_steps=16
      --policy.input_perturbation=0.1
      --policy.use_ema=true
      --policy.optimizer_lr=0.0003
      --policy.optimizer_backbone_lr=0.00003
      --policy.scheduler_warmup_steps=2000
    )
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

COMMON+=(--batch_size="$BATCH_SIZE")

echo "[$(date '+%F %T')] starting $RUN_NAME on host GPU" | tee "$LOG"
HF_HUB_OFFLINE="$HF_HUB_OFFLINE_VALUE" PYTHONPATH=src uv run python \
  "${COMMON[@]}" "${POLICY[@]}" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] completed $RUN_NAME" | tee -a "$LOG"
