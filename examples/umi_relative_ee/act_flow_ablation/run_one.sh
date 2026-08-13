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
TRAIN_ROOT="${UMI_TRAIN_ROOT:-/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1459_occlusion}"
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
VAL_ROOT="${UMI_VAL_ROOT:-/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation}"
VAL_FREQ="${UMI_VAL_FREQ:-10000}"
BATCH_SIZE=8
NUM_WORKERS="${UMI_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${UMI_PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${UMI_PERSISTENT_WORKERS:-true}"
HF_HUB_OFFLINE_VALUE=0
HF_HOME_VALUE=""
SAVE_CHECKPOINT="${UMI_SAVE_CHECKPOINT:-true}"
# Long research runs need durable recovery points: failures at validation,
# dependency import, or host interruption must not discard an entire 100k run.
# Callers can still override this (for example, smoke tests disable saving).
DEFAULT_SAVE_FREQ=$((STEPS < 10000 ? STEPS : 10000))
SAVE_FREQ="${UMI_SAVE_FREQ:-$DEFAULT_SAVE_FREQ}"
RUN_NAME="${VARIANT}_seed${SEED}_${STEPS}steps"
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
  --num_workers="$NUM_WORKERS"
  --prefetch_factor="$PREFETCH_FACTOR"
  --persistent_workers="$PERSISTENT_WORKERS"
  --log_freq=200
  --val_freq="$VAL_FREQ"
  --eval_freq=0
  --save_checkpoint="$SAVE_CHECKPOINT"
  --save_freq="$SAVE_FREQ"
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
  smolvla_rot6d)
    BATCH_SIZE="${UMI_SMOLVLA_BATCH_SIZE:-8}"
    HF_HUB_OFFLINE_VALUE=1
    POLICY=(
      --policy.path=lerobot/smolvla_base
      --policy.input_features=null
      --policy.chunk_size=30
      --policy.n_action_steps=30
      --policy.umi_rotation_representation=rot6d
      --policy.flow_matching_padding_mode=openpi_full_width
      --policy.train_state_proj=true
      --policy.optimizer_lr=0.0001
      --policy.scheduler_warmup_steps=1000
      --policy.scheduler_decay_steps="$STEPS"
      --policy.scheduler_decay_lr=0.0000025
    )
    ;;
  smolvla_axis_angle)
    BATCH_SIZE="${UMI_SMOLVLA_BATCH_SIZE:-8}"
    HF_HUB_OFFLINE_VALUE=1
    POLICY=(
      --policy.path=lerobot/smolvla_base
      --policy.input_features=null
      --policy.chunk_size=30
      --policy.n_action_steps=30
      --policy.umi_rotation_representation=axis_angle
      --policy.flow_matching_padding_mode=openpi_full_width
      --policy.train_state_proj=true
      --policy.optimizer_lr=0.0001
      --policy.scheduler_warmup_steps=1000
      --policy.scheduler_decay_steps="$STEPS"
      --policy.scheduler_decay_lr=0.0000025
    )
    ;;
  lingbot_va_axis_angle)
    BATCH_SIZE="${UMI_LINGBOT_BATCH_SIZE:-1}"
    HF_HUB_OFFLINE_VALUE=0
    HF_HOME_VALUE="${UMI_LINGBOT_HF_HOME:-$ARTIFACT_ROOT/hf-cache}"
    POLICY=(
      --policy.path="${UMI_LINGBOT_CHECKPOINT:-$ARTIFACT_ROOT/pretrained/lingbot_va_libero_long}"
      --policy.wan_pretrained_path="${UMI_LINGBOT_FROZEN:-$ARTIFACT_ROOT/pretrained/lingbot_va_frozen_libero_long}"
      --policy.attn_mode=flex
      --policy.obs_cam_keys=[observation.images.camera]
      --policy.camera_layout=width_concat
      --policy.image_hflip=false
      --policy.frame_chunk_size=4
      --policy.action_per_frame=4
      --policy.used_action_channel_ids=[0,1,2,3,4,5,6]
      --policy.text_encoder_device=cpu
      --policy.optimizer_lr=0.00001
      --policy.scheduler_warmup_steps=1000
      --peft.method_type=LORA
      --peft.r=8
      --peft.lora_alpha=8
    )
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

COMMON+=(--batch_size="$BATCH_SIZE")

echo "[$(date '+%F %T')] starting $RUN_NAME on host GPU" | tee "$LOG"
if [[ -n "$HF_HOME_VALUE" ]]; then
  export HF_HOME="$HF_HOME_VALUE"
  mkdir -p "$HF_HOME"
fi
if [[ "${UMI_RESUME:-false}" == "true" ]]; then
  # Canonical LeRobot resume: reload the FULL config (including the polymorphic
  # policy subclass + all hyperparameters) from the checkpoint's train_config.json,
  # then --resume=true loads optimizer / scheduler / global-step from
  # checkpoints/last/training_state and continues training to --steps. The --policy.*
  # flags are NOT re-passed here -- the saved config is the single source of truth
  # for the model, so a mismatch is impossible. Only pure-infra knobs (workers,
  # save/val freq, output dir) are overridden on top. Passing --resume without
  # --config_path raises "A config_path is expected when resuming a run."
  RESUME_CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"
  if [[ ! -f "$RESUME_CFG" ]]; then
    echo "[run_one] resume requested but $RESUME_CFG missing; aborting (checkpoint left intact)" >&2
    exit 3
  fi
  echo "[run_one] resume=true for $RUN_NAME from $RESUME_CFG (preserving checkpoint)" >&2
  HF_HUB_OFFLINE="$HF_HUB_OFFLINE_VALUE" PYTHONPATH=src uv run python \
    examples/umi_relative_ee/train_umi_relative_ee.py \
    --config_path="$RESUME_CFG" \
    --resume=true \
    --output_dir="$OUT" \
    --job_name="$RUN_NAME" \
    --num_workers="$NUM_WORKERS" \
    --prefetch_factor="$PREFETCH_FACTOR" \
    --persistent_workers="$PERSISTENT_WORKERS" \
    --log_freq=200 \
    --val_freq="$VAL_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint="$SAVE_CHECKPOINT" \
    --policy.device=cuda \
    --wandb.enable=false \
    2>&1 | tee -a "$LOG"
  echo "[$(date '+%F %T')] completed $RUN_NAME" | tee -a "$LOG"
  exit 0
fi
HF_HUB_OFFLINE="$HF_HUB_OFFLINE_VALUE" PYTHONPATH=src uv run python \
  "${COMMON[@]}" "${POLICY[@]}" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] completed $RUN_NAME" | tee -a "$LOG"
