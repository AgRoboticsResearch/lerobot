#!/usr/bin/env bash
# Evaluate an explicitly selected recovery checkpoint in a non-canonical
# namespace. This is for checkpoint-selection analysis; primary tables continue
# to use evaluate_one.sh and the numerically greatest completed checkpoint.
set -euo pipefail

RUN_NAME="${1:?usage: evaluate_checkpoint.sh RUN_NAME CHECKPOINT_STEP [EVAL_SEED] [SAMPLES_PER_EPISODE]}"
CHECKPOINT_STEP="${2:?usage: evaluate_checkpoint.sh RUN_NAME CHECKPOINT_STEP [EVAL_SEED] [SAMPLES_PER_EPISODE]}"
EVAL_SEED="${3:-1000}"
SAMPLES_PER_EPISODE="${4:-5}"

REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
printf -v PADDED_STEP '%06d' "$CHECKPOINT_STEP"
CHECKPOINT="$ARTIFACT_ROOT/train/$RUN_NAME/checkpoints/$CHECKPOINT_STEP/pretrained_model"
[[ -d "$CHECKPOINT" ]] || CHECKPOINT="$ARTIFACT_ROOT/train/$RUN_NAME/checkpoints/$PADDED_STEP/pretrained_model"
if [[ ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo "No materialized checkpoint at step $CHECKPOINT_STEP for $RUN_NAME" >&2
  exit 2
fi

NAMESPACE="eval_checkpoint_h32"
OUT="$ARTIFACT_ROOT/$NAMESPACE/$RUN_NAME/step$PADDED_STEP/seed$EVAL_SEED"
LOG="$ARTIFACT_ROOT/logs/${NAMESPACE}_${RUN_NAME}_step${PADDED_STEP}_seed${EVAL_SEED}.log"
if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "Refusing to overwrite existing checkpoint evaluation: $OUT or $LOG" >&2
  exit 2
fi
mkdir -p "$(dirname -- "$OUT")" "$ARTIFACT_ROOT/logs"
cd "$REPO"

record_exit() {
  status=$?
  echo "[$(date '+%F %T')] checkpoint evaluation exited $RUN_NAME step $CHECKPOINT_STEP seed $EVAL_SEED status=$status" | tee -a "$LOG"
}
trap record_exit EXIT

echo "[$(date '+%F %T')] evaluating checkpoint $RUN_NAME step $CHECKPOINT_STEP seed $EVAL_SEED" | tee "$LOG"
PYTHONPATH=src uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
  --pretrained_path="$CHECKPOINT" \
  --dataset_root="$VAL_ROOT" \
  --repo_id="$VAL_REPO" \
  --samples_per_episode="$SAMPLES_PER_EPISODE" \
  --query_min_action_offset=-1 \
  --query_max_action_offset=31 \
  --seed="$EVAL_SEED" \
  --device=cuda \
  --video_backend=pyav \
  --output_dir="$OUT" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] completed checkpoint evaluation $RUN_NAME step $CHECKPOINT_STEP seed $EVAL_SEED" | tee -a "$LOG"
