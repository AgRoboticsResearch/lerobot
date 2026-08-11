#!/usr/bin/env bash
# Evaluate one completed run on fixed validation queries using the host GPU.
set -euo pipefail

RUN_NAME="${1:?usage: evaluate_one.sh RUN_NAME [EVAL_SEED] [SAMPLES_PER_EPISODE]}"
EVAL_SEED="${2:-1000}"
SAMPLES_PER_EPISODE="${3:-5}"

REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
RUN_DIR="$ARTIFACT_ROOT/train/$RUN_NAME"
CHECKPOINTS_DIR="$RUN_DIR/checkpoints"

if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
  echo "No checkpoint directory found: $CHECKPOINTS_DIR" >&2
  exit 2
fi
mapfile -t CHECKPOINTS < <(find "$CHECKPOINTS_DIR" -mindepth 2 -maxdepth 2 -type d -name pretrained_model | sort)
if [[ "${#CHECKPOINTS[@]}" -ne 1 ]]; then
  echo "Expected exactly one saved checkpoint under $CHECKPOINTS_DIR, found ${#CHECKPOINTS[@]}" >&2
  exit 2
fi

OUT="$ARTIFACT_ROOT/eval/$RUN_NAME/seed$EVAL_SEED"
LOG="$ARTIFACT_ROOT/logs/eval_${RUN_NAME}_seed${EVAL_SEED}.log"
if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "Refusing to overwrite existing evaluation: $OUT or $LOG" >&2
  exit 2
fi
mkdir -p "$(dirname -- "$OUT")" "$ARTIFACT_ROOT/logs"
cd "$REPO"

echo "[$(date '+%F %T')] evaluating $RUN_NAME with inference seed $EVAL_SEED" | tee "$LOG"
PYTHONPATH=src uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
  --pretrained_path="${CHECKPOINTS[0]}" \
  --dataset_root="$VAL_ROOT" \
  --repo_id="$VAL_REPO" \
  --samples_per_episode="$SAMPLES_PER_EPISODE" \
  --seed="$EVAL_SEED" \
  --device=cuda \
  --video_backend=pyav \
  --output_dir="$OUT" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] completed evaluation $RUN_NAME seed $EVAL_SEED" | tee -a "$LOG"
