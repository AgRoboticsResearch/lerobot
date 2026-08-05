#!/usr/bin/env bash
# OpenPI full-width flow-matching baseline: train ACT, SmolVLA, and pi0.5 for
# 100K optimizer steps on sroiv2_strawberry_picking_lab_1302_occlusion, with
# sroiv2_strawberry_picking_lab_validation as the offline validation set.
#
# Follows examples/umi_relative_ee/doc/padded_noise_strategy.md:
#   - SmolVLA/pi0.5 train with full-width (32-D) flow loss and full-width
#     inference (mask_padded_action_dims_at_inference defaults to false).
#   - No umi_rot6d_identity_norm flag (default per-dim normalization), i.e. the
#     maintained standard UMI relative-EE entrypoints.
# After each run finishes, eval_open_loop_dataset.py evaluates the final
# checkpoint on validation episodes 0-4 (5 query frames each, seed 1000),
# matching the "Latest matched comparison" setup in within_chunk_jitter_analysis.md.
#
# Usage:
#   bash examples/umi_relative_ee/shell_scripts/run_openpi_fullwidth_100k.sh {act|smolvla|pi05|all}
#
# Overridable env vars: PYTHON_BIN, OUTPUT_PREFIX (default outputs/train),
# EVAL_OUTPUT_DIR (default outputs/debug/open_loop_eval_fullwidth_100k).
set -euo pipefail

POLICY="${1:?policy required: act|smolvla|pi05|all}"
PY="${PYTHON_BIN:-/home/zfei/anaconda3/envs/py312/bin/python}"
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/train}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/debug/open_loop_eval_fullwidth_100k}"
TRAIN_REPO=sroi/sroiv2_strawberry_picking_lab_1302_occlusion
TRAIN_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
LOG_DIR="$REPO/examples/umi_relative_ee/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"
export PYTHONPATH=src

DATA_ARGS=(
  --dataset.repo_id="$TRAIN_REPO"
  --dataset.root="$TRAIN_ROOT"
  --validation_dataset.repo_id="$VAL_REPO"
  --validation_dataset.root="$VAL_ROOT"
  --val_freq=10000
)

COMMON_ARGS=(
  --policy.use_umi_relative_ee=true
  --policy.device=cuda
  --policy.chunk_size=30
  --policy.n_action_steps=30
  --policy.push_to_hub=false
  --seed=1000
  --steps=100000
  --save_freq=20000
  --eval_freq=0
  --wandb.enable=true
  --wandb.project=lerobot
)

run_act() {
  local OUT="$OUTPUT_PREFIX/act_openpi_fullwidth_1302_100k"
  local LOG="$LOG_DIR/act_openpi_fullwidth_100k.log"
  echo "[$(date '+%F %T')] ACT: training 100K steps -> $OUT (log $LOG)"
  "$PY" examples/umi_relative_ee/train_relative_ee_processor.py \
    "${DATA_ARGS[@]}" "${COMMON_ARGS[@]}" \
    --policy.type=act \
    --policy.repo_id=zfff/act_openpi_fullwidth_1302_100k \
    --batch_size=8 --num_workers=4 --log_freq=200 \
    --output_dir="$OUT" --job_name=act_openpi_fullwidth_1302_100k \
    > "$LOG" 2>&1
  eval_final "$OUT" "act_openpi_fullwidth_1302_100k"
}

run_smolvla() {
  local OUT="$OUTPUT_PREFIX/smolvla_openpi_fullwidth_1302_100k"
  local LOG="$LOG_DIR/smolvla_openpi_fullwidth_100k.log"
  echo "[$(date '+%F %T')] SmolVLA: training 100K steps -> $OUT (log $LOG)"
  env HF_HUB_OFFLINE=1 "$PY" examples/umi_relative_ee/train_relative_ee_processor.py \
    "${DATA_ARGS[@]}" "${COMMON_ARGS[@]}" \
    --policy.path=lerobot/smolvla_base --policy.input_features=null \
    --policy.train_state_proj=true \
    --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=100000 --policy.scheduler_decay_lr=0.0000025 \
    --policy.repo_id=zfff/smolvla_openpi_fullwidth_1302_100k \
    --batch_size=8 --num_workers=4 --log_freq=200 \
    --output_dir="$OUT" --job_name=smolvla_openpi_fullwidth_1302_100k \
    > "$LOG" 2>&1
  eval_final "$OUT" "smolvla_openpi_fullwidth_1302_100k"
}

run_pi05() {
  local OUT="$OUTPUT_PREFIX/pi05_lora_openpi_fullwidth_1302_100k"
  local LOG="$LOG_DIR/pi05_openpi_fullwidth_100k.log"
  echo "[$(date '+%F %T')] pi0.5: training 100K steps -> $OUT (log $LOG)"
  env HF_HUB_OFFLINE=1 "$PY" examples/umi_relative_ee/train_pi05_lora.py \
    "${DATA_ARGS[@]}" "${COMMON_ARGS[@]}" \
    --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=false \
    --policy.optimizer_lr=0.0001 --policy.scheduler_decay_lr=0.00001 \
    --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=100000 \
    --policy.repo_id=zfff/pi05_lora_openpi_fullwidth_1302_100k \
    --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=16 \
    --batch_size=2 --num_workers=8 --prefetch_factor=2 --log_freq=50 \
    --output_dir="$OUT" --job_name=pi05_lora_openpi_fullwidth_1302_100k \
    > "$LOG" 2>&1
  eval_final "$OUT" "pi05_lora_openpi_fullwidth_1302_100k"
}

# Evaluate the highest-step checkpoint under $OUT with the matched-comparison
# setup from within_chunk_jitter_analysis.md (validation episodes 0-4).
eval_final() {
  local OUT="$1"
  local TAG="$2"
  local CKPT_DIR
  CKPT_DIR=$(find "$OUT/checkpoints" -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+$' -printf '%f\n' | sort -n | tail -1)
  if [ -z "$CKPT_DIR" ]; then
    echo "[$(date '+%F %T')] WARN: no numeric checkpoint dir found under $OUT/checkpoints; skipping eval"
    return 0
  fi
  local PRETRAINED="$OUT/checkpoints/$CKPT_DIR/pretrained_model"
  echo "[$(date '+%F %T')] $TAG: evaluating $PRETRAINED (checkpoint $CKPT_DIR)"
  env HF_HUB_OFFLINE=1 "$PY" examples/umi_relative_ee/eval_open_loop_dataset.py \
    --pretrained_path="$PRETRAINED" \
    --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
    --episode_indices 0 1 2 3 4 \
    --samples_per_episode=5 \
    --seed=1000 --device=cuda \
    --output_dir="$EVAL_OUTPUT_DIR" \
    > "$LOG_DIR/eval_${TAG}.log" 2>&1
  echo "[$(date '+%F %T')] $TAG eval summary:"
  cat "$LOG_DIR/eval_${TAG}.log"
}

case "$POLICY" in
  act)     run_act ;;
  smolvla) run_smolvla ;;
  pi05)    run_pi05 ;;
  all)
    run_act
    run_smolvla
    run_pi05
    echo "[$(date '+%F %T')] All three runs finished. Eval reports in $EVAL_OUTPUT_DIR"
    ;;
  *) echo "unknown policy $POLICY (use act|smolvla|pi05|all)"; exit 1 ;;
esac
