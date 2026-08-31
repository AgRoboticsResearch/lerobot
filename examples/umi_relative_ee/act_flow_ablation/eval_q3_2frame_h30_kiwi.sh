#!/usr/bin/env bash
# Q3 two-frame ACT R50-VAE (ImageNet-V1): canonical native-h30 evaluation.
#
# Runs the five retained 100k..500k checkpoints serially so the sweep can
# safely share kiwi with the Q4 trainer. Native h30 means full chunk scoring:
# deliberately do NOT pass --eval_horizon. Existing completed rows are skipped.
set -uo pipefail

export PYTHONPATH=/home/zfei/code/lerobot-fei-v5.0-umi-unified/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
EVAL=/home/zfei/code/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/eval_open_loop_dataset.py
CKPTS=/mnt/data/zfei/lerobot-act-flow-ablation/train/act_r50_v1_vae_2frame_seed1000_1000000steps/checkpoints
DATASET=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
OUT_ROOT=/mnt/data/zfei/lerobot-act-flow-ablation/eval/q3_2frame_h30

mkdir -p "$OUT_ROOT/logs"
for STEP in 0100000 0200000 0300000 0400000 0500000; do
  RUN="act_r50_v1_vae_2frame_seed1000_${STEP}steps"
  OUT="$OUT_ROOT/$RUN/seed1000"
  LOG="$OUT_ROOT/logs/${RUN}.log"
  if compgen -G "$OUT/*_open_loop_metrics.json" >/dev/null; then
    echo "[$(date '+%F %T')] done already: $RUN"
    continue
  fi
  CKPT="$CKPTS/$STEP/pretrained_model"
  if [ ! -d "$CKPT" ]; then
    echo "[$(date '+%F %T')] MISSING CKPT: $CKPT"
    continue
  fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] eval native-h30: $RUN"
  if timeout 7200 "$PY" "$EVAL" \
      --pretrained_path "$CKPT" \
      --dataset_root "$DATASET" \
      --repo_id sroi/sroiv2_strawberry_picking_lab_validation \
      --samples_per_episode 5 \
      --query_min_action_offset -1 --query_max_action_offset 31 \
      --seed 1000 --device cuda --video_backend pyav \
      --output_dir "$OUT" >"$LOG" 2>&1; then
    echo "[$(date '+%F %T')] exit=0 $RUN"
  else
    echo "[$(date '+%F %T')] FAILED $RUN (see $LOG)"
  fi
done

echo "[$(date '+%F %T')] Q3 native-h30 sweep finished"
