#!/usr/bin/env bash
# Re-evaluate the lerobot-port pi0.5 700K checkpoint with chunks TRUNCATED to
# the first 10 steps (--eval_horizon 10) for an equal-footing comparison with
# the official-openpi arms (horizon 10). Same 100-episode / 500-query protocol,
# three inference seeds (1000/2000/3000) mirroring the original eval_common_h32.
set -uo pipefail
export PYTHONPATH=/home/zfei/code/lerobot-fei-v5.0-umi-unified/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
EVAL=/home/zfei/kiwi-ablation/examples/umi_relative_ee/eval_open_loop_dataset.py
CKPT=/home/zfei/code/lerobot-fei-v5.0-umi-unified/outputs/train/pi05_openpi_split_lora_masked_1459_bs4_1m/checkpoints/0700000/pretrained_model
DATASET=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
OUT=/home/zfei/kiwi-ablation/artifacts/eval_common_h32/pi05_port_700k_h10
mkdir -p "$OUT"
LOG="$OUT/_driver.log"

for seed in 1000 2000 3000; do
  d="$OUT/seed$seed"; mkdir -p "$d"
  json="$d/pi05_port_700k_h10_open_loop_metrics.json"
  [ -f "$json" ] && { echo "SKIP seed$seed"; continue; }
  echo "=== EVAL seed $seed (horizon 10) ==="
  "$PY" "$EVAL" \
    --pretrained_path "$CKPT" \
    --dataset_root "$DATASET" \
    --samples_per_episode 5 \
    --seed "$seed" \
    --eval_horizon 10 \
    --output_dir "$d" >> "$LOG" 2>&1 \
    && mv "$d"/*_open_loop_metrics.json "$json" 2>/dev/null || { echo "FAILED seed$seed"; tail -n 12 "$LOG"; }
done
echo "ALL_DONE -> $OUT"
