#!/usr/bin/env bash
# Open-loop eval of the two SmolVLA notation checkpoints (rot6d vs axis_angle,
# 100k steps each) on kiwi. Writes one JSON per checkpoint into a shared folder.
set -uo pipefail
export PYTHONPATH=/home/zfei/code/lerobot-fei-v5.0-umi-unified/src
# kiwi is offline for HF Hub; SmolVLA loads its VLM backbone via transformers which
# otherwise pings the Hub (httpx crash). Force local-cache use (smolvla_base is cached).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
EVAL=/home/zfei/kiwi-ablation/examples/umi_relative_ee/eval_open_loop_dataset.py
DATASET=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
OUT=/home/zfei/kiwi-ablation/outputs/research_report/smolvla_notation_eval_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
LOG="$OUT/_eval_driver.log"
SAMPLES_PER_EP="${SAMPLES_PER_EP:-5}"

CKPTS=(
  "smolvla_rot6d_seed1000:/home/zfei/kiwi-ablation/artifacts/train/smolvla_rot6d_seed1000_100000steps/checkpoints/100000/pretrained_model"
  "smolvla_axis_angle_seed1000:/home/zfei/kiwi-ablation/artifacts/train/smolvla_axis_angle_seed1000_100000steps/checkpoints/100000/pretrained_model"
)

echo "=== SmolVLA notation open-loop eval: ${#CKPTS[@]} ckpts, samples/ep=$SAMPLES_PER_EP, OUT=$OUT ==="
i=0
for entry in "${CKPTS[@]}"; do
  i=$((i+1))
  name="${entry%%:*}"
  pm="${entry#*:}"
  json="$OUT/${name}_100000_open_loop_metrics.json"
  [ -f "$pm/config.json" ] || { echo "[$i] SKIP $name (no config.json at $pm)"; continue; }
  if [ -f "$json" ]; then echo "[$i] SKIP existing $name"; continue; fi
  echo "[$i/${#CKPTS[@]}] EVAL $name -> $(basename "$json")"
  "$PY" "$EVAL" \
    --pretrained_path "$pm" \
    --dataset_root "$DATASET" \
    --samples_per_episode "$SAMPLES_PER_EP" \
    --output_dir "$OUT" \
    >> "$LOG" 2>&1 \
    || { echo "[$i] FAILED $name (continuing)"; tail -n 15 "$LOG"; continue; }
  echo "[$i] done $name"
done

echo "=== building figures + report ==="
"$PY" /home/zfei/kiwi-ablation/examples/umi_relative_ee/make_report_figures.py "$OUT" || echo "WARN figures"
"$PY" /home/zfei/kiwi-ablation/examples/umi_relative_ee/compile_open_loop_report.py "$OUT" || echo "WARN report"
echo "ALL_DONE -> $OUT"
