#!/usr/bin/env bash
# Run eval_open_loop_dataset.py on every local UMI relative-EE checkpoint under
# outputs/train/*, on the full validation set, writing all JSONs into one folder.
# Resumable (skips checkpoints whose JSON already exists). Sequential, continue-on-error.
set -uo pipefail
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"
PY=/home/zfei/anaconda3/envs/py312/bin/python
DATASET=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
# Each run lands in its own folder: <llm-suggested slug>_<datetime> (unique, no clobbering).
# Override with OUT=... to resume into an existing folder. name_run.py prefers the `claude`
# CLI for the slug, then $RUN_NAME, then a generic fallback.
OUT="${OUT:-outputs/research_report/$("$PY" examples/umi_relative_ee/name_run.py)}"
SAMPLES_PER_EP="${SAMPLES_PER_EP:-5}"
WITH_REPORT="${WITH_REPORT:-1}"   # 1 = build figures + report in $OUT after eval
mkdir -p "$OUT"
LOG="$OUT/_eval_driver.log"

# Enumerate .../checkpoints/<step>/pretrained_model dirs (exclude 'last').
mapfile -t PMS < <(find outputs/train -type d -name pretrained_model \
  | grep '/checkpoints/' \
  | grep -v '/last/pretrained_model' \
  | sort)

echo "=== open-loop eval batch: ${#PMS[@]} checkpoints, samples_per_episode=$SAMPLES_PER_EP, all episodes ==="
i=0
for pm in "${PMS[@]}"; do
  i=$((i+1))
  step=$(basename "$(dirname "$pm")")
  model=$(basename "$(dirname "$(dirname "$(dirname "$pm")")")")
  [ -f "$pm/config.json" ] || { echo "[$i] SKIP $model/$step (no config.json)"; continue; }
  json="$OUT/${model}_${step}_open_loop_metrics.json"
  if [ -f "$json" ]; then
    echo "[$i/${#PMS[@]}] SKIP existing ${model}_${step}"
    continue
  fi
  echo "[$i/${#PMS[@]}] EVAL ${model} ${step} -> $(basename "$json")"
  "$PY" examples/umi_relative_ee/eval_open_loop_dataset.py \
    --pretrained_path "$pm" \
    --dataset_root "$DATASET" \
    --samples_per_episode "$SAMPLES_PER_EP" \
    --output_dir "$OUT" \
    >> "$LOG" 2>&1 \
    || { echo "[$i] FAILED ${model} ${step} (continuing)"; continue; }
done

if [ "$WITH_REPORT" = "1" ]; then
  echo "=== building figures + report in $OUT ==="
  "$PY" examples/umi_relative_ee/make_report_figures.py "$OUT" || echo "WARN: figures step failed"
  "$PY" examples/umi_relative_ee/compile_open_loop_report.py "$OUT" || echo "WARN: report step failed"
fi
echo "ALL_DONE -> $OUT"
