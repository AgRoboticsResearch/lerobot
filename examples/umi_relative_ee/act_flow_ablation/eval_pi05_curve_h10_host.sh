#!/usr/bin/env bash
# Host-side execution of the π0.5-port budget-curve rows of the unified
# horizon-10 sweep (§9.2.9) — front-run of kiwi K1, user-directed 2026-08-18
# ("full curve 50k–900K"). The kiwi run pi05_openpi_split_lora_masked_1459_bs4_1m
# (§9.2.2) holds real 50k-spaced checkpoints 0050000–0900000; they are copied
# (weights only; no training happens on the host) into the canonical host
# tree and evaluated with flags identical to eval_unified_h10_sweep.sh, so
# the reports pass compile_unified_h10.py's protocol assertions. 650K/700K
# are re-scored here under the canonical query window because their earlier
# t+10 evals (§9.2.4) predate it; 1M lands with K1 when the trainer exits.
# Stochastic flow sampler → three inference seeds per checkpoint.
set -uo pipefail

CANON=${UMI_ABLATION_ROOT:-/mnt/data1/projects/lerobot-arch-exp}
SRC=zfei@10.98.19.22:/home/zfei/code/lerobot-fei-v5.0-umi-unified/outputs/train/pi05_openpi_split_lora_masked_1459_bs4_1m/checkpoints
STAGE=$CANON/train/pi05_port_curve_kiwi_seed1000/checkpoints
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
OUT_ROOT=$CANON/reeval_v2metrics/eval_unified_h10
mkdir -p "$STAGE" "$OUT_ROOT/logs"

STEPS="0050000 0100000 0150000 0200000 0250000 0300000 0350000 0400000 0450000 0500000 0550000 0600000 0650000 0700000 0750000 0800000 0850000 0900000"

for STEP in $STEPS; do
  RUN="pi05_port_seed1000_${STEP}steps"
  # idempotent: skip only when ALL three seed reports exist
  DONE=0
  for SEED in 1000 2000 3000; do
    compgen -G "$OUT_ROOT/$RUN/seed$SEED"/*_open_loop_metrics.json >/dev/null || DONE=1
  done
  if [ "$DONE" -eq 0 ]; then
    echo "[$(date '+%F %T')] all seeds done already: $RUN"
    continue
  fi
  if [ ! -d "$STAGE/$STEP/pretrained_model" ]; then
    echo "[$(date '+%F %T')] scp $STEP"
    scp -P 2203 -q -r "$SRC/$STEP" "$STAGE/" || { echo "[$(date '+%F %T')] scp FAILED $STEP"; continue; }
  fi
  for SEED in 1000 2000 3000; do
    OUT="$OUT_ROOT/$RUN/seed$SEED"
    LOG="$OUT_ROOT/logs/${RUN}_seed${SEED}.log"
    if compgen -G "$OUT"/*_open_loop_metrics.json >/dev/null; then
      echo "[$(date '+%F %T')] done already: $RUN seed$SEED"
      continue
    fi
    [ -d "$OUT" ] && rm -rf "$OUT"   # stale dir from a killed attempt
    echo "[$(date '+%F %T')] eval: $RUN seed$SEED"
    if PYTHONPATH=src timeout 3600 uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
        --pretrained_path="$STAGE/$STEP/pretrained_model" \
        --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
        --samples_per_episode=5 --query_min_action_offset=-1 --query_max_action_offset=31 \
        --eval_horizon=10 \
        --seed="$SEED" --device=cuda --video_backend=pyav --output_dir="$OUT" \
        >"$LOG" 2>&1; then
      echo "[$(date '+%F %T')] exit=0 $RUN seed$SEED"
    else
      echo "[$(date '+%F %T')] FAILED $RUN seed$SEED (see $LOG)"
    fi
  done
done

echo "=== π0.5-port host curve sweep COMPLETE: $(find "$OUT_ROOT" -path '*pi05_port_seed1000*' -name '*_open_loop_metrics.json' | wc -l) report files ==="
