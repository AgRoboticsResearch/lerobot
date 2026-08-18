#!/usr/bin/env bash
# π0.5-port budget-curve rows of the unified NATIVE-h30 (full-chunk)
# evaluation (§9.2.11), user-directed 2026-08-19 ("all models that support
# chunk30: full eval on h30"). Same staged 50k-spaced checkpoints as the h10
# front-run (already under train/pi05_port_curve_kiwi_seed1000/), evaluated
# with the canonical query window but NO --eval_horizon → full 30-step
# scoring. 650K/700K/1M are owned by the kiwi K2 pass (same
# pi05_port_<STEP>_h30_v2 naming) and are deliberately skipped here so every
# row has exactly one owner. Stochastic flow sampler → three inference seeds
# per checkpoint. VRAM-gated at ≥4 GiB free so it backfills alongside the
# SmolVLA trainer. Idempotent per (run, seed).
set -uo pipefail

CANON=${UMI_ABLATION_ROOT:-/mnt/data1/projects/lerobot-arch-exp}
STAGE=$CANON/train/pi05_port_curve_kiwi_seed1000/checkpoints
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
OUT_ROOT=$CANON/reeval_v2metrics/eval_common_h32
LOG_ROOT=$OUT_ROOT/logs
mkdir -p "$LOG_ROOT"

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }

# 18 curve points minus 0650000/0700000 (kiwi K2 owns 650K/700K/1M)
STEPS="0050000 0100000 0150000 0200000 0250000 0300000 0350000 0400000 0450000 0500000 0550000 0600000 0750000 0800000 0850000 0900000"

cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
for STEP in $STEPS; do
  RUN="pi05_port_${STEP}_h30_v2"
  for SEED in 1000 2000 3000; do
    OUT="$OUT_ROOT/$RUN/seed$SEED"
    LOG="$LOG_ROOT/${RUN}_seed${SEED}.log"
    if compgen -G "$OUT"/*_open_loop_metrics.json >/dev/null; then
      echo "[$(date '+%F %T')] done already: $RUN seed$SEED"
      continue
    fi
    [ -d "$STAGE/$STEP/pretrained_model" ] || { echo "[$(date '+%F %T')] missing staged ckpt $STEP — skipping"; continue; }
    local_fm=0
    while true; do
      local_fm=$(free_mib); [ "$local_fm" -ge 4000 ] && break
      echo "[$(date '+%F %T')] waiting: ${local_fm} MiB free < 4000"; sleep 120
    done
    [ -d "$OUT" ] && rm -rf "$OUT"   # stale dir from a killed attempt
    echo "[$(date '+%F %T')] eval: $RUN seed$SEED"
    if PYTHONPATH=src timeout 3600 uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
        --pretrained_path="$STAGE/$STEP/pretrained_model" \
        --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
        --samples_per_episode=5 --query_min_action_offset=-1 --query_max_action_offset=31 \
        --seed="$SEED" --device=cuda --video_backend=pyav --output_dir="$OUT" \
        >"$LOG" 2>&1; then
      echo "[$(date '+%F %T')] exit=0 $RUN seed$SEED"
    else
      echo "[$(date '+%F %T')] FAILED $RUN seed$SEED (see $LOG)"
    fi
  done
done
echo "=== π0.5-port h30 host curve sweep COMPLETE: $(find "$OUT_ROOT" -path '*pi05_port_*_h30_v2*' -name '*_open_loop_metrics.json' | wc -l) report files (48 expected) ==="
