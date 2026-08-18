#!/usr/bin/env bash
# Complete evaluation of the SmolVLA rot6d 1M budget run (§9.2.3 extension,
# user-directed 2026-08-18): every 100k checkpoint (0100000..1000000) under
# BOTH protocols, mirroring the R50-V1 1M treatment (§9.2.8):
#
#  1. h30 native curve — full 30-step scoring (NO --eval_horizon), run names
#     smolvla_rot6d_1m_seed1000_<STEP>steps under eval_common_h32/, outside the
#     unified t+10 tree. Directly comparable with the §9.2.7/§9.2.8 curves.
#  2. unified t+10 rows — canonical §9.2.9 protocol (--eval_horizon 10, bounds
#     [-1,31], 500 queries), run names smolvla_rot6d_seed1000_0<STEP>steps
#     (7-digit steps, so they never collide with the 6-digit notation-run name
#     smolvla_rot6d_seed1000_100000steps) under eval_unified_h10/.
#
# SmolVLA samples stochastically (flow) → inference seeds 1000/2000/3000 on
# every checkpoint, per the unified protocol. Idempotent per (run, seed).
# VRAM-gated at ≥4 GiB free so it can coexist with any stray GPU job.
set -uo pipefail

CANON=/mnt/data1/projects/lerobot-arch-exp
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
CKPTS=$CANON/train/smolvla_rot6d_seed1000_1000000steps/checkpoints
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
LOG_ROOT=$CANON/reeval_v2metrics/logs
mkdir -p "$LOG_ROOT"

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }

run_eval() {  # out_root run step seed horizon
  local OUT_ROOT="$1" RUN="$2" STEP="$3" SEED="$4" H="$5"
  local OUT="$OUT_ROOT/$RUN/seed$SEED"
  local LOG="$LOG_ROOT/${RUN}_seed${SEED}_h${H}.log"
  if compgen -G "$OUT"/*_open_loop_metrics.json >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] skip $RUN seed$SEED h$H (already evaluated)"
    return 0
  fi
  local fm
  while true; do
    fm=$(free_mib); [ "$fm" -ge 4000 ] && break
    echo "[$(date '+%F %T')] waiting: ${fm} MiB free < 4000"; sleep 120
  done
  [ -d "$OUT" ] && rm -rf "$OUT"   # stale dir from a killed attempt
  echo "[$(date '+%F %T')] eval: $RUN seed$SEED h$H"
  local args=()
  [ "$H" != "30" ] && args+=(--eval_horizon="$H")
  PYTHONPATH=src timeout 3600 uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
      --pretrained_path="$CKPTS/$STEP/pretrained_model" \
      --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
      --samples_per_episode=5 --query_min_action_offset=-1 --query_max_action_offset=31 \
      "${args[@]}" \
      --seed="$SEED" --device=cuda --video_backend=pyav --output_dir="$OUT" \
      >"$LOG" 2>&1 \
    && echo "[$(date '+%F %T')] exit=0 $RUN seed$SEED h$H" \
    || echo "[$(date '+%F %T')] FAILED $RUN seed$SEED h$H (see $LOG)"
}

cd "$REPO"
for STEP in 0100000 0200000 0300000 0400000 0500000 0600000 0700000 0800000 0900000 1000000; do
  [ -d "$CKPTS/$STEP/pretrained_model" ] || { echo "[$(date '+%F %T')] missing $STEP — skipping"; continue; }
  for SEED in 1000 2000 3000; do
    # 1) h30 native curve (outside the unified tree)
    run_eval "$CANON/reeval_v2metrics/eval_common_h32" "smolvla_rot6d_1m_seed1000_${STEP}steps" "$STEP" "$SEED" 30
    # 2) unified t+10 row
    run_eval "$CANON/reeval_v2metrics/eval_unified_h10" "smolvla_rot6d_seed1000_${STEP}steps" "$STEP" "$SEED" 10
  done
done
echo "=== SmolVLA rot6d 1M eval COMPLETE: $(find "$CANON/reeval_v2metrics" -path '*smolvla_rot6d_seed1000_0*steps*' -o -path '*smolvla_rot6d_1m_seed1000*' | grep -c '_open_loop_metrics.json$') / 60 report files ==="
