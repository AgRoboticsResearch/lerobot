#!/usr/bin/env bash
# ACT R50-V1 1M budget-curve evaluation at native horizon 30 (§9.2.8, R1).
#
# Evaluates every 100k checkpoint (0100000..1000000) of the fresh seed-1000
# R50-V1 run with the v2 metric set under the same fixed 100-episode /
# 500-query protocol as eval_historical_act_curve.sh — full 30-step scoring
# (NO --eval_horizon), so the curve is directly comparable with the §9.2.7
# historical R18 curve (same horizon, same queries, same metric set).
#
# Run names deliberately carry the `_1m` variant suffix
# (act_r50_v1_vae_1m_seed1000_<STEP>steps) and land under eval_common_h32/,
# NOT eval_unified_h10/ — the t+30 rows must never mix with the unified t+10
# tree. Deterministic ACT: inference seed 1000 only.
#
# Idempotent: skips checkpoints whose report JSON already exists.
set -uo pipefail
CANON=/mnt/data1/projects/lerobot-arch-exp
SHADOW=$CANON/reeval_v2metrics
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
R50RUN=$CANON/train/act_r50_v1_vae_seed1000_1000000steps/checkpoints
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
r50v1_eval_busy() { pgrep -f 'eval_open_loop_dataset\.py.*act_r50_v1_vae_seed1000_1000000steps' >/dev/null 2>&1; }

cd "$REPO"
for STEP in 0100000 0200000 0300000 0400000 0500000 0600000 0700000 0800000 0900000 1000000; do
  RUN=act_r50_v1_vae_1m_seed1000_${STEP}steps
  OUT=$SHADOW/eval_common_h32/$RUN/seed1000
  if compgen -G "$OUT"/*.json >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] skip $RUN (already evaluated)"
    continue
  fi
  fm=""
  while true; do
    if r50v1_eval_busy; then sleep 60; continue; fi   # serialize with stray evals
    fm=$(free_mib)
    [ "$fm" -ge 4000 ] && break
    echo "[$(date '+%F %T')] waiting: ${fm} MiB free < 4000"
    sleep 120
  done
  echo "[$(date '+%F %T')] evaluating $RUN (free ${fm} MiB)"
  PYTHONPATH=src timeout 3600 uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
    --pretrained_path="$R50RUN/$STEP/pretrained_model" \
    --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
    --samples_per_episode=5 --query_min_action_offset=-1 --query_max_action_offset=31 \
    --seed=1000 --device=cuda --video_backend=pyav --output_dir="$OUT"
  echo "[$(date '+%F %T')] exit=$? $RUN"
done
echo "[$(date '+%F %T')] R50-V1 1M h30 CURVE DONE - $(ls -d $SHADOW/eval_common_h32/act_r50_v1_vae_1m_*steps 2>/dev/null | wc -l)/10 report dirs"
