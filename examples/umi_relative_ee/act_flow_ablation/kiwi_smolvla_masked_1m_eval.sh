#!/usr/bin/env bash
# Complete evaluation of the SmolVLA masked-subspace 1M budget run (§9.2.10),
# mirroring the host full-width chain (eval_smolvla_rot6d_1m_curve.sh):
#
#  1. h30 native curve — full 30-step scoring (NO --eval_horizon), run names
#     smolvla_masked_1m_seed1000_<STEP>steps under eval_common_h32/.
#  2. unified t+10 rows — canonical §9.2.9 protocol (--eval_horizon 10, bounds
#     [-1,31], 500 queries), run names smolvla_masked_seed1000_0<STEP>steps
#     (7-digit steps) under eval_unified_h10/.
#
# SmolVLA samples stochastically (flow) → inference seeds 1000/2000/3000 on
# every checkpoint. Idempotent per (run, seed); VRAM-gated at ≥4 GiB free.
set -uo pipefail
export PYTHONPATH=/home/zfei/code/lerobot-fei-v5.0-umi-unified/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
EVAL=/home/zfei/kiwi-ablation/examples/umi_relative_ee/eval_open_loop_dataset.py
CKPTS=/home/zfei/kiwi-ablation/artifacts/train/smolvla_rot6d_masked_seed1000_1000000steps/checkpoints
DATASET=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
LOG_ROOT=/home/zfei/kiwi-ablation/artifacts/logs
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
  cd /home/zfei/kiwi-ablation
  if timeout 3600 "$PY" "$EVAL" \
      --pretrained_path="$CKPTS/$STEP/pretrained_model" \
      --dataset_root="$DATASET" \
      --samples_per_episode=5 --query_min_action_offset=-1 --query_max_action_offset=31 \
      "${args[@]}" \
      --seed="$SEED" --device=cuda --video_backend=pyav --output_dir="$OUT" \
      >"$LOG" 2>&1; then
    echo "[$(date '+%F %T')] exit=0 $RUN seed$SEED h$H"
  else
    echo "[$(date '+%F %T')] FAILED $RUN seed$SEED h$H (see $LOG)"
  fi
}

for STEP in 0100000 0200000 0300000 0400000 0500000 0600000 0700000 0800000 0900000 1000000; do
  [ -d "$CKPTS/$STEP/pretrained_model" ] || { echo "[$(date '+%F %T')] missing $STEP — skipping"; continue; }
  for SEED in 1000 2000 3000; do
    # 1) h30 native curve (outside the unified tree)
    run_eval /home/zfei/kiwi-ablation/artifacts/eval_common_h32 "smolvla_masked_1m_seed1000_${STEP}steps" "$STEP" "$SEED" 30
    # 2) unified t+10 row
    run_eval /home/zfei/kiwi-ablation/artifacts/eval_unified_h10 "smolvla_masked_seed1000_${STEP}steps" "$STEP" "$SEED" 10
  done
done
echo "=== SmolVLA rot6d masked 1M eval COMPLETE: $(find /home/zfei/kiwi-ablation/artifacts/eval_unified_h10 /home/zfei/kiwi-ablation/artifacts/eval_common_h32 -path '*smolvla_masked*' 2>/dev/null | grep -c '_open_loop_metrics.json$') / 60 report files ==="
