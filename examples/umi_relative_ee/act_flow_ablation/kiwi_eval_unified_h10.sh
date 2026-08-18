#!/usr/bin/env bash
# Unified horizon-10 re-evaluation sweep (kiwi side) — §9.4. Run on kiwi AFTER
# the 1M training finishes (K-phase). Same protocol as the host sweep
# eval_unified_h10_sweep.sh: fixed 500-query set (100 episodes x 5, explicit
# bounds [-1,31]), --eval_horizon 10, full v2 metric set, pyav-equivalent
# decode, deterministic models at seed 1000, stochastic port at 3 seeds.
#
# Rows (kiwi):
#   - pi0.5 port 1M / 700K / 650K   (3 inference seeds each)
#   - SmolVLA rot6d + axis_angle @100k (seed 1000; original notation protocol)
#
# Outputs -> ~/kiwi-ablation/artifacts/eval_unified_h10/<RUN>/seed<k>/ with
# RUN_RE-compatible names. After completion, copy the tree back to the host:
#   scp -r ~/kiwi-ablation/artifacts/eval_unified_h10/* <host>:$CANON/reeval_v2metrics/eval_unified_h10/
# Idempotent: finished JSONs are skipped.
set -uo pipefail

export PYTHONPATH=/home/zfei/code/lerobot-fei-v5.0-umi-unified/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
EVAL=/home/zfei/kiwi-ablation/examples/umi_relative_ee/eval_open_loop_dataset.py
PORT_RUN=/home/zfei/code/lerobot-fei-v5.0-umi-unified/outputs/train/pi05_openpi_split_lora_masked_1459_bs4_1m/checkpoints
SMOL=/home/zfei/kiwi-ablation/artifacts/train
DATASET=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
OUT_ROOT=/home/zfei/kiwi-ablation/artifacts/eval_unified_h10
mkdir -p "$OUT_ROOT/logs"

wait_gpu_free() {  # the trainer holds the GPU until ~1M; wait for it to exit
  while pgrep -f train_pi05_lora.py >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] trainer still running; waiting"
    sleep 600
  done
}

run_eval() {  # run_dir ckpt_path seed
  local RUN="$1" CKPT="$2" SEED="$3"
  local OUT="$OUT_ROOT/$RUN/seed$SEED"
  local LOG="$OUT_ROOT/logs/${RUN}_seed${SEED}.log"
  if compgen -G "$OUT"/*_open_loop_metrics.json >/dev/null; then
    echo "[$(date '+%F %T')] done already: $RUN seed$SEED"
    return 0
  fi
  [ -d "$OUT" ] && rm -rf "$OUT"
  echo "[$(date '+%F %T')] eval: $RUN seed$SEED"
  if timeout 7200 "$PY" "$EVAL" \
      --pretrained_path "$CKPT" \
      --dataset_root "$DATASET" \
      --samples_per_episode 5 \
      --query_min_action_offset -1 --query_max_action_offset 31 \
      --eval_horizon 10 \
      --seed "$SEED" --device cuda \
      --output_dir "$OUT" >"$LOG" 2>&1; then
    echo "[$(date '+%F %T')] exit=0 $RUN seed$SEED"
  else
    echo "[$(date '+%F %T')] FAILED $RUN seed$SEED (see $LOG)"
  fi
}

wait_gpu_free

for STEP in 1000000 0700000 0650000; do
  for SEED in 1000 2000 3000; do
    run_eval "pi05_port_seed1000_${STEP}steps" "$PORT_RUN/$STEP/pretrained_model" "$SEED"
  done
done

run_eval smolvla_rot6d_seed1000_100000steps      "$SMOL/smolvla_rot6d_seed1000_100000steps/checkpoints/100000/pretrained_model" 1000
run_eval smolvla_axis_angle_seed1000_100000steps "$SMOL/smolvla_axis_angle_seed1000_100000steps/checkpoints/100000/pretrained_model" 1000
# SmolVLA samples stochastically (flow): mirror the 3-inference-seed protocol
for SEED in 2000 3000; do
  run_eval smolvla_rot6d_seed1000_100000steps      "$SMOL/smolvla_rot6d_seed1000_100000steps/checkpoints/100000/pretrained_model" "$SEED"
  run_eval smolvla_axis_angle_seed1000_100000steps "$SMOL/smolvla_axis_angle_seed1000_100000steps/checkpoints/100000/pretrained_model" "$SEED"
done

echo "=== kiwi unified h10 sweep COMPLETE: $(find "$OUT_ROOT" -name '*_open_loop_metrics.json' | wc -l) report files ==="
