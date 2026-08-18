#!/usr/bin/env bash
# Host-side execution of the SmolVLA rows of the unified horizon-10 sweep
# (§9.2.9) — front-run of kiwi K3. The two notation-ablation checkpoints
# (seed 1000, 100k; trained on kiwi, §9.2.3) were copied to the canonical
# host train/ tree (scp of pretrained_model only; no training happened on
# the host). Flags mirror eval_unified_h10_sweep.sh exactly so the reports
# pass compile_unified_h10.py's protocol assertions; SmolVLA samples
# stochastically (flow), hence three inference seeds per checkpoint.
# The kiwi script remains the fallback if these rows are ever missing.
set -uo pipefail

CANON=${UMI_ABLATION_ROOT:-/mnt/data1/projects/lerobot-arch-exp}
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
OUT_ROOT=$CANON/reeval_v2metrics/eval_unified_h10
mkdir -p "$OUT_ROOT/logs"

run_eval() {  # run_dir seed
  local RUN="$1" SEED="$2"
  local CKPT="$CANON/train/$RUN/checkpoints/100000/pretrained_model"
  local OUT="$OUT_ROOT/$RUN/seed$SEED"
  local LOG="$OUT_ROOT/logs/${RUN}_seed${SEED}.log"
  if compgen -G "$OUT"/*_open_loop_metrics.json >/dev/null; then
    echo "[$(date '+%F %T')] done already: $RUN seed$SEED"
    return 0
  fi
  [ -d "$OUT" ] && rm -rf "$OUT"   # stale dir from a killed attempt
  echo "[$(date '+%F %T')] eval: $RUN seed$SEED"
  if PYTHONPATH=src timeout 3600 uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
      --pretrained_path="$CKPT" \
      --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
      --samples_per_episode=5 --query_min_action_offset=-1 --query_max_action_offset=31 \
      --eval_horizon=10 \
      --seed="$SEED" --device=cuda --video_backend=pyav --output_dir="$OUT" \
      >"$LOG" 2>&1; then
    echo "[$(date '+%F %T')] exit=0 $RUN seed$SEED"
  else
    echo "[$(date '+%F %T')] FAILED $RUN seed$SEED (see $LOG)"
  fi
}

for SEED in 1000 2000 3000; do
  run_eval smolvla_rot6d_seed1000_100000steps "$SEED"
  run_eval smolvla_axis_angle_seed1000_100000steps "$SEED"
done

echo "=== SmolVLA host h10 sweep COMPLETE: $(find "$OUT_ROOT" -path '*smolvla*' -name '*_open_loop_metrics.json' | wc -l) report files ==="
