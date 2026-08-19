#!/usr/bin/env bash
# K2+K3 chain of the kiwi h30 phase (§9.2.11, user-directed 2026-08-19):
#
#  K2 — π0.5-port h30 re-evals at the three retained budgets (1M/700K/650K),
#       full 30-step scoring (NO --eval_horizon), canonical window, three
#       inference seeds. Run names pi05_port_<STEP>_h30_v2 — IDENTICAL to the
#       host front-run (eval_pi05_curve_h30_host.sh), so both sources feed the
#       same §9.2.11 compiler grammar; each row has exactly one owner.
#  K3 — SmolVLA notation h30 re-evals (rot6d + axis_angle @100k), three
#       inference seeds, run names smolvla_<rep>_seed1000_100000steps —
#       RUN_RE-parseable by compile_unified_h30.py.
#
# Idempotent per (run, seed); coexists with the masked 1M trainer (~2.8 GB)
# on the 16 GB 5080.
set -uo pipefail
export PYTHONPATH=/home/zfei/code/lerobot-fei-v5.0-umi-unified/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
EVAL=/home/zfei/kiwi-ablation/examples/umi_relative_ee/eval_open_loop_dataset.py
DATASET=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
PORT_CKPTS=/home/zfei/code/lerobot-fei-v5.0-umi-unified/outputs/train/pi05_openpi_split_lora_masked_1459_bs4_1m/checkpoints
SMOL_CKPTS=/home/zfei/kiwi-ablation/artifacts/train
OUT_ROOT=/home/zfei/kiwi-ablation/artifacts/eval_common_h32
mkdir -p "$OUT_ROOT/logs"

run_eval() {  # out_run ckpt
  local RUN="$1" CKPT="$2"
  [ -d "$CKPT" ] || { echo "[$(date '+%F %T')] missing ckpt $CKPT — skipping"; return 0; }
  for SEED in 1000 2000 3000; do
    local OUT="$OUT_ROOT/$RUN/seed$SEED"
    local LOG="$OUT_ROOT/logs/${RUN}_seed${SEED}_h30.log"
    if compgen -G "$OUT"/*_open_loop_metrics.json >/dev/null; then
      echo "[$(date '+%F %T')] skip $RUN seed$SEED (done)"
      continue
    fi
    [ -d "$OUT" ] && rm -rf "$OUT"
    echo "[$(date '+%F %T')] eval: $RUN seed$SEED (h30)"
    if timeout 3600 "$PY" "$EVAL" \
        --pretrained_path="$CKPT" \
        --dataset_root="$DATASET" \
        --samples_per_episode=5 --query_min_action_offset=-1 --query_max_action_offset=31 \
        --seed="$SEED" --device=cuda --video_backend=pyav --output_dir="$OUT" \
        >"$LOG" 2>&1; then
      echo "[$(date '+%F %T')] exit=0 $RUN seed$SEED"
    else
      echo "[$(date '+%F %T')] FAILED $RUN seed$SEED (see $LOG)"
    fi
  done
}

echo "[$(date '+%F %T')] === K2: port h30 re-evals (1M/700K/650K x3 seeds) ==="
for STEP in 1000000 0700000 0650000; do
  run_eval "pi05_port_${STEP}_h30_v2" "$PORT_CKPTS/$STEP/pretrained_model"
done

echo "[$(date '+%F %T')] === K3: SmolVLA notation h30 re-evals (x3 seeds) ==="
run_eval smolvla_rot6d_seed1000_100000steps      "$SMOL_CKPTS/smolvla_rot6d_seed1000_100000steps/checkpoints/100000/pretrained_model"
run_eval smolvla_axis_angle_seed1000_100000steps "$SMOL_CKPTS/smolvla_axis_angle_seed1000_100000steps/checkpoints/100000/pretrained_model"

echo "[$(date '+%F %T')] === K2+K3 COMPLETE: $(find "$OUT_ROOT" \( -path '*pi05_port_*_h30_v2*' -o -path '*smolvla_*seed1000_100000steps*' \) -name '*_open_loop_metrics.json' 2>/dev/null | wc -l) report files (15 expected) ==="
