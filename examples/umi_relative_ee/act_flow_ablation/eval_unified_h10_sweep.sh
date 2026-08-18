#!/usr/bin/env bash
# Unified horizon-10 re-evaluation sweep (host side) — §9.4.
#
# ONE protocol for every surviving model with a >=10-step chunk:
#   fixed 500-query set (100 episodes x 5, explicit action-offset bounds
#   [-1, 31]), scoring horizon t+10 (--eval_horizon 10: only the first 10
#   steps of each decoded chunk are scored, endpoint = t+10), full v2 metric
#   set (endpoint pose, chunk means, per-component L1 / per-dim MSE,
#   accuracy@0.5/0.1, jerk), PyAV, cuda. Deterministic ACT at inference seed
#   1000; stochastic flow models at seeds 1000/2000/3000.
#
# Rows (host):
#   - historical production ACT R18-VAE, 30 checkpoints 100k..3M (repo disk)
#   - six seed-23k companions (real weights only; ACT @100k/80k, flow @50k)
#   - ACT R50-V1 1M run, 100k-spaced checkpoints (fills in as training
#     completes; re-run this driver at R-phase to pick up 0900000/1000000)
#   - official openpi arms via the patched eval_openpi_open_loop.py with the
#     canonical query window so their frames are IDENTICAL to the LeRobot rows
#
# NOT included (weights stranded by the artifact-disk failures, §8 incident
# 12): the entire seed-1000 ACT/flow/DP matrix incl. both umi_official ports.
# Kiwi-side rows (pi0.5 port 650K/700K/1M, SmolVLA notation) run in K-phase
# via kiwi_eval_unified_h10.sh.
#
# Outputs -> $CANON/reeval_v2metrics/eval_unified_h10/<RUN>/seed<k>/ with
# RUN_RE-compatible names so collect_results.py --v2_eval_roots reads them.
# Arm B (pi05_port_openpi_args h30 @t+10) already conforms and is copied in.
#
# Safe alongside the R50 trainer: every LeRobot eval waits for >=4 GiB free;
# openpi (JAX) evals run with XLA_PYTHON_CLIENT_PREALLOCATE=false. Idempotent:
# finished reports are skipped. Usage: eval_unified_h10_sweep.sh
set -uo pipefail

CANON=/mnt/data1/projects/lerobot-arch-exp
SHADOW=$CANON/reeval_v2metrics
UNIFIED=$SHADOW/eval_unified_h10
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation
HIST=$REPO/outputs/train/act_umi_identity_rot6d_1459/checkpoints
R50RUN=$CANON/train/act_r50_v1_vae_seed1000_1000000steps

mkdir -p "$UNIFIED/logs" "$SHADOW/logs"
ln -sfn "$CANON/train" "$SHADOW/train"
cd "$REPO"

r50_alive() { pgrep -f 'act_r50_v1_vae_seed1000_1000000steps' >/dev/null 2>&1; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gate_vram() {  # only while the R50 trainer owns part of the card
  while r50_alive && [ "$(free_mib)" -lt 4000 ]; do sleep 120; done
  return 0
}

run_lerobot() {  # run_dir ckpt_path seed
  local RUN="$1" CKPT="$2" SEED="$3"
  local OUT="$UNIFIED/$RUN/seed$SEED"
  local LOG="$UNIFIED/logs/${RUN}_seed${SEED}.log"
  if compgen -G "$OUT"/*_open_loop_metrics.json >/dev/null; then
    echo "[$(date '+%F %T')] done already: $RUN seed$SEED"
    return 0
  fi
  gate_vram
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

echo "=== unified h10 sweep: LeRobot rows ==="

# 1) historical production ACT (30 checkpoints, deterministic)
for STEP in 0100000 0200000 0300000 0400000 0500000 0600000 0700000 0800000 \
            0900000 1000000 1100000 1200000 1300000 1400000 1500000 1600000 \
            1700000 1800000 1900000 2000000 2100000 2200000 2300000 2400000 \
            2500000 2600000 2700000 2800000 2900000 3000000; do
  CKPT="$HIST/$STEP/pretrained_model"
  compgen -G "$CKPT"/*.safetensors >/dev/null || continue
  run_lerobot "act_umi_identity_rot6d_1459_${STEP}steps" "$CKPT" 1000
done

# 2) seed-23k companions (real weights; note flow stopped at 50k, r50_vae at 80k)
run_lerobot act_r18_l1_seed2000_100000steps "$CANON/train/act_r18_l1_seed2000_100000steps/checkpoints/100000/pretrained_model" 1000
run_lerobot act_r18_l1_seed3000_100000steps "$CANON/train/act_r18_l1_seed3000_100000steps/checkpoints/100000/pretrained_model" 1000
run_lerobot act_r50_vae_seed2000_100000steps "$CANON/train/act_r50_vae_seed2000_100000steps/checkpoints/080000/pretrained_model" 1000
run_lerobot act_r50_vae_seed3000_100000steps "$CANON/train/act_r50_vae_seed3000_100000steps/checkpoints/080000/pretrained_model" 1000
for SEED in 1000 2000 3000; do
  run_lerobot act_r18_flow_u_lr1e5_seed2000_100000steps "$CANON/train/act_r18_flow_u_lr1e5_seed2000_100000steps/checkpoints/050000/pretrained_model" "$SEED"
  run_lerobot act_r18_flow_u_lr1e5_seed3000_100000steps "$CANON/train/act_r18_flow_u_lr1e5_seed3000_100000steps/checkpoints/050000/pretrained_model" "$SEED"
done

# 3) ACT R50-V1 1M curve (100k-spaced; skips checkpoints that do not exist yet)
for STEP in 0100000 0200000 0300000 0400000 0500000 0600000 0700000 0800000 \
            0900000 1000000; do
  CKPT="$R50RUN/checkpoints/$STEP/pretrained_model"
  compgen -G "$CKPT"/*.safetensors >/dev/null || { echo "[$(date '+%F %T')] wait: no ckpt yet $STEP"; continue; }
  run_lerobot "act_r50_v1_vae_seed1000_${STEP}steps" "$CKPT" 1000
done

# 4) Arm B (port h30-trained, openpi recipe) — existing conforming t+10 JSONs
for SEED in 1000 2000 3000; do
  SRC="$CANON/eval_common_h32/pi05_port_openpi_args_h30_h10/seed$SEED"
  DST="$UNIFIED/pi05_port_openpi_recipe_seed1000_020000steps/seed$SEED"
  if compgen -G "$DST"/*_open_loop_metrics.json >/dev/null; then continue; fi
  if compgen -G "$SRC"/*_open_loop_metrics.json >/dev/null; then
    mkdir -p "$DST" && cp "$SRC"/*_open_loop_metrics.json "$DST"/
    echo "[$(date '+%F %T')] adopted Arm B t+10 seed$SEED from eval_common_h32"
  fi
done

echo "=== unified h10 sweep: openpi rows (JAX, canonical query window) ==="
OPENPI_PY=/home/zfei/codes/openpi/.venv/bin/python
OPENPI_EVAL=$REPO/examples/umi_relative_ee/act_flow_ablation/eval_openpi_open_loop.py
OPENPI_CKPT_ROOT=/home/zfei/codes/openpi/checkpoints
OPENPI_DATA=/mnt/data1/sroi/lerobot/sroiv2_strawberry_validation_rotvec

run_openpi() {  # config_name action_horizon eval_horizon run_dir out_step
  local CFG="$1" AH="$2" EH="$3" RUN="$4" STEP="$5"
  local OUT="$UNIFIED/$RUN/seed1000"
  local JSON="$OUT/${RUN}_${STEP}_open_loop_metrics.json"
  local LOG="$UNIFIED/logs/${RUN}_openpi.log"
  if [ -f "$JSON" ]; then
    echo "[$(date '+%F %T')] done already: $RUN (openpi)"
    return 0
  fi
  local CKPT="$OPENPI_CKPT_ROOT/$CFG/run1/19999"
  compgen -G "$CKPT"/params/* >/dev/null || { echo "[$(date '+%F %T')] WARN no openpi ckpt $CFG"; return 0; }
  local EHFLAG=""
  [ "$EH" != "-" ] && EHFLAG="--eval_horizon $EH"
  gate_vram
  [ -d "$OUT" ] && rm -rf "$OUT"
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] eval (openpi): $RUN"
  if ( cd /home/zfei/codes/openpi && \
       HF_LEROBOT_HOME=/mnt/data1/sroi/lerobot HF_HUB_OFFLINE=1 \
       XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 5400 \
       "$OPENPI_PY" "$OPENPI_EVAL" --config-name "$CFG" --checkpoint "$CKPT" \
         --dataset_root "$OPENPI_DATA" \
         --samples_per_episode 5 --action_horizon "$AH" $EHFLAG \
         --query_min_action_offset -1 --query_max_action_offset 31 \
         --seed 1000 --output "$JSON" ) \
      >"$LOG" 2>&1; then
    echo "[$(date '+%F %T')] exit=0 $RUN (openpi)"
  else
    echo "[$(date '+%F %T')] FAILED $RUN (openpi, see $LOG)"
  fi
}

run_openpi pi05_lora_sroi_rot6d      10 -  pi05_lora_sroi_rot6d_seed1000_0020000steps 0020000
run_openpi pi05_lora_sroi_rotvec     10 -  pi05_lora_sroi_rotvec_seed1000_0020000steps 0020000
run_openpi pi05_lora_sroi_rot6d_h30  30 10 pi05_lora_sroi_rot6d_h30_seed1000_0020000steps 0020000

echo "=== unified h10 sweep COMPLETE: $(find "$UNIFIED" -name '*_open_loop_metrics.json' | wc -l) report files under $UNIFIED ==="
