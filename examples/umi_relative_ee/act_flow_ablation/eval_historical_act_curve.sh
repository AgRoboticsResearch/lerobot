#!/usr/bin/env bash
# Historical 1459 ACT budget-curve evaluation (user directive 2026-08-17).
#
# Evaluates every 100k checkpoint (0100000..3000000) of the original
# production run outputs/train/act_umi_identity_rot6d_1459 — the only ACT
# run besides the six seed-23k companions whose weights survived the
# artifact-disk failures — with the v2 metric set (per-component L1,
# per-dim MSE, accuracy@tau) under the same fixed 100-episode / 500-query
# protocol as reeval_seed23k_v2metrics.sh.
#
# GPU-scheduling contract (single host RTX 4090 shared with the h30 chain):
# - While the port-h30 trainer is alive: backfill mode, needs >=4 GiB free.
#   No cron-firing clearance needed — no JAX phase can start before the
#   trainer ends (the babysit cron gates P1 on it).
# - After the trainer exits: WAIT until the openpi JAX phases are complete
#   (P4's *_v2metrics JSONs exist) before each eval — JAX preallocates 95%
#   of the card and would OOM against a concurrent ACT eval. Safety net:
#   after 2026-08-18 06:00 assume the phases are done (or failed) anyway.
# - Idempotent: skips any checkpoint whose report JSON already exists, so
#   the two anchor evals (0100000, 3000000) launched separately are reused.
set -uo pipefail
CANON=/mnt/data1/projects/lerobot-arch-exp
SHADOW=$CANON/reeval_v2metrics
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
HIST=$REPO/outputs/train/act_umi_identity_rot6d_1459/checkpoints
VAL_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
VAL_REPO=sroi/sroiv2_strawberry_picking_lab_validation

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
port_alive() { pgrep -f 'pi05_port_openpi_args_rot6d_h30' >/dev/null 2>&1; }
hist_eval_busy() { pgrep -f 'eval_open_loop_dataset\.py.*act_umi_identity_rot6d_1459' >/dev/null 2>&1; }
p4_done() {
  local n
  n=$(find "$REPO/outputs/research_report/openpi_sroi_eval" -maxdepth 2 -name '*.json' -path '*_v2metrics*' 2>/dev/null | wc -l)
  [ "$n" -ge 2 ]
}
deadline_passed() { [ "$(date +%s)" -ge "$(date -d '2026-08-18 06:00' +%s)" ]; }

cd "$REPO"
for STEP in 0100000 0200000 0300000 0400000 0500000 0600000 0700000 0800000 \
            0900000 1000000 1100000 1200000 1300000 1400000 1500000 1600000 \
            1700000 1800000 1900000 2000000 2100000 2200000 2300000 2400000 \
            2500000 2600000 2700000 2800000 2900000 3000000; do
  RUN=act_umi_identity_rot6d_1459_${STEP}steps
  OUT=$SHADOW/eval_common_h32/$RUN/seed1000
  if compgen -G "$OUT"/*.json >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] skip $RUN (already evaluated)"
    continue
  fi
  fm=""
  while true; do
    if hist_eval_busy; then sleep 120; continue; fi   # serialize with the anchor driver / stragglers
    fm=$(free_mib)
    if port_alive; then
      [ "$fm" -ge 4000 ] && break
      echo "[$(date '+%F %T')] waiting: port trainer alive, ${fm} MiB free < 4000"
    else
      if p4_done || deadline_passed; then
        [ "$fm" -ge 6000 ] && break
        echo "[$(date '+%F %T')] waiting: JAX phases done, ${fm} MiB free < 6000"
      else
        echo "[$(date '+%F %T')] holding: port trainer finished, openpi JAX phases (P1/P4) not yet complete - deferring to avoid OOM against their 95% preallocation"
      fi
    fi
    sleep 300
  done
  echo "[$(date '+%F %T')] evaluating $RUN (free ${fm} MiB)"
  PYTHONPATH=src uv run python examples/umi_relative_ee/eval_open_loop_dataset.py \
    --pretrained_path="$HIST/$STEP/pretrained_model" \
    --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
    --samples_per_episode=5 --query_min_action_offset=-1 --query_max_action_offset=31 \
    --seed=1000 --device=cuda --video_backend=pyav --output_dir="$OUT"
  echo "[$(date '+%F %T')] exit=$? $RUN"
done
echo "[$(date '+%F %T')] HISTORICAL ACT CURVE DONE - $(ls -d $SHADOW/eval_common_h32/act_umi_identity_rot6d_1459_*steps 2>/dev/null | wc -l)/30 report dirs"
