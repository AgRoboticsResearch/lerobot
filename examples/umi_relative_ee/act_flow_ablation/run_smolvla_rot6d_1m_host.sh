#!/usr/bin/env bash
# SmolVLA rot6d 1M-step budget run (user-directed 2026-08-18): fresh seed-1000
# training to 1M with 100k-spaced checkpoints, extending the §9.2.3 notation
# finding with a budget curve in the exact shape of the §9.2.8 R50-V1 run.
# FRESH, not a resume of the 100k notation run: that run's cosine schedule
# (--policy.scheduler_decay_steps=100000) has already decayed 1e-4 → 2.5e-6, so
# extending it under a 1M schedule would be scheduler-incompatible
# (same reasoning as §9.3's stage-two fresh runs). The fresh run's 100k point
# doubles as a replication check of the notation run, as in §9.2.8 read-out 2.
#
# This wrapper first waits for the in-flight π0.5-port host curve sweep
# (eval_pi05_curve_h10_host.sh) to finish so the SmolVLA trainer never contends
# with the sweep's evals (keeps their recorded latency/memory columns clean),
# then launches run_one.sh smolvla_rot6d 1000000 1000 in the foreground.
# Relaunch-safe: if this wrapper is re-run while the trainer is alive it exits;
# if the trainer died, re-run with UMI_RESUME=true to resume from
# checkpoints/last (the 1M schedule is unchanged, so resume is valid here).
set -uo pipefail

REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
CANON=${UMI_ABLATION_ROOT:-/mnt/data1/projects/lerobot-arch-exp}
SWEEP_LOG=$CANON/reeval_v2metrics/eval_unified_h10/logs/pi05_curve_host_sweep.log

if pgrep -f 'train_umi_relative_ee\.py.*smolvla_rot6d' >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] trainer already alive — nothing to do"
  exit 0
fi

echo "[$(date '+%F %T')] waiting for π0.5-port curve sweep to complete"
while true; do
  if grep -q '=== π0.5-port host curve sweep COMPLETE' "$SWEEP_LOG" 2>/dev/null \
     && ! pgrep -f 'eval_open_loop_dataset\.py' >/dev/null 2>&1; then
    break
  fi
  sleep 120
done
echo "[$(date '+%F %T')] sweep done — launching SmolVLA rot6d 1M training"

cd "$REPO"
export UMI_ABLATION_ROOT=$CANON
export UMI_SAVE_FREQ=100000
exec bash examples/umi_relative_ee/act_flow_ablation/run_one.sh smolvla_rot6d 1000000 1000
