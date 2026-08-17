#!/usr/bin/env bash
# Re-eval driver v2 — the ONLY host-side checkpoints with real weights after the
# disk-failure salvage: the six never-evaluated seed-2000/3000 companions.
# (The entire seed-1000 matrix under train/ is husk directories — empty
# pretrained_model skeletons; see RESEARCH_REPORT §8 incident 12. This driver
# supersedes reeval_v2metrics_backfill.sh, whose run list is all husks.)
#
# Runs against the UPDATED evaluator (per-component L1 / per-dim MSE, commit
# 51ff19f5) while the port h30 training leaves VRAM free; outputs land in the
# shadow root reeval_v2metrics/ (legacy metrics untouched).
#
# Checkpoint steps are the greatest REAL step per run — flow stopped at 50k and
# r50_vae at 80k when the artifact disk failed, so these rows are
# budget-mismatched vs the seed-1000 full-budget table and must be compared
# across the seed pair at matched steps, not against seed-1000@100k.
#
# Usage: reeval_seed23k_v2metrics.sh [backfill|standalone]  (default backfill:
# stops when the port trainer exits so P1/P2 phase evals get the whole GPU)
set -uo pipefail

MODE="${1:-backfill}"
CANON=/mnt/data1/projects/lerobot-arch-exp
SHADOW=$CANON/reeval_v2metrics
REPO=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified

mkdir -p "$SHADOW/logs"
ln -sfn "$CANON/train" "$SHADOW/train"
cd "$REPO"

port_alive() { pgrep -f 'pi05_port_openpi_args_rot6d_h30' >/dev/null 2>&1; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
may_continue() {
  if [ "$MODE" != "standalone" ] && ! port_alive; then
    echo "[$(date '+%F %T')] port training gone; yielding GPU to phase evals"
    return 1
  fi
  return 0
}
gate_vram() {
  while [ "$(free_mib)" -lt 4000 ]; do
    sleep 120
    may_continue || return 1
  done
  return 0
}
has_weights() {  # greatest numeric checkpoint dir with a real model file
  local pm ck
  ck=$(ls "$CANON/train/$1/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  pm="$CANON/train/$1/checkpoints/$ck/pretrained_model"
  [ -n "$ck" ] && compgen -G "$pm"/*.safetensors >/dev/null
}

# run_name:inference_seeds  (deterministic ACT = 1000; generative flow = 3)
ITEMS=(
  "act_r18_l1_seed2000_100000steps:1000"
  "act_r18_l1_seed3000_100000steps:1000"
  "act_r50_vae_seed2000_100000steps:1000"
  "act_r50_vae_seed3000_100000steps:1000"
  "act_r18_flow_u_lr1e5_seed2000_100000steps:1000 2000 3000"
  "act_r18_flow_u_lr1e5_seed3000_100000steps:1000 2000 3000"
)

for ITEM in "${ITEMS[@]}"; do
  RUN="${ITEM%%:*}"
  SEEDS="${ITEM##*:}"
  if ! has_weights "$RUN"; then
    echo "[$(date '+%F %T')] skip $RUN (no real weights — husk)"
    continue
  fi
  for SEED in $SEEDS; do
    OUT="$SHADOW/eval_common_h32/$RUN/seed$SEED"
    LOG="$SHADOW/logs/eval_common_h32_${RUN}_seed${SEED}.log"
    if compgen -G "$OUT"/*.json >/dev/null; then
      echo "[$(date '+%F %T')] done already: $RUN seed$SEED"
      continue
    fi
    may_continue || exit 0
    gate_vram || exit 0
    # stale output dir from a killed attempt (shadow only): remove and retry
    if [ -d "$OUT" ]; then rm -rf "$OUT"; fi
    if [ -e "$LOG" ]; then
      mkdir -p "$SHADOW/logs/failed_attempts"
      mv "$LOG" "$SHADOW/logs/failed_attempts/${RUN}_seed${SEED}_$(date +%s).log"
    fi
    echo "[$(date '+%F %T')] eval: $RUN seed$SEED"
    if ! UMI_ABLATION_ROOT="$SHADOW" bash examples/umi_relative_ee/act_flow_ablation/evaluate_one.sh "$RUN" "$SEED" 5; then
      echo "[$(date '+%F %T')] FAILED $RUN seed$SEED (continuing)"
    fi
  done
done
echo "[$(date '+%F %T')] seed23k reeval driver ($MODE) complete"
