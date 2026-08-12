#!/usr/bin/env bash
# Compare ACT's best held-out-loss recovery checkpoint with the fixed-budget
# final checkpoint only after the canonical final evaluation releases the GPU.
set -uo pipefail

WAIT_TMUX="${UMI_WAIT_FOR_TMUX:-umi_act_l1_100k_early_eval_20260812}"
RUN_NAME="${UMI_ACT_SELECTION_RUN:-act_r18_l1_seed1000_100000steps}"
CHECKPOINT_STEP="${UMI_ACT_SELECTION_STEP:-60000}"
EVAL_SEED="${UMI_ACT_SELECTION_SEED:-1000}"
SAMPLES_PER_EPISODE="${UMI_EVAL_SAMPLES_PER_EPISODE:-5}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
printf -v PADDED_STEP '%06d' "$CHECKPOINT_STEP"
OUT="$ARTIFACT_ROOT/eval_checkpoint_h32/$RUN_NAME/step$PADDED_STEP/seed$EVAL_SEED"
LOG="$ARTIFACT_ROOT/logs/eval_checkpoint_h32_${RUN_NAME}_step${PADDED_STEP}_seed${EVAL_SEED}.log"

echo "[$(date '+%F %T')] ACT checkpoint-selection evaluator waiting for $WAIT_TMUX"
while tmux has-session -t "$WAIT_TMUX" 2>/dev/null; do sleep 15; done

reports="$(find "$OUT" -maxdepth 1 -type f -name '*_open_loop_metrics.json' -size +0c 2>/dev/null | wc -l)"
if [[ "$reports" -eq 1 ]] && grep -Fq "] completed checkpoint evaluation $RUN_NAME step $CHECKPOINT_STEP seed $EVAL_SEED" "$LOG"; then
  echo "[$(date '+%F %T')] checkpoint-selection evaluation already complete"
  exit 0
fi

# The canonical evaluator has exited, but another queue may be starting. Require
# modest inference headroom before loading the small deterministic ACT policy.
while true; do
  free_gpu_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
  if [[ "$free_gpu_mib" =~ ^[0-9]+$ && "$free_gpu_mib" -ge 4096 ]]; then break; fi
  sleep 30
done

"$SCRIPT_DIR/evaluate_checkpoint.sh" "$RUN_NAME" "$CHECKPOINT_STEP" "$EVAL_SEED" "$SAMPLES_PER_EPISODE"
