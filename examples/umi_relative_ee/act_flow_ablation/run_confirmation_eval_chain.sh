#!/usr/bin/env bash
# Chain: wait for all multi-seed confirmation training to finish, then evaluate
# seeds 2000 + 3000, then collect + plot. Launched in the background so the
# pipeline advances autonomously overnight while the host GPU is saturated by
# training (evals cannot share the GPU until training releases it).
#
# Completion is the durable check from training_completion.sh, so a run whose
# shell wrapper died after writing its final checkpoint still counts as done.
# Training itself is driven by the resume-aware companions; this script only
# gates the eval phase on their completion.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/training_completion.sh"
ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
CHAIN_LOG="$ROOT/logs/confirmation_eval_chain_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$CHAIN_LOG") 2>&1

timestamp() { date '+%F %T'; }
STEPS=100000
VARIANTS=(
  act_r18_vae act_r50_vae act_r50_v1_vae act_r18_l1
  act_r18_flow_u_lr1e5 act_r18_diffusion_lr1e5 diffusion_r18
)
SEEDS=(2000 3000)

echo "[$(timestamp)] confirmation eval-chain started; waiting for all confirmation training to complete"
poll_count=0
while :; do
  missing=()
  for s in "${SEEDS[@]}"; do
    for v in "${VARIANTS[@]}"; do
      name="${v}_seed${s}_${STEPS}steps"
      if ! training_is_complete "$ROOT" "$name" "$STEPS" 2>/dev/null; then
        missing+=("$name")
      fi
    done
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "[$(timestamp)] all confirmation runs complete; proceeding to evaluation"
    break
  fi
  poll_count=$((poll_count + 1))
  # Log a concise status every ~hour (12 polls * 300s).
  if (( poll_count % 12 == 1 )); then
    echo "[$(timestamp)] still waiting on ${#missing[@]} run(s): ${missing[*]}"
  fi
  sleep 300
done

# Evaluate seed 2000 then seed 3000. Each call runs collect_results + plot at the
# end; the second call therefore reflects the full matrix.
echo "[$(timestamp)] === evaluating seed 2000 ==="
"$SCRIPT_DIR/supervise_evaluations.sh" 2000
echo "[$(timestamp)] === evaluating seed 3000 ==="
"$SCRIPT_DIR/supervise_evaluations.sh" 3000

# Final authoritative collection + figures across all seeds (1000/2000/3000).
echo "[$(timestamp)] === final collect + plot ==="
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
UV_CACHE_DIR=/tmp/uv-cache-umi-ablation uv run python \
  "$SCRIPT_DIR/collect_results.py" --artifact_root "$ROOT"
MPLCONFIGDIR=/tmp/lerobot-matplotlib UV_CACHE_DIR=/tmp/uv-cache-umi-ablation uv run --with matplotlib python \
  "$SCRIPT_DIR/plot_results.py" --artifact_root "$ROOT"

echo "[$(timestamp)] confirmation eval-chain finished"
