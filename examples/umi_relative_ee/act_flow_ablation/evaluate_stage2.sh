#!/usr/bin/env bash
# Corrected common-horizon evaluation for the longer-budget promoted controls.
set -euo pipefail

STEPS="${1:-100000}"
TRAINING_SEED="${2:-1000}"
SAMPLES_PER_EPISODE="${3:-5}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DETERMINISTIC_VARIANTS=(
  act_r18_vae
  act_r50_vae
  act_r50_v1_vae
  act_r18_l1
)
GENERATIVE_VARIANTS=(
  act_r18_flow_u_lr1e5
  act_r18_diffusion_lr1e5
  diffusion_r18
)

for variant in "${DETERMINISTIC_VARIANTS[@]}"; do
  run_name="${variant}_seed${TRAINING_SEED}_${STEPS}steps"
  "$SCRIPT_DIR/evaluate_one.sh" "$run_name" 1000 "$SAMPLES_PER_EPISODE"
done

for variant in "${GENERATIVE_VARIANTS[@]}"; do
  run_name="${variant}_seed${TRAINING_SEED}_${STEPS}steps"
  for inference_seed in 1000 2000 3000; do
    "$SCRIPT_DIR/evaluate_one.sh" "$run_name" "$inference_seed" "$SAMPLES_PER_EPISODE"
  done
done
