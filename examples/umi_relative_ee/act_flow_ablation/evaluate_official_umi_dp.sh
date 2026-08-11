#!/usr/bin/env bash
# Evaluate stochastic UMI diffusion candidates with three fixed inference seeds.
set -euo pipefail

STEPS="${1:-30000}"
TRAINING_SEED="${2:-1000}"
SAMPLES_PER_EPISODE="${3:-5}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for variant in umi_official_dp umi_official_transformer_dp; do
  run_name="${variant}_seed${TRAINING_SEED}_${STEPS}steps"
  for inference_seed in 1000 2000 3000; do
    "$SCRIPT_DIR/evaluate_one.sh" "$run_name" "$inference_seed" "$SAMPLES_PER_EPISODE"
  done
done
