#!/usr/bin/env bash
# Longer-budget promotion of the stage-one winners and matched controls.
set -euo pipefail

STEPS="${1:-100000}"
TRAINING_SEED="${2:-1000}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VARIANTS=(
  act_r18_vae
  act_r50_vae
  act_r50_v1_vae
  act_r18_l1
  act_r18_flow_u_lr1e5
  act_r18_diffusion_lr1e5
  diffusion_r18
)

for variant in "${VARIANTS[@]}"; do
  "$SCRIPT_DIR/run_one.sh" "$variant" "$STEPS" "$TRAINING_SEED"
done
