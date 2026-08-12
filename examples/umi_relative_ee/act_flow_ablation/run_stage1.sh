#!/usr/bin/env bash
# Sequential screen. Keeping one process on the GPU avoids contention.
set -euo pipefail

STEPS="${1:-30000}"
SEED="${2:-1000}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VARIANTS=(
  act_r18_vae
  act_r34_vae
  act_r50_vae
  act_r50_v1_vae
  act_r50_large
  act_r18_l1
  act_r18_flow_u_lr1e5
  act_r18_flow_u_lr1e4
  act_r18_flow_beta_lr1e4
  act_r18_diffusion_lr1e5
  diffusion_r18
)

for variant in "${VARIANTS[@]}"; do
  "$SCRIPT_DIR/run_one.sh" "$variant" "$STEPS" "$SEED"
done
