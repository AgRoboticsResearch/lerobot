#!/usr/bin/env bash
# Screen both released UMI diffusion architectures after the existing queue.
set -euo pipefail

STEPS="${1:-30000}"
TRAINING_SEED="${2:-1000}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for variant in umi_official_dp umi_official_transformer_dp; do
  "$SCRIPT_DIR/run_one.sh" "$variant" "$STEPS" "$TRAINING_SEED"
done
