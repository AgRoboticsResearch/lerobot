#!/usr/bin/env bash
# Find the largest safe released-recipe batch, then run each full candidate.
set -euo pipefail

STEPS="${1:-30000}"
TRAINING_SEED="${2:-1000}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"

for variant in umi_official_dp umi_official_transformer_dp; do
  selected_batch=""
  for batch_size in 64 32 16 8; do
    smoke_root="$ARTIFACT_ROOT/smoke_official_dp/batch${batch_size}"
    if UMI_ABLATION_ROOT="$smoke_root" \
      UMI_OFFICIAL_BATCH_SIZE="$batch_size" \
      UMI_SAVE_CHECKPOINT=false \
      UMI_VAL_FREQ=1000 \
      "$SCRIPT_DIR/run_one.sh" "$variant" 2 "$TRAINING_SEED"; then
      selected_batch="$batch_size"
      break
    fi
    echo "[$(date '+%F %T')] $variant batch $batch_size failed; trying a smaller batch" >&2
  done
  if [[ -z "$selected_batch" ]]; then
    echo "No safe smoke batch found for $variant" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] $variant selected batch $selected_batch for the full run"
  UMI_OFFICIAL_BATCH_SIZE="$selected_batch" \
    "$SCRIPT_DIR/run_one.sh" "$variant" "$STEPS" "$TRAINING_SEED"
done
