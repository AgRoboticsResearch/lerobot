#!/usr/bin/env bash
# Resumably fetch and validate LingBot's trainable checkpoint and frozen assets.
# Hugging Face may return success with an incomplete local_dir during a network
# outage, so exit status alone is deliberately not treated as completion.
set -uo pipefail

ARTIFACT_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
HF_HOME="${UMI_LINGBOT_HF_HOME:-$ARTIFACT_ROOT/hf-cache}"
TRAINABLE_DIR="${UMI_LINGBOT_CHECKPOINT:-$ARTIFACT_ROOT/pretrained/lingbot_va_libero_long}"
FROZEN_DIR="${UMI_LINGBOT_FROZEN:-$ARTIFACT_ROOT/pretrained/lingbot_va_frozen_libero_long}"
RETRY_SECONDS="${UMI_LINGBOT_RETRY_SECONDS:-30}"
HF_BIN="${UMI_HF_BIN:-/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified/.venv/bin/hf}"
LOG="$ARTIFACT_ROOT/logs/lingbot_prefetch_$(date '+%Y%m%d_%H%M%S').log"
export HF_HOME
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/home/zfei/.cache/huggingface/token}"

mkdir -p "$ARTIFACT_ROOT/logs" "$TRAINABLE_DIR" "$FROZEN_DIR"
exec > >(tee -a "$LOG") 2>&1

timestamp() { date '+%F %T'; }

has_weight() {
  local directory="$1"
  find "$directory" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' \) -size +1M -print -quit 2>/dev/null | grep -q .
}

trainable_complete() {
  [[ -s "$TRAINABLE_DIR/config.json" ]] &&
    [[ -s "$TRAINABLE_DIR/model.safetensors" ]] &&
    [[ "$(stat -c '%s' "$TRAINABLE_DIR/model.safetensors")" -gt 9000000000 ]] &&
    [[ -s "$TRAINABLE_DIR/policy_preprocessor.json" ]] &&
    [[ -s "$TRAINABLE_DIR/policy_postprocessor.json" ]] &&
    ! find "$TRAINABLE_DIR/.cache/huggingface/download" -maxdepth 1 -type f \
      -name '*.incomplete' -print -quit 2>/dev/null | grep -q .
}

frozen_complete() {
  [[ -s "$FROZEN_DIR/vae/config.json" ]] && has_weight "$FROZEN_DIR/vae" &&
    [[ -s "$FROZEN_DIR/text_encoder/config.json" ]] && has_weight "$FROZEN_DIR/text_encoder" &&
    [[ -d "$FROZEN_DIR/tokenizer" ]] &&
    find "$FROZEN_DIR/tokenizer" -maxdepth 1 -type f -size +0c -print -quit 2>/dev/null | grep -q . &&
    ! find "$FROZEN_DIR/.cache/huggingface/download" -type f \
      -name '*.incomplete' -print -quit 2>/dev/null | grep -q .
}

echo "[$(timestamp)] LingBot prefetch supervisor started; log=$LOG"
attempt=0
until trainable_complete; do
  attempt=$((attempt + 1))
  echo "[$(timestamp)] trainable download attempt $attempt (partial files are retained)"
  "$HF_BIN" download lerobot/lingbot_va_libero_long \
    model.safetensors config.json policy_preprocessor.json policy_postprocessor.json \
    policy_postprocessor_step_0_unnormalizer_processor.safetensors \
    --local-dir "$TRAINABLE_DIR" --max-workers 1 || true
  trainable_complete || sleep "$RETRY_SECONDS"
done
echo "[$(timestamp)] verified trainable LingBot checkpoint"

attempt=0
until frozen_complete; do
  attempt=$((attempt + 1))
  echo "[$(timestamp)] frozen-asset download attempt $attempt (transformer weights excluded)"
  "$HF_BIN" download robbyant/lingbot-va-posttrain-libero-long \
    --include 'vae/*' --include 'text_encoder/*' --include 'tokenizer/*' \
    --local-dir "$FROZEN_DIR" --max-workers 2 || true
  frozen_complete || sleep "$RETRY_SECONDS"
done
echo "[$(timestamp)] verified frozen VAE, text encoder, and tokenizer assets"
echo "[$(timestamp)] LingBot prefetch complete"
