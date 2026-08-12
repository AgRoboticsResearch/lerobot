#!/usr/bin/env bash
# Shared structural validation for materialized LingBot trainable/frozen assets.

lingbot_has_weight() {
  find "$1" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) \
    -size +1M -print -quit 2>/dev/null | grep -q .
}

lingbot_trainable_complete() {
  local trainable="$1"
  [[ -s "$trainable/config.json" ]] &&
    [[ -s "$trainable/model.safetensors" ]] &&
    [[ "$(stat -c '%s' "$trainable/model.safetensors")" -gt 9000000000 ]] &&
    [[ -s "$trainable/policy_preprocessor.json" ]] &&
    [[ -s "$trainable/policy_postprocessor.json" ]] &&
    ! find "$trainable/.cache/huggingface/download" -type f \
      -name '*.incomplete' -print -quit 2>/dev/null | grep -q .
}

lingbot_frozen_complete() {
  local frozen="$1"
  [[ -s "$frozen/vae/config.json" ]] && lingbot_has_weight "$frozen/vae" &&
    [[ -s "$frozen/text_encoder/config.json" ]] && lingbot_has_weight "$frozen/text_encoder" &&
    [[ -d "$frozen/tokenizer" ]] &&
    find "$frozen/tokenizer" -maxdepth 1 -type f -size +0c -print -quit 2>/dev/null | grep -q . &&
    ! find "$frozen/.cache/huggingface/download" -type f \
      -name '*.incomplete' -print -quit 2>/dev/null | grep -q .
}

lingbot_assets_complete() {
  lingbot_trainable_complete "$1" && lingbot_frozen_complete "$2"
}
