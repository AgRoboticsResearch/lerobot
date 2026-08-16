#!/usr/bin/env bash
set -euo pipefail

# RESUME the kiwi π0.5 1M run (pi05_openpi_split_lora_masked_1459_bs4_1m) from
# its last checkpoint (0700000 of 1M as of 2026-08-16). All hyperparameters are
# reloaded verbatim from the checkpoint's saved train_config.json — masked-
# subspace flow (flow_matching_padding_mode=masked_subspace), split-rank LoRA
# (PaliGemma r/α 16, gemma_expert r/α 32), bs4, lr 5e-5→5e-6 cosine over the
# full 1M steps, seed 1000, chunk 30 — so nothing can drift from the original
# launch (run_pi05_split_lora_kiwi_1459_1m.sh). --resume=true restores
# step/optimizer/scheduler state from checkpoints/last and continues the SAME
# W&B run (WandBLogger picks the id up from wandb/latest-run).
#
# Usage (on kiwi, no tmux there):
#   cd /home/zfei/code/lerobot-fei-v5.0-umi-unified
#   LOG=outputs/train/pi05_openpi_split_lora_masked_1459_bs4_1m/train_resume_$(date +%Y%m%d_%H%M%S).log
#   nohup bash examples/umi_relative_ee/shell_scripts/resume_pi05_split_lora_kiwi_1459_1m.sh \
#     > "$LOG" 2>&1 &

PY="${PYTHON_BIN:-.venv/bin/python}"
REPO="${REPO:-/home/zfei/code/lerobot-fei-v5.0-umi-unified}"
RUN_NAME="pi05_openpi_split_lora_masked_1459_bs4_1m"
CKPT_LAST="${REPO}/outputs/train/${RUN_NAME}/checkpoints/last/pretrained_model"

cd "$REPO"

exec env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  "${PY}" examples/umi_relative_ee/train_pi05_lora.py \
  --config_path="${CKPT_LAST}/train_config.json" \
  --resume=true
