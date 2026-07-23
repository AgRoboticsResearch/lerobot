#!/usr/bin/env bash
set -euo pipefail

# LIBERO ACT Training
# Task suite: libero_object (280 max steps, object manipulation)
# Dataset: HuggingFaceVLA/libero (Local: /mnt/data0/data/libero/LIBERO_HF)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use /mnt/data0 for cache (root / is 90% full)
export HF_DATASETS_CACHE=/mnt/data0/.cache/huggingface/datasets
export HF_HOME=/mnt/data0/.cache/huggingface
export TMPDIR=/mnt/data0/.tmp
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$TMPDIR"

# LIBERO assets will be downloaded to ~/.cache/libero/assets (symlinked to /mnt/data0/data/libero/LIBERO_HF/assets)

lerobot-train \
    --policy.type=act \
    --policy.repo_id=libero-local \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=/mnt/data0/data/libero/LIBERO_HF \
    --env.type=libero \
    --env.task=libero_object \
    --output_dir="${SCRIPT_DIR}/checkpoints" \
    --batch_size=8 \
    --steps=100000 \
    --eval_freq=10000 \
    --save_freq=10000 \
    --eval.batch_size=1 \
    --eval.n_episodes=3 \
    --log_freq=200
