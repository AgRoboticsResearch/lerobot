#!/usr/bin/env bash
# Chain for the §9.2.5 horizon-30 stack A/B on the single host RTX 4090:
#   1) official openpi (JAX) pi05_lora_sroi_rot6d_h30 -- 20k steps @ bs16, ~4.0 s/it (~22 h)
#   2) lerobot-port (PyTorch) openpi-matched run        -- 20k steps @ bs16, ~2.15 s/it (~12 h)
# Both smoke-tested 2026-08-16 (12 steps each, clean end, no OOM). Norm stats for
# the h30 openpi config were copied from the h10 rot6d arm (per-dim quantiles are
# horizon-independent). Disk guard: openpi keeps 10k+20k (keep_period=10000);
# the 10000 intermediate is deleted after training like the previous arms.
set -uo pipefail

ts() { date '+%F %T'; }
LOGDIR=/mnt/data1/projects/lerobot-arch-exp/logs
mkdir -p "$LOGDIR"

# ---- stage 1: official openpi JAX h30 -------------------------------------- #
cd /home/zfei/codes/openpi
export HF_LEROBOT_HOME=/mnt/data1/sroi/lerobot
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

if [ -d checkpoints/pi05_lora_sroi_rot6d_h30/run1/19999 ]; then
  echo "[$(ts)] stage 1 already complete (19999 exists), skipping"
else
  echo "[$(ts)] stage 1: openpi pi05_lora_sroi_rot6d_h30 (run1)"
  uv run scripts/train.py pi05_lora_sroi_rot6d_h30 --exp-name=run1 --overwrite \
    || { echo "[$(ts)] stage 1 FAILED"; exit 1; }
  # disk guard: drop the 10k intermediate (keep 19999 final) like the h10 arms
  rm -rf checkpoints/pi05_lora_sroi_rot6d_h30/run1/10000
  echo "[$(ts)] stage 1 done"
fi

# ---- stage 2: lerobot-port PyTorch, openpi-matched args -------------------- #
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
PORT_OUT=/mnt/data1/projects/lerobot-arch-exp/outputs/train/pi05_port_openpi_args_rot6d_h30_bs16_20k
if [ -f "${PORT_OUT}/checkpoints/020000/pretrained_model/config.json" ]; then
  echo "[$(ts)] stage 2 already complete, skipping"
else
  echo "[$(ts)] stage 2: lerobot-port openpi-matched h30"
  bash examples/umi_relative_ee/act_flow_ablation/run_pi05_port_openpi_matched_h30.sh \
    || { echo "[$(ts)] stage 2 FAILED"; exit 1; }
  echo "[$(ts)] stage 2 done"
fi

echo "[$(ts)] === h30 stack A/B chain finished ==="
