#!/usr/bin/env bash
# Persistent lightweight health monitor for the confirmation training/eval push.
# Appends one concise status line every 5 min to a single log so a supervisor can
# tail it cheaply. Detects: job count, per-job latest step + errors, VRAM, disk
# free, eval-chain phase, and completion. Never kills anything.
set -uo pipefail
ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
MONLOG="$ROOT/logs/confirmation_monitor.log"
mkdir -p "$ROOT/logs"
echo "[monitor $(date '+%F %T')] started" >> "$MONLOG"

while :; do
  ts=$(date '+%F %T')
  # training jobs: name + last step + error count
  jobs=""
  ntrain=0
  for name in act_r50_v1_vae_seed3000_100000steps act_r50_vae_seed3000_100000steps \
              act_r50_v1_vae_seed2000_100000steps act_r18_flow_u_lr1e5_seed3000_100000steps \
              diffusion_r18_seed3000_100000steps; do
    log="$ROOT/logs/$name.log"
    if pgrep -fa train_umi_relative_ee.py >/dev/null 2>&1 && \
       pgrep -fa "job_name=$name" >/dev/null 2>&1; then
      st=$(grep -oE '[0-9]+/[0-9]+ ' "$log" 2>/dev/null | tail -1 | tr -d ' \r')
      jobs="$jobs ${name%_seed*}:${st:-?}"
      ntrain=$((ntrain + 1))
    fi
  done
  vram=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  dfree=$(df --output=avail -BG /media/zfei/Glowat512 2>/dev/null | tail -1 | tr -d ' G')
  # eval-chain phase
  phase=$(grep -hE 'all confirmation runs complete|evaluating seed|final collect|eval-chain finished' \
    "$ROOT"/logs/confirmation_eval_chain_*.log 2>/dev/null | tail -1)
  echo "[monitor $ts] ntrain=$ntrain vram(used,free,util,temp)=[$vram] diskfree=${dfree}G jobs:$jobs ${phase:+phase:$phase}" >> "$MONLOG"
  sleep 300
done
