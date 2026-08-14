#!/usr/bin/env bash
# Wait for both openpi SROI trainings to finish, then run the open-loop eval
# (eval_openpi_open_loop.py, same protocol as the SmolVLA notation eval) on the
# 10k and 20k checkpoints of both variants. Results -> /mnt/data1 artifacts.
set -uo pipefail
ts(){ date '+%F %T'; }
RUNLOG=/mnt/data1/projects/lerobot-arch-exp/logs/openpi_sroi_run.log
LOG=/mnt/data1/projects/lerobot-arch-exp/logs/openpi_eval_chain.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(ts)] eval-chain: waiting for both trainings..."
while true; do
  if grep -q "ALL openpi SROI runs finished" "$RUNLOG"; then break; fi
  if grep -q "train FAILED" "$RUNLOG" && tail -5 "$RUNLOG" | grep -q "FAILED"; then
    : # orchestrator exits on failure; detect via process check below
  fi
  pgrep -f "sroi_run_all[.]sh" >/dev/null || pgrep -f "scripts/train[.]py" >/dev/null || break
  sleep 60
done
sleep 30
if ! grep -q "ALL openpi SROI runs finished" "$RUNLOG"; then
  echo "[$(ts)] eval-chain: training did NOT complete cleanly -- aborting"; exit 1
fi
echo "[$(ts)] eval-chain: trainings done. running evals..."

cd /home/zfei/codes/openpi
export HF_LEROBOT_HOME=/mnt/data1/sroi/lerobot
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=/home/zfei/codes/openpi/.venv/bin/python
EVAL=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/act_flow_ablation/eval_openpi_open_loop.py
OUT=/mnt/data1/projects/lerobot-arch-exp/outputs/research_report/openpi_sroi_eval
mkdir -p "$OUT"

for cfg in pi05_lora_sroi_rotvec pi05_lora_sroi_rot6d; do
  for step in 010000 020000; do
    ckpt=/home/zfei/codes/openpi/checkpoints/$cfg/run1/$step
    json="$OUT/${cfg}_${step}_open_loop_metrics.json"
    if [ -f "$json" ]; then echo "[$(ts)] SKIP existing $(basename "$json")"; continue; fi
    if [ ! -d "$ckpt/params" ]; then echo "[$(ts)] WARN no ckpt $ckpt"; continue; fi
    echo "[$(ts)] EVAL $cfg@$step"
    "$PY" "$EVAL" --config-name "$cfg" --checkpoint "$ckpt" --output "$json" \
      || echo "[$(ts)] EVAL FAILED for $cfg@$step (continuing)"
  done
done
echo "[$(ts)] eval-chain: ALL DONE -> $OUT"
