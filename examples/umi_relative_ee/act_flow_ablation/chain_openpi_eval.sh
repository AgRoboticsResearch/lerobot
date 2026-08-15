#!/usr/bin/env bash
# Supervise the openpi SROI runs end-to-end (v3, corrected checkpoint names):
#  - openpi names the FINAL 20k checkpoint "19999" (0-indexed last step) and the
#    keep_period checkpoint "10000"; the earlier v2 guessed 010000/020000.
#  1. when each arm's training finishes, free its 10000 checkpoint (disk guard)
#  2. when BOTH finish, run the open-loop eval on each arm's FINAL (19999)
#     checkpoint. Results -> /mnt/data1 artifacts.
set -uo pipefail
ts(){ date '+%F %T'; }
RUNLOG=/mnt/data1/projects/lerobot-arch-exp/logs/openpi_sroi_run.log
LOG=/mnt/data1/projects/lerobot-arch-exp/logs/openpi_eval_chain.log
exec > >(tee -a "$LOG") 2>&1

CKPT_ROOT=/home/zfei/codes/openpi/checkpoints
echo "[$(ts)] chain v3: supervising..."
freed_rotvec=0
freed_rot6d=0
while true; do
  if [ "$freed_rotvec" = "0" ] && grep -q "=== done pi05_lora_sroi_rotvec ===" "$RUNLOG"; then
    echo "[$(ts)] chain: rotvec done -- freeing rotvec/10000 (disk guard)"
    rm -rf "$CKPT_ROOT/pi05_lora_sroi_rotvec/run1/10000" && freed_rotvec=1
  fi
  if [ "$freed_rot6d" = "0" ] && grep -q "=== done pi05_lora_sroi_rot6d ===" "$RUNLOG"; then
    echo "[$(ts)] chain: rot6d done -- freeing rot6d/10000 (disk guard)"
    rm -rf "$CKPT_ROOT/pi05_lora_sroi_rot6d/run1/10000" && freed_rot6d=1
  fi
  grep -q "ALL openpi SROI runs finished" "$RUNLOG" && break
  pgrep -f "sroi_run_all[.]sh" >/dev/null || pgrep -f "scripts/train[.]py" >/dev/null || { sleep 10; break; }
  sleep 60
done
sleep 30
if ! grep -q "ALL openpi SROI runs finished" "$RUNLOG"; then
  echo "[$(ts)] chain: training did NOT complete cleanly -- aborting"; exit 1
fi
echo "[$(ts)] chain: trainings done. running evals..."

cd /home/zfei/codes/openpi
export HF_LEROBOT_HOME=/mnt/data1/sroi/lerobot
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=/home/zfei/codes/openpi/.venv/bin/python
EVAL=/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/act_flow_ablation/eval_openpi_open_loop.py
OUT=/mnt/data1/projects/lerobot-arch-exp/outputs/research_report/openpi_sroi_eval
mkdir -p "$OUT"

for cfg in pi05_lora_sroi_rotvec pi05_lora_sroi_rot6d; do
  step=19999
  ckpt=$CKPT_ROOT/$cfg/run1/$step
  json="$OUT/${cfg}_final_open_loop_metrics.json"
  if [ -f "$json" ]; then echo "[$(ts)] SKIP existing $(basename "$json")"; continue; fi
  if [ ! -d "$ckpt/params" ]; then echo "[$(ts)] WARN no ckpt $ckpt"; continue; fi
  echo "[$(ts)] EVAL $cfg@$step"
  "$PY" "$EVAL" --config-name "$cfg" --checkpoint "$ckpt" --output "$json" \
    || echo "[$(ts)] EVAL FAILED for $cfg@$step (continuing)"
done
echo "[$(ts)] chain: ALL DONE -> $OUT"
