#!/usr/bin/env bash
# SmolVLA masked-subspace padding 1M budget run (user-directed 2026-08-19).
# IDENTICAL recipe to the host full-width 1M run (smolvla_rot6d seed 1000,
# §9.2.10) except --policy.flow_matching_padding_mode=masked_subspace —
# the flow-matching padding A/B (Option B) extended to the 1M budget regime.
#
# Gate: waits for the pi0.5 1M trainer (train_pi05_lora.py) to exit — the
# same gate kiwi_eval_unified_h10.sh uses. The SmolVLA trainer needs ~3 GB
# and coexists with the K1–K3 torch evals (~4–6 GB) on the 16 GB 5080.
# After "End of training" it chains the dual-protocol eval driver
# (kiwi_smolvla_masked_1m_eval.sh: h30 curve + unified h10, 3 seeds x 10 ckpts).
#
# Relaunch-safe: if the trainer died mid-run, re-running this script resumes
# from checkpoints/last (--resume=true is added automatically); the run log
# is appended, never truncated.
set -uo pipefail
REPO=/home/zfei/kiwi-ablation
VENV=/home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python
ART=/home/zfei/kiwi-ablation/artifacts
TRAIN_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_1459_occlusion
VAL_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
STEPS=1000000
NAME=smolvla_rot6d_masked_seed1000_${STEPS}steps
OUT=$ART/train/$NAME
LOG=$ART/logs/$NAME.log
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$ART/train" "$ART/logs"

if [ -f "$OUT/checkpoints/1000000/pretrained_model/config.json" ] \
   && grep -q 'End of training' "$LOG" 2>/dev/null; then
  echo "[$(date '+%F %T')] $NAME already complete — chaining evals only"
  exec bash ~/kiwi-ablation/kiwi_smolvla_masked_1m_eval.sh
fi

echo "[$(date '+%F %T')] waiting for pi0.5 trainer (train_pi05_lora.py) to exit"
while pgrep -f train_pi05_lora.py >/dev/null 2>&1; do sleep 300; done
echo "[$(date '+%F %T')] GPU free of pi0.5 — starting $NAME"

RESUME=()
[ -d "$OUT/checkpoints/last/pretrained_model" ] && RESUME=(--resume=true)

cd "$REPO" && PYTHONPATH=src "$VENV" examples/umi_relative_ee/train_umi_relative_ee.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1459_occlusion --dataset.root="$TRAIN_ROOT" \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation --validation_dataset.root="$VAL_ROOT" \
  --dataset.use_imagenet_stats=true --validation_dataset.use_imagenet_stats=true \
  --dataset.video_backend=pyav --validation_dataset.video_backend=pyav \
  --policy.device=cuda --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
  --policy.push_to_hub=false --seed=1000 --steps="$STEPS" \
  --num_workers=4 --prefetch_factor=4 --persistent_workers=true \
  --log_freq=200 --val_freq=10000 --eval_freq=0 --save_checkpoint=true --save_freq=100000 \
  --output_dir="$OUT" --job_name="$NAME" --wandb.enable=false --batch_size=8 \
  --policy.path=lerobot/smolvla_base --policy.input_features=null \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.umi_rotation_representation=rot6d \
  --policy.flow_matching_padding_mode=masked_subspace --policy.train_state_proj=true \
  --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="$STEPS" --policy.scheduler_decay_lr=0.0000025 \
  "${RESUME[@]}" \
  >> "$LOG" 2>&1
status=$?
if [ "$status" -ne 0 ] || ! grep -q 'End of training' "$LOG"; then
  echo "[$(date '+%F %T')] TRAINING FAILED (exit=$status, see $LOG) — NOT chaining evals"
  exit 1
fi
echo "[$(date '+%F %T')] === $NAME training COMPLETE ==="
exec bash ~/kiwi-ablation/kiwi_smolvla_masked_1m_eval.sh
