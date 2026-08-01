#!/usr/bin/env bash
# Zero-noise inference visualizations for SmolVLA@0200000 and pi0.5@100000.
# Same models / dataset / episodes as run_jerk_ab.sh, but emits prediction MP4s
# (predicted vs GT gripper trajectory) under zero-noise flow-matching decoding.
set -uo pipefail
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
PY=/home/zfei/anaconda3/envs/py312/bin/python
DS=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
RID=sroi/sroiv2_strawberry_picking_lab_validation
# D405 is fixed-calibration, so any episode's intrinsics work for --project.
CAM=$DS/meta/camera_info/validation_20260714_160922-png__episode_001/camera_info_color.json
SMOL=outputs/train/smolvla_umi_identity_rot6d_1302/checkpoints/0200000/pretrained_model
PI=outputs/train/pi05_lora_r16_umi_identity_rot6d_1302/checkpoints/100000/pretrained_model
mkdir -p outputs/debug/zero_noise_vis

run() {  # $1=name $2=ckpt
  out=outputs/debug/zero_noise_vis/$1
  rm -rf "$out"; mkdir -p "$out"
  PYTHONPATH=src "$PY" examples/umi_relative_ee/visualize_predictions.py \
    --pretrained_path "$2" --dataset_root "$DS" --repo_id "$RID" \
    --episode_indices 0 1 2 3 4 --zero_noise --project \
    --camera_info_path "$CAM" --device cuda --output_dir "$out"
}

run smolvla "$SMOL"
run pi05    "$PI"

echo "DONE" > outputs/debug/zero_noise_vis/_done
echo
echo "=== zero-noise videos ==="
find outputs/debug/zero_noise_vis -name 'pred_episode_*.mp4' | sort
