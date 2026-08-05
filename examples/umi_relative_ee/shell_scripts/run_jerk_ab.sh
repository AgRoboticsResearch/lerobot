#!/usr/bin/env bash
# Within-chunk rotation-jitter A/B: masked (default) vs legacy full-width-noise
# inference, for SmolVLA@0200000 and pi0.5@100000. 5 eps / 25 samples each.
set -uo pipefail
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
PY=/home/zfei/anaconda3/envs/py312/bin/python
DS=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
RID=sroi/sroiv2_strawberry_picking_lab_validation
SMOL=outputs/train/smolvla_umi_identity_rot6d_1302/checkpoints/0200000/pretrained_model
PI=outputs/train/pi05_lora_r16_umi_identity_rot6d_1302/checkpoints/100000/pretrained_model
mkdir -p outputs/debug/jerk_ab
run() {  # $1=name $2=ckpt $3=extra
  out=outputs/debug/jerk_ab/$1; rm -rf "$out"
  PYTHONPATH=src "$PY" examples/umi_relative_ee/eval_open_loop_dataset.py \
    --pretrained_path "$2" --dataset_root "$DS" --repo_id "$RID" \
    --episode_indices 0 1 2 3 4 --samples_per_episode 5 --device cuda $3 \
    --output_dir "$out" >/dev/null 2>&1
}
run smolvla_masked   "$SMOL" ""
run smolvla_unmasked "$SMOL" "--legacy_full_action_noise"
run pi05_masked      "$PI" ""
run pi05_unmasked    "$PI" "--legacy_full_action_noise"
{
echo "name,rot_jerk_deg,gt_rot_jerk_deg,rotation_end_deg,xyz_jerk_m"
for n in smolvla_masked smolvla_unmasked pi05_masked pi05_unmasked; do
  f=$(ls outputs/debug/jerk_ab/$n/*_open_loop_metrics.json 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    "$PY" -c "import json;d=json.load(open('$f'))['summary']['episode_balanced'];print('$n,%.4f,%.4f,%.4f,%.5f'%(d['rot_jerk_deg'],d['gt_rot_jerk_deg'],d['rotation_end_deg'],d['xyz_jerk_m']))"
  else echo "$n,MISSING,,,"
  fi
done
} > outputs/debug/jerk_ab/results.csv
echo DONE > outputs/debug/jerk_ab/_done
