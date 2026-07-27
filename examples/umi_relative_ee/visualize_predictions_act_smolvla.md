# Visualizing ACT / SmolVLA predictions (UMI relative-EE) on the validation set

`examples/umi_relative_ee/visualize_predictions.py` (this repo, **py310**) runs
**open-loop prediction** on the validation dataset for the ACT and SmolVLA
UMI-relative-EE models and renders, per episode, a video with the predicted and
GT gripper-tip trajectories projected onto the camera image (predicted
green→red, GT overlaid) plus a separate 3D trajectory video.

## Requirements
- Python env: `/home/zfei/anaconda3/envs/py310/bin/python`
- Validation dataset: `/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation`
- Intrinsics `K`: any `camera_info_color.json` under the dataset's `meta/camera_info/`
  (D405 uses fixed calibration, so any episode's file works)
- Hand-eye: `examples/umi_relative_ee/camera_gripper_extrinsics_sroi_v2_d405.json`
  (D405 rig; default for `--extrinsics_config`, copied from `sroi_rosbag_utilities`)

## Checkpoints
| model | best-val | final |
|-------|----------|-------|
| ACT | `outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/0200000` (best ~180k, val ≈0.046) | `.../checkpoints/2500000` (val 0.0480) |
| SmolVLA | `outputs/train/smolvla_relative_ee_chunk30_strawberry_1M/checkpoints/0500000` (val 0.01418) | `.../checkpoints/1000000` (val 0.01555) |

## Run

ACT (no task needed):
```bash
/home/zfei/anaconda3/envs/py310/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --inference \
  --pretrained_path outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/0200000/pretrained_model \
  --gt --mp4 \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_018/camera_info_color.json \
  --output_dir outputs/debug/viz_act
```

SmolVLA (needs `--task`):
```bash
/home/zfei/anaconda3/envs/py310/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --inference \
  --pretrained_path outputs/train/smolvla_relative_ee_chunk30_strawberry_1M/checkpoints/0500000/pretrained_model \
  --task "pick the strawberry" --gt --mp4 \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_018/camera_info_color.json \
  --output_dir outputs/debug/viz_smolvla
```

ACT trained on the 1125-episode dataset (1M-step checkpoint):
```bash
/home/zfei/anaconda3/envs/py310/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --inference \
  --pretrained_path outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1000000/pretrained_model \
  --gt --gripper --mp4 \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_040/camera_info_color.json \
  --output_dir outputs/debug/viz_act_1125_ckpt_1m_validation
```

This command was verified on validation episodes 0, 1, and 2. It rendered 164
valid frames and produced all six expected H.264 videos (one projected-camera
video and one 3D video per episode). The open-loop, chunk-end errors aggregated
over those frames were:

| metric | value |
|--------|-------|
| Mean translation error | 0.02887 m |
| Median translation error | 0.02662 m |
| Mean rotation error | 0.06110 rad |
| Mean gripper error | 0.23945 |

Output (per episode): `<output_dir>/<repo_id>/proj_inference_episode_<N>.mp4`
(camera + projected trajectory) and `traj3d_inference_episode_<N>.mp4` (3D).
When `--gt` is enabled, aggregate and per-frame errors are also written to
`<output_dir>/<repo_id>/prediction_metrics.json`.

## Fixes applied to this script (vs upstream)
- **ACT VAE prior**: at inference the preprocessor left `ACTION=None`, which sent
  ACT's VAE down the posterior branch and crashed (`vae_encoder_action_input_proj`
  got `None`). Added a pop so it uses the prior path:
  ```python
  if processed.get(ACTION) is None:
      processed.pop(ACTION, None)   # right before policy.predict_action_chunk(processed)
  ```
- **D405 JSON hand-eye**: the projection now loads `T_opt_cam` / `T_cam_ee` from
  `camera_gripper_extrinsics_sroi_v2_d405.json` (via copied `load_tip_kin` /
  `_load_rigid_transform`) through `--extrinsics_config`, instead of the Piper URDF
  via `placo` (`get_kinematic_transforms`). The URDF's `camera_link→ee_link`
  transform differs from the measured D405 rig and shifts the projected start from
  **(327, 321)** to (355, 229). The JSON is copied next to the script — no runtime
  dependency on the `sroi_rosbag_utilities` folder.

Both changes make the projected start point **(327, 321)**, matching
`sroi_rosbag_utilities`'s `visualize_traj_video.py --extrinsics-config configs/camera_gripper_extrinsics_sroi_v2_d405.json`.

## Note on checkpoint choice
ACT best-val was ~180k (val ≈0.046) vs final 0.0480; SmolVLA best-val was 500k
(0.01418) vs final 0.01555. For the tightest predictions use the **best-val**
checkpoints; the final ones show the fully-trained model.
