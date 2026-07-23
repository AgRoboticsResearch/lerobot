# Visualizing π0.5 LoRA predictions (UMI relative-EE) on the validation set

`visualize_predictions.py` (this folder) runs **open-loop prediction** on the
validation dataset: for each frame it feeds the π0.5 LoRA model the GT
observation (camera image + two-pose state derived from the action delta + the
task string), predicts the 30-step action chunk, and renders a per-frame video.
With `--project` it also draws the predicted and GT gripper-tip trajectories
directly on the camera image.

Built on the sibling `lerobot/examples/umi_relative_ee/visualize_predictions.py`,
adapted for π0.5's UMI processor and LoRA checkpoints.

## Requirements
- Python env: `/home/zfei/anaconda3/envs/py312/bin/python`
- Trained checkpoint: `outputs/train/pi05_lora_umi_relative_ee/checkpoints/050000/pretrained_model`
  (final = best-val; val loss 0.0572 — π0.5 improved monotonically to 50K)
- Validation dataset: `/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation`
- Hand-eye: `examples/umi_relative_ee/camera_gripper_extrinsics_sroi_v2_d405.json`
  (D405 rig; copied from `sroi_rosbag_utilities` — self-contained, no folder dependency)

## Run

Panel-composite (camera + 3D trajectory + per-dim curves), no on-image projection:
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/pi05_lora_umi_relative_ee/checkpoints/050000/pretrained_model \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2
```

With on-image pixel projection (predicted green→red + GT cyan gripper-tip trajectory):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/pi05_lora_umi_relative_ee/checkpoints/050000/pretrained_model \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --project
```

Output: `outputs/debug/viz_pi05_umi/<repo_id>/pred_episode_<N>.mp4` (override with `--output_dir`).

`--project` auto-finds the color intrinsics `K` under the dataset's
`meta/camera_info/` (override with `--camera_info_path`) and loads the hand-eye
from `--extrinsics_config` (default the local JSON). The projected start point is
**(327, 321)** — identical to `sroi_rosbag_utilities`'s `visualize_traj_video.py`.

## π0.5-specific notes (gotchas solved)
- **LoRA loading**: `from_pretrained(base, config=policy_config)` then
  `PeftModel.from_pretrained(adapter)`. The `config=policy_config` is essential —
  without it the model uses base π0.5's `base_0_rgb` image keys instead of
  `observation.images.camera`.
- **State derivation**: π0.5's UMI processor derives a 20D state from
  `action[t-1:t+1]`, so the val set is built with
  `resolve_delta_timestamps(policy.config, meta)` and driven through the
  **dataloader + `lerobot_collate_fn`** (a hand-built batch dict doesn't survive
  the processor's transition mapping — `validate_policy` is the template).
- **Task**: π0.5 is PaliGemma-language-conditioned; `--task "pick the strawberry"`
  is injected for tokenization (default).
- **Projection**: uses the D405 JSON hand-eye (copied sroi `load_tip_kin` /
  `project_future` / `_green_red_gradient`), **not** the Piper URDF — the URDF's
  `camera→ee` transform differs and shifts the start to (355, 229).

## Options
`--no_gt`, `--task`, `--fps`, `--device`, `--extrinsics_config`,
`--camera_info_path`, `--output_dir`, `--first_frame_debug`.
