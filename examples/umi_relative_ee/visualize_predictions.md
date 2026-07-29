# Visualizing UMI relative-EE predictions (ACT / SmolVLA / π0.5)

`visualize_predictions.py` (this folder) runs **open-loop prediction** on a recorded
dataset: for each frame it feeds the policy the GT observation (camera image + the
two-pose state derived from the action delta + the task string), predicts the
`chunk`-step action chunk through the checkpoint's saved UMI relative-EE
processors, and renders a per-episode **panel-composite** video
(camera | 3D trajectory | per-dimension curves). Pass `--project` to also draw the
predicted (green→red) and GT (cyan) gripper-tip trajectories on the camera image.

One script covers all three policies. **LoRA adapters are detected automatically**
(if `adapter_config.json` is present it loads the base named there and applies the
adapter), so the same command works for full-fine-tuned ACT / SmolVLA and for
π0.5 LoRA.

## Requirements
- **Python env**: any lerobot env with the policy plus `imageio` / `matplotlib` / `cv2`.
  Locally: `/home/zfei/anaconda3/envs/py312/bin/python`. On kiwi: the unified
  checkout's `.venv/bin/python` (run `uv pip install matplotlib imageio imageio-ffmpeg`
  if missing — `cv2`/`av` come from the `dataset`/`av-dep` extras).
- **Trained checkpoint** that saved its processors (`<ckpt>/pretrained_model`).
- **Dataset root** — the validation set for generalization, or the training set for
  an overfit check.
- **Intrinsics `K`** (`--project` only): a `camera_info_color.json` under the
  dataset's `meta/camera_info/`, auto-found if you omit `--camera_info_path`. D405
  uses fixed calibration, so any episode's file works.
- **Hand-eye** (`--project` only): `camera_gripper_extrinsics_sroi_v2_d405.json`,
  the default for `--extrinsics_config` (lives next to this script; copied from
  `sroi_rosbag_utilities`).

## Run

Panel composite only — no calibration needed:
```bash
python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path <CKPT>/pretrained_model \
  --dataset_root <DATASET_ROOT> \
  --episode_indices 0 1 2
```

Panel composite **+ on-image trajectory projection**:
```bash
python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path <CKPT>/pretrained_model \
  --dataset_root <DATASET_ROOT> \
  --episode_indices 0 1 2 --project
```

Output: `<output_dir>/<repo_id>/pred_episode_<N>.mp4` (default
`outputs/debug/viz_umi`; override with `--output_dir`).

### Per-policy notes
- **ACT** — no `--task` needed. Best-val checkpoint usually fits tightest.
- **SmolVLA** — needs the language task; `--task "pick the strawberry"` is the
  default so you can usually omit it. PaliGemma-tokenized internally.
- **π0.5 LoRA** — point `--pretrained_path` at the LoRA adapter dir; the base model
  named in `adapter_config.json` is loaded automatically. No extra flags.

## Overfit / training-episode check
To confirm the pipeline can fit a specific episode (e.g. after a one-episode
overfit run), point `--dataset_root` at the **training** set and select that
episode. A correct overfit reproduces the GT trajectory near-perfectly (predicted
overlays GT):
```bash
python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/smolvla_one_ep_debug/checkpoints/050000/pretrained_model \
  --dataset_root /path/to/sroiv2_strawberry_picking_lab_1000onesb_1125 \
  --episode_indices 0 --project
```

## Gotchas solved in this script
- **π0.5 LoRA loading**: `policy_class.from_pretrained(base, config=policy_config)`
  then `PeftModel.from_pretrained(adapter)`. The `config=policy_config` is essential
  — without it the model uses base π0.5's `base_0_rgb` image keys instead of
  `observation.images.camera`.
- **π0.5 state derivation**: π0.5's UMI processor derives a 20D state from
  `action[t-1:t+1]`, so the dataset is built with
  `resolve_delta_timestamps(policy.config, meta)` and driven through the
  **dataloader + `lerobot_collate_fn`** (a hand-built batch dict does not survive
  the processor's transition mapping — `validate_policy` is the template).
- **ACT VAE prior at inference**: the preprocessor would otherwise leave
  `ACTION=None` and send ACT's VAE down the posterior branch; the script pops it so
  inference uses the prior path.
- **Projection hand-eye**: loads `T_opt_cam` / `T_cam_ee` from the D405 JSON (copied
  `load_tip_kin` / `project_future`), **not** the Piper URDF — the URDF's
  camera→ee transform shifts the projected start from **(327, 321)** to (355, 229).
  The JSON default matches `sroi_rosbag_utilities`'s `visualize_traj_video.py`.

## Options
`--no_gt`, `--task`, `--fps`, `--device`, `--seed`, `--repo_id`, `--output_dir`,
`--extrinsics_config`, `--camera_info_path`, `--max_frames_per_episode`,
`--first_frame_debug`, `--project`.

## Checkpoint choice
For ACT and SmolVLA the **best-val** checkpoint usually predicts tighter than the
final one (both overfit late). Use best-val for the cleanest trajectories; final to
see the fully-trained model.
