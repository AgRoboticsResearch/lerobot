> Historical runbook preserved from `fei`. For the maintained v5 command and shared `use_umi_relative_ee` interface, start with `README.md`.

# SmolVLA UMI Relative-EE Training

This guide documents the validated SmolVLA training workflow for the strawberry-picking datasets in this checkout. It uses the processor-based UMI relative end-effector (EE) representation implemented by `train_relative_ee_processor.py`.

## What this pipeline trains

The source dataset stores absolute 7D EE poses:

```text
[x, y, z, wx, wy, wz, gripper]
```

The first six values are an absolute position plus axis-angle rotation. The final value is the absolute gripper state. The dataset does not need an `observation.state` column because the training preprocessor derives state from adjacent actions.

The processor converts the raw data into the representation consumed by SmolVLA:

```text
Raw action:       [B, chunk + 1, 7] absolute axis-angle EE poses
Derived state:    [B, 2, 7]
Model state:      [B, 20]            relative rot6d
Model action:     [B, chunk, 10]     relative rot6d
```

Each 10D action is:

```text
[dx, dy, dz, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]
```

All actions in a predicted chunk are relative to the same pose at the chunk boundary:

```text
T_relative[i] = inverse(T_chunk_base) @ T_target[i]
T_target[i]   = T_chunk_base @ T_relative[i]
```

They are not chained from one prediction to the next.

## Dataset used by the validated run

Training dataset:

```text
repo_id:        sroi/sroiv2_strawberry_picking_lab_1000onesb
root:           /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb
episodes:       1012
frames:         88218
fps:            30
camera:         observation.images.camera, 480x640 RGB
action:         7D absolute EE pose
task:           "pick the strawberry"
```

Validation dataset:

```text
repo_id:        sroi/sroiv2_strawberry_picking_lab_validation
root:           /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
episodes:       100
frames:         9274
task:           "pick the strawberry"
```

Validation uses normalization statistics computed from the training dataset. It does not recompute statistics from validation data.

## Processor order

The SmolVLA relative-EE preprocessor runs these steps:

1. Rename observations.
2. Add a batch dimension when needed.
3. Ensure each language task ends with a newline.
4. Tokenize the language task with the SmolVLM tokenizer.
5. Move tensors to the policy device.
6. Derive the two-step state from actions at `t-1` and `t`.
7. Convert absolute 7D actions to relative 10D rot6d actions.
8. Convert and flatten the two-step state to 20D relative rot6d.
9. Apply min-max normalization to state and action.

The postprocessor performs the inverse action transformation:

1. Unnormalize the predicted 10D actions.
2. Compose each prediction with the cached chunk-base pose and convert it to an absolute 7D EE pose.
3. Move output to CPU.

The preprocessor and postprocessor, including training statistics, are saved with every checkpoint.

## Model initialization

Use the pretrained SmolVLA checkpoint:

```text
lerobot/smolvla_base
```

Specify it with `--policy.path`, not `--policy.type`:

```bash
--policy.path=lerobot/smolvla_base
```

The pretrained configuration uses the `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` VLM. Under the recommended fine-tuning configuration:

```text
Total parameters:       approximately 450M
Trainable parameters:   approximately 100M
freeze_vision_encoder:  true
train_expert_only:      true
train_state_proj:       true
```

This uses the transformer vision encoder for image features while keeping the pretrained VLM, including its vision encoder, frozen. The action expert and state projection are trained.

## Full training command

Run from the LeRobot repository root:

```bash
HF_HUB_DISABLE_XET=1 python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=10000 \
  --policy.path=lerobot/smolvla_base \
  --output_dir=outputs/train/smolvla_relative_ee_chunk30_strawberry \
  --job_name=smolvla_relative_ee_chunk30_strawberry \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.repo_id=zfff/smolvla_relative_ee_strawberry \
  --wandb.enable=true \
  --save_freq=5000 \
  --steps=20000 \
  --batch_size=8 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true
```

`HF_HUB_DISABLE_XET=1` avoids Xet transfer failures and uses the standard Hugging Face download path. Once every model file is cached, `HF_HUB_OFFLINE=1` can additionally be used for fully local startup.

## Why these training settings are different from ACT

Do not reuse the 2.5-million-step ACT schedule for SmolVLA. SmolVLA starts from a pretrained VLM and action expert, and the recommended first experiment is approximately 20,000 fine-tuning steps.

Suggested progression:

1. Train for 20,000 steps and inspect training loss, validation loss, and rollout performance.
2. Compare checkpoints at 5,000-step intervals.
3. Continue toward 50,000 or 100,000 only if validation and real-robot performance are still improving.

Batch size 8 was verified on an NVIDIA RTX 4090 with one 480x640 camera. Reduce it to 4, 2, or 1 if image count, resolution, augmentation, or trainable model scope increases.

## Validation behavior

At every positive multiple of `val_freq`, the trainer:

1. Switches the policy to evaluation mode.
2. Resets the stateful preprocessor.
3. Processes the full selected validation dataset without gradients.
4. Computes sample-weighted validation metrics.
5. Resets the preprocessor again and restores training mode.
6. Logs metrics to W&B with the `val/` prefix.

For SmolVLA, the main metric is:

```text
val/loss
```

This is a flow-matching mean squared error. ACT also logs a metric called `val/loss`, but ACT's value is an L1 reconstruction loss plus a weighted VAE KL-divergence term. The raw ACT and SmolVLA loss values therefore must not be compared directly.

Use `val/loss` to compare checkpoints from the same policy and configuration. For a policy-independent ACT-versus-SmolVLA comparison, use metrics such as:

- Absolute EE translation error.
- Rotation error in degrees.
- Gripper error.
- Closed-loop task success rate.

To smoke-test validation on a single episode:

```bash
--validation_dataset.episodes=[0] \
--val_freq=1
```

The validated one-episode smoke run completed with:

```text
val/loss=0.690596
```

This number only confirms that the complete validation path works. It is not a trained-policy benchmark.

## Expected startup sequence

A healthy run should log the following stages:

```text
Creating dataset
Training with UMI-style: Processor Pipeline + rot6d (10D)
Recomputing stats for relative rot6d actions
Updated action metadata shape: (7,) -> [10]
Added derived observation.state to metadata (shape [20])
Creating validation dataset
Creating policy
Creating optimizer and scheduler
Start offline training on a fixed dataset
```

The relative-action statistics pass took approximately 15-16 seconds on the validated dataset. It runs before model construction.

## Checkpoints and resume

With `save_freq=5000`, checkpoints are written under:

```text
outputs/train/smolvla_relative_ee_chunk30_strawberry/checkpoints/
```

Each checkpoint contains:

```text
005000/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── train_config.json
│   ├── policy_preprocessor.json
│   ├── policy_postprocessor.json
│   └── processor state .safetensors files
└── training_state/
    ├── optimizer_state.safetensors
    ├── optimizer_param_groups.json
    ├── scheduler_state.json
    ├── rng_state.safetensors
    └── training_step.json
```

Resume from the latest checkpoint with:

```bash
HF_HUB_DISABLE_XET=1 python examples/umi_relative_ee/train_relative_ee_processor.py \
  --config_path=outputs/train/smolvla_relative_ee_chunk30_strawberry/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

Resume uses the saved policy configuration, processor state, normalization statistics, optimizer, scheduler, RNG state, and training step. Do not replace the saved relative-EE processors with processors from `lerobot/smolvla_base`.

## Visualizing SmolVLA predictions

Use `visualize_predictions.py` with a trained relative-EE checkpoint. Do not point it directly at `lerobot/smolvla_base`, because the base checkpoint does not contain this dataset's 10D relative-action statistics or relative-EE processors.

### Dataset prediction versus ground truth

The following command runs SmolVLA on episode 0, draws predicted and ground-truth trajectories, and saves MP4 files:

```bash
HF_HUB_DISABLE_XET=1 python examples/umi_relative_ee/visualize_predictions.py \
  --dataset_root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --episode_indices=0 \
  --inference \
  --gt \
  --pretrained_path=outputs/train/smolvla_relative_ee_chunk30_strawberry/checkpoints/last/pretrained_model \
  --task="pick the strawberry" \
  --device=cuda \
  --output_dir=outputs/debug/smolvla_visualization \
  --mp4
```

Use `--max_frames=100` to render only the first 100 frames of each selected episode during a quick check. A value of `0`, the default, renders the full episode.

Output files are grouped by dataset name:

```text
outputs/debug/smolvla_visualization/
└── sroiv2_strawberry_picking_lab_1000onesb/
    ├── proj_inference_episode_0.mp4
    └── traj3d_inference_episode_0.mp4
```

`traj3d_inference_episode_0.mp4` shows the relative 3D prediction and ground-truth trajectory. The image projection in `proj_inference_episode_0.mp4` requires calibrated camera intrinsics. Supply them with:

```bash
--camera_info_path=/path/to/camera_info.json
```

The file must contain a flattened 3x3 intrinsic matrix:

```json
{
  "K": [fx, 0, cx, 0, fy, cy, 0, 0, 1]
}
```

Use real calibration values for the dataset camera. If no calibration file exists, the visualizer still saves the 3D trajectory video and the unprojected camera stream, while logging that the 2D overlay is disabled.

Omit `--mp4` for interactive OpenCV and Matplotlib windows. Press Escape to stop.

### Live camera visualization

SmolVLA also works in camera-only mode. The language task remains mandatory:

```bash
HF_HUB_DISABLE_XET=1 python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path=outputs/train/smolvla_relative_ee_chunk30_strawberry/checkpoints/last/pretrained_model \
  --task="pick the strawberry" \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
  --device=cuda
```

For a RealSense camera, intrinsics are detected from the connected device. Camera-only mode uses the configured `--initial_state`, or an identity EE pose if it is omitted; it does not read the robot state.

## Troubleshooting

### Missing Hub repository ID

The pretrained base configuration may inherit Hub-pushing defaults. If startup reports that `policy.repo_id` is missing, explicitly pass:

```bash
--policy.push_to_hub=false
```

### Xet TLS or CAS download errors

Use the standard Hugging Face transfer path:

```bash
HF_HUB_DISABLE_XET=1 python ...
```

The first startup needs both the SmolVLM weights and the separate `lerobot/smolvla_base` action-expert checkpoint. Later runs reuse the local cache.

### Invalid safetensors header

An interrupted ranged download can leave a sparse or corrupt cache blob. A valid file must open successfully with `safetensors.safe_open`; do not repeatedly resume a confirmed corrupt blob. Preserve or remove the exact corrupt blob, then force-download that model file with Xet disabled.

### CUDA out of memory

First reduce `--batch_size`. Keeping `freeze_vision_encoder=true` and `train_expert_only=true` is important for the validated memory profile. Unfreezing the VLM changes the optimization problem and consumes substantially more memory.

### `cfg.policy` is `None` with `--policy.path`

The relative-EE entry point must defer policy-specific setup until the standard trainer resolves the pretrained path. This checkout performs that setup from the training-dataset hook. Older copies of `train_relative_ee_processor.py` that inspect `cfg.policy` before calling the standard trainer will fail for pretrained policy paths.

## Verified checks

The following checks passed in the validated environment:

- SmolVLA processor tests: 10 passed, 1 skipped.
- Policy-generic relative-EE processor tests: 7 passed.
- Two optimizer steps at batch size 1.
- Two optimizer steps at batch size 8.
- Validation-dataset initialization with all 100 episodes.
- An actual validation pass on episode 0.
- Ruff lint and formatting checks for the modified training script.

The batch-size-8 smoke run produced finite training losses of `0.584` and `0.537`. These values are startup checks, not final training results.

## Related files

- `examples/umi_relative_ee/train_relative_ee_processor.py`: training entry point and dataset metadata adaptation.
- `examples/umi_relative_ee/visualize_predictions.py`: ACT/SmolVLA camera and dataset trajectory visualization.
- `src/lerobot/processor/relative_action_processor_smolvla.py`: SmolVLA processor composition.
- `src/lerobot/processor/relative_action_processor.py`: shared SE(3), state derivation, and cache steps.
- `src/lerobot/policies/smolvla/configuration_smolvla.py`: SmolVLA and relative-EE configuration fields.
- `src/lerobot/policies/smolvla/modeling_smolvla.py`: SmolVLA training and inference implementation.
- `examples/umi_relative_ee/doc/umi_style_ee_processor_pipeline.md`: policy-independent relative-EE math and deployment details.
