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

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

> 从 `fei` 保留的历史操作手册。有关维护中的 v5 命令和共享 `use_umi_relative_ee` 接口，请从 `README.md` 开始。

# SmolVLA UMI 相对末端执行器（EE）训练

本指南记录了在此检出版本中，针对草莓采摘数据集所验证的 SmolVLA 训练工作流。它使用由 `train_relative_ee_processor.py` 实现的、基于处理器的 UMI 相对末端执行器（EE）表示。

## 此流水线训练什么

源数据集存储绝对 7D EE 位姿：

```text
[x, y, z, wx, wy, wz, gripper]
```

前六个值是绝对位置加轴角旋转。最后一个值是绝对夹爪状态。数据集不需要 `observation.state` 列，因为训练预处理器会从相邻动作推导状态。

处理器将原始数据转换为 SmolVLA 使用的表示：

```text
Raw action:       [B, chunk + 1, 7] absolute axis-angle EE poses
Derived state:    [B, 2, 7]
Model state:      [B, 20]            relative rot6d
Model action:     [B, chunk, 10]     relative rot6d
```

每个 10D 动作是：

```text
[dx, dy, dz, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]
```

预测动作块中的所有动作都相对于动作块边界处的同一位姿：

```text
T_relative[i] = inverse(T_chunk_base) @ T_target[i]
T_target[i]   = T_chunk_base @ T_relative[i]
```

它们不是从一个预测到下一个预测链式连接的。

## 验证运行所使用的数据集

训练数据集：

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

验证数据集：

```text
repo_id:        sroi/sroiv2_strawberry_picking_lab_validation
root:           /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
episodes:       100
frames:         9274
task:           "pick the strawberry"
```

验证使用根据训练数据集计算的归一化统计量。它不会从验证数据重新计算统计量。

## 处理器顺序

SmolVLA 相对 EE 预处理器执行以下步骤：

1. 重命名观测。
2. 在需要时添加批次维度。
3. 确保每个语言任务以换行符结尾。
4. 使用 SmolVLM 分词器对语言任务进行分词。
5. 将张量移动到策略设备。
6. 从 `t-1` 和 `t` 处的动作推导两步状态。
7. 将绝对 7D 动作转换为相对 10D rot6d 动作。
8. 将两步状态转换并展平为 20D 相对 rot6d。
9. 对状态和动作应用最小-最大归一化。

后处理器执行逆动作变换：

1. 对预测的 10D 动作进行反归一化。
2. 将每个预测与缓存的块基位姿组合，并转换为绝对 7D EE 位姿。
3. 将输出移动到 CPU。

预处理器和后处理器（包括训练统计量）会随每个检查点保存。

## 模型初始化

使用预训练的 SmolVLA 检查点：

```text
lerobot/smolvla_base
```

通过 `--policy.path` 指定，而不是 `--policy.type`：

```bash
--policy.path=lerobot/smolvla_base
```

预训练配置使用 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` VLM。在推荐的微调配置下：

```text
Total parameters:       approximately 450M
Trainable parameters:   approximately 100M
freeze_vision_encoder:  true
train_expert_only:      true
train_state_proj:       true
```

这会使用 transformer 视觉编码器提取图像特征，同时保持预训练 VLM（包括其视觉编码器）冻结。动作专家和状态投影会被训练。

## 完整训练命令

在 LeRobot 仓库根目录运行：

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

`HF_HUB_DISABLE_XET=1` 可避免 Xet 传输失败，并使用标准 Hugging Face 下载路径。当所有模型文件都已缓存后，还可以使用 `HF_HUB_OFFLINE=1` 实现完全本地启动。

## 为什么这些训练设置与 ACT 不同

不要对 SmolVLA 复用 250 万步的 ACT 训练计划。SmolVLA 从预训练 VLM 和动作专家开始，推荐的首次实验约为 20,000 个微调步。

建议的推进方式：

1. 训练 20,000 步，并检查训练损失、验证损失和 rollout 性能。
2. 以 5,000 步为间隔比较检查点。
3. 仅当验证和真实机器人性能仍在提升时，才继续训练到 50,000 或 100,000 步。

批次大小 8 已在配备一个 480x640 相机的 NVIDIA RTX 4090 上验证。如果图像数量、分辨率、数据增强或可训练模型范围增加，请将其降低到 4、2 或 1。

## 验证行为

在每个 `val_freq` 的正整数倍处，训练器会：

1. 将策略切换为评估模式。
2. 重置有状态预处理器。
3. 在无梯度的情况下处理完整的所选验证数据集。
4. 计算样本加权的验证指标。
5. 再次重置预处理器并恢复训练模式。
6. 将指标以 `val/` 前缀记录到 W&B。

对于 SmolVLA，主要指标是：

```text
val/loss
```

这是流匹配均方误差。ACT 也记录一个名为 `val/loss` 的指标，但 ACT 的值是 L1 重建损失加上加权的 VAE KL 散度项。因此，ACT 和 SmolVLA 的原始损失值不得直接比较。

使用 `val/loss` 比较同一策略和配置下的检查点。对于与策略无关的 ACT 与 SmolVLA 对比，请使用以下指标：

- 绝对 EE 平移误差。
- 以度表示的旋转误差。
- 夹爪误差。
- 闭环任务成功率。

要在单个回合上对验证进行冒烟测试：

```bash
--validation_dataset.episodes=[0] \
--val_freq=1
```

经验证的单回合冒烟测试运行完成，结果为：

```text
val/loss=0.690596
```

该数值仅确认完整验证路径可正常工作。它不是训练后策略的基准。

## 预期启动序列

健康的运行应记录以下阶段：

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

相对动作统计遍历在验证数据集上耗时约 15-16 秒。它在模型构建之前运行。

## 检查点与恢复

当 `save_freq=5000` 时，检查点写入：

```text
outputs/train/smolvla_relative_ee_chunk30_strawberry/checkpoints/
```

每个检查点包含：

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

使用以下命令从最新检查点恢复：

```bash
HF_HUB_DISABLE_XET=1 python examples/umi_relative_ee/train_relative_ee_processor.py \
  --config_path=outputs/train/smolvla_relative_ee_chunk30_strawberry/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

恢复会使用保存的策略配置、处理器状态、归一化统计量、优化器、调度器、RNG 状态和训练步数。不要用 `lerobot/smolvla_base` 中的处理器替换已保存的相对 EE 处理器。

## 可视化 SmolVLA 预测

使用 `visualize_predictions.py` 和已训练的相对 EE 检查点。不要直接指向 `lerobot/smolvla_base`，因为基础检查点不包含此数据集的 10D 相对动作统计量或相对 EE 处理器。

### 数据集预测与真值对比

以下命令在第 0 回合运行 SmolVLA，绘制预测轨迹和真值轨迹，并保存 MP4 文件：

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

在快速检查时，使用 `--max_frames=100` 仅渲染每个所选回合的前 100 帧。默认值 `0` 会渲染完整回合。

输出文件按数据集名称分组：

```text
outputs/debug/smolvla_visualization/
└── sroiv2_strawberry_picking_lab_1000onesb/
    ├── proj_inference_episode_0.mp4
    └── traj3d_inference_episode_0.mp4
```

`traj3d_inference_episode_0.mp4` 显示相对 3D 预测和真值轨迹。`proj_inference_episode_0.mp4` 中的图像投影需要标定后的相机内参。通过以下方式提供：

```bash
--camera_info_path=/path/to/camera_info.json
```

该文件必须包含展平的 3x3 内参矩阵：

```json
{
  "K": [fx, 0, cx, 0, fy, cy, 0, 0, 1]
}
```

请使用数据集相机的真实标定值。如果不存在标定文件，可视化器仍会保存 3D 轨迹视频和未投影的相机流，同时记录 2D 叠加已禁用。

省略 `--mp4` 可使用交互式 OpenCV 和 Matplotlib 窗口。按 Escape 停止。

### 实时相机可视化

SmolVLA 也可在仅相机模式下工作。语言任务仍然是必需的：

```bash
HF_HUB_DISABLE_XET=1 python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path=outputs/train/smolvla_relative_ee_chunk30_strawberry/checkpoints/last/pretrained_model \
  --task="pick the strawberry" \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
  --device=cuda
```

对于 RealSense 相机，内参会从连接的设备检测。仅相机模式使用配置的 `--initial_state`，如果省略则使用 identity EE 位姿；它不会读取机器人状态。

## 故障排除

### 缺少 Hub 仓库 ID

预训练基础配置可能继承 Hub 推送默认值。如果启动时报告缺少 `policy.repo_id`，请显式传入：

```bash
--policy.push_to_hub=false
```

### Xet TLS 或 CAS 下载错误

使用标准 Hugging Face 传输路径：

```bash
HF_HUB_DISABLE_XET=1 python ...
```

首次启动需要 SmolVLM 权重以及单独的 `lerobot/smolvla_base` 动作专家检查点。后续运行会复用本地缓存。

### 无效的 safetensors 头

中断的范围下载可能留下稀疏或损坏的缓存 blob。有效文件必须能通过 `safetensors.safe_open` 成功打开；不要反复恢复已确认损坏的 blob。保留或删除确切的损坏 blob，然后在禁用 Xet 的情况下强制下载该模型文件。

### CUDA 内存不足

首先降低 `--batch_size`。保持 `freeze_vision_encoder=true` 和 `train_expert_only=true` 对于已验证的内存配置很重要。解冻 VLM 会改变优化问题，并消耗显著更多内存。

### 使用 `--policy.path` 时 `cfg.policy` 为 `None`

相对 EE 入口点必须将策略特定设置推迟到标准训练器解析预训练路径之后。此检出版本从训练数据集钩子执行该设置。较旧版本的 `train_relative_ee_processor.py` 会在调用标准训练器之前检查 `cfg.policy`，对于预训练策略路径会失败。

## 已核验的检查

以下检查在验证环境中通过：

- SmolVLA 处理器测试：10 通过，1 跳过。
- 策略通用的相对 EE 处理器测试：7 通过。
- 批次大小为 1 时执行两个优化器步。
- 批次大小为 8 时执行两个优化器步。
- 使用全部 100 个回合初始化验证数据集。
- 在第 0 回合执行实际验证遍历。
- 对修改后的训练脚本执行 Ruff lint 和格式检查。

批次大小为 8 的冒烟测试运行产生了有限的训练损失 `0.584` 和 `0.537`。这些数值是启动检查，不是最终训练结果。

## 相关文件

- `examples/umi_relative_ee/train_relative_ee_processor.py`：训练入口点与数据集元数据适配。
- `examples/umi_relative_ee/visualize_predictions.py`：ACT/SmolVLA 相机和数据集轨迹可视化。
- `src/lerobot/processor/relative_action_processor_smolvla.py`：SmolVLA 处理器组合。
- `src/lerobot/processor/relative_action_processor.py`：共享的 SE(3)、状态推导和缓存步骤。
- `src/lerobot/policies/smolvla/configuration_smolvla.py`：SmolVLA 与相对 EE 配置字段。
- `src/lerobot/policies/smolvla/modeling_smolvla.py`：SmolVLA 训练与推理实现。
- `examples/umi_relative_ee/doc/umi_style_ee_processor_pipeline.md`：与策略无关的相对 EE 数学和部署细节。
