# RTC with UMI Relative End-Effector Policies

This note describes Real-Time Chunking (RTC) support for Pi0.5 and SmolVLA checkpoints trained with the UMI relative end-effector pipeline in this directory.

## Current compatibility

| Policy and action representation | Current RTC status |
| --- | --- |
| Pi0, Pi0.5, or SmolVLA with absolute actions | Supported |
| Ordinary relative joint actions using `RelativeActionsProcessorStep` | Supported; the RTC rollout engine re-anchors the leftover prefix |
| Pi0.5 with `use_umi_relative_ee=true` | Supported through `RTCInferenceEngine` |
| SmolVLA with `use_umi_relative_ee=true` | Supported through `RTCInferenceEngine` |

Existing UMI checkpoints and LoRA adapters do not require retraining. The support is entirely in the inference-time processor and RTC queue path.

## Why re-anchoring is required

RTC guides the start of a newly generated chunk toward the unexecuted portion of the previous chunk. This requires both chunks to use the same action representation and reference frame.

UMI relative-EE preprocessing expresses every target in a chunk relative to the end-effector pose at the start of that chunk. For the same absolute target:

- A leftover action from the old chunk is represented as `T_old_base^-1 @ T_target`.
- The corresponding action in the new chunk must be represented as `T_current_base^-1 @ T_target`.

These relative values differ whenever the robot has moved between chunk starts. Comparing them directly does not preserve the same absolute target.

The RTC action queue keeps two synchronized representations:

- model-space actions used by the flow-matching denoiser;
- postprocessed absolute 7D actions used for robot execution.

For a new UMI chunk, `RTCInferenceEngine` detects `UmiRelativeActionsStep`, reads the unexecuted absolute actions, and rebuilds the model-space prefix using the latest cached EE pose. It then applies the checkpoint's action normalization before passing the prefix to Pi0.5 or SmolVLA.

Pi0.5 and SmolVLA retain their existing native RTC denoising hooks. Those hooks receive the corrected 10D normalized UMI prefix and do not need policy-specific SE(3) code.

## Recommended rollout path

Use the standard rollout RTC backend and CUDA:

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-rollout \
  --strategy.type=base \
  --policy.path=/path/to/umi_checkpoint/pretrained_model \
  --device=cuda \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=<robot_type> \
  <robot and camera options> \
  --task="pick the strawberry" \
  --duration=120
```

Start with the policy defaults of 10 flow-matching denoising steps, `execution_horizon=10`, and `max_guidance_weight=10.0`. Measure actual inference latency and inspect motion continuity before tuning the horizon or guidance weight.

## Behavior boundaries

- Selecting `--inference.type=sync` uses the existing synchronous path and does not run any RTC re-anchoring code.
- If `rtc_config.enabled` is false, the UMI RTC conversion is skipped.
- Training, validation, checkpoint serialization, and ordinary `predict_action_chunk()` calls are unchanged.
- The existing `deploy_umi_relative_ee_piper.py` custom loop does not automatically use `RTCInferenceEngine`.
- A custom direct call to `policy.predict_action_chunk(prev_chunk_left_over=...)` must not pass a stale UMI prefix from an older chunk frame. Custom RTC code must perform the same absolute-to-current-frame conversion described below.

## Implementation flow

Before applying RTC guidance, the rollout inference path:

1. Read the leftover actions in absolute coordinates from the postprocessed action queue.
2. Re-express every leftover end-effector target relative to the latest UMI chunk-start pose using SE(3) composition.
3. Convert the relative rotations to the row-based 6D rotation representation used by the UMI training pipeline.
4. Normalize the re-anchored actions using the checkpoint's training statistics.
5. Pass this new model-space prefix to the Pi0.5 or SmolVLA RTC denoising hook.

The queue continues to keep both representations so robot execution remains absolute while future guidance can be reconstructed exactly.

## Offline dataset visualization

Use the UMI-aware evaluator in this directory instead of
`examples/rtc/eval_dataset.py`. The generic evaluator compares unrelated random
samples and does not re-anchor an old UMI chunk into the current EE frame.

```bash
PYTHONPATH=src uv run python examples/umi_relative_ee/eval_rtc_dataset.py \
  --pretrained_path=/path/to/checkpoint/pretrained_model \
  --dataset_root=/path/to/validation_dataset \
  --repo_id=<dataset_repo_id> \
  --episode_indices 0 \
  --query_stride=5 \
  --inference_delay=4 \
  --execution_horizon=10 \
  --max_guidance_weight=10.0 \
  --device=cuda \
  --video_fps=6 \
  --project \
  --output_dir=outputs/debug/rtc_umi_dataset
```
`--project` uses the default SROI v2 hand-eye extrinsics and auto-discovers the
dataset's `camera_info_color.json`. Override `--extrinsics_config` or
`--camera_info_path` when using different calibration files.


For each sequential transition, the evaluator:

1. Predicts a preceding chunk and stores its postprocessed absolute EE targets.
2. Simulates executing `query_stride` actions.
3. Re-anchors the remaining targets into the current UMI chunk frame.
4. Runs no-RTC and RTC generation with identical sampled noise.
5. Saves camera-and-trajectory panels comparing the previous tail, no-RTC output,
   RTC output, and dataset ground truth.
6. Combines those panels into an H.264 episode MP4 and saves a JSON report with
   overlap, ground-truth, re-anchoring, roughness, and latency metrics.

RTC is intended to improve agreement with the still-executing previous chunk.
It is not guaranteed to reduce single-sample dataset ground-truth error.

## Verification coverage

`tests/policies/rtc/test_rtc_umi_relative_ee.py` covers both Pi0.5 and SmolVLA processor pipelines and verifies that:

1. Re-anchoring preserves each leftover action's absolute SE(3) target.
2. Rotation conversion uses the same row-based 6D convention as training.
3. Normalization matches the checkpoint processor statistics.
4. Moving the chunk-start pose changes the model-space prefix to the correct new frame.

## Relevant implementation files

- `examples/umi_relative_ee/eval_rtc_dataset.py`: sequential recorded-dataset metrics and visualization.
- `src/lerobot/rollout/inference/rtc.py`: RTC inference engine and UMI processor detection.
- `src/lerobot/policies/rtc/relative.py`: ordinary relative-action and UMI SE(3) re-anchoring helpers.
- `src/lerobot/processor/umi_relative_ee_processor.py`: `UmiRelativeActionsStep` and UMI SE(3) transforms.
- `src/lerobot/policies/pi05/modeling_pi05.py`: Pi0.5 RTC denoising hook.
- `src/lerobot/policies/smolvla/modeling_smolvla.py`: SmolVLA RTC denoising hook.
- `tests/policies/rtc/test_rtc_relative_actions.py`: ordinary relative-action RTC coverage.
- `tests/policies/rtc/test_rtc_umi_relative_ee.py`: Pi0.5 and SmolVLA UMI RTC re-anchoring coverage.
- `tests/processor/test_umi_relative_ee_processor.py`: general UMI processor coverage.

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 使用 UMI 相对末端执行器策略的 RTC

本文说明在本目录中使用 UMI 相对末端执行器流程训练的 Pi0.5 和 SmolVLA checkpoint 的实时分块（Real-Time Chunking, RTC）支持。

## 当前兼容性

| 策略与动作表示 | 当前 RTC 状态 |
| --- | --- |
| Pi0、Pi0.5 或 SmolVLA 使用绝对动作 | 支持 |
| 使用 `RelativeActionsProcessorStep` 的普通相对关节动作 | 支持；RTC rollout 引擎会重新锚定剩余前缀 |
| Pi0.5 使用 `use_umi_relative_ee=true` | 通过 `RTCInferenceEngine` 支持 |
| SmolVLA 使用 `use_umi_relative_ee=true` | 通过 `RTCInferenceEngine` 支持 |

现有的 UMI checkpoint 和 LoRA adapter 无需重新训练。该支持完全位于推理时的 processor 和 RTC 队列路径中。

## 为什么需要重新锚定

RTC 会将新生成 chunk 的开头引导向上一 chunk 中尚未执行的部分。这要求两个 chunk 使用相同的动作表示和参考坐标系。

UMI 相对 EE 预处理会将 chunk 中的每个目标表示为相对于该 chunk 起始时刻末端执行器位姿的量。对于同一个绝对目标：

- 来自旧 chunk 的剩余动作表示为 `T_old_base^-1 @ T_target`。
- 新 chunk 中对应的动作必须表示为 `T_current_base^-1 @ T_target`。

只要机器人在 chunk 起始时刻之间发生了移动，这些相对值就会不同。直接比较它们无法保持相同的绝对目标。

RTC 动作队列保留两种同步的表示：

- flow-matching 去噪器使用的模型空间动作；
- 用于机器人执行的、经后处理的绝对 7D 动作。

对于新的 UMI chunk，`RTCInferenceEngine` 会检测 `UmiRelativeActionsStep`，读取未执行的绝对动作，并使用最新缓存的 EE 位姿重建模型空间前缀。随后在将前缀传给 Pi0.5 或 SmolVLA 之前，应用 checkpoint 的动作归一化。

Pi0.5 和 SmolVLA 保留其现有的原生 RTC 去噪钩子。这些钩子接收修正后的 10D 归一化 UMI 前缀，无需策略特定的 SE(3) 代码。

## 推荐的 rollout 路径

使用标准的 rollout RTC 后端和 CUDA：

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-rollout \
  --strategy.type=base \
  --policy.path=/path/to/umi_checkpoint/pretrained_model \
  --device=cuda \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=<robot_type> \
  <robot and camera options> \
  --task="pick the strawberry" \
  --duration=120
```

从策略默认值开始：10 步 flow-matching 去噪、`execution_horizon=10` 和 `max_guidance_weight=10.0`。在调整 horizon 或 guidance weight 之前，先测量实际推理延迟并检查运动连续性。

## 行为边界

- 选择 `--inference.type=sync` 会使用现有的同步路径，不会运行任何 RTC 重新锚定代码。
- 如果 `rtc_config.enabled` 为 false，UMI RTC 转换会被跳过。
- 训练、验证、checkpoint 序列化以及普通的 `predict_action_chunk()` 调用均保持不变。
- 现有的 `deploy_umi_relative_ee_piper.py` 自定义循环不会自动使用 `RTCInferenceEngine`。
- 对 `policy.predict_action_chunk(prev_chunk_left_over=...)` 的自定义直接调用不得传入来自旧 chunk 帧的过期 UMI 前缀。自定义 RTC 代码必须执行下文所述的相同的绝对坐标到当前坐标系的转换。

## 实现流程

在应用 RTC 引导之前，rollout 推理路径会：

1. 从后处理动作队列中读取绝对坐标下的剩余动作。
2. 使用 SE(3) 复合将每个剩余末端执行器目标重新表示为相对于最新 UMI chunk 起始位姿的量。
3. 将相对旋转转换为 UMI 训练流程使用的基于行的 6D 旋转表示。
4. 使用 checkpoint 的训练统计量对重新锚定的动作进行归一化。
5. 将这个新的模型空间前缀传给 Pi0.5 或 SmolVLA 的 RTC 去噪钩子。

队列继续保留两种表示，使机器人执行保持绝对坐标，同时未来可以精确重建引导。

## 离线数据集可视化

使用本目录中的 UMI 感知评估器，而不是
`examples/rtc/eval_dataset.py`。通用评估器比较的是互不相关的随机样本，不会将旧的 UMI chunk 重新锚定到当前 EE 坐标系。

```bash
PYTHONPATH=src uv run python examples/umi_relative_ee/eval_rtc_dataset.py \
  --pretrained_path=/path/to/checkpoint/pretrained_model \
  --dataset_root=/path/to/validation_dataset \
  --repo_id=<dataset_repo_id> \
  --episode_indices 0 \
  --query_stride=5 \
  --inference_delay=4 \
  --execution_horizon=10 \
  --max_guidance_weight=10.0 \
  --device=cuda \
  --video_fps=6 \
  --project \
  --output_dir=outputs/debug/rtc_umi_dataset
```
`--project` 使用默认的 SROI v2 手眼外参，并自动发现数据集的 `camera_info_color.json`。使用不同的标定文件时，覆盖 `--extrinsics_config` 或 `--camera_info_path`。


对于每个顺序转换，评估器会：

1. 预测前一个 chunk，并存储其后处理的绝对 EE 目标。
2. 模拟执行 `query_stride` 个动作。
3. 将剩余目标重新锚定到当前 UMI chunk 坐标系。
4. 使用相同的采样噪声运行 no-RTC 和 RTC 生成。
5. 保存相机与轨迹面板，比较上一 chunk 尾部、no-RTC 输出、RTC 输出和数据集真值。
6. 将这些面板合并为一个 H.264 episode MP4，并保存包含 overlap、真值、重新锚定、粗糙度和延迟指标的 JSON 报告。

RTC 旨在改善与仍在执行的前一 chunk 的一致性。它不保证降低单样本数据集真值误差。

## 验证覆盖

`tests/policies/rtc/test_rtc_umi_relative_ee.py` 覆盖 Pi0.5 和 SmolVLA 两条 processor 流水线，并验证：

1. 重新锚定会保留每个剩余动作的绝对 SE(3) 目标。
2. 旋转转换使用与训练相同的基于行的 6D 约定。
3. 归一化与 checkpoint processor 的统计量一致。
4. 移动 chunk 起始位姿会将模型空间前缀更改为正确的新坐标系。

## 相关实现文件

- `examples/umi_relative_ee/eval_rtc_dataset.py`：顺序录制数据集的指标与可视化。
- `src/lerobot/rollout/inference/rtc.py`：RTC 推理引擎和 UMI processor 检测。
- `src/lerobot/policies/rtc/relative.py`：普通相对动作和 UMI SE(3) 重新锚定辅助函数。
- `src/lerobot/processor/umi_relative_ee_processor.py`：`UmiRelativeActionsStep` 和 UMI SE(3) 变换。
- `src/lerobot/policies/pi05/modeling_pi05.py`：Pi0.5 RTC 去噪钩子。
- `src/lerobot/policies/smolvla/modeling_smolvla.py`：SmolVLA RTC 去噪钩子。
- `tests/policies/rtc/test_rtc_relative_actions.py`：普通相对动作 RTC 覆盖。
- `tests/policies/rtc/test_rtc_umi_relative_ee.py`：Pi0.5 和 SmolVLA UMI RTC 重新锚定覆盖。
- `tests/processor/test_umi_relative_ee_processor.py`：通用 UMI processor 覆盖。
