# Debugging UMI relative-EE rotation predictions

This guide documents the numeric GT-versus-prediction rotation diagnostic in
`generate_rotation_comparison.py`. It complements the projected episode videos
described in `prediction_visualization.md`:

- Use the videos to inspect where a predicted trajectory appears in the camera
  image.
- Use this diagnostic to inspect the model's raw relative-pose outputs and
  measure rotation error directly on SO(3).

The commands below are for the `fei-v5.0-umi-unified` branch in:

```text
/home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
```

## What is being compared

For every valid dataset frame, the script asks the policy for one 30-step
action chunk. Ground truth and prediction use the same UMI convention:

```text
T_rel[k] = inverse(T_base) @ T_target[k]
```

All 30 targets are relative to the same chunk-start `T_base`. They are not
chained from one predicted action to the next.

The model's relative pose is 10D:

```text
[dx, dy, dz, r00, r01, r02, r10, r11, r12, gripper]
```

The diagnostic calls the first nine values **raw 9D**:

```text
[dx, dy, dz, r00, r01, r02, r10, r11, r12]
```

- `dx, dy, dz` are relative translation in meters.
- The six `r` values are the first two rows of the relative rotation matrix.
- `gripper` is excluded because this diagnostic is about pose and rotation.

The six stored rotation values are converted to an orthonormal 3×3 rotation
matrix before computing angular error. Thus, the plotted SO(3) error is not a
per-channel rot6d difference.

## Data and normalization path

The diagnostic intentionally follows the training/inference pipeline:

1. `resolve_delta_timestamps()` requests the observation history and action
   horizon required by the saved policy configuration.
2. The checkpoint's saved preprocessor derives the temporal state and converts
   absolute 7D EE actions to same-base relative 10D actions.
3. The saved checkpoint statistics normalize the input.
4. `policy.predict_action_chunk()` predicts a normalized relative-action chunk.
5. The prediction is unnormalized with the checkpoint's action statistics.
6. Raw 9D values and SO(3) rotations are compared in physical, unnormalized
   units.

For ACT, the action field is removed after preprocessing so inference uses the
VAE prior rather than the training-only posterior.

Frames whose 30-step ground-truth action horizon contains padding are skipped.
This avoids treating repeated end-of-episode padding as real motion.

## Outputs

Figures and the CSV are written to `<output_dir>/<repo_id>/` — the same layout
`visualize_predictions.py` uses for its projected-trajectory MP4s and
`prediction_metrics.json`. Run both scripts with the same output directory to
collect the video, metrics, rotation summary/raw-9D PNGs, and CSV in one folder
(e.g. `outputs/debug/viz_act_…/<repo_id>/`). That directory contains:

### `rotation_gt_vs_prediction_summary.png`

For each selected episode:

- rows 1–3 plot the endpoint rotation vector's x, y, and z components;
- dashed black is ground truth;
- blue is the prediction;
- row 4 plots endpoint geodesic SO(3) error in degrees;
- the dashed red line and label report the episode mean and median.

The endpoint is action-chunk step 29 for a 30-step chunk.

Rotation-vector components are useful for seeing directional bias, but their
coordinates can be discontinuous near the angle-axis branch boundary. Use the
geodesic SO(3) error as the primary accuracy measure.

### `raw_9d_gt_vs_prediction_episode_<N>.png`

One figure is generated per episode. It shows all nine unnormalized channels
across the complete action chunk:

```text
dx, dy, dz, r00, r01, r02, r10, r11, r12
```

Rather than selecting an unusually easy or hard frame, the script chooses the
frame whose endpoint rotation error is closest to that episode's median.

The rotation entries are matrix components and therefore unitless. Values such
as `r00` and `r11` may remain close to `+1`; Matplotlib can show an axis offset
such as `+1` in those panels.

### `raw_9d_gt_vs_prediction.csv`

This contains every retained dataset frame and every action-chunk step. Columns:

```text
episode
frame
chunk_step
rotation_error_deg
gt_dx ... gt_r12
pred_dx ... pred_r12
```

Use the CSV for per-step distributions, per-channel bias, or comparisons
between checkpoints.

## Reproduce the current ACT diagnostics

Activate the unified worktree and make sure its source tree is imported:

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The most recent complete checkpoint from the stopped ACT run is:

```text
outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model
```

### Training episodes 0, 1, and 2

Video and rotation outputs are co-located by passing the same directory to both
scripts (`visualize_predictions.py` takes `--output_dir`; the rotation script
takes `--output-dir`). Everything lands under `<OUT>/<repo_id>/`:

```bash
CKPT=outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model
DS=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb_1125
OUT=outputs/debug/viz_act_unified_ckpt_1p1m_train_ep0_1_2

# projected-trajectory MP4 + prediction_metrics.json
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path "$CKPT" --dataset_root "$DS" --episode_indices 0 1 2 --project \
  --output_dir "$OUT"

# raw-9D + SO(3) rotation diagnostics, into the same folder
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint "$CKPT" --dataset-root "$DS" --episodes 0 1 2 \
  --output-dir "$OUT" --device cuda
```

### Validation episodes 0, 1, and 2

Same as above with the validation dataset and a different `OUT`:

```bash
CKPT=outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model
DS=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
OUT=outputs/debug/viz_act_unified_ckpt_1p1m_validation_ep0_1_2

/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path "$CKPT" --dataset_root "$DS" --episode_indices 0 1 2 --project \
  --output_dir "$OUT"

/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint "$CKPT" --dataset-root "$DS" --episodes 0 1 2 \
  --output-dir "$OUT" --device cuda
```

Change `--episodes` / `--episode_indices` to inspect other episodes. `--device cpu`
is supported but will be slower.

Camera intrinsics and the D405 hand-eye calibration are not required. Those are
only needed when projecting trajectories onto camera images.

## Results from checkpoint 1,100,000

The commands above produced:

| split | retained frames | mean error over all chunk steps | endpoint mean | endpoint median |
|---|---:|---:|---:|---:|
| train, episodes 0–2 | 215 | 0.594° | 0.758° | 0.673° |
| validation, episodes 0–2 | 161 | 2.820° | 4.797° | 3.679° |

Validation endpoint means by episode:

| episode | frames | endpoint mean |
|---:|---:|---:|
| 0 | 53 | 3.039° |
| 1 | 54 | 2.734° |
| 2 | 54 | 8.586° |

Episode 2 is substantially harder than episodes 0 and 1. Inspecting more
validation episodes is necessary before treating these three episodes as an
overall validation estimate.

## How to interpret a failure

### Raw rot6d differs but SO(3) error is small

Do not diagnose rotation quality from a single raw channel alone. The six
values jointly form a rotation representation and are projected to a valid
rotation matrix. Prefer the geodesic error.

### Translation is good but rotation is consistently offset

Check:

- that GT and prediction use the same chunk-start base pose;
- that rotation rows/columns use the same convention;
- that checkpoint statistics belong to the same dataset/action representation;
- whether the error is concentrated in particular episodes or poses.

### Errors grow with chunk step

This usually means long-horizon endpoint prediction is harder. Compare error by
`chunk_step` in the CSV rather than looking only at step 29.

### Validation is much worse than training

Possible causes include domain shift, different motion distributions, camera
or scene differences that affect visual inference, and rare rotations in the
training data. First expand the episode sample and compare the per-episode
distribution.

### Very large normalized values are suspected

Inspect the checkpoint/dataset normalization statistics for narrow ranges or
very small standard deviations. The figures and CSV here contain unnormalized
outputs, so they reveal the physical result but not the internal normalized
magnitude.

## Scope and limitations

- This is recorded-observation, open-loop evaluation. It is not a robot rollout.
- GT observations condition every frame independently, so the diagnostic does
  not measure closed-loop compounding error.
- The representative raw-9D plot shows one frame per episode; use the CSV for
  the full distribution.
- Three episodes are useful for debugging, not for reporting a final benchmark.
- The current script has been exercised with the ACT checkpoint above. Other
  unified UMI policies can use the same numeric approach when their saved
  processor and `predict_action_chunk()` interface are compatible.

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 调试 UMI relative-EE 旋转预测

本指南记录了 `generate_rotation_comparison.py` 中的 GT 与预测旋转数值诊断。它补充了 `prediction_visualization.md` 中描述的投影 episode 视频：

- 使用视频检查预测轨迹在相机图像中出现的位置。
- 使用本诊断检查模型的原始相对位姿输出，并直接在 SO(3) 上测量旋转误差。

以下命令适用于 `fei-v5.0-umi-unified` 分支中的：

```text
/home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
```

## 比较的内容

对于每个有效的数据集帧，脚本向策略请求一个 30 步的动作 chunk。Ground truth 和预测使用相同的 UMI 约定：

```text
T_rel[k] = inverse(T_base) @ T_target[k]
```

全部 30 个目标都相对于同一个 chunk 起始 `T_base`。它们不是从一个预测动作链式传递到下一个。

模型的相对位姿是 10D：

```text
[dx, dy, dz, r00, r01, r02, r10, r11, r12, gripper]
```

诊断将前九个值称为 **raw 9D**：

```text
[dx, dy, dz, r00, r01, r02, r10, r11, r12]
```

- `dx, dy, dz` 是以米为单位的相对平移。
- 六个 `r` 值是相对旋转矩阵的前两行。
- `gripper` 被排除，因为本诊断关注的是位姿与旋转。

在计算角度误差之前，存储的六个旋转值会转换为正交的 3×3 旋转矩阵。因此，绘制的 SO(3) 误差不是逐通道的 rot6d 差值。

## 数据与归一化路径

该诊断有意遵循训练/推理流水线：

1. `resolve_delta_timestamps()` 请求保存的策略配置所需的观测历史和动作视野。
2. checkpoint 保存的预处理器推导时序状态，并将绝对 7D EE 动作转换为同一基准的相对 10D 动作。
3. 保存的 checkpoint 统计量对输入进行归一化。
4. `policy.predict_action_chunk()` 预测一个归一化的相对动作 chunk。
5. 使用 checkpoint 的动作统计量对预测进行反归一化。
6. 在物理、未归一化的单位下比较 raw 9D 值和 SO(3) 旋转。

对于 ACT，预处理后会移除动作字段，因此推理使用 VAE 先验而不是仅用于训练的后验。

如果某一帧的 30 步 ground-truth 动作视野包含 padding，则跳过该帧。这可以避免将重复的 episode 末端 padding 当作真实运动。

## 输出

图和 CSV 写入 `<output_dir>/<repo_id>/`——与 `visualize_predictions.py` 用于其投影轨迹 MP4 和 `prediction_metrics.json` 的布局相同。用相同的输出目录运行两个脚本，即可将视频、指标、旋转摘要/raw-9D PNG 和 CSV 收集到同一个文件夹中（例如 `outputs/debug/viz_act_…/<repo_id>/`）。该目录包含：

### `rotation_gt_vs_prediction_summary.png`

对于每个选定的 episode：

- 第 1–3 行绘制端点旋转向量的 x、y 和 z 分量；
- 黑色虚线是 ground truth；
- 蓝色是预测；
- 第 4 行绘制端点测地 SO(3) 误差（以度为单位）；
- 红色虚线和标签报告该 episode 的均值和中位数。

端点对于 30 步 chunk 而言是动作 chunk 的第 29 步。

旋转向量分量有助于观察方向偏差，但在角度-轴分支边界附近其坐标可能不连续。请将测地 SO(3) 误差作为主要的精度度量。

### `raw_9d_gt_vs_prediction_episode_<N>.png`

每个 episode 生成一张图。它显示完整动作 chunk 上全部九个未归一化通道：

```text
dx, dy, dz, r00, r01, r02, r10, r11, r12
```

脚本不是选择一个异常简单或困难的帧，而是选择端点旋转误差最接近该 episode 中位数的帧。

旋转项是矩阵分量，因此无量纲。诸如 `r00` 和 `r11` 的值可能保持接近 `+1`；Matplotlib 可能在这些面板中显示诸如 `+1` 的轴偏移。

### `raw_9d_gt_vs_prediction.csv`

其中包含每个保留的数据集帧和每个动作 chunk 步。列：

```text
episode
frame
chunk_step
rotation_error_deg
gt_dx ... gt_r12
pred_dx ... pred_r12
```

使用 CSV 分析逐步分布、逐通道偏差或不同 checkpoint 之间的比较。

## 复现当前的 ACT 诊断

激活统一的 worktree，并确保导入其源码树：

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

已停止的 ACT 运行中最近一个完整 checkpoint 是：

```text
outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model
```

### 训练 episode 0、1 和 2

通过向两个脚本传递相同目录，可将视频和旋转输出放在一起（`visualize_predictions.py` 使用 `--output_dir`；旋转脚本使用 `--output-dir`）。所有内容都落在 `<OUT>/<repo_id>/` 下：

```bash
CKPT=outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model
DS=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb_1125
OUT=outputs/debug/viz_act_unified_ckpt_1p1m_train_ep0_1_2

# projected-trajectory MP4 + prediction_metrics.json
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path "$CKPT" --dataset_root "$DS" --episode_indices 0 1 2 --project \
  --output_dir "$OUT"

# raw-9D + SO(3) rotation diagnostics, into the same folder
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint "$CKPT" --dataset-root "$DS" --episodes 0 1 2 \
  --output-dir "$OUT" --device cuda
```

### 验证 episode 0、1 和 2

与上面相同，但使用验证数据集和不同的 `OUT`：

```bash
CKPT=outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model
DS=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
OUT=outputs/debug/viz_act_unified_ckpt_1p1m_validation_ep0_1_2

/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path "$CKPT" --dataset_root "$DS" --episode_indices 0 1 2 --project \
  --output_dir "$OUT"

/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint "$CKPT" --dataset-root "$DS" --episodes 0 1 2 \
  --output-dir "$OUT" --device cuda
```

更改 `--episodes` / `--episode_indices` 可检查其他 episode。支持 `--device cpu`，但会更慢。

不需要相机内参和 D405 手眼标定。只有在将轨迹投影到相机图像时才需要它们。

## checkpoint 1,100,000 的结果

上述命令产生了：

| split | 保留帧数 | 所有 chunk 步的平均误差 | 端点均值 | 端点中位数 |
|---|---:|---:|---:|---:|
| 训练，episode 0–2 | 215 | 0.594° | 0.758° | 0.673° |
| 验证，episode 0–2 | 161 | 2.820° | 4.797° | 3.679° |

各 episode 的验证端点均值：

| episode | 帧数 | 端点均值 |
|---:|---:|---:|
| 0 | 53 | 3.039° |
| 1 | 54 | 2.734° |
| 2 | 54 | 8.586° |

Episode 2 明显比 episode 0 和 1 更难。在将这三个 episode 视为整体验证估计之前，有必要检查更多验证 episode。

## 如何解读失败情形

### 原始 rot6d 不同但 SO(3) 误差很小

不要仅凭单个原始通道来诊断旋转质量。这六个值共同构成一种旋转表示，并会被投影为有效的旋转矩阵。优先使用测地误差。

### 平移良好但旋转持续偏移

检查：

- GT 和预测是否使用相同的 chunk 起始基准位姿；
- 旋转的行/列是否使用相同的约定；
- checkpoint 统计量是否属于相同的数据集/动作表示；
- 误差是否集中在特定的 episode 或位姿上。

### 误差随 chunk step 增大

这通常意味着长视野端点预测更难。应按 CSV 中的 `chunk_step` 比较误差，而不是只看第 29 步。

### 验证结果明显差于训练

可能的原因包括域偏移、不同的运动分布、影响视觉推理的相机或场景差异，以及训练数据中的稀有旋转。首先扩大 episode 样本，并比较逐 episode 分布。

### 怀疑归一化值非常大

检查 checkpoint/数据集的归一化统计量是否存在范围过窄或标准差非常小的情况。此处的图和 CSV 包含未归一化的输出，因此它们揭示的是物理结果，而不是内部的归一化量级。

## 范围与局限

- 这是基于记录观测的开环评估。它不是机器人 rollout。
- GT 观测独立地条件化每一帧，因此该诊断不测量闭环累积误差。
- 代表性的 raw-9D 图每个 episode 只显示一帧；完整分布请使用 CSV。
- 三个 episode 适合调试，不适合报告最终基准。
- 当前脚本已使用上述 ACT checkpoint 进行过验证。当其他统一 UMI 策略保存的处理器和 `predict_action_chunk()` 接口兼容时，可以采用相同的数值方法。

