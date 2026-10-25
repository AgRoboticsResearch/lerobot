# Prediction visualization for UMI relative-EE policies (π0.5 / ACT / SmolVLA)

For the matched narrow-full-width versus broad-masked π0.5 comparison, including
why flow validation loss disagrees with decoded endpoint and smoothness metrics,
see the combined
[`pi0.5_finetunning.md`](./pi0.5_finetunning.md#22-why-validation-flow-loss-disagreed-with-prediction-quality)
guide.

How the open-loop prediction videos are produced for the strawberry-picking
models, so the predicted gripper motion can be compared to ground truth on the
camera image. This document describes the shared method implemented by
`visualize_predictions.py` and doubles as its **run guide** (commands, flags,
output paths). The preserved policy-specific notes remain in
`visualize_predictions_act_smolvla.md` and `../../../docs/legacy/fei-v5.0/pi05_umi_README.md`.

For raw unnormalized 9D pose plots and geodesic SO(3) rotation-error analysis,
see `debug_rotation_visualization.md`. Both scripts write under
`<output_dir>/<repo_id>/`, so passing the same output directory to each
co-locates the video, metrics, rotation PNGs, and CSV in one folder.

## What it produces
Per dataset episode, an MP4 where each frame overlays the **predicted** and **GT**
gripper-tip trajectory on the camera image (predicted = green→red gradient, GT =
cyan/yellow), plus a 3D trajectory panel. Used to qualitatively check a model's
predictions on validation or training data.

Output path: `<output_dir>/<repo_id>/pred_episode_<N>.mp4` (default
`outputs/debug/viz_umi`; override with `--output_dir`). Pass `--project` to also
draw the predicted (green→red) and GT (cyan) gripper-tip trajectories on the
camera image; omit it for the panel composite only (no calibration needed).

## Why open-loop (no environment)
UMI-style data has no robot/sim env to roll out in. So instead of a closed-loop
rollout, we condition the policy on the **recorded** observation at each frame,
predict the action chunk, and compare it to the GT motion — an open-loop
prediction check.

## Per-frame pipeline
1. **Load frame with the action delta** `action[t-1, t, t+1, …, t+chunk-1]`
   (built via `resolve_delta_timestamps(policy.config, meta)`; the `[t-1]` is
   needed for state derivation).
2. **Preprocess** through the policy's saved processor:
   - derive the 20D two-pose state from `action[t-1], action[t]` (no state column
     on disk);
   - convert absolute 7D action chunk → relative 10D rot6d, relative to `action[t]`;
   - normalize (QUANTILES, stats from the checkpoint's normalizer);
   - tokenize the language task (π0.5 / SmolVLA are language-conditioned).
3. **`predict_action_chunk`** → predicted relative 10D chunk (normalized).
4. **Unnormalize** → relative 10D actions in real units.
5. **Trajectory**: each chunk target → `rot6d_to_matrix` → 4×4 relative transform;
   positions live in the chunk-start (`action[t]`) frame.
6. **Project** onto the camera image via the D405 hand-eye + intrinsics.
7. **Render** the composite frame; collect frames → MP4.

## The projection (SROI v2 D405 rig)
Hand-eye from `camera_gripper_extrinsics_sroi_v2_d405.json` (copied next to each
script — **no dependency on the `sroi_rosbag_utilities` folder**):
- `T_opt_cam` (optical→camera), `T_cam_ee` (camera→gripper-tip).
- For a relative target `T_rel`:
  `p_tip = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]` — the gripper-tip in the current
  camera optical frame.
- Pinhole with the color intrinsics `K`: `px = fx·x/z + cx`, `py = fy·y/z + cy`;
  masked to NaN when behind the camera.
- The chunk-start tip `(T_opt_cam @ T_cam_ee)[:3, 3]` projects to a fixed
  **(327, 321)** — identical to `sroi_rosbag_utilities`'s `visualize_traj_video.py
  --extrinsics-config …sroi_v2_d405.json`. This is the reference the projections
  are matched against.

> ⚠️ Use the **D405 JSON**, not the Piper URDF. The URDF's `camera_link→ee_link`
> transform differs from the measured rig and shifts the start to (355, 229).
> The preserved legacy tools may still use URDF/Placo; the maintained unified
> visualizer uses this JSON via `--extrinsics_config`.

## The unified script
| script (repo / env) | models | output | notes |
|---------------------|--------|--------|-------|
| `examples/umi_relative_ee/visualize_predictions.py` (py312) | ACT, SmolVLA, π0.5/LoRA | `pred_episode_<N>.mp4` | panel composite (camera + 3D + per-dim curves); on-image projection with `--project` |

The script includes the SROI projection helpers (`load_tip_kin`, `project_future`,
`_green_red_gradient`) and the D405 JSON locally. **LoRA adapters are detected
automatically** (if `adapter_config.json` is present it loads the base named there
and applies the adapter), so the same command works for full-fine-tuned ACT /
SmolVLA and for π0.5 LoRA.

## Model-specific notes
- **π0.5 (LoRA)**: load via `PeftConfig` → `from_pretrained(base, config=policy_config)`
  → `PeftModel.from_pretrained(adapter)`. The `config=policy_config` is essential —
  without it the model uses base π0.5's `base_0_rgb` image keys instead of
  `observation.images.camera`. Drive inference through the **dataloader +
  `lerobot_collate_fn`** path (a hand-built batch dict doesn't survive the
  processor's transition mapping; `validate_policy` is the template). Task
  `"pick the strawberry"` is injected for PaliGemma. Point `--pretrained_path`
  at the LoRA adapter dir; the base model named in `adapter_config.json` is loaded
  automatically — no extra flags.
- **ACT**: at inference, pop the `None` ACTION the preprocessor leaves
  (`if processed.get(ACTION) is None: processed.pop(ACTION, None)`), else the VAE
  runs the posterior branch and crashes — inference must use the prior. No `--task`
  needed; best-val checkpoint usually fits tightest.
- **SmolVLA**: needs `--task "pick the strawberry"` (the default, so you can
  usually omit it). PaliGemma-tokenized internally.

## Checkpoints
- **π0.5** — `outputs/train/pi05_lora_umi_relative_ee/checkpoints/`: loss-fix rerun
  best `040000` (val 0.0571), final `050000` (0.0577).
- **SmolVLA** — `outputs/train/smolvla_relative_ee_chunk30_strawberry_1M/checkpoints/`:
  loss-fix rerun `0250000`, `0500000` (training ongoing toward 1M).
- **ACT** — `outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/`:
  best `0200000` (~180k), final `2500000`.

Use **best-val** for the tightest predictions; **final** for the fully-trained view
(both overfit late).

## Validation vs train data
- Validation: `/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation` (100 ep).
- Train: `/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb` (1012 ep).

Swap `--dataset_root` (and `--camera_info_path` when auto-discovery is unavailable). Train
predictions are expected to track GT more tightly (the model saw them).

## Overfit / training-episode check
To confirm the pipeline can fit a specific episode (e.g. after a one-episode
overfit run), point `--dataset_root` at the **training** set and select that
episode. A correct overfit reproduces the GT trajectory near-perfectly (predicted
overlays GT):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/smolvla_one_ep_debug/checkpoints/050000/pretrained_model \
  --dataset_root /path/to/sroiv2_strawberry_picking_lab_1000onesb_1125 \
  --episode_indices 0 --project
```

## Requirements
- **Intrinsics K**: any `camera_info_color.json` under the dataset's
  `meta/camera_info/` (D405 is fixed-calibration, so any episode's file works).
- **Hand-eye**: `camera_gripper_extrinsics_sroi_v2_d405.json` next to the script.
- **Environment**: any lerobot env with the policy plus `imageio` / `matplotlib` /
  `cv2`. Locally: `/home/zfei/anaconda3/envs/py312/bin/python`. On kiwi: the unified
  checkout's `.venv/bin/python` (run `uv pip install matplotlib imageio
  imageio-ffmpeg` if missing — `cv2`/`av` come from the `dataset`/`av-dep` extras).
- Running the viz while a training job is active is fine as long as there's GPU
  headroom (~5–6 GB for eval; check `nvidia-smi` first to avoid OOM-ing the run).

## Quick reference (commands)
Minimal run — panel composite only (no calibration needed):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path <CKPT>/pretrained_model \
  --dataset_root <DATASET_ROOT> \
  --episode_indices 0 1 2
```
Panel composite **+ on-image trajectory projection** (`--project`):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path <CKPT>/pretrained_model \
  --dataset_root <DATASET_ROOT> \
  --episode_indices 0 1 2 --project
```

π0.5 on validation (unified worktree, py312). Video and rotation diagnostics are
co-located by passing the same directory to both scripts (`visualize_predictions.py`
takes `--output_dir`; the rotation script takes `--output-dir`); everything lands
under `<OUT>/<repo_id>/`:
```bash
# projected-trajectory video + prediction_metrics.json
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/pi05_lora_umi_relative_ee/checkpoints/040000/pretrained_model \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --project \
  --output_dir outputs/debug/viz_pi05

# raw-9D + SO(3) rotation diagnostics, into the same folder
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint outputs/train/pi05_lora_umi_relative_ee/checkpoints/040000/pretrained_model \
  --dataset-root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episodes 0 1 2 \
  --output-dir outputs/debug/viz_pi05
```
SmolVLA on validation (same CLI and environment):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 \
  --pretrained_path outputs/train/smolvla_relative_ee_chunk30_strawberry_1M/checkpoints/0500000/pretrained_model \
  --task "pick the strawberry" \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_018/camera_info_color.json \
  --output_dir outputs/debug/viz_smolvla
```
(ACT uses the same command with an ACT checkpoint; `--task` is harmless and may be omitted.)

## Options
`--no_gt`, `--task`, `--fps`, `--device`, `--seed`, `--repo_id`, `--output_dir`,
`--extrinsics_config`, `--camera_info_path`, `--max_frames_per_episode`,
`--first_frame_debug`, `--project`, `--num_steps`, `--zero_noise`.

## Zero-noise decoding (`--zero_noise`)
Flow policies (SmolVLA, π0.5) normally seed the flow integrator with a random
Gaussian latent. `--zero_noise` feeds an **all-zero latent** instead — the
"zero all dims" arm of the within-chunk-jitter audit
([`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)), which
collapses SmolVLA's rotation-jitter proxy from ~1.9° to ~0.085° and improves its
rotation endpoint from ~4.4° to ~2.6°. It is ignored for non-flow policies (ACT).
The latent shape is derived from the checkpoint config
(`(1, chunk_size, max_action_dim)`). The production sampler integrates the
complete latent width; zero-noise remains a diagnostic only. Diagnostics
retain `--legacy_full_action_noise` for report compatibility, but it no longer
changes SmolVLA/π0.5 inference. See the historical
[`run_jerk_ab.sh`](../shell_scripts/run_jerk_ab.sh) and the
zero-noise video runner [`run_zero_noise_vis.sh`](../shell_scripts/run_zero_noise_vis.sh).

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# UMI 相对末端执行器策略的预测可视化（π0.5 / ACT / SmolVLA）

关于 narrow-full-width 与 broad-masked 的 π0.5 配对比较——包括为什么 flow validation loss 与解码后的端点及平滑度指标不一致——参见合并指南
[`pi0.5_finetunning.md`](./pi0.5_finetunning.md#22-why-validation-flow-loss-disagreed-with-prediction-quality)。

本文档说明草莓采摘模型的开环预测视频是如何生成的，以便将预测的夹爪运动与相机图像上的真值进行比较。本文档描述由 `visualize_predictions.py` 实现的共享方法，并兼作其**运行指南**（命令、标志、输出路径）。保留的策略特定说明仍位于 `visualize_predictions_act_smolvla.md` 和 `../../../docs/legacy/fei-v5.0/pi05_umi_README.md`。

对于原始未归一化的 9D 位姿图和测地 SO(3) 旋转误差分析，参见 `debug_rotation_visualization.md`。两个脚本都写入 `<output_dir>/<repo_id>/`，因此向二者传入相同的输出目录即可将视频、指标、旋转 PNG 和 CSV 放在同一个文件夹中。

## 输出内容
对数据集的每个 episode，生成一个 MP4，其中每一帧在相机图像上叠加**预测**和 **GT** 夹爪尖端轨迹（预测 = 绿→红渐变，GT = 青色/黄色），外加一个 3D 轨迹面板。用于在验证数据或训练数据上定性检查模型的预测。

输出路径：`<output_dir>/<repo_id>/pred_episode_<N>.mp4`（默认 `outputs/debug/viz_umi`；用 `--output_dir` 覆盖）。传入 `--project` 还会在相机图像上绘制预测（绿→红）和 GT（青色）夹爪尖端轨迹；省略它则仅生成面板合成图（无需标定）。

## 为什么是开环（无环境）
UMI 风格的数据没有机器人/仿真环境可用于 rollout。因此我们不做闭环 rollout，而是在每一帧以**录制的**观测为条件对策略进行条件化，预测动作 chunk，并与 GT 运动比较——这是一种开环预测检查。

## 逐帧流程
1. **加载带动作增量的帧** `action[t-1, t, t+1, …, t+chunk-1]`
   （通过 `resolve_delta_timestamps(policy.config, meta)` 构建；`[t-1]` 是
   状态推导所必需的）。
2. **预处理**，通过策略保存的 processor：
   - 从 `action[t-1], action[t]` 推导 20D 双位姿状态（磁盘上没有 state 列）；
   - 将绝对 7D 动作 chunk 转换为相对于 `action[t]` 的相对 10D rot6d；
   - 归一化（QUANTILES，统计量来自 checkpoint 的 normalizer）；
   - 对语言任务进行 tokenize（π0.5 / SmolVLA 是语言条件化的）。
3. **`predict_action_chunk`** → 预测的相对 10D chunk（已归一化）。
4. **反归一化** → 真实单位下的相对 10D 动作。
5. **轨迹**：每个 chunk 目标 → `rot6d_to_matrix` → 4×4 相对变换；
   位置位于 chunk 起始（`action[t]`）坐标系。
6. 通过 D405 手眼和内外参**投影**到相机图像上。
7. **渲染**合成帧；收集帧 → MP4。

## 投影（SROI v2 D405 装置）
手眼来自 `camera_gripper_extrinsics_sroi_v2_d405.json`（复制在每个脚本旁边——**不依赖 `sroi_rosbag_utilities` 文件夹**）：
- `T_opt_cam`（optical→camera），`T_cam_ee`（camera→gripper-tip）。
- 对于相对目标 `T_rel`：
  `p_tip = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]` —— 当前相机光学坐标系中的夹爪尖端。
- 使用彩色内参 `K` 的针孔模型：`px = fx·x/z + cx`，`py = fy·y/z + cy`；
  当位于相机后方时掩码为 NaN。
- chunk 起始尖端 `(T_opt_cam @ T_cam_ee)[:3, 3]` 投影到固定的 **(327, 321)** —— 与 `sroi_rosbag_utilities` 的 `visualize_traj_video.py --extrinsics-config …sroi_v2_d405.json` 完全一致。这是投影所对照的参考。

> ⚠️ 使用 **D405 JSON**，而不是 Piper URDF。URDF 的 `camera_link→ee_link` 变换与实测装置不同，会将起点偏移到 (355, 229)。保留的 legacy 工具可能仍使用 URDF/Placo；维护中的统一可视化器通过 `--extrinsics_config` 使用此 JSON。

## 统一脚本
| 脚本（仓库 / 环境） | 模型 | 输出 | 说明 |
|---------------------|--------|--------|-------|
| `examples/umi_relative_ee/visualize_predictions.py`（py312） | ACT、SmolVLA、π0.5/LoRA | `pred_episode_<N>.mp4` | 面板合成图（相机 + 3D + 各维曲线）；使用 `--project` 时在图像上投影 |

该脚本在本地包含 SROI 投影辅助函数（`load_tip_kin`、`project_future`、`_green_red_gradient`）和 D405 JSON。**LoRA adapter 会被自动检测**（如果存在 `adapter_config.json`，则加载其中指定的基础模型并应用 adapter），因此同一条命令既适用于全量微调的 ACT / SmolVLA，也适用于 π0.5 LoRA。

## 模型特定说明
- **π0.5（LoRA）**：通过 `PeftConfig` 加载 → `from_pretrained(base, config=policy_config)`
  → `PeftModel.from_pretrained(adapter)`。`config=policy_config` 至关重要——
  没有它，模型会使用基础 π0.5 的 `base_0_rgb` 图像键，而不是
  `observation.images.camera`。通过 **dataloader +
  `lerobot_collate_fn`** 路径驱动推理（手工构建的 batch dict 无法通过
  processor 的 transition 映射；`validate_policy` 是模板）。为 PaliGemma 注入任务
  `"pick the strawberry"`。将 `--pretrained_path` 指向 LoRA adapter 目录；
  `adapter_config.json` 中指定的基础模型会自动加载——无需额外标志。
- **ACT**：推理时弹出预处理器留下的 `None` ACTION
  （`if processed.get(ACTION) is None: processed.pop(ACTION, None)`），否则 VAE
  会运行后验分支并崩溃——推理必须使用先验。不需要 `--task`；
  best-val checkpoint 通常拟合最紧。
- **SmolVLA**：需要 `--task "pick the strawberry"`（这是默认值，所以通常可以
  省略）。内部使用 PaliGemma tokenize。

## Checkpoints
- **π0.5** —— `outputs/train/pi05_lora_umi_relative_ee/checkpoints/`：loss 修复后的重跑
  best `040000`（val 0.0571），final `050000`（0.0577）。
- **SmolVLA** —— `outputs/train/smolvla_relative_ee_chunk30_strawberry_1M/checkpoints/`：
  loss 修复后的重跑 `0250000`、`0500000`（训练仍在向 1M 推进）。
- **ACT** —— `outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/`：
  best `0200000`（约 180k），final `2500000`。

使用 **best-val** 获得最紧的预测；使用 **final** 查看完全训练后的结果
（两者后期都会过拟合）。

## 验证集 vs 训练数据
- 验证集：`/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation`（100 ep）。
- 训练集：`/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb`（1012 ep）。

切换 `--dataset_root`（以及当自动发现不可用时的 `--camera_info_path`）。训练集上的预测预计会更紧密地跟随 GT（模型见过这些数据）。

## 过拟合 / 训练 episode 检查
要确认该流程能够拟合某个特定 episode（例如在一次单 episode 过拟合运行之后），将 `--dataset_root` 指向**训练**集并选择该 episode。正确的过拟合会近乎完美地复现 GT 轨迹（预测覆盖 GT）：
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/smolvla_one_ep_debug/checkpoints/050000/pretrained_model \
  --dataset_root /path/to/sroiv2_strawberry_picking_lab_1000onesb_1125 \
  --episode_indices 0 --project
```

## 要求
- **内参 K**：数据集 `meta/camera_info/` 下的任意 `camera_info_color.json`（D405 为固定标定，因此任意 episode 的文件都可用）。
- **手眼**：脚本旁边的 `camera_gripper_extrinsics_sroi_v2_d405.json`。
- **环境**：任意包含该策略以及 `imageio` / `matplotlib` / `cv2` 的 lerobot 环境。本地：`/home/zfei/anaconda3/envs/py312/bin/python`。在 kiwi 上：统一 checkout 的 `.venv/bin/python`（若缺失，运行 `uv pip install matplotlib imageio imageio-ffmpeg` —— `cv2`/`av` 来自 `dataset`/`av-dep` extras）。
- 只要 GPU 有余量，在训练任务运行时执行可视化是没问题的（评估约需 5–6 GB；先检查 `nvidia-smi`，避免让训练运行 OOM）。

## 快速参考（命令）
最小运行——仅面板合成图（无需标定）：
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path <CKPT>/pretrained_model \
  --dataset_root <DATASET_ROOT> \
  --episode_indices 0 1 2
```
面板合成图 **+ 图像上轨迹投影**（`--project`）：
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path <CKPT>/pretrained_model \
  --dataset_root <DATASET_ROOT> \
  --episode_indices 0 1 2 --project
```

在验证集上运行 π0.5（统一 worktree，py312）。通过向两个脚本传入相同目录，视频和旋转诊断结果会被放在一起（`visualize_predictions.py` 使用 `--output_dir`；旋转脚本使用 `--output-dir`）；所有内容都会落在 `<OUT>/<repo_id>/` 下：
```bash
# projected-trajectory video + prediction_metrics.json
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/pi05_lora_umi_relative_ee/checkpoints/040000/pretrained_model \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --project \
  --output_dir outputs/debug/viz_pi05

# raw-9D + SO(3) rotation diagnostics, into the same folder
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint outputs/train/pi05_lora_umi_relative_ee/checkpoints/040000/pretrained_model \
  --dataset-root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episodes 0 1 2 \
  --output-dir outputs/debug/viz_pi05
```
在验证集上运行 SmolVLA（相同的 CLI 和环境）：
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 \
  --pretrained_path outputs/train/smolvla_relative_ee_chunk30_strawberry_1M/checkpoints/0500000/pretrained_model \
  --task "pick the strawberry" \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_018/camera_info_color.json \
  --output_dir outputs/debug/viz_smolvla
```
（ACT 使用相同的命令，只需换成 ACT checkpoint；`--task` 无害，可以省略。）

## 选项
`--no_gt`, `--task`, `--fps`, `--device`, `--seed`, `--repo_id`, `--output_dir`,
`--extrinsics_config`, `--camera_info_path`, `--max_frames_per_episode`,
`--first_frame_debug`, `--project`, `--num_steps`, `--zero_noise`。

## 零噪声解码（`--zero_noise`）
Flow 策略（SmolVLA、π0.5）通常用随机高斯 latent 为 flow 积分器播种。`--zero_noise` 改为输入**全零 latent**——这是 within-chunk-jitter 审计的“zero all dims”分支
（[`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)），它使 SmolVLA 的旋转抖动代理从约 1.9° 降至约 0.085°，并将其旋转端点从约 4.4° 改善到约 2.6°。对非 flow 策略（ACT）会被忽略。
Latent 形状由 checkpoint 配置推导
（`(1, chunk_size, max_action_dim)`）。生产采样器会对完整 latent 宽度进行积分；零噪声仅作为诊断保留。诊断代码保留 `--legacy_full_action_noise` 以保持报告兼容，但它不再改变 SmolVLA/π0.5 的推理。参见历史脚本
[`run_jerk_ab.sh`](../shell_scripts/run_jerk_ab.sh) 和
零噪声视频运行脚本 [`run_zero_noise_vis.sh`](../shell_scripts/run_zero_noise_vis.sh)。
