# OpenPI full-width 100K experiment log

Baseline per [`padded_noise_strategy.md`](./padded_noise_strategy.md):
SmolVLA and pi0.5 train with full-width (32-D) flow loss and full-width
inference (`mask_padded_action_dims_at_inference=false`). ACT unchanged.
All three train on `sroiv2_strawberry_picking_lab_1302_occlusion` (121,262
frames / 1302 episodes) and validate on `sroiv2_strawberry_picking_lab_validation`
(9,274 frames / 100 episodes), val every 10K steps, seed 1000, chunk 30.

Launcher: [`run_openpi_fullwidth_100k.sh`](../shell_scripts/run_openpi_fullwidth_100k.sh)

## Runs (all 100,000 optimizer steps, run in parallel on one RTX 4090)

| Policy | Entrypoint | Batch | LR | Output dir | Log |
| --- | --- | ---: | --- | --- | --- |
| ACT | train_relative_ee_processor.py | 8 | 1e-5 | outputs/train/act_openpi_fullwidth_1302_100k | logs/act_openpi_fullwidth_100k.log |
| SmolVLA | train_relative_ee_processor.py | 8 | 1e-4 (decay 100K -> 2.5e-6) | outputs/train/smolvla_openpi_fullwidth_1302_100k | logs/smolvla_openpi_fullwidth_100k.log |
| pi0.5 LoRA r16 | train_pi05_lora.py | 2 | 1e-4 (decay 100K -> 1e-5) | outputs/train/pi05_lora_openpi_fullwidth_1302_100k | logs/pi05_openpi_fullwidth_100k.log |

Notes:
- No `umi_rot6d_identity_norm` flag: default per-dim MIN_MAX (ACT/SmolVLA) /
  QUANTILES (pi0.5) normalization, i.e. the maintained standard UMI entrypoints.
- SmolVLA/pi0.5 `scheduler_decay_steps=100000` (LR decays over the full run,
  per README guidance).
- Checkpoints every 20K steps (save_freq=20000); final checkpoint at 100000.

## Validation loss progression

| Policy | 10K | 20K | 30K | 40K | 50K | 60K | 70K | 80K | 90K | 100K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | 0.0674 | 0.0535 | 0.0483 | 0.0435 | 0.0439 | 0.0427 | 0.0416 | 0.0401 | 0.0458 | 0.0394 |
| SmolVLA | - | 0.0153 | - | 0.0143 | 0.0140 | 0.0152 | 0.0144 | 0.0143 | 0.0137 | 0.0143 |
| pi0.5 | 0.0159 | 0.0145 | - | 0.0135 | 0.0135 | 0.0135 | 0.0135 | - | 0.0127 | 0.0131 |

## Jitter evaluation (validation episodes 0-4, 5 frames/episode, seed 1000)

`eval_open_loop_dataset.py` reports, `episode_balanced`:

| Policy @step | rotation end (deg) | rot jerk (deg) | xyz jerk (m) | GT rot jerk (deg) | GT xyz jerk (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT @100000 | 3.737 | 0.0791 | 0.000664 | 0.1376 | 0.000585 |
| SmolVLA @100000 | 4.856 | 0.7323 | 0.004066 | 0.1376 | 0.000585 |
| pi0.5 @100000 | 5.282 | 0.1132 | 0.000702 | 0.1376 | 0.000585 |

Reference (previous identity-rot6d checkpoints, masked/full-width inference):
ACT @0400000: rot end 5.82°, rot jerk 0.041°, xyz jerk 0.00030 m.
SmolVLA @0200000: rot end 4.42°, rot jerk 1.919°, xyz jerk 0.00658 m.

Reports: outputs/debug/open_loop_eval_fullwidth_100k/

## Prediction visualization (validation episodes 0-2, --project)

Generated 2026-08-01 with `visualize_predictions.py` (see
`prediction_visualization.md`); D405 hand-eye + auto-discovered color K.
All videos: 1500x800 panel composite with on-image trajectory projection.

`outputs/debug/viz_openpi_fullwidth_100k/<model>/sroiv2_strawberry_picking_lab_validation/pred_episode_{0,1,2}.mp4`

| Model | episode 0 | episode 1 | episode 2 |
| --- | --- | --- | --- |
| ACT | outputs/debug/viz_openpi_fullwidth_100k/act/sroiv2_strawberry_picking_lab_validation/pred_episode_0.mp4 | ..._1.mp4 | ..._2.mp4 |
| SmolVLA | outputs/debug/viz_openpi_fullwidth_100k/smolvla/.../pred_episode_0.mp4 | ... | ... |
| pi0.5 | outputs/debug/viz_openpi_fullwidth_100k/pi05/.../pred_episode_0.mp4 | ... | ... |

## 1M-step full-width runs (launched 2026-08-01 ~23:45)

| Model | Machine | Launcher | Output dir | Log |
| --- | --- | --- | --- | --- |
| SmolVLA 1M | kiwi (RTX 5080) | examples/umi_relative_ee/shell_scripts/run_smolvla_fullwidth_1m_kiwi.sh | outputs/train/smolvla_openpi_fullwidth_1302_1M (on kiwi) | examples/umi_relative_ee/logs/smolvla_openpi_fullwidth_1M.log |
| pi0.5 LoRA 1M | this host (RTX 4090) | examples/umi_relative_ee/shell_scripts/run_pi05_fullwidth_1m.sh | outputs/train/pi05_lora_openpi_fullwidth_1302_1M | examples/umi_relative_ee/logs/pi05_openpi_fullwidth_1M.log |

Same contract as the 100K runs: 1302_occlusion train + validation val, chunk 30, seed 1000,
`mask_padded_action_dims_at_inference=false`, `scheduler_decay_steps=1000000`,
save every 100K, val every 50K, batch 8 (SmolVLA) / 2 (pi0.5).

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# OpenPI 全宽 100K 实验日志

基线依据 [`padded_noise_strategy.md`](./padded_noise_strategy.md)：SmolVLA 和 pi0.5 使用全宽（32-D）flow 损失和全宽推理（`mask_padded_action_dims_at_inference=false`）训练。ACT 不变。三者均在 `sroiv2_strawberry_picking_lab_1302_occlusion`（121,262 帧 / 1302 episodes）上训练，并在 `sroiv2_strawberry_picking_lab_validation`（9,274 帧 / 100 episodes）上验证，每 10K 步验证一次，seed 1000，chunk 30。

启动器：[`run_openpi_fullwidth_100k.sh`](../shell_scripts/run_openpi_fullwidth_100k.sh)

## 运行（均为 100,000 优化步，在一张 RTX 4090 上并行运行）

| 策略 | 入口 | Batch | LR | 输出目录 | 日志 |
| --- | --- | ---: | --- | --- | --- |
| ACT | train_relative_ee_processor.py | 8 | 1e-5 | outputs/train/act_openpi_fullwidth_1302_100k | logs/act_openpi_fullwidth_100k.log |
| SmolVLA | train_relative_ee_processor.py | 8 | 1e-4（衰减 100K -> 2.5e-6） | outputs/train/smolvla_openpi_fullwidth_1302_100k | logs/smolvla_openpi_fullwidth_100k.log |
| pi0.5 LoRA r16 | train_pi05_lora.py | 2 | 1e-4（衰减 100K -> 1e-5） | outputs/train/pi05_lora_openpi_fullwidth_1302_100k | logs/pi05_openpi_fullwidth_100k.log |

说明：
- 无 `umi_rot6d_identity_norm` 标志：默认为逐维 MIN_MAX（ACT/SmolVLA）/ QUANTILES（pi0.5）归一化，即维护中的标准 UMI 入口。
- SmolVLA/pi0.5 `scheduler_decay_steps=100000`（LR 在整个运行期间衰减，按 README 指导）。
- 每 20K 步保存检查点（save_freq=20000）；最终检查点在 100000。

## 验证损失进展

| 策略 | 10K | 20K | 30K | 40K | 50K | 60K | 70K | 80K | 90K | 100K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | 0.0674 | 0.0535 | 0.0483 | 0.0435 | 0.0439 | 0.0427 | 0.0416 | 0.0401 | 0.0458 | 0.0394 |
| SmolVLA | - | 0.0153 | - | 0.0143 | 0.0140 | 0.0152 | 0.0144 | 0.0143 | 0.0137 | 0.0143 |
| pi0.5 | 0.0159 | 0.0145 | - | 0.0135 | 0.0135 | 0.0135 | 0.0135 | - | 0.0127 | 0.0131 |

## 抖动评估（验证 episode 0-4，每 episode 5 帧，seed 1000）

`eval_open_loop_dataset.py` 报告，`episode_balanced`：

| 策略 @步数 | 旋转终点（度） | 旋转 jerk（度） | xyz jerk（m） | GT 旋转 jerk（度） | GT xyz jerk（m） |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT @100000 | 3.737 | 0.0791 | 0.000664 | 0.1376 | 0.000585 |
| SmolVLA @100000 | 4.856 | 0.7323 | 0.004066 | 0.1376 | 0.000585 |
| pi0.5 @100000 | 5.282 | 0.1132 | 0.000702 | 0.1376 | 0.000585 |

参考（先前的 identity-rot6d 检查点，masked/full-width 推理）：
ACT @0400000：旋转终点 5.82°，旋转 jerk 0.041°，xyz jerk 0.00030 m。
SmolVLA @0200000：旋转终点 4.42°，旋转 jerk 1.919°，xyz jerk 0.00658 m。

报告：outputs/debug/open_loop_eval_fullwidth_100k/

## 预测可视化（验证 episode 0-2，--project）

生成于 2026-08-01，使用 `visualize_predictions.py`（见 `prediction_visualization.md`）；D405 手眼 + 自动发现的彩色 K。所有视频：1500x800 面板合成，带图像上的轨迹投影。

`outputs/debug/viz_openpi_fullwidth_100k/<model>/sroiv2_strawberry_picking_lab_validation/pred_episode_{0,1,2}.mp4`

| 模型 | episode 0 | episode 1 | episode 2 |
| --- | --- | --- | --- |
| ACT | outputs/debug/viz_openpi_fullwidth_100k/act/sroiv2_strawberry_picking_lab_validation/pred_episode_0.mp4 | ..._1.mp4 | ..._2.mp4 |
| SmolVLA | outputs/debug/viz_openpi_fullwidth_100k/smolvla/.../pred_episode_0.mp4 | ... | ... |
| pi0.5 | outputs/debug/viz_openpi_fullwidth_100k/pi05/.../pred_episode_0.mp4 | ... | ... |

## 1M 步全宽运行（2026-08-01 ~23:45 启动）

| 模型 | 机器 | 启动器 | 输出目录 | 日志 |
| --- | --- | --- | --- | --- |
| SmolVLA 1M | kiwi（RTX 5080） | examples/umi_relative_ee/shell_scripts/run_smolvla_fullwidth_1m_kiwi.sh | outputs/train/smolvla_openpi_fullwidth_1302_1M（在 kiwi 上） | examples/umi_relative_ee/logs/smolvla_openpi_fullwidth_1M.log |
| pi0.5 LoRA 1M | 本机（RTX 4090） | examples/umi_relative_ee/shell_scripts/run_pi05_fullwidth_1m.sh | outputs/train/pi05_lora_openpi_fullwidth_1302_1M | examples/umi_relative_ee/logs/pi05_openpi_fullwidth_1M.log |

与 100K 运行相同的约定：1302_occlusion 训练 + validation 验证，chunk 30，seed 1000，`mask_padded_action_dims_at_inference=false`，`scheduler_decay_steps=1000000`，每 100K 保存，每 50K 验证，batch 8（SmolVLA）/ 2（pi0.5）。
