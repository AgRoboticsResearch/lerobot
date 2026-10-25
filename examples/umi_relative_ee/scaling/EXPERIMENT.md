# Scaling Experiment — ACT, relative-EE, strawberry picking

**Status:** IN PROGRESS (1012-ep run done; 100-ep & 500-ep training in parallel).
**Started:** 2026-07-18. **Expected completion:** ~2026-07-21.
**Owner:** Zhenghao. **Machine:** pangolin (1× RTX 4090, 24 GB).

## 1. Goal / hypothesis

Measure how **validation loss** of an ACT policy (UMI-style relative end-effector
actions, rot6d, action-chunk 30) scales with **training dataset size**, holding
everything else fixed (model, optimizer, steps, validation set).

Hypothesis: validation loss follows an approximately power-law decay in dataset
size, `L(D) ≈ a·D^(−b) + c`, and the model overfits the smaller subsets well
before the 2.5M-step budget — so **best (early-stopped) val loss** is the fair
comparison metric, not final.

## 2. Method

**Policy / training (identical across all three runs):**
- Model: ACT, 52M params
- Action representation: UMI-style relative EE, **rot6d (10D)**, `chunk_size=30`, `n_action_steps=30`
- `derive_state_from_action=true`, `use_relative_actions=true`, `pose_dim=6`, `use_rot6d=true`
- Normalization: MIN_MAX (UMI-style); ImageNet stats for cameras
- Batch size 8, AdamW lr 1e-5, **2.5M steps**, `val_freq=10000`
- **No checkpoints saved** (`--save_checkpoint=false`) — disk conservation; the
  validation curve (console log + wandb) is the durable artifact.

**Only the training subset varies:** 100 / 500 / 1012 episodes (nested prefix slices —
100 ⊂ 500 ⊂ 1012, identical episode content). Each run recomputes its own
normalization stats over its own subset. The validation set is **the same 100
episodes / 9274 frames** for all three runs.

## 3. Datasets (all LeRobot v3, 30 fps, RealSense D405, task "pick the strawberry")

| train subset | episodes | frames | path (root) |
|---|---|---|---|
| 100-ep | 100 | 11,732 | `sroiv2_strawberry_picking_lab_1000onesb_100` |
| 500-ep | 500 | 46,640 | `sroiv2_strawberry_picking_lab_1000onesb_500` |
| 1012-ep (full) | 1012 | 88,218 | `sroiv2_strawberry_picking_lab_1000onesb` |
| **validation (shared)** | 100 | 9,274 | `sroiv2_strawberry_picking_lab_validation` |

All under `/mnt/data1/sroi/lerobot/`. The full set is a **masked** rebuild (trajectories
rebuilt 2026-07-14). Per-dataset lineage in each `DATA_SOURCES.md`.

**Caveat:** subsets are **leading-prefix slices** by session/episode order, so at small N
early sessions are over-represented (100-ep averages ~117 frames/ep vs ~87 for the full
set). Fine for a nested scaling sweep; acknowledged in the report.

## 4. Runs

| tag | status | best val loss (so far) | best-val step | final val loss |
|---|---|---|---|---|
| 100-ep | training (~76%) | 0.0833 | 380k | TBD |
| 500-ep | training (~75%) | 0.0572 | 70k | TBD |
| 1012-ep | **done** | 0.0456 | 180k | 0.0480 |

**Preliminary scaling signal (best val): 0.083 → 0.057 → 0.046** for 100→500→1012 ep —
monotonic, consistent with a power law. (Final-loss values are noisier due to overfit.)

**Key observation:** every run converges then overfits well before 2.5M steps — best val
at 70k/180k/380k for 500/1012/100-ep respectively. Extra steps add nothing to val loss
(and slightly hurt). Train loss keeps dropping (e.g. 100-ep: 0.014→0.006) while val is
flat → classic memorization of the limited demos.

## 5. Reproduce

```bash
# Env: /home/zfei/anaconda3/envs/py310/bin/python (lerobot editable install of this repo)
bash outputs/train/ee_vs_joints/scaling_analysis/run_scaling_subset.sh <100|500> <dataset_dirname>
```
`run_scaling_subset.sh` hard-codes the full config above (only the dataset + output dir
change). The 1012-run used the same config directly via
`examples/umi_relative_ee/train_relative_ee_processor.py`.

## 6. Automation / monitoring

- Both new runs are **detached** (`nohup setsid`); survive session restarts.
- A recurring cron (every 30 min, `:08`/`:38`, **session-only**) babysits and, when both
  runs hit 2.5M, auto-compiles the deliverable per `FIGURE_SPEC.md` then self-deletes.
- Logs: `run_100ep.console.log`, `run_500ep.console.log`. 1012-run curve in
  `val_curve_1000ep.csv` (val) and `_1012_history.csv` (train+val, 10k pts from cloud).
- **wandb:** all three runs grouped in project
  [`biorobotlab/lerobot_scaling_ee`](https://wandb.ai/biorobotlab/lerobot_scaling_ee):
  `act_scaling_ee_chunk30_{100,500,1012}ep`. The 1012 run is a **mirror/replay**
  (`5wkorw6c`, tagged `mirror-of:lerobot/mgnsa3nw`) of the original finished run
  (`biorobotlab/lerobot/mgnsa3nw`, logged under a different default project) — wandb
  can't move runs across projects, so the real metrics were re-logged at the same steps.

## 7. Deliverable

A standard scaling-report figure set + `SCALING_REPORT.md` (see **`FIGURE_SPEC.md`**):
val & train loss vs steps (3 dataset sizes), val loss vs dataset-size (log-log, power-law
fit + exponent), generalization gap vs steps, and loss components.

## 8. Caveats / known issues
- Leading-prefix subsets (§3) — small-N over-representation of early sessions.
- Large transient grad-norm spikes on sampled steps (up to ~900) but train loss stays
  flat — grad clipping contains them; no divergence.
- Single GPU, runs are parallel (each ~0.066 s/step under contention vs 0.034 solo).

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 扩展实验 — ACT、相对末端执行器、草莓采摘

**状态：** 进行中（1012-ep 运行已完成；100-ep 与 500-ep 训练并行进行中）。
**开始时间：** 2026-07-18。**预计完成：** 约 2026-07-21。
**负责人：** Zhenghao。**机器：** pangolin（1× RTX 4090，24 GB）。

## 1. 目标 / 假设

衡量 ACT 策略（UMI 风格的相对末端执行器动作、rot6d、action-chunk 30）的**验证损失**
如何随**训练数据集规模**变化，同时保持其他一切不变（模型、优化器、步数、验证集）。

假设：验证损失随数据集规模近似呈幂律衰减，`L(D) ≈ a·D^(−b) + c`，并且模型会在 2.5M 步预算之前就很早地在较小子集上过拟合——因此**最佳（早停）验证损失**才是公平的比较指标，而非最终损失。

## 2. 方法

**策略 / 训练（三次运行完全一致）：**
- 模型：ACT，52M 参数
- 动作表示：UMI 风格相对 EE，**rot6d（10D）**，`chunk_size=30`，`n_action_steps=30`
- `derive_state_from_action=true`，`use_relative_actions=true`，`pose_dim=6`，`use_rot6d=true`
- 归一化：MIN_MAX（UMI 风格）；相机使用 ImageNet 统计量
- 批大小 8，AdamW lr 1e-5，**2.5M 步**，`val_freq=10000`
- **不保存 checkpoint**（`--save_checkpoint=false`）——节省磁盘；验证曲线（控制台日志 + wandb）是可持久保存的产物。

**仅训练子集变化：** 100 / 500 / 1012 episodes（嵌套前缀切片——100 ⊂ 500 ⊂ 1012，episode 内容相同）。每次运行在自己的子集上重新计算自己的归一化统计量。验证集对三次运行均为**相同的 100 个 episode / 9274 帧**。

## 3. 数据集（均为 LeRobot v3，30 fps，RealSense D405，任务 "pick the strawberry"）

| 训练子集 | episodes | 帧数 | 路径（root） |
|---|---|---|---|
| 100-ep | 100 | 11,732 | `sroiv2_strawberry_picking_lab_1000onesb_100` |
| 500-ep | 500 | 46,640 | `sroiv2_strawberry_picking_lab_1000onesb_500` |
| 1012-ep（完整） | 1012 | 88,218 | `sroiv2_strawberry_picking_lab_1000onesb` |
| **验证集（共享）** | 100 | 9,274 | `sroiv2_strawberry_picking_lab_validation` |

均位于 `/mnt/data1/sroi/lerobot/` 下。完整数据集是**经过掩码处理**的重建版本（轨迹于 2026-07-14 重建）。每个数据集的来源信息见各自的 `DATA_SOURCES.md`。

**注意事项：** 子集是按 session/episode 顺序的**前导前缀切片**，因此在较小的 N 下，早期 session 被过度代表（100-ep 平均约 117 帧/ep，而完整集约为 87）。对于嵌套的扩展扫描而言可以接受；已在报告中说明。

## 4. 运行

| 标签 | 状态 | 最佳验证损失（截至目前） | 最佳验证步数 | 最终验证损失 |
|---|---|---|---|---|
| 100-ep | 训练中（约 76%） | 0.0833 | 380k | TBD |
| 500-ep | 训练中（约 75%） | 0.0572 | 70k | TBD |
| 1012-ep | **已完成** | 0.0456 | 180k | 0.0480 |

**初步扩展信号（最佳验证损失）：0.083 → 0.057 → 0.046**，对应 100→500→1012 ep——单调，与幂律一致。（最终损失值因过拟合而噪声更大。）

**关键观察：** 每次运行都会收敛，然后在远早于 2.5M 步时过拟合——500/1012/100-ep 分别在 70k/180k/380k 达到最佳验证损失。额外的步数对验证损失没有帮助（甚至略有损害）。训练损失持续下降（例如 100-ep：0.014→0.006），而验证损失保持平坦 → 这是对有限示教的典型记忆化。

## 5. 复现

```bash
# Env: /home/zfei/anaconda3/envs/py310/bin/python (lerobot editable install of this repo)
bash outputs/train/ee_vs_joints/scaling_analysis/run_scaling_subset.sh <100|500> <dataset_dirname>
```
`run_scaling_subset.sh` 将上述完整配置硬编码（只有数据集 + 输出目录会变）。1012 次运行通过
`examples/umi_relative_ee/train_relative_ee_processor.py` 直接使用相同配置。

## 6. 自动化 / 监控

- 两个新运行均为**分离（detached）**运行（`nohup setsid`）；可在会话重启后继续存活。
- 一个循环 cron（每 30 分钟，`:08`/`:38`，**仅当前会话**）负责照看，并在两个运行都达到 2.5M 时，按 `FIGURE_SPEC.md` 自动编译交付物，然后自删除。
- 日志：`run_100ep.console.log`、`run_500ep.console.log`。1012 运行的曲线在
  `val_curve_1000ep.csv`（验证）和 `_1012_history.csv`（训练+验证，来自云端的 10k 个点）。
- **wandb：** 三次运行都分组在项目
  [`biorobotlab/lerobot_scaling_ee`](https://wandb.ai/biorobotlab/lerobot_scaling_ee) 下：
  `act_scaling_ee_chunk30_{100,500,1012}ep`。1012 运行是原始已完成运行
  （`biorobotlab/lerobot/mgnsa3nw`，记录在另一个默认项目下）的**镜像/重放**
  （`5wkorw6c`，标记为 `mirror-of:lerobot/mgnsa3nw`）——wandb
  无法跨项目移动运行，因此真实指标在相同步数上被重新记录。

## 7. 交付物

一套标准的扩展报告图集 + `SCALING_REPORT.md`（参见 **`FIGURE_SPEC.md`**）：
验证与训练损失随步数变化（3 种数据集规模）、验证损失随数据集规模变化（log-log，幂律
拟合 + 指数）、泛化差距随步数变化，以及损失分量。

## 8. 注意事项 / 已知问题
- 前导前缀子集（§3）——小 N 下早期 session 被过度代表。
- 在采样的步数上出现大的瞬态梯度范数尖峰（最高约 900），但训练损失保持
  平坦——梯度裁剪将其抑制；没有发散。
- 单 GPU，运行并行（竞争下每步约 0.066 秒，而单独运行时为 0.034 秒）。
