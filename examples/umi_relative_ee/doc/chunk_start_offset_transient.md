# Transient chunk-start offset (~300K) — not a masked-subspace bug

**Date:** 2026-08-03. Companion to [`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)
(the ACT-vs-flow / padding A/B audit). This documents a separate observation from the A/B viz.

## Observation
In the A/B viz videos, `viz_ab_masked_*` (Option **B**, masked_subspace, ckpt **300K**) showed an
**almost-fixed "move to the left" segment at the start of every predicted chunk**, on every frame —
i.e. an input-independent initial jump. `viz_ab_openpi_*` (Option **A**, ckpt **600K**) did not.

## Method
Open-loop prediction on validation episodes 0–2, 40 frames. Metric: the mean predicted first action of
the chunk (`pred_rel[0]`, the relative translation that becomes the first trajectory segment), in the
chunk-start frame, averaged across frames. A real motion has `a[0] ≈ 0` (no instantaneous jump);
**GT `a[0]` y = 0.00 mm**. Measured for A and B across checkpoints, plus GT.

## Finding: it is a transient training artifact, identical in A and B
Mean `a[0]` **y** (mm) per checkpoint (GT = 0.0; the artifact axis is +y, with a smaller −x ≈ −1.7 mm):

| step | A (openpi_full_width) | B (masked_subspace) |
|------|------|------|
| 100K | +0.82 | −1.31 |
| 200K | −2.22 | −3.04 |
| **300K** | **+5.77** | **+5.78** |
| 400K | +0.66 | *(not trained yet)* |
| 500K | −2.05 | |
| 600K | −0.41 | |

(GT overall chunk-mean y = +1.34 mm; biased checkpoints also inflate this — A@300K +4.62, B@300K +8.86.)

- The offset is **wrong** (GT `a[0]` = 0; it is not real gripper motion) and **input-independent**
  (constant across frames), so it is a genuine model artifact.
- **It is NOT caused by `masked_subspace`.** Option A (openpi_full_width) has the **identical +5.77 mm
  spike at 300K**. Both formulations pass through it.
- **It is transient.** Absent at 100K/200K, **peaks right at ~300K**, then **resolves by 400–600K**
  (A@400K +0.66 → A@600K −0.41, i.e. back to ~0).

The reason it appeared only in the B videos: **B was at 300K (inside the spike) while A was viz'd at
600K (past it).**

## Root cause
A training-dynamics wobble: during the ~300K phase the model temporarily learns a near-constant **+y
velocity component at the chunk start**, which the flow-matching Euler integrator turns into a fixed
endpoint offset on `a[0]` (a constant velocity bias `b` integrates to a constant output offset `−b`).
As training continues the model corrects it.

Both runs spike at the *same step* and *same magnitude* because they share the **same seed / data
order and the same LR schedule**, so their trajectories are correlated and hit the same transient
phase together. The flow-matching integration, the `masked_subspace` masking, and the loss are all
mathematically correct (verified in `src/lerobot/policies/flow_matching.py::integrate_flow_matching`
and `modeling_smolvla.py` `forward`/`sample_actions`) — this is not a code bug and not a property of
the padding mode. Note this is distinct from the within-chunk *jitter* (per-step noise) analyzed in
the companion doc; this is a *constant endpoint offset* on the first action.

## Recommendation
- It **self-resolves** — A is clean by 400–600K; B will be too past ~400K.
- For deployment or clean viz, **use a checkpoint ≥ 400K**. Re-render the B videos once B has a
  400K+ checkpoint and the "left move" will be gone.
- If many checkpoints are saved through this phase (save_freq=100K), prefer the ones on either side
  of the ~300K spike (e.g. 100–200K or ≥400K) when picking a deploy checkpoint by val loss.

## Reproduce
Mean `a[0]` translation per checkpoint (val eps 0–2, 40 frames), via the shared viz machinery:
```bash
# on kiwi (.venv) or host (py312), HF_HUB_OFFLINE=1
python - <<'PY'   # see measure script pattern in within_chunk_jitter_analysis.md
# load checkpoint -> predict_action_chunk on val eps 0,1,2 (delta-windowed) ->
# unnormalize -> mean of pred_rel[0, :3] across ~40 frames; compare to GT a[0] (≈0).
PY
```
Checkpoint roots on kiwi:
`outputs/train/smolvla_openpi_fullwidth_1302_1M/checkpoints/<step>/pretrained_model` (A) and
`outputs/train/smolvla_masked_subspace_1302_1M/checkpoints/<step>/pretrained_model` (B);
dataset `~/data/sroiv2_strawberry_picking_lab_validation`.

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 块起始偏移瞬态（~300K）——并非掩码子空间缺陷

**日期：** 2026-08-03。与 [`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)（ACT 对 flow / padding A/B 审计）配套。本文记录 A/B 可视化中一个单独的观察。

## 观察
在 A/B 可视化视频中，`viz_ab_masked_*`（选项 **B**，masked_subspace，ckpt **300K**）在每一帧的每个预测块起始处都显示出**几乎固定的“向左移动”片段**——即一个与输入无关的初始跳变。`viz_ab_openpi_*`（选项 **A**，ckpt **600K**）没有这种现象。

## 方法
对验证 episode 0–2、40 帧进行开环预测。指标：块起始帧中块的平均预测首动作（`pred_rel[0]`，即成为第一个轨迹片段的相对平移），跨帧平均。真实运动有 `a[0] ≈ 0`（无瞬时跳变）；**GT `a[0]` y = 0.00 mm**。对 A 和 B 跨检查点测量，外加 GT。

## 发现：这是一个瞬态训练伪影，A 和 B 完全相同
每个检查点的平均 `a[0]` **y**（mm）（GT = 0.0；伪影轴为 +y，另有较小的 −x ≈ −1.7 mm）：

| 步数 | A（openpi_full_width） | B（masked_subspace） |
|------|------|------|
| 100K | +0.82 | −1.31 |
| 200K | −2.22 | −3.04 |
| **300K** | **+5.77** | **+5.78** |
| 400K | +0.66 | *（尚未训练）* |
| 500K | −2.05 | |
| 600K | −0.41 | |

（GT 整体块均值 y = +1.34 mm；有偏检查点也会放大该值——A@300K +4.62，B@300K +8.86。）

- 该偏移是**错误的**（GT `a[0]` = 0；它不是真实的夹爪运动），并且**与输入无关**（跨帧恒定），因此是真正的模型伪影。
- **它不是由 `masked_subspace` 引起的。** 选项 A（openpi_full_width）在 300K 时具有**完全相同的 +5.77 mm 尖峰**。两种形式都会经历它。
- **它是瞬态的。** 100K/200K 时不存在，**恰好在 ~300K 达到峰值**，然后**在 400–600K 前消退**（A@400K +0.66 → A@600K −0.41，即回到 ~0）。

它只出现在 B 视频中的原因：**B 处于 300K（尖峰内），而 A 是在 600K（已过尖峰）时可视化的。**

## 根本原因
一种训练动态抖动：在 ~300K 阶段，模型暂时学到了块起始处近乎恒定的 **+y 速度分量**，flow-matching 欧拉积分器将其转化为 `a[0]` 上的固定端点偏移（恒定速度偏差 `b` 积分后成为恒定输出偏移 `−b`）。随着训练继续，模型会纠正它。

两个运行在*相同步数*和*相同幅度*出现尖峰，是因为它们共享**相同的 seed / 数据顺序和相同的 LR 调度**，因此轨迹相关，并一起进入同一瞬态阶段。flow-matching 积分、`masked_subspace` 掩码和损失在数学上都是正确的（已在 `src/lerobot/policies/flow_matching.py::integrate_flow_matching` 和 `modeling_smolvla.py` 的 `forward`/`sample_actions` 中验证）——这不是代码缺陷，也不是 padding 模式的属性。请注意，这与配套文档中分析的块内*抖动*（每步噪声）不同；这是首个动作上的*恒定端点偏移*。

## 建议
- 它**会自行消退**——A 在 400–600K 时已干净；B 在 ~400K 之后也会如此。
- 用于部署或干净的可视化时，**使用 ≥ 400K 的检查点**。等 B 有 400K+ 检查点后重新渲染 B 视频，“左移”就会消失。
- 如果在此阶段保存了许多检查点（save_freq=100K），按 val loss 挑选部署检查点时，优先选择 ~300K 尖峰两侧的检查点（例如 100–200K 或 ≥400K）。

## 复现
每个检查点的平均 `a[0]` 平移（val eps 0–2，40 帧），通过共享的可视化机制：
```bash
# on kiwi (.venv) or host (py312), HF_HUB_OFFLINE=1
python - <<'PY'   # see measure script pattern in within_chunk_jitter_analysis.md
# load checkpoint -> predict_action_chunk on val eps 0,1,2 (delta-windowed) ->
# unnormalize -> mean of pred_rel[0, :3] across ~40 frames; compare to GT a[0] (≈0).
PY
```
kiwi 上的检查点根目录：
`outputs/train/smolvla_openpi_fullwidth_1302_1M/checkpoints/<step>/pretrained_model`（A）和
`outputs/train/smolvla_masked_subspace_1302_1M/checkpoints/<step>/pretrained_model`（B）；
数据集 `~/data/sroiv2_strawberry_picking_lab_validation`。
