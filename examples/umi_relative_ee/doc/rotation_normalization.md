# Rotation normalization: UMI identity vs. our per-dim scaling

> **Status:** analysis + running A/B experiment (2026-07-31). This documents the
> one place our `umi_relative_ee` pipeline materially diverges from canonical UMI,
> why it is a candidate cause of full-data rotation "jumpiness," and the reversible
> experiment tracked in [`rot6d_identity_norm_experiment.md`](./rot6d_identity_norm_experiment.md).

## TL;DR

UMI leaves the 6D-rotation action components **unscaled (identity)** and only
normalizes position and gripper. We scale **all 10 dims uniformly** — including
the 6 rotation ones — with per-dimension MIN_MAX (ACT/SmolVLA) or QUANTILES (π0.5).
π0.5's quantile normalizer was designed for **joint-space** actions and has never
been validated on a 6D rotation representation. For *relative, near-identity*
rotations this per-dim scaling over-weights the near-constant rotation components
and is a plausible contributor to the rotation jumpiness observed on full-data
runs.

## The divergence

### Our pipeline — uniform per-dim scaling of all 10 action dims

The UMI stats are computed **flat over the whole 10D action** with no sub-field
split, so the 6 `rot6d` dims get their own statistics just like position:

- `src/lerobot/datasets/umi_relative_ee_stats.py:75-86` — one
  `RunningQuantileStats` over `[pos(3), rot6d(6), gripper(1)]`.
- `src/lerobot/processor/normalize_processor.py:329` — the normalizer selects a
  mode by **FeatureType** (`ACTION`), then applies that single mode to every dim.
- Per-policy mode:
  - π0.5: `src/lerobot/policies/pi05/configuration_pi05.py:80` → `QUANTILES`
    (q01/q99 → [-1,1]).
  - ACT: `src/lerobot/policies/act/configuration_act.py:172` → `MIN_MAX`.
  - SmolVLA: `src/lerobot/policies/smolvla/configuration_smolvla.py:142` → `MIN_MAX`.
- The 20D state is treated identically; its `rot6d` components live at dims
  `[3:9]` and `[13:19]`.

### Canonical UMI — rotation at identity

UMI scales **per sub-field**: position → range [-1,1], **rot6d → identity
(scale=1, offset=0)**, gripper → range [-1,1]:

- `detached-umi-policy/diffusion_policy/dataset/umi_dataset.py:221-227`
  (`get_range_normalizer_from_stat` for pos/gripper,
  `get_identity_normalizer_from_stat` for the 6D rotation).
- Rotation matrix entries are already bounded in [-1,1] (two orthonormal unit
  rows), so scaling them is unnecessary and distorts their geometry.

### openpi / π0.5 — no rotation in the action space at all

Every action space in openpi is **joint-space** (joint angles + a scalar gripper):
Aloha 14D, DROID 8D, Libero 7D, base π0.5 padded to 32D. There is no quaternion,
axis-angle, euler, or 6D rotation anywhere in the pipeline.

- π0.5 / π0-FAST default normalization is **quantile per-dim**:
  `(x - q01)/(q99 - q01) * 2 - 1` → [-1,1]
  (`openpi/src/openpi/transforms.py:137-145`, mode switch at
  `openpi/src/openpi/training/config.py:187`: `use_quantile_norm = model_type != PI0`).
- Stats are computed flat, keyed only by `"state"`/`"actions"`, one statistic per
  dimension (`openpi/scripts/compute_norm_stats.py:102`).

**Implication:** our LeRobot π0.5 path matches openpi's default mode, so we did
*not* pick the wrong mode — the mode is correct *for π0.5*. The mismatch is that
we feed a 6D rotation representation into a normalizer that was tuned for
well-spread, independent joint dimensions and never designed to handle rotation.

## Why per-dim scaling can hurt *relative* rot6d

`rot6d` is the first two rows of the rotation matrix. A relative action
`T_curr⁻¹ @ T_future` is **near identity** for most of a chunk (small per-frame
motion), so its `rot6d` clusters tightly around `[1, 0, 0, 0, 1, 0]`:

- diagonal dims (`rot6d[0]`, `rot6d[4]`) sit near **+1** with a **narrow** spread,
- off-diagonal dims sit near **0**.

Applying **per-dimension quantile/min-max** scaling to that:

- The diagonal "1" components have a thin `[q01, q99]` band (e.g. ~[0.7, 0.99]),
  which quantile scaling **stretches across all of [-1,1]** — a large multiplicative
  amplification of a near-constant signal.
- In the regression / flow-matching loss each dim's effective weight is
  `1/(q99 - q01)` (or `1/std` for mean/std). The near-constant diagonal dims get the
  **largest** weight; the off-diagonal dims that actually encode *which way* the
  gripper rotated get the **smallest**.

Net effect: the loss is dominated by predicting the near-constant `[1, 1]`
diagonal components to many significant figures, while the dimensions carrying the
real rotation direction are comparatively under-fit. The model snaps toward
identity and is noisy on the informative axes — i.e. rotation jumpiness. UMI's
identity normalization keeps every component in natural scale with balanced
weighting and never creates this imbalance.

This is consistent with existing debug results:

- Single-episode overfit reaches a ~0.9° rotation floor (memorization wins
  despite the bad weighting).
- Full-data generalization stalls at ~4.6–5.6°, and LoRA rank is **not** the
  bottleneck.
- A loss-weighting / target-distribution problem cripples generalization while
  leaving overfitting intact — exactly this signature.

> **Caveat:** the magnitude of the imbalance depends on how tightly the relative
> rotations cluster (offset-1 targets are ~identity, offset-30 less so, and stats
> are pooled across the whole chunk). The principle — per-dim affine scaling of a
> coupled rotation representation distorts component weighting, worst on
> near-identity data — holds regardless; the open question is how large the effect
> is for this dataset. It is cheap to measure.

## Do not bother with a mode swap

Switching π0.5 → π0 (quantiles → mean/std) does **not** fix this: mean/std has the
same per-dim weighting flaw (`1/std` amplifies the low-variance diagonal
components just as `1/(q99 - q01)` does). The issue is per-dim affine scaling of a
rotation representation, independent of which affine. The fix is identity on the
rotation sub-field, not a different global mode.

## Experiment (reversible, minimal)

Force the `rot6d` slice to identity in the stats **before** normalization. This
reproduces UMI's behavior exactly and requires no architecture change:

- **Action (10D):** dims `[3:9]` → `q01 = -1, q99 = 1` (π0.5) / `min = -1, max = 1`
  (ACT, SmolVLA).
- **State (20D):** dims `[3:9]` and `[13:19]` → same.

Cheapest hook: overwrite those slices in the return value of
`compute_umi_relative_ee_stats` (`src/lerobot/datasets/umi_relative_ee_stats.py`,
the `return` near line 93). Gate it behind a config flag so it can be toggled per
run without code edits.

Decision criterion: compare matched checkpoints with the policy-neutral
`eval_open_loop_dataset.py` on all 100 validation episodes (10 evenly spaced valid
queries per episode). The primary value is the episode-balanced within-chunk
rotation second-difference/acceleration proxy (currently stored under the legacy
key `rot_jerk_deg`); chunk-end rotation error is secondary accuracy. A meaningful
smoothness reduction (target: at least 20% relative, beyond seed noise) supports
the jumpiness hypothesis, provided the paired checkpoints used the same action
padding/noise objective. RTC evaluation remains a separate SmolVLA/π0.5 deployment
metric and does not support ACT.

The padding/noise qualification matters because a later implementation audit
found that SmolVLA historically sampled an unconstrained 32-D latent for a 10-D
UMI action but trained loss only on the ten real dimensions. See
[`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md) for the
OpenPI comparison, GPU ablations, and the two coherent fixed-width formulations.
Inference now masks the 22 unused coordinates throughout the flow ODE by default
for SmolVLA, π0, and π0.5, including the velocity exposed to RTC. This applies to
existing checkpoints without retraining. Use `--legacy_full_action_noise` in the
UMI evaluators only for historical reproduction; training-side padding
objectives remain a deferred retraining A/B.

### Long-term form

A per-sub-field normalizer step (position → range, **rotation → identity**,
gripper → range) so the identity treatment is explicit rather than a stats
override. This mirrors `detached-umi-policy`'s
`UmiDataset.get_normalizer` exactly.

## Evidence index

| Claim | Location |
| --- | --- |
| Our stats computed flat over 10D | `src/lerobot/datasets/umi_relative_ee_stats.py:75-86` |
| Our normalizer: one mode per FeatureType, per-dim | `src/lerobot/processor/normalize_processor.py:329` |
| π0.5 action mode = QUANTILES | `src/lerobot/policies/pi05/configuration_pi05.py:80` |
| ACT action mode = MIN_MAX (UMI path) | `src/lerobot/policies/act/configuration_act.py:172` |
| SmolVLA action mode = MIN_MAX (UMI path) | `src/lerobot/policies/smolvla/configuration_smolvla.py:142` |
| UMI: rot6d → identity, pos/gripper → range | `detached-umi-policy/diffusion_policy/dataset/umi_dataset.py:221-227` |
| openpi quantile formula | `openpi/src/openpi/transforms.py:137-145` |
| openpi mode switch (non-PI0 → quantile) | `openpi/src/openpi/training/config.py:187` |
| openpi stats keyed only by state/actions | `openpi/scripts/compute_norm_stats.py:102` |

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 旋转归一化：UMI identity 与我们的逐维缩放

> **状态：** 分析 + 正在运行的 A/B 实验（2026-07-31）。本文记录了我们的 `umi_relative_ee` 流程与标准 UMI 实质分歧的唯一一处，说明为什么它是全量数据旋转“抖动”的一个候选原因，以及记录在 [`rot6d_identity_norm_experiment.md`](./rot6d_identity_norm_experiment.md) 中的可逆实验。

## 摘要

UMI 让 6D 旋转动作分量保持**不缩放（identity）**，只对位置和夹爪做归一化。我们对**全部 10 个维度统一缩放**——包括 6 个旋转维度——使用逐维 MIN_MAX（ACT/SmolVLA）或 QUANTILES（π0.5）。π0.5 的分位数归一化器是为**关节空间**动作设计的，从未在 6D 旋转表示上验证过。对于*相对、接近 identity* 的旋转，这种逐维缩放会过度加权接近恒定的旋转分量，很可能是全量数据运行中观察到的旋转抖动的一个促成因素。

## 分歧所在

### 我们的流程——对全部 10 个动作维度统一逐维缩放

UMI 统计量是在**整个 10D 动作上扁平**计算的，没有按子字段拆分，因此 6 个 `rot6d` 维度和位置一样获得各自的统计量：

- `src/lerobot/datasets/umi_relative_ee_stats.py:75-86` —— 一个作用于
  `[pos(3), rot6d(6), gripper(1)]` 的 `RunningQuantileStats`。
- `src/lerobot/processor/normalize_processor.py:329` —— normalizer 按 **FeatureType**（`ACTION`）选择一种模式，然后将该单一模式应用于每个维度。
- 各策略的模式：
  - π0.5：`src/lerobot/policies/pi05/configuration_pi05.py:80` → `QUANTILES`
    （q01/q99 → [-1,1]）。
  - ACT：`src/lerobot/policies/act/configuration_act.py:172` → `MIN_MAX`。
  - SmolVLA：`src/lerobot/policies/smolvla/configuration_smolvla.py:142` → `MIN_MAX`。
- 20D 状态也以相同方式处理；其 `rot6d` 分量位于维度
  `[3:9]` 和 `[13:19]`。

### 标准 UMI——旋转保持 identity

UMI **按子字段**缩放：位置 → 范围 [-1,1]，**rot6d → identity
（scale=1, offset=0）**，夹爪 → 范围 [-1,1]：

- `detached-umi-policy/diffusion_policy/dataset/umi_dataset.py:221-227`
  （位置/夹爪使用 `get_range_normalizer_from_stat`，
  6D 旋转使用 `get_identity_normalizer_from_stat`）。
- 旋转矩阵元素本来就被限制在 [-1,1] 内（两行正交单位向量），因此缩放它们是不必要的，并且会扭曲其几何结构。

### openpi / π0.5——动作空间中完全没有旋转

openpi 中的每个动作空间都是**关节空间**（关节角 + 一个标量夹爪）：
Aloha 14D、DROID 8D、Libero 7D、基础 π0.5 填充到 32D。整个流程中没有任何四元数、轴角、欧拉角或 6D 旋转。

- π0.5 / π0-FAST 的默认归一化是**逐维分位数**：
  `(x - q01)/(q99 - q01) * 2 - 1` → [-1,1]
  （`openpi/src/openpi/transforms.py:137-145`，模式切换位于
  `openpi/src/openpi/training/config.py:187`：`use_quantile_norm = model_type != PI0`）。
- 统计量按扁平方式计算，仅以 `"state"`/`"actions"` 为键，每个维度一个统计量
  （`openpi/scripts/compute_norm_stats.py:102`）。

**含义：** 我们的 LeRobot π0.5 路径与 openpi 的默认模式一致，所以*并非*选错了模式——该模式*对 π0.5 而言*是正确的。不匹配之处在于，我们把 6D 旋转表示输入给了一个为分布良好、相互独立的关节维度调校、且从未针对旋转设计的归一化器。

## 为什么逐维缩放会损害*相对* rot6d

`rot6d` 是旋转矩阵的前两行。相对动作
`T_curr⁻¹ @ T_future` 在一个 chunk 的大部分时间内都**接近 identity**（每帧运动很小），因此其 `rot6d` 紧密聚集在 `[1, 0, 0, 0, 1, 0]` 附近：

- 对角维度（`rot6d[0]`、`rot6d[4]`）位于 **+1** 附近，分布**很窄**，
- 非对角维度位于 **0** 附近。

对其应用**逐维分位数/min-max**缩放：

- 对角方向的 “1” 分量具有很薄的 `[q01, q99]` 区间（例如约 [0.7, 0.99]），
  分位数缩放会把它**拉伸到整个 [-1,1]**——对一个接近恒定的信号做大幅乘性放大。
- 在回归 / flow-matching loss 中，每个维度的有效权重是
  `1/(q99 - q01)`（对于 mean/std 则是 `1/std`）。接近恒定的对角维度获得
  **最大**权重；而真正编码夹爪*朝哪个方向*旋转的非对角维度获得
  **最小**权重。

净效果：loss 被“将接近恒定的 `[1, 1]` 对角分量预测到很多有效数字”所主导，而承载真实旋转方向的分量相对而言拟合不足。模型会向 identity 塌缩，并在有信息量的轴上表现嘈杂——也就是旋转抖动。UMI 的 identity 归一化让每个分量保持自然尺度，权重均衡，永远不会造成这种失衡。

这与已有的调试结果一致：

- 单 episode 过拟合可达到约 0.9° 的旋转下限（尽管权重设置不佳，记忆仍然胜出）。
- 全量数据泛化停滞在约 4.6–5.6°，而 LoRA rank **不是**瓶颈。
- loss 加权 / 目标分布问题会严重损害泛化，同时让过拟合保持完好——正是这种特征。

> **注意：** 失衡的严重程度取决于相对旋转聚集得有多紧（offset-1 的目标接近 identity，offset-30 则不那么接近，而且统计量是在整个 chunk 上汇总的）。其原理——对耦合旋转表示做逐维仿射缩放会扭曲分量权重，在接近 identity 的数据上最严重——无论具体情况如何都成立；开放的问题是，该效应在这个数据集上有多大。测量成本很低。

## 不必尝试切换模式

将 π0.5 → π0（分位数 → mean/std）**不能**解决这个问题：mean/std 有同样的逐维加权缺陷（`1/std` 会像 `1/(q99 - q01)` 一样放大低方差的对角分量）。问题在于对旋转表示做逐维仿射缩放，与具体是哪种仿射无关。修复方法是让旋转子字段保持 identity，而不是换一个全局模式。

## 实验（可逆、最小改动）

在归一化**之前**，将统计量中的 `rot6d` 切片强制设为 identity。这能精确复现 UMI 的行为，且无需任何架构改动：

- **动作（10D）：** 维度 `[3:9]` → `q01 = -1, q99 = 1`（π0.5） / `min = -1, max = 1`
  （ACT、SmolVLA）。
- **状态（20D）：** 维度 `[3:9]` 和 `[13:19]` → 同上。

最省事的钩子：覆盖 `compute_umi_relative_ee_stats` 返回值中的这些切片
（`src/lerobot/datasets/umi_relative_ee_stats.py`，约第 93 行的 `return`）。将其放在一个配置标志之后，以便无需改代码即可按运行切换。

判定标准：用策略无关的 `eval_open_loop_dataset.py` 在全部 100 个验证 episode 上比较配对 checkpoint（每个 episode 取 10 个均匀间隔的有效查询）。主要指标是 episode 均衡的 chunk 内旋转二阶差分/加速度代理（目前存储在 legacy 键 `rot_jerk_deg` 下）；chunk 末端旋转误差是次要的准确性指标。平滑度的显著降低（目标：相对至少 20%，超出随机种子噪声）支持抖动假设，前提是配对的 checkpoint 使用了相同的动作填充/噪声目标。RTC 评估仍是独立的 SmolVLA/π0.5 部署指标，不支持 ACT。

填充/噪声这一限定很重要，因为后来的一次实现审计发现，SmolVLA 历史上为 10-D UMI 动作采样了无约束的 32-D latent，但只在十个真实维度上计算训练 loss。参见
[`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md) 了解 OpenPI 对比、GPU 消融，以及两种一致的固定宽度公式。
现在，SmolVLA、π0 和 π0.5 的推理默认会在整个 flow ODE 中屏蔽 22 个未使用的坐标，包括暴露给 RTC 的速度。这适用于现有 checkpoint，无需重新训练。仅在历史复现时于 UMI 评估器中使用 `--legacy_full_action_noise`；训练侧的填充目标仍是一个推迟的重新训练 A/B。

### 长期形态

一个按子字段的归一化器步骤（位置 → range，**旋转 → identity**，夹爪 → range），使 identity 处理显式化，而不是通过统计量覆盖实现。这完全对应 `detached-umi-policy` 的 `UmiDataset.get_normalizer`。

## 证据索引

| 论断 | 位置 |
| --- | --- |
| 我们的统计量在 10D 上扁平计算 | `src/lerobot/datasets/umi_relative_ee_stats.py:75-86` |
| 我们的 normalizer：每个 FeatureType 一种模式，逐维 | `src/lerobot/processor/normalize_processor.py:329` |
| π0.5 动作模式 = QUANTILES | `src/lerobot/policies/pi05/configuration_pi05.py:80` |
| ACT 动作模式 = MIN_MAX（UMI 路径） | `src/lerobot/policies/act/configuration_act.py:172` |
| SmolVLA 动作模式 = MIN_MAX（UMI 路径） | `src/lerobot/policies/smolvla/configuration_smolvla.py:142` |
| UMI：rot6d → identity，pos/gripper → range | `detached-umi-policy/diffusion_policy/dataset/umi_dataset.py:221-227` |
| openpi 分位数公式 | `openpi/src/openpi/transforms.py:137-145` |
| openpi 模式切换（非 PI0 → 分位数） | `openpi/src/openpi/training/config.py:187` |
| openpi 统计量仅以 state/actions 为键 | `openpi/scripts/compute_norm_stats.py:102` |
