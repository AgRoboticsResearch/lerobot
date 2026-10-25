# Within-chunk action jitter: ACT vs flow matching

> **Status:** inference fix implemented after mathematical audit, updated 2026-07-31.
> Companion to [`rot6d_identity_norm_experiment.md`](./rot6d_identity_norm_experiment.md).
>
> **Superseded runtime:** the masked-inference experiment documented below is
> historical. SmolVLA and π0.5 now follow the full-width OpenPI contract in
> [`padded_noise_strategy.md`](./padded_noise_strategy.md).

## TL;DR

The large SmolVLA within-chunk jitter is real, but it is **not an unavoidable
flow-matching noise floor**. The SE(3) relative transform, row-rot6d convention,
rot6d projection, and flow ODE direction are consistent with their references.
Increasing the Euler step count does not fix the jitter.

The audit found a concrete fixed-width action-space mismatch:

1. UMI actions have 10 real dimensions and are zero-padded to SmolVLA's 32-D
   model width.
2. Training and inference inject independent Gaussian noise into all 32
   dimensions.
3. The training wrapper crops the flow loss to the first 10 dimensions, leaving
   padded dimensions 10–31 unsupervised.
4. `action_in_proj` consumes all 32 dimensions jointly, so the 22 random,
   unsupervised coordinates can perturb the ten real outputs.

Upstream OpenPI does **not** use this mixed formulation. It zero-pads the target
to the model width, samples full-width noise, and supervises the full-width flow
loss. A second valid formulation is a masked 10-D flow embedded in 32-D, but
that requires padded noise/state/velocity to remain zero throughout training
and inference. SmolVLA currently combines the loss rule from the second design
with the noise rule from the first.

On a fixed validation frame, removing only padded-coordinate noise reduced the
rotation acceleration proxy from **2.00° to 1.08°**. Removing all sampling noise
reduced it to **0.093°**. Across the same 25 samples used in the latest ACT vs
SmolVLA comparison, zero-noise decoding reduced SmolVLA from **1.919° to
0.085°**, versus **0.138° GT**, while also improving rotation endpoint error
from **4.42° to 2.59°**. This is strong diagnostic evidence, not yet a deployment
recommendation by itself; the implemented masked-subspace inference path still
needs the full 100-episode evaluation and real rollouts.

The production fix now follows OpenPI for SmolVLA and π0.5: normalized actions
are zero-padded to the 32D model width, training supervises all 32 flow
coordinates, and inference integrates full-width Gaussian noise without
dimension masking. The old config field remains loadable but is ignored by
those two policies. Existing mixed-formulation checkpoints can run with the new
sampler, but retraining is required to supervise their padded outputs. π0 keeps
its existing behavior. The UMI evaluator's `--legacy_full_action_noise` flag is
therefore a compatibility no-op for SmolVLA and π0.5.

## Metric terminology and limitations

Both existing scripts call their metric "jerk", but they compute a **second
difference**, which is a discrete acceleration/change-in-velocity proxy:

```text
Δ²p[t] = p[t+1] - 2 p[t] + p[t-1]
```

True jerk is a third derivative. The current report keys
`rot_jerk_deg`/`xyz_jerk_m` are retained for compatibility, but should be read as
approximately:

- `rotation_accel_deg_per_step2`
- `xyz_accel_m_per_step2`

The current `eval_open_loop_dataset.py` rotation metric forms adjacent relative
increments and measures the geodesic difference between them. It is much better
than reducing rotation to a scalar angle from the chunk start, but it still:

- compares increments from adjacent local frames without explicit frame
  transport;
- uses float32 `trace -> acos`, which is numerically weak near zero;
- omits the boundary from the observed current pose to predicted action 0;
- reports per-step units without multiplying by the control frequency squared.

These limitations may matter for ACT's very small value, but cannot plausibly
explain the roughly 47× ACT/SmolVLA gap. A follow-up metric should use an
`SO(3)` logarithm in float64, express angular increments in a consistent world
or transported body frame, and separately report within-chunk acceleration,
start jump, and cross-chunk discontinuity.

The older `measure_chunk_jitter.py` tables below use an additional approximation:
rotation is collapsed to the unsigned scalar angle from the chunk-start frame
before second differencing. That loses rotation axis and sign, so those historical
rotation numbers show the trend but are not directly interchangeable with the
new absolute-pose `SO(3)` metric.

## Latest matched comparison

Open-loop GPU evaluation, validation episodes 0–4, five query frames per episode,
same query frames and seed, most-recent checkpoints:

| Policy | rotation end | rotation acceleration proxy | xyz acceleration proxy | GT rotation proxy |
| --- | ---: | ---: | ---: | ---: |
| ACT @0400000 | 5.82° | 0.041° | 0.00030 m | 0.138° |
| SmolVLA @0200000 | 4.42° | 1.919° | 0.00658 m | 0.138° |

Endpoint accuracy and trajectory smoothness disagree: SmolVLA has the better
endpoint but is about **47× less smooth in rotation** and about **14× above GT**.
The 25 SmolVLA values are consistently high rather than being driven by one
outlier.

Reports:

- `outputs/debug/jerk_test_act/act_umi_identity_rot6d_1302_0400000_open_loop_metrics.json`
- `outputs/debug/jerk_test_smolvla/smolvla_umi_identity_rot6d_1302_0200000_open_loop_metrics.json`

## Fixed-width action spaces: two valid formulations

Let the model width be \(D=32\), the real action dimension be \(d=10\), and
\(m=[1,\ldots,1,0,\ldots,0]\) be the active-dimension mask.

### A. Full-width degenerate target distribution (OpenPI)

Embed the real target as:

\[
y=[a,0_{D-d}]
\]

and use full-width Gaussian noise:

\[
\epsilon\sim\mathcal N(0,I_D),\quad
x_t=t\epsilon+(1-t)y,\quad
u_t=\epsilon-y.
\]

Compute the flow loss on **all \(D\) dimensions**. For padded dimensions the
target endpoint is zero, so the model explicitly learns to transport random
noise back to zero. At inference, sample all \(D\) dimensions, integrate all
\(D\), then trim the output to the first \(d\).

This is what the local OpenPI checkout at commit `15a9616` does:

- `openpi/src/openpi/transforms.py:328-337` zero-pads state/actions;
- JAX `openpi/src/openpi/models/pi0.py:189-214` samples noise with
  `actions.shape` and reduces MSE over every action dimension;
- PyTorch `openpi/src/openpi/models_pytorch/pi0_pytorch.py:317-374` returns the
  full elementwise MSE with no real-DoF crop;
- both samplers initialize a full `action_dim` Gaussian;
- embodiment output adapters trim to the real dimension, e.g.
  `openpi/src/openpi/policies/libero_policy.py:95-100`.

The online OpenPI sources show the same behavior:
[padding transform](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py),
[JAX flow loss and sampler](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py),
and [PyTorch flow loss and sampler](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py).

### B. Masked lower-dimensional flow embedded in 32-D

Alternatively, define:

\[
\epsilon_m=m\odot\epsilon,\quad
y=m\odot y,\quad
x_t=t\epsilon_m+(1-t)y,\quad
u_t=\epsilon_m-y.
\]

Only active dimensions need a loss, but inactive dimensions must remain exactly
zero. In practice:

1. zero padded noise during training and inference;
2. zero padded `x_t` initially;
3. mask predicted velocity and/or clamp padded `x_t` after every ODE step;
4. calculate the objective only on real dimensions;
5. trim outputs as usual.

This is a flow on a \(d\)-dimensional subspace. It avoids spending capacity on
meaningless coordinates while preserving the pretrained 32-D projection shapes.
LeRobot X-VLA's documented `auto` action mode similarly pads to the pretrained
width while applying loss only to real dimensions, although its exact internal
dynamics are policy-specific:
[X-VLA action-space documentation](https://github.com/huggingface/lerobot/blob/main/docs/source/xvla.mdx).

### Invalid mixed formulation used by legacy inference and current training

Before the inference fix, the SmolVLA path did this at both training and inference:

\[
\epsilon\sim\mathcal N(0,I_{32}),\qquad
\mathcal L=\mathcal L_{0:10}.
\]

Relevant code:

- `prepare_action()` pads to `max_action_dim=32`:
  `src/lerobot/policies/smolvla/modeling_smolvla.py:490-493`;
- training samples `noise` with that padded shape:
  `src/lerobot/policies/smolvla/modeling_smolvla.py:774-786`;
- the wrapper crops losses to `original_action_dim=10`:
  `src/lerobot/policies/smolvla/modeling_smolvla.py:382-385`;
- inference samples all 32 coordinates:
  `src/lerobot/policies/smolvla/modeling_smolvla.py:826-829`;
- `action_in_proj` maps the full 32-D vector into each action token:
  `src/lerobot/policies/smolvla/modeling_smolvla.py:586`.

During training, padded coordinates are still not trained to terminate at zero.
Legacy inference also did not hold them at zero. Cropping only the final output
does not prevent such coordinates from changing hidden action tokens and
therefore the first ten outputs.

This is inherited from current LeRobot rather than introduced by UMI. The
real-DoF loss crop was merged for SmolVLA, π0 and π0.5 in
[LeRobot PR #3133](https://github.com/huggingface/lerobot/pull/3133) on
2026-03-11. Its review says learning padding is unnecessary and leaves
customizable handling as a future task, but the samplers continue to initialize
full-width noise. The present upstream SmolVLA source contains both the
[real-DoF loss crop](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py)
and full-width sampling. This repository now overrides that inference behavior
with the masked-subspace integration described below; the loss is intentionally
unchanged until the retraining experiment.

The local UMI runs include the same crop. Local commit `fbaeeb53` correctly fixed
**temporal** `action_is_pad` masking and per-sample reduction, but also retained
the dimension crop. Temporal padding (invalid future timesteps near an episode
boundary) and action-dimension padding are separate problems and need separate
masks.

## Implemented inference fix (no retraining required)

`src/lerobot/policies/flow_matching.py` now supplies one integration path used by
SmolVLA, π0, and π0.5. For an action feature of width `d` and model width `D`, it:

1. derives `d` from the checkpoint's `action_feature`;
2. leaves behavior unchanged when `d == D`;
3. zeroes caller-provided or generated noise in coordinates `d:D` without
   mutating the caller's tensor;
4. masks the base denoiser velocity before RTC sees it;
5. masks RTC's guided velocity and clamps the Euler state after every step.

Step 4 keeps RTC's endpoint estimate and prefix error in the same masked
subspace; masking only the final guided velocity would allow its internal
calculation to reintroduce padded coordinates. It also prevents
cross-coordinate leakage if RTC guidance differentiates the denoiser velocity.

The new config field is
`mask_padded_action_dims_at_inference: bool = True` for all three policies. It is
automatic for every smaller action space, not UMI-specific. Existing checkpoint
JSON files do not contain the field, so config loading uses the new `True`
default. For exact historical reproduction, set it to `false`; the shared
integrator then performs the original full-width Euler update.

Both UMI evaluators record active/model dimensions and the selected mask mode in
`action_dimension_inference`. Their `--legacy_full_action_noise` flag performs a
checkpoint-fixed A/B against the old sampler. ACT reports the mode as not
applicable because it is not a flow policy.

This is deliberately inference-only. It removes the known 22-coordinate random
nuisance from existing UMI checkpoints, but does not make their historical
training objective mathematically coherent after the fact. A later retraining
A/B should compare full-width OpenPI supervision with masked-subspace training.

## GPU isolation experiments

All probes used SmolVLA identity-rot6d checkpoint `0200000`, GPU inference, and
the same validation input unless stated otherwise.

### Post-fix matched validation

The implemented default mask and `--legacy_full_action_noise` opt-out were run
on the same 25 queries (episodes 0–4, five queries each, seed 1000):

| Inference mode | rotation end | rotation proxy | xyz proxy | GT rotation proxy |
| --- | ---: | ---: | ---: | ---: |
| Legacy full 32-D noise | 4.418° | 1.919° | 0.00658 m | 0.138° |
| Default masked 10-of-32 flow | **4.281°** | **0.963°** | **0.00388 m** | 0.138° |

The proper mask cuts the rotation proxy by **49.8%** and the XYZ proxy by
**41.0%**, while slightly improving endpoint rotation accuracy. The legacy row
exactly reproduces the earlier reported values, confirming that the opt-out is
behaviorally compatible.

This also shows the fix is necessary but not sufficient: masked SmolVLA remains
about 7× above recorded rotation acceleration and about 23× above ACT's 0.041°
on this subset. The remaining active-coordinate stochastic sensitivity should
be addressed by the deferred retraining/smoothness experiments, not by restoring
padded noise.

Reports:

- `outputs/debug/padded_action_mask_validation_smolvla/smolvla_umi_identity_rot6d_1302_0200000_open_loop_metrics.json`
- `outputs/debug/padded_action_mask_validation_smolvla_legacy/smolvla_umi_identity_rot6d_1302_0200000_open_loop_metrics.json`

### Noise-coordinate ablation

| Initial latent | rotation proxy | xyz proxy | rotation end |
| --- | ---: | ---: | ---: |
| Random all 32 dims | 1.996° | 0.00670 m | 4.57° |
| Random only real dims 0–9 | 1.077° | 0.00458 m | 4.07° |
| Random only padded dims 10–31 | 0.524° | 0.00174 m | 2.60° |
| Zero all dims | 0.093° | 0.00030 m | 2.31° |
| Ground truth | 0.174° | 0.00058 m | — |

The components are not additive because the network is nonlinear, but padded
noise clearly and materially perturbs real output coordinates. Clamping padded
coordinates to zero after every Euler step gave a smaller additional improvement
over zero padded initialization, confirming that initialization is the dominant
padding intervention for this checkpoint.

### Zero-noise decoding on all 25 matched queries

| SmolVLA decoding | rotation end | rotation proxy | xyz end | xyz proxy |
| --- | ---: | ---: | ---: | ---: |
| Standard random | 4.418° | 1.919° | 0.0345 m | 0.00658 m |
| Zero noise | 2.587° | 0.085° | 0.0331 m | 0.000326 m |
| Ground truth | — | 0.138° | — | 0.000585 m |

Zero noise is a useful diagnostic and immediate experimental arm. It is not
guaranteed to be a statistically meaningful representative sample: the exact
zero vector is atypical under a 960-D Gaussian
(`30 steps × 32 dimensions`), and deterministic decoding can behave poorly for
multimodal actions. Validate it over all 100 episodes and real task success.

### Other hypotheses ruled out

- **Euler resolution:** using 5, 10, and 30 steps gave approximately 1.90°,
  2.00°, and 2.11°. More solver steps do not remove the oscillation.
- **Degenerate rot6d projection:** predicted row norms and Gram-Schmidt residual
  norms stay near one, so projection is not amplifying near-zero vectors.
- **SE(3)/rot6d convention:** the local row convention, Gram-Schmidt
  reconstruction, and `T_ref^-1 @ T_target` relative transform match
  `universal_manipulation_interface` and `detached-umi-policy`.
- **Normalization mismatch between the compared checkpoints:** ACT and SmolVLA
  identity checkpoints both contain identity stats for rotation dimensions
  3–8. Identity normalization therefore does not explain their cross-policy gap.
- **Flow direction/sign:** training and sampling use a consistent rectified-flow
  convention. Solver direction is correct.

## What remains after removing padded noise

Random noise in the ten real coordinates still produces about 1.08° on the
single-frame ablation, well above GT. That is not a mathematical inevitability:
an accurately learned conditional flow can map independent Gaussian source noise
to coherent trajectory samples. It instead indicates that this learned velocity
field remains sensitive in high-frequency temporal directions.

Plausible contributors include:

- no explicit translation or `SO(3)` temporal-acceleration objective;
- weaker local temporal inductive bias than canonical UMI's temporal-convolution
  diffusion UNet;
- full 30-action execution/evaluation rather than canonical UMI's shorter
  receding execution horizon;
- no EMA in this SmolVLA training path;
- a scheduler that reaches its floor at 30K while the run continues for millions
  of updates;
- frozen backbone/expert-only fine-tuning limiting adaptation.

The strong endpoint accuracy shows that more generic accuracy training alone is
unlikely to target the remaining failure mode.

## Recommended experiments

### Existing checkpoint, no retraining

Run the same 100-episode/1,000-query evaluation with identical query indices:

1. standard 32-D random noise;
2. random noise in dimensions 0–9, zeros in 10–31, padded state clamped as today;
3. all-zero noise;
4. optionally several random samples ranked by smoothness under an endpoint or
   plausibility constraint.

Do not average raw rot6d coordinates without projecting each step back to
`SO(3)` or using a rotation-aware mean. Smoothness-based sample selection better
preserves multimodal branches than unconditional averaging.

### Deferred retraining A/B/C

Keep model width and pretrained weights at 32, but compare:

| Arm | Noise | Loss | ODE padded coordinates |
| --- | --- | --- | --- |
| Current mixed baseline | all 32 | real 10 | unconstrained |
| OpenPI full-width | all 32 | all 32 | learned to end at zero |
| Masked subspace | real 10 only | real 10 | forced zero |

The full-width and masked-subspace arms are both mathematically coherent. Do not
combine results from checkpoints trained with different padding objectives
without labeling that difference.

For UMI-only training, masked-subspace flow is the cheaper targeted design. For
maximum consistency with OpenPI and older pretrained π checkpoints, full-width
loss is the conservative compatibility design. The empirical A/B should decide
which transfers better from the particular SmolVLA base checkpoint.

After fixing padding, test a rotation-aware temporal acceleration penalty,
shorter execution horizon with replanning, and downstream translation
interpolation/quaternion SLERP. These address the remaining active-coordinate
jitter rather than hiding the padding bug.

## Historical checkpoint series

These older `measure_chunk_jitter.py` results remain useful as a trend study,
but use the scalar-angle/second-difference approximation described above.
Episodes 0–2, 60 valid frames each, `num_steps=10`.

GT baselines: validation translation 0.538 mm, rotation 8.95 mrad; training
translation 0.709 mm, rotation 9.42 mrad.

### Translation proxy (mm) — prediction (prediction/GT)

| model · checkpoint | validation | training |
| --- | ---: | ---: |
| ACT 100K | 0.88 (1.6×) | 1.05 (1.5×) |
| ACT 500K | 0.56 (1.0×) | 0.61 (0.9×) |
| ACT 1100K | 0.34 (0.6×) | 0.53 (0.7×) |
| SmolVLA frozen 100K | 7.16 (13.3×) | 6.92 (9.8×) |
| SmolVLA frozen 500K | 6.50 (12.1×) | 6.48 (9.1×) |
| SmolVLA frozen 900K | 5.65 (10.5×) | 5.67 (8.0×) |
| SmolVLA unfrozen 100K | 5.40 (10.0×) | 6.94 (9.8×) |
| SmolVLA unfrozen 200K | 5.11 (9.5×) | 6.83 (9.6×) |
| SmolVLA unfrozen 400K | 4.75 (8.8×) | 5.98 (8.4×) |

### Rotation proxy (mrad) — prediction (prediction/GT)

| model · checkpoint | validation | training |
| --- | ---: | ---: |
| ACT 100K | 4.45 (0.50×) | 5.67 (0.60×) |
| ACT 500K | 4.22 (0.47×) | 3.22 (0.34×) |
| ACT 1100K | 2.99 (0.33×) | 2.71 (0.29×) |
| SmolVLA frozen 100K | 91.0 (10.2×) | 76.3 (8.1×) |
| SmolVLA frozen 500K | 83.1 (9.3×) | 69.1 (7.3×) |
| SmolVLA frozen 900K | 77.3 (8.6×) | 61.5 (6.5×) |
| SmolVLA unfrozen 100K | 63.4 (7.1×) | 77.8 (8.3×) |
| SmolVLA unfrozen 200K | 61.9 (6.9×) | 75.8 (8.0×) |
| SmolVLA unfrozen 400K | 58.3 (6.5×) | 64.7 (6.9×) |

The historical conclusion that ACT becomes much smoother while SmolVLA
plateaus is still supported. The causal interpretation changes: this is not
proof of an inherent flow-matching floor. It is consistent with the
fixed-width padding mismatch plus residual active-coordinate sensitivity and
the lack of an explicit smoothness objective.

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 动作块内抖动：ACT 与流匹配

> **状态：** 经数学审计后已实现推理修复，更新于 2026-07-31。
> 本文是 [`rot6d_identity_norm_experiment.md`](./rot6d_identity_norm_experiment.md) 的配套文档。
>
> **已废弃的运行时：** 下文记录的掩码推理实验已是历史。SmolVLA 和 π0.5 现在遵循 [`padded_noise_strategy.md`](./padded_noise_strategy.md) 中全宽度 OpenPI 约定。

## 摘要

SmolVLA 较大的动作块内抖动是真实存在的，但**并非不可避免的流匹配噪声下限**。SE(3) 相对变换、行 rot6d 约定、rot6d 投影以及流 ODE 方向均与其参考实现一致。增加 Euler 步数并不能解决该抖动。

审计发现了一个具体的固定宽度动作空间不匹配问题：

1. UMI 动作有 10 个真实维度，并被零填充到 SmolVLA 的 32 维模型宽度。
2. 训练和推理都会向全部 32 个维度注入独立的高斯噪声。
3. 训练封装将流损失裁剪到前 10 个维度，使填充维度 10–31 处于无监督状态。
4. `action_in_proj` 联合使用全部 32 个维度，因此这 22 个随机的、无监督的坐标可能扰动十个真实输出。

上游 OpenPI **并不**使用这种混合形式。它将目标零填充到模型宽度，采样全宽度噪声，并对全宽度流损失进行监督。第二种有效形式是嵌入 32 维中的掩码 10 维流，但这要求填充的噪声/状态/速度在训练和推理全程保持为零。SmolVLA 目前把第二种设计的损失规则与第一种设计的噪声规则结合在一起。

在一个固定验证帧上，仅去除填充坐标的噪声，就将旋转加速度代理指标从 **2.00° 降至 1.08°**。去除所有采样噪声后，该指标降至 **0.093°**。在最新 ACT 与 SmolVLA 对比所使用的同样 25 个样本上，零噪声解码将 SmolVLA 从 **1.919° 降至 0.085°**，而 **GT 为 0.138°**，同时将旋转终点误差从 **4.42° 改善至 2.59°**。这是强有力的诊断证据，但本身尚不构成部署建议；已实现的掩码子空间推理路径仍需完整 100 回合评估和真实机器人 rollout。

生产修复现在让 SmolVLA 和 π0.5 遵循 OpenPI：归一化动作被零填充到 32D 模型宽度，训练监督全部 32 个流坐标，推理对全宽度高斯噪声进行积分而不做维度掩码。旧配置字段仍可加载，但会被这两个策略忽略。现有混合形式检查点可以用新采样器运行，但需要重新训练才能监督其填充输出。π0 保持现有行为。因此，UMI 评估器的 `--legacy_full_action_noise` 标志对 SmolVLA 和 π0.5 只是兼容性空操作。

## 指标术语与局限性

两个现有脚本都把其指标称为 "jerk"，但它们计算的是**二阶差分**，即离散的加速度/速度变化代理：

```text
Δ²p[t] = p[t+1] - 2 p[t] + p[t-1]
```

真正的 jerk 是三阶导数。当前报告中的键 `rot_jerk_deg`/`xyz_jerk_m` 为兼容性保留，但应近似读作：

- `rotation_accel_deg_per_step2`
- `xyz_accel_m_per_step2`

当前的 `eval_open_loop_dataset.py` 旋转指标构造相邻的相对增量，并测量它们之间的测地线差异。它比把旋转简化为相对动作块起点的标量角度要好得多，但仍存在以下问题：

- 比较相邻局部坐标系中的增量，而没有显式的坐标系传输（frame transport）；
- 使用 float32 的 `trace -> acos`，在接近零时数值上较弱；
- 忽略了从观测到的当前位姿到预测动作 0 的边界；
- 报告的是每步单位，而未乘以控制频率的平方。

这些局限对 ACT 的极小值可能有影响，但无法合理解释约 47× 的 ACT/SmolVLA 差距。后续指标应在 float64 中使用 `SO(3)` 对数，在一致的世界坐标系或传输后的机体坐标系中表达角增量，并分别报告动作块内加速度、起始跳变和跨块不连续。

下文较早的 `measure_chunk_jitter.py` 表格还使用了额外近似：在二阶差分之前，把旋转折叠为相对动作块起始坐标系的无符号标量角度。这会丢失旋转轴和符号，因此这些历史旋转数值只显示趋势，不能与新的绝对位姿 `SO(3)` 指标直接互换。

## 最新配对比较

开环 GPU 评估，验证回合 0–4，每回合五个查询帧，相同查询帧与随机种子，最新检查点：

| 策略 | 旋转终点 | 旋转加速度代理 | xyz 加速度代理 | GT 旋转代理 |
| --- | ---: | ---: | ---: | ---: |
| ACT @0400000 | 5.82° | 0.041° | 0.00030 m | 0.138° |
| SmolVLA @0200000 | 4.42° | 1.919° | 0.00658 m | 0.138° |

终点精度与轨迹平滑度呈现矛盾：SmolVLA 终点更好，但旋转平滑度约差 **47×**，约为 GT 的 **14×**。这 25 个 SmolVLA 数值整体偏高，并非由单个离群值造成。

报告：

- `outputs/debug/jerk_test_act/act_umi_identity_rot6d_1302_0400000_open_loop_metrics.json`
- `outputs/debug/jerk_test_smolvla/smolvla_umi_identity_rot6d_1302_0200000_open_loop_metrics.json`

## 固定宽度动作空间：两种有效形式

设模型宽度为 \(D=32\)，真实动作维度为 \(d=10\)，\(m=[1,\ldots,1,0,\ldots,0]\) 为活跃维度掩码。

### A. 全宽度退化目标分布（OpenPI）

将真实目标嵌入为：

\[
y=[a,0_{D-d}]
\]

并使用全宽度高斯噪声：

\[
\epsilon\sim\mathcal N(0,I_D),\quad
x_t=t\epsilon+(1-t)y,\quad
u_t=\epsilon-y.
\]

在**全部 \(D\) 个维度**上计算流损失。对于填充维度，目标终点为零，因此模型会显式学习将随机噪声输运回零。推理时，对全部 \(D\) 个维度采样，积分全部 \(D\) 个维度，然后将输出裁剪为前 \(d\) 个。

本地 OpenPI 检出（commit `15a9616`）就是这样做的：

- `openpi/src/openpi/transforms.py:328-337` 对状态/动作进行零填充；
- JAX 版本 `openpi/src/openpi/models/pi0.py:189-214` 使用 `actions.shape` 采样噪声，并在每个动作维度上计算 MSE；
- PyTorch 版本 `openpi/src/openpi/models_pytorch/pi0_pytorch.py:317-374` 返回完整的逐元素 MSE，不按真实自由度裁剪；
- 两个采样器都初始化完整的 `action_dim` 高斯；
- 实施例输出适配器裁剪到真实维度，例如 `openpi/src/openpi/policies/libero_policy.py:95-100`。

在线 OpenPI 源码显示了相同行为：[填充变换](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)、[JAX 流损失与采样器](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)，以及 [PyTorch 流损失与采样器](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)。

### B. 嵌入 32 维中的掩码低维流

或者，定义：

\[
\epsilon_m=m\odot\epsilon,\quad
y=m\odot y,\quad
x_t=t\epsilon_m+(1-t)y,\quad
u_t=\epsilon_m-y.
\]

只有活跃维度需要损失，但非活跃维度必须保持严格为零。实践中：

1. 在训练和推理期间将填充噪声置零；
2. 初始时将填充的 `x_t` 置零；
3. 在每个 ODE 步之后，掩码预测速度并/或钳制填充的 `x_t`；
4. 仅在真实维度上计算目标函数；
5. 像往常一样裁剪输出。

这是 \(d\) 维子空间上的流。它避免将容量花费在无意义的坐标上，同时保留预训练的 32 维投影形状。LeRobot X-VLA 文档中的 `auto` 动作模式同样会填充到预训练宽度，同时只在真实维度上应用损失，尽管其确切内部动态因策略而异：[X-VLA 动作空间文档](https://github.com/huggingface/lerobot/blob/main/docs/source/xvla.mdx)。

### 旧版推理与当前训练使用的无效混合形式

在推理修复之前，SmolVLA 路径在训练和推理时都这样做：

\[
\epsilon\sim\mathcal N(0,I_{32}),\qquad
\mathcal L=\mathcal L_{0:10}.
\]

相关代码：

- `prepare_action()` 填充到 `max_action_dim=32`：`src/lerobot/policies/smolvla/modeling_smolvla.py:490-493`；
- 训练使用该填充形状采样 `noise`：`src/lerobot/policies/smolvla/modeling_smolvla.py:774-786`；
- 封装将损失裁剪到 `original_action_dim=10`：`src/lerobot/policies/smolvla/modeling_smolvla.py:382-385`；
- 推理采样全部 32 个坐标：`src/lerobot/policies/smolvla/modeling_smolvla.py:826-829`；
- `action_in_proj` 将完整的 32 维向量映射到每个动作 token：`src/lerobot/policies/smolvla/modeling_smolvla.py:586`。

训练期间，填充坐标仍未被训练为终止于零。旧版推理也没有将它们保持为零。仅裁剪最终输出并不能阻止这些坐标改变隐藏动作 token，从而改变前十个输出。

这是继承自当前 LeRobot 而非 UMI 引入的。真实自由度损失裁剪于 2026-03-11 在 [LeRobot PR #3133](https://github.com/huggingface/lerobot/pull/3133) 中为 SmolVLA、π0 和 π0.5 合并。其审查意见称学习填充是不必要的，并将可定制处理留作未来任务，但采样器仍初始化全宽度噪声。当前上游 SmolVLA 源码同时包含[真实自由度损失裁剪](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py)和全宽度采样。本仓库现在用下文所述的掩码子空间积分覆盖该推理行为；在重新训练实验之前，损失有意保持不变。

本地 UMI 运行包含相同的裁剪。本地 commit `fbaeeb53` 正确修复了**时间** `action_is_pad` 掩码和逐样本归约，但也保留了维度裁剪。时间填充（回合边界附近无效的未来时间步）和动作维度填充是两个独立问题，需要分别掩码。

## 已实现的推理修复（无需重新训练）

`src/lerobot/policies/flow_matching.py` 现在提供一条由 SmolVLA、π0 和 π0.5 共用的积分路径。对于宽度为 `d` 的动作特征和模型宽度 `D`，它会：

1. 从检查点的 `action_feature` 推导 `d`；
2. 当 `d == D` 时保持行为不变；
3. 将调用者提供或生成的噪声在坐标 `d:D` 上置零，而不修改调用者的张量；
4. 在 RTC 看到基础去噪器速度之前对其进行掩码；
5. 对 RTC 的引导速度进行掩码，并在每一步之后钳制 Euler 状态。

第 4 步使 RTC 的终点估计和前缀误差保持在同一掩码子空间中；如果只掩码最终引导速度，其内部计算可能重新引入填充坐标。如果 RTC 引导对去噪器速度求导，它也能防止跨坐标泄漏。

新的配置字段为 `mask_padded_action_dims_at_inference: bool = True`，适用于全部三个策略。它对每个更小的动作空间都是自动的，并非 UMI 专用。现有检查点 JSON 文件不包含该字段，因此配置加载使用新的默认值 `True`。若需精确复现历史结果，将其设为 `false`；共享积分器随后执行原始全宽度 Euler 更新。

两个 UMI 评估器都会在 `action_dimension_inference` 中记录活跃/模型维度以及所选掩码模式。它们的 `--legacy_full_action_noise` 标志会针对旧采样器执行检查点固定的 A/B 对比。ACT 报告该模式不适用，因为它不是流策略。

这有意仅限推理。它消除了现有 UMI 检查点中已知的 22 坐标随机干扰，但并不能在事后使其历史训练目标在数学上自洽。后续重新训练 A/B 应比较全宽度 OpenPI 监督与掩码子空间训练。

## GPU 隔离实验

所有探针均使用 SmolVLA identity-rot6d 检查点 `0200000`、GPU 推理和相同的验证输入，除非另有说明。

### 修复后的配对验证

在相同的 25 个查询（回合 0–4，每个五个查询，种子 1000）上运行了已实现的默认掩码和 `--legacy_full_action_noise` 退出选项：

| 推理模式 | 旋转终点 | 旋转代理 | xyz 代理 | GT 旋转代理 |
| --- | ---: | ---: | ---: | ---: |
| 旧版全 32 维噪声 | 4.418° | 1.919° | 0.00658 m | 0.138° |
| 默认 10-of-32 掩码流 | **4.281°** | **0.963°** | **0.00388 m** | 0.138° |

正确的掩码将旋转代理降低 **49.8%**，将 XYZ 代理降低 **41.0%**，同时略微改善终点旋转精度。旧版行精确复现了先前报告的数值，确认退出选项在行为上兼容。

这也表明该修复必要但不充分：在该子集上，掩码后的 SmolVLA 仍比记录的旋转加速度高约 7×，比 ACT 的 0.041° 高约 23×。剩余的活跃坐标随机敏感性应通过后续重新训练/平滑度实验解决，而不是恢复填充噪声。

报告：

- `outputs/debug/padded_action_mask_validation_smolvla/smolvla_umi_identity_rot6d_1302_0200000_open_loop_metrics.json`
- `outputs/debug/padded_action_mask_validation_smolvla_legacy/smolvla_umi_identity_rot6d_1302_0200000_open_loop_metrics.json`

### 噪声坐标消融

| 初始潜变量 | 旋转代理 | xyz 代理 | 旋转终点 |
| --- | ---: | ---: | ---: |
| 全部 32 维随机 | 1.996° | 0.00670 m | 4.57° |
| 仅真实维度 0–9 随机 | 1.077° | 0.00458 m | 4.07° |
| 仅填充维度 10–31 随机 | 0.524° | 0.00174 m | 2.60° |
| 全部维度置零 | 0.093° | 0.00030 m | 2.31° |
| 真值 | 0.174° | 0.00058 m | — |

各分量不可加，因为网络是非线性的，但填充噪声明显且实质性地扰动了真实输出坐标。在每个 Euler 步后将填充坐标钳制为零，相比零填充初始化只带来较小的额外改善，确认初始化是该检查点的主要填充干预。

### 在全部 25 个配对查询上的零噪声解码

| SmolVLA 解码 | 旋转终点 | 旋转代理 | xyz 终点 | xyz 代理 |
| --- | ---: | ---: | ---: | ---: |
| 标准随机 | 4.418° | 1.919° | 0.0345 m | 0.00658 m |
| 零噪声 | 2.587° | 0.085° | 0.0331 m | 0.000326 m |
| 真值 | — | 0.138° | — | 0.000585 m |

零噪声是有用的诊断和即时实验分支。它不保证是统计上有意义的代表性样本：精确的零向量在 960 维高斯（`30 steps × 32 dimensions`）下是非典型的，并且确定性解码对多模态动作可能表现不佳。应在全部 100 个回合和真实任务成功率上验证。

### 已排除的其他假设

- **Euler 分辨率：** 使用 5、10 和 30 步分别得到约 1.90°、2.00° 和 2.11°。更多求解器步数不能消除振荡。
- **退化的 rot6d 投影：** 预测行范数和 Gram-Schmidt 残差范数保持在 1 附近，因此投影并未放大小向量。
- **SE(3)/rot6d 约定：** 本地行约定、Gram-Schmidt 重建以及 `T_ref^-1 @ T_target` 相对变换与 `universal_manipulation_interface` 和 `detached-umi-policy` 匹配。
- **被比较检查点之间的归一化不匹配：** ACT 和 SmolVLA identity 检查点都包含旋转维度 3–8 的 identity 统计量。因此 identity 归一化不能解释它们的跨策略差距。
- **流方向/符号：** 训练和采样使用一致的整流流约定。求解器方向正确。

## 去除填充噪声后仍存在的问题

十个真实坐标中的随机噪声在单帧消融中仍产生约 1.08°，远高于 GT。这不是数学必然性：准确学习到的条件流可以将独立高斯源噪声映射为连贯的轨迹样本。它反而表明，这个学习到的速度场在高频时间方向上仍然敏感。

可能的贡献因素包括：

- 没有显式的平移或 `SO(3)` 时间加速度目标；
- 局部时间归纳偏置弱于规范 UMI 的时间卷积扩散 UNet；
- 执行/评估完整的 30 个动作，而不是规范 UMI 更短的滚动执行时域；
- 此 SmolVLA 训练路径中没有 EMA；
- 调度器在 30K 时达到下限，而运行继续数百万次更新；
- 冻结骨干、仅微调专家限制了适应能力。

较强的终点精度表明，仅靠更通用的精度训练不太可能针对剩余的失败模式。

## 建议的实验

### 现有检查点，不重新训练

使用相同的查询索引运行相同的 100 回合/1,000 查询评估：

1. 标准 32 维随机噪声；
2. 维度 0–9 使用随机噪声，10–31 置零，填充状态按当前方式钳制；
3. 全零噪声；
4. 可选地，在终点或合理性约束下按平滑度排序的若干随机样本。

不要直接平均原始 rot6d 坐标，除非将每一步投影回 `SO(3)` 或使用旋转感知均值。基于平滑度的样本选择比无条件平均更能保留多模态分支。

### 后续重新训练 A/B/C

保持模型宽度和预训练权重为 32，但比较：

| 分支 | 噪声 | 损失 | ODE 填充坐标 |
| --- | --- | --- | --- |
| 当前混合基线 | 全部 32 | 真实 10 | 无约束 |
| OpenPI 全宽度 | 全部 32 | 全部 32 | 学习终止于零 |
| 掩码子空间 | 仅真实 10 | 真实 10 | 强制为零 |

全宽度和掩码子空间两个分支在数学上都是自洽的。不要将在不同填充目标下训练的检查点结果混合，除非标注该差异。

对于仅 UMI 训练，掩码子空间流是更经济的针对性设计。为与 OpenPI 和较旧预训练 π 检查点保持最大一致性，全宽度损失是保守的兼容性设计。实证 A/B 应决定哪种设计从特定 SmolVLA 基础检查点迁移得更好。

修复填充后，测试旋转感知的时间加速度惩罚、带重规划的更短执行时域，以及下游平移插值/四元数 SLERP。这些针对剩余的活跃坐标抖动，而不是掩盖填充缺陷。

## 历史检查点序列

这些较早的 `measure_chunk_jitter.py` 结果仍可作为趋势研究有用，但使用上述标量角度/二阶差分近似。回合 0–2，每个 60 个有效帧，`num_steps=10`。

GT 基线：验证平移 0.538 mm，旋转 8.95 mrad；训练平移 0.709 mm，旋转 9.42 mrad。

### 平移代理（mm）——预测（预测/GT）

| 模型 · 检查点 | 验证 | 训练 |
| --- | ---: | ---: |
| ACT 100K | 0.88 (1.6×) | 1.05 (1.5×) |
| ACT 500K | 0.56 (1.0×) | 0.61 (0.9×) |
| ACT 1100K | 0.34 (0.6×) | 0.53 (0.7×) |
| SmolVLA frozen 100K | 7.16 (13.3×) | 6.92 (9.8×) |
| SmolVLA frozen 500K | 6.50 (12.1×) | 6.48 (9.1×) |
| SmolVLA frozen 900K | 5.65 (10.5×) | 5.67 (8.0×) |
| SmolVLA unfrozen 100K | 5.40 (10.0×) | 6.94 (9.8×) |
| SmolVLA unfrozen 200K | 5.11 (9.5×) | 6.83 (9.6×) |
| SmolVLA unfrozen 400K | 4.75 (8.8×) | 5.98 (8.4×) |

### 旋转代理（mrad）——预测（预测/GT）

| 模型 · 检查点 | 验证 | 训练 |
| --- | ---: | ---: |
| ACT 100K | 4.45 (0.50×) | 5.67 (0.60×) |
| ACT 500K | 4.22 (0.47×) | 3.22 (0.34×) |
| ACT 1100K | 2.99 (0.33×) | 2.71 (0.29×) |
| SmolVLA frozen 100K | 91.0 (10.2×) | 76.3 (8.1×) |
| SmolVLA frozen 500K | 83.1 (9.3×) | 69.1 (7.3×) |
| SmolVLA frozen 900K | 77.3 (8.6×) | 61.5 (6.5×) |
| SmolVLA unfrozen 100K | 63.4 (7.1×) | 77.8 (8.3×) |
| SmolVLA unfrozen 200K | 61.9 (6.9×) | 75.8 (8.0×) |
| SmolVLA unfrozen 400K | 58.3 (6.5×) | 64.7 (6.9×) |

ACT 变得平滑得多而 SmolVLA 趋于平台的历史结论仍然成立。因果解释发生了变化：这并不是固有流匹配下限的证明。它与固定宽度填充不匹配、残余活跃坐标敏感性以及缺乏显式平滑目标相一致。
