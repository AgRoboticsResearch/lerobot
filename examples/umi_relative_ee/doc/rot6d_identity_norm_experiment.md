# Experiment: identity rot6d normalization (rotation-jumpiness A/B)

> Companion to [`rotation_normalization.md`](./rotation_normalization.md) (the analysis).
> This file is the **experiment plan + run log**. Last updated: 2026-07-30.

## Hypothesis

- **H1** — Per-dimension MIN_MAX/QUANTILES normalization of the 6D-rotation action
  components is a material cause of the full-data rotation "jumpiness" floor
  (~4.6–5.6°). Leaving rotation at **identity** (UMI-style) reduces that floor.
- **H0 (null)** — Normalization of rot6d has no measurable effect on rotation
  generalization.

Mechanism (full derivation in `rotation_normalization.md`): relative rot6d clusters
near identity `[1,0,0,0,1,0]`; per-dim quantile/min-max scaling stretches the thin
near-constant diagonal bands and gives those dims the largest loss weight
`1/(q99−q01)`, under-weighting the off-diagonals that carry rotation direction.

## Arms

| Arm | `umi_rot6d_identity_norm` | rot6d treatment | Everything else |
| --- | --- | --- | --- |
| **A — baseline** | `false` | per-dim MIN_MAX (ACT/SmolVLA) / QUANTILES (π0.5) | operator's existing runs |
| **B — identity** | `true` | identity (unscaled), like UMI | trained here |

## Metrics & decision criterion

The hypothesis is about **jumpiness (within-chunk smoothness)**, not endpoint accuracy
— and the two are independent. On a 5-episode spot check SmolVLA had the *better*
chunk-end rotation error (4.4° vs ACT 5.8°) yet its within-chunk rotation
acceleration proxy was **~47× larger** (`rot_jerk_deg` 1.9 vs 0.04). So an
endpoint metric ranks the jittery model higher and is blind to the thing we're
testing. **Within-chunk smoothness is the decision metric; accuracy is secondary.**

- **Primary (decision)**: `summary.episode_balanced.rot_jerk_deg` — legacy key for
  the mean within-chunk rotation **second-difference/acceleration proxy** of one
  predicted chunk (**no** open-loop/cross-frame accumulation), lower = smoother.
  `within_chunk_jerk()` also reports the legacy `gt_rot_jerk_deg` key as a
  real-motion reference. Compare identity vs baseline **within the same policy**
  at matched steps. Cross-policy smoothness is descriptive, but cannot decide
  whether normalization caused the difference because the decoders differ.
- **Decision**: H1 supported if Arm B's `rot_jerk_deg` is meaningfully lower than the
  same policy's baseline at matched steps (≥~20% relative, beyond seed noise). Else H0.
- **Secondary (accuracy)**: `episode_balanced.rotation_end_deg` (chunk-end, 30-ahead)
  and `rotation_chunk_mean_deg` — does identity-rot6d also keep/improve endpoint accuracy.
- ⚠ `val/loss` is **not** comparable across arms (different action normalization →
  different MSE scale).
- **Tertiary**: RTC metrics from `eval_rtc_dataset.py` for SmolVLA and π0.5 only
  (deployment/overlap; ACT has no RTC).

### SmolVLA padding/noise confound discovered during jitter review

The follow-up implementation audit found a separate source of flow-policy
within-chunk jitter; see
[`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md).
SmolVLA pads this 10-D UMI action to 32 dimensions and samples noise in all 32,
but its training wrapper computes loss only on the ten real dimensions. The
22 random, unsupervised coordinates enter the shared action projection and
measurably perturb the real outputs.

This does **not** invalidate the identity-normalization hypothesis, but it changes
how results must be attributed:

- identity vs scaled rot6d remains a valid normalization A/B only when both
  checkpoints used the same action-padding loss/noise formulation;
- ACT vs SmolVLA smoothness is not evidence for or against rotation
  normalization because their decoder mechanisms differ;
- comparisons against older SmolVLA baselines must record whether they were
  trained before or after the real-DoF loss crop was introduced;
- SmolVLA padding/noise ablations must hold normalization and checkpoint fixed.

OpenPI's reference implementation uses a coherent full-width formulation:
zero-pad targets to 32, sample 32-D noise, and supervise all 32 flow outputs so
the padded coordinates learn to terminate at zero. A coherent masked alternative
would sample and integrate noise only in the ten real dimensions while forcing
the remaining coordinates to zero. Current SmolVLA mixes the two designs.

The inference confound is now controlled by default: SmolVLA, π0, and π0.5 mask
coordinates beyond the checkpoint's real action width throughout the flow ODE,
including inside RTC. This works with existing checkpoints and does not require
retraining. Training still uses the historical real-DoF loss with full-width
noise, so future retraining results must continue to report their training
padding objective. Do not claim that identity rot6d alone caused or fixed the
flow-policy jitter.

Both evaluators write `action_dimension_inference` into the JSON report. Use the
default masked mode for primary results. Add `--legacy_full_action_noise` only
for a checkpoint-fixed diagnostic A/B against the pre-fix sampler.

## Phase 0 — stats distortion on real data (GPU-free)

`measure_rot6d_normalization_distortion.py`. Effective per-dim loss weight
`1/(q99−q01)`:

| dataset | rot6d QUANTILES max/min | rot6d diag/off-diag | rot6d MIN_MAX max/min |
| --- | --- | --- | --- |
| `1000onesb` (88,218 fr / 1,012 ep) | **10.6×** | 3.18× | ~7.8× |
| **`1302_occlusion` (121,262 fr / 1,302 ep) ← experiment dataset** | **7.8×** | 2.72× | ~4.1× |

On the experiment dataset the diagonal `rot6d_0/4` dims have q99−q01 ≈ 0.06–0.07
(near-constant) vs off-diagonal ≈ 0.35–0.49 → they get ~3× the loss weight of the
dims that encode rotation direction. Mechanism confirmed; A/B justified.

## Phase 1 — training A/B: RUNNING 2026-07-30 (identity arm only)

All three policies train on **`sroiv2_strawberry_picking_lab_1302_occlusion`**,
validate on the **100-episode validation set**, with `--policy.umi_rot6d_identity_norm=true`
and `--val_freq=10000`. Concurrent on one RTX 4090 (~16/24 GB). Verified: every log
contains "UMI relative-EE stats … forced to identity". Baselines (Arm A) are the
operator's existing runs.

| Policy | Entrypoint | Norm | bs | steps | save | ~rate | ETA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | `train_relative_ee_processor.py --policy.type=act` | MIN_MAX | 8 | 2,500,000 | 100k | ~8–26/s | days |
| SmolVLA | `train_relative_ee_processor.py --policy.path=lerobot/smolvla_base` | MIN_MAX | 8 | 2,500,000 | 100k | ~7/s | days |
| π0.5 | `train_pi05_lora.py --policy.pretrained_path=lerobot/pi05_base` (LoRA **r16**, bf16, grad-ckpt) | QUANTILES | 2 | 500,000 | 100k | ~2.5/s | ~55h |

Output dirs: `outputs/train/{act_umi_identity_rot6d_1302, smolvla_umi_identity_rot6d_1302,
pi05_lora_r16_umi_identity_rot6d_1302}`. Logs: `logs/{act,smolvla,pi05}_identity_1302.log`.
Checkpoints at `.../checkpoints/<step>/pretrained_model` every 100k → first eval point ~100k.

### Exact commands (as run)

ACT:
```bash
PYTHONPATH=src python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=10000 \
  --policy.type=act --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
  --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.repo_id=zfff/act_umi_identity_rot6d_1302 --policy.push_to_hub=false \
  --seed=1000 --save_freq=100000 --steps=2500000 --batch_size=8 --num_workers=4 \
  --log_freq=200 --eval_freq=0 \
  --output_dir=outputs/train/act_umi_identity_rot6d_1302 --job_name=act_umi_identity_rot6d_1302 \
  --wandb.enable=true --wandb.project=lerobot
```

SmolVLA (`HF_HUB_OFFLINE=1` — both `smolvla_base` and `SmolVLM2-500M-Video-Instruct` are cached):
```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=10000 \
  --policy.path=lerobot/smolvla_base --policy.input_features=null \
  --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
  --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 --policy.train_state_proj=true \
  --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 --policy.scheduler_decay_lr=0.0000025 \
  --policy.repo_id=zfff/smolvla_umi_identity_rot6d_1302 --policy.push_to_hub=false \
  --seed=1000 --batch_size=8 --num_workers=4 --steps=2500000 --save_freq=100000 \
  --log_freq=200 --eval_freq=0 \
  --output_dir=outputs/train/smolvla_umi_identity_rot6d_1302 --job_name=smolvla_umi_identity_rot6d_1302 \
  --wandb.enable=true --wandb.project=lerobot
```

π0.5 (LoRA **rank 16**, `scheduler_decay_steps=500000`):
```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src python examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=10000 \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.0001 --policy.scheduler_decay_lr=0.00001 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=500000 \
  --policy.repo_id=zfff/pi05_lora_r16_umi_identity_rot6d_1302 --policy.push_to_hub=false \
  --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=16 \
  --batch_size=2 --num_workers=8 --prefetch_factor=2 \
  --seed=1000 --steps=500000 --save_freq=100000 --log_freq=50 --eval_freq=0 \
  --output_dir=outputs/train/pi05_lora_r16_umi_identity_rot6d_1302 --job_name=pi05_lora_r16_umi_identity_rot6d_1302 \
  --wandb.enable=true --wandb.project=lerobot
```

Launcher (reproduces all three): `bash examples/umi_relative_ee/shell_scripts/run_identity_ab.sh {act|smolvla|pi05}`.

### Evaluation (post-hoc, primary metric — all policies, all 100 episodes)
```bash
PYTHONPATH=src python examples/umi_relative_ee/eval_open_loop_dataset.py \
  --pretrained_path=outputs/train/<policy>_umi_identity_rot6d_1302/checkpoints/<step>/pretrained_model \
  --dataset_root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --samples_per_episode=10 --seed=1000 --device=cuda \
  --output_dir=outputs/debug/open_loop_identity_<policy>_<step>
```

Omitting `--episode_indices` intentionally selects every episode in the dataset.
For this validation dataset the report must show `"dataset_total_episodes": 100`,
`"summary.num_episodes": 100`, and normally `"summary.num_samples": 1000`.
The decision value is `"summary.episode_balanced.rot_jerk_deg"` (legacy key for
the within-chunk rotation acceleration proxy, lower = smoother; see Metrics).
The output also records `rotation_end_deg` / `rotation_chunk_mean_deg` (secondary
accuracy), XYZ, gripper, the legacy `gt_*_jerk` real-motion-reference keys,
per-episode, per-sample, and padded-action inference-mode metadata. Flow policies
use masked padded dimensions by default; no retraining is needed to evaluate an
existing checkpoint. To reproduce the historical sampler, add
`--legacy_full_action_noise` and label that result as legacy.

Optional RTC evaluation for SmolVLA/π0.5 deployment behavior (not ACT, and not the
primary A/B metric):
```bash
PYTHONPATH=src python examples/umi_relative_ee/eval_rtc_dataset.py \
  --pretrained_path=outputs/train/<policy>_umi_identity_rot6d_1302/checkpoints/<step>/pretrained_model \
  --dataset_root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --transitions_per_episode=3 \
  --query_stride=5 --inference_delay=4 --execution_horizon=10 \
  --max_guidance_weight=10.0 --device=cuda \
  --output_dir=outputs/debug/rtc_identity_<policy>_<step>
```

### Full-validation identity-arm accuracy results (2026-07-31)

GPU evaluation with `eval_open_loop_dataset.py`, all 100 validation episodes,
10 evenly spaced valid queries per episode (1,000 queries), seed 1000. Values are
episode-balanced; centimetres below are converted from the report's metres.

| Policy | Ckpt | rot chunk mean | rot chunk end (secondary) | xyz chunk mean | xyz chunk end | gripper chunk/end |
| --- | --- | --- | --- | --- | --- | --- |
| ACT | 0300000 | 2.793° | 4.857° | 1.48 cm | 2.30 cm | 0.096 / 0.141 |
| SmolVLA | 0200000 | 2.958° | 5.059° | 1.53 cm | 2.73 cm | 0.108 / 0.172 |

Reports:

- `outputs/debug/open_loop_identity_act_0300000/act_umi_identity_rot6d_1302_0300000_open_loop_metrics.json`
- `outputs/debug/open_loop_identity_smolvla_0200000/smolvla_umi_identity_rot6d_1302_0200000_open_loop_metrics.json`

These establish identity-arm values only. They do **not** support or reject H1 until
the same evaluator is run on matched non-identity checkpoints at the same steps.
π0.5 is intentionally pending because the concurrent GPU did not have enough free
memory for that evaluation.

### Monitoring
Hourly session cron checks the three runs and updates the live status below; flags
crashes; runs the post-hoc eval and records results once all three finish.

### Live status
_Checked by babysit cron (e9f9fe49). 2026-07-31 ~14:45._
- ACT — RUNNING, identity active (1302). ~518.6K / 2.5M (21%); val 0.0357 @510K (flat). no errors.
- SmolVLA — RUNNING, identity active (1302). ~295K / 2.5M (12%); val 0.0271 @290K (flat ~0.027). no errors.
- π0.5 — RUNNING, identity active (1302, r16). ~130.4K / 500K (26%); val 0.0260 @130K (recovered; best 0.0254 @90K). no errors.
GPU 16.1 / 24 GB, 100% util. (val/loss is within-arm health only — not comparable to
the scaled-rot6d baseline arm; the decision metric is the post-hoc within-chunk
rotation acceleration proxy.)

### Visualizations (open-loop prediction, validation eps 0–2)

Per `prediction_visualization.md`, most-recent checkpoints → projected trajectory
videos + `prediction_metrics.json` (mean **chunk-end** error, 161 frames over 3 val
episodes). Task `"pick the strawberry"`, D405 rig `--project`.

| Policy | Ckpt | rot end-err | xyz end-err | gripper err | output dir |
| --- | --- | --- | --- | --- | --- |
| ACT | 0400000 | **5.8°** (0.101 rad) ↑from 5.2°@200k | 2.9 cm | 0.238 | `outputs/debug/viz_act_identity_1302_0400000/…/pred_episode_{0,1,2}.mp4` |
| SmolVLA | 0200000 | **4.3°** masked (4.3° unmasked) | 3.6 cm | 0.241 | `…/viz_smolvla_identity_1302_0200000_masked/…` (unmasked: `…_0200000`) |
| π0.5 | 100000 | **4.8°** masked (5.3° unmasked, −9%) | 4.4 cm | 0.268 | `…/viz_pi05_identity_1302_100000_masked/…` (unmasked: `…/viz_pi05_identity_1302`) |

These are **early** checkpoints on a 3-episode sample — directional only, not the
policy-neutral full-validation result (that is `eval_open_loop_dataset.py`; its reports
are listed above this section). Sample trend: ACT drifting up (slight overfit; val flat
~0.035), SmolVLA edging down (improving). Compare each to the operator's baseline
`prediction_metrics.json` at matched steps. Earlier-checkpoint viz kept in step-specific
dirs (`viz_act_identity_1302_0200000`, `viz_smolvla_identity_1302_0100000`).

## Phase 2 — confirmation: _pending Phase 1 result_

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 实验：identity rot6d 归一化（旋转抖动性 A/B）

> [`rotation_normalization.md`](./rotation_normalization.md)（分析文档）的配套文档。
> 本文件是**实验计划 + 运行日志**。最后更新：2026-07-30。

## 假设

- **H1** — 对 6D 旋转动作分量按维度做 MIN_MAX/QUANTILES 归一化，是全量数据旋转“抖动性”下限（约 4.6–5.6°）的一个重要原因。将旋转保持为 **identity**（UMI 风格）可降低该下限。
- **H0（零假设）** — rot6d 的归一化对旋转泛化没有可测量的影响。

机理（完整推导见 `rotation_normalization.md`）：相对 rot6d 聚集在 identity `[1,0,0,0,1,0]` 附近；按维度的分位数/min-max 缩放会拉伸那条狭窄的近常数对角线带，并给这些维度赋予最大的损失权重 `1/(q99−q01)`，从而相对低估了携带旋转方向的非对角线分量。

## 分支

| 分支 | `umi_rot6d_identity_norm` | rot6d 处理 | 其余一切 |
| --- | --- | --- | --- |
| **A — baseline** | `false` | 逐维 MIN_MAX（ACT/SmolVLA）/ QUANTILES（π0.5） | 操作员已有的运行 |
| **B — identity** | `true` | identity（不缩放），与 UMI 相同 | 在此训练 |

## 指标与判定标准

该假设关注的是**抖动性（chunk 内平滑度）**，而不是端点精度——两者相互独立。在一次 5 个 episode 的抽查中，SmolVLA 的 chunk 末端旋转误差*更好*（4.4° 对 ACT 5.8°），但其 chunk 内旋转加速度代理指标**约大 47 倍**（`rot_jerk_deg` 1.9 对 0.04）。因此端点指标会把抖动更大的模型排在前面，而对我们要检验的东西不可见。**chunk 内平滑度是判定指标；精度是次要的。**

- **主要（判定）**：`summary.episode_balanced.rot_jerk_deg` —— 取预测的单个 chunk 的 chunk 内旋转**二阶差分/加速度代理**的均值所用的旧版键（**不含**开环/跨帧累积），越低越平滑。`within_chunk_jerk()` 还会报告旧版 `gt_rot_jerk_deg` 键，作为真实运动的参照。在匹配的 step 上**在同一策略内部**比较 identity 与 baseline。跨策略的平滑度只是描述性的，不能用来判定归一化是否造成了差异，因为解码器不同。
- **判定**：如果在匹配 step 上分支 B 的 `rot_jerk_deg` 显著低于同一策略的 baseline（相对值 ≥~20%，超出随机种子噪声），则支持 H1。否则接受 H0。
- **次要（精度）**：`episode_balanced.rotation_end_deg`（chunk 末端，提前 30 步）和 `rotation_chunk_mean_deg` —— identity-rot6d 是否也能保持/改善端点精度。
- ⚠ `val/loss` 在各分支之间**不可**比较（动作归一化不同 → MSE 尺度不同）。
- **第三级**：仅对 SmolVLA 和 π0.5 使用 `eval_rtc_dataset.py` 的 RTC 指标（部署/重叠；ACT 没有 RTC）。

### 抖动审查中发现的 SmolVLA padding/noise 混淆因素

后续的实现审计发现了 flow 策略 chunk 内抖动的另一个来源；参见 [`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)。SmolVLA 将这个 10 维 UMI 动作 padding 到 32 维，并在全部 32 维上采样噪声，但其训练包装器只在十个真实维度上计算损失。这 22 个随机、无监督的坐标进入共享的动作投影，并会可测量地扰动真实输出。

这**不会**使 identity 归一化假设失效，但会改变结果的归因方式：

- identity 与缩放 rot6d 的对比仍然是有效的归一化 A/B，但仅当两个 checkpoint 使用相同的动作 padding 损失/噪声公式时；
- ACT 与 SmolVLA 的平滑度对比不能作为支持或反对旋转归一化的证据，因为它们的解码器机理不同；
- 与更早 SmolVLA baseline 的对比必须记录它们是在真实自由度损失裁剪引入之前还是之后训练的；
- SmolVLA 的 padding/noise 消融必须固定归一化和 checkpoint。

OpenPI 的参考实现使用一致的 full-width 公式：将目标零填充到 32 维，采样 32 维噪声，并对全部 32 个 flow 输出进行监督，使填充坐标学会收敛到零。一个一致的 masked 替代方案是只在十个真实维度上采样并积分噪声，同时强制剩余坐标为零。当前 SmolVLA 混用了这两种设计。

现在推理端的混淆因素已默认受控：SmolVLA、π0 和 π0.5 在整个 flow ODE 中（包括 RTC 内部）会屏蔽超出 checkpoint 真实动作宽度的坐标。这适用于现有 checkpoint，且无需重新训练。训练仍然使用历史性的真实自由度损失与 full-width 噪声，因此未来的重训结果必须继续报告其训练 padding 目标。不要声称仅凭 identity rot6d 就造成或修复了 flow 策略的抖动。

两个评估器都会将 `action_dimension_inference` 写入 JSON 报告。主要结果使用默认的 masked 模式。仅针对修复前采样器进行固定 checkpoint 的诊断性 A/B 时，才添加 `--legacy_full_action_noise`。

## 阶段 0 — 真实数据上的统计失真（无需 GPU）

`measure_rot6d_normalization_distortion.py`。有效逐维损失权重 `1/(q99−q01)`：

| 数据集 | rot6d QUANTILES max/min | rot6d diag/off-diag | rot6d MIN_MAX max/min |
| --- | --- | --- | --- |
| `1000onesb`（88,218 帧 / 1,012 ep） | **10.6×** | 3.18× | ~7.8× |
| **`1302_occlusion`（121,262 帧 / 1,302 ep）← 实验数据集** | **7.8×** | 2.72× | ~4.1× |

在实验数据集上，对角线 `rot6d_0/4` 维度的 q99−q01 ≈ 0.06–0.07（近常数），而非对角线 ≈ 0.35–0.49 → 它们获得的损失权重约为编码旋转方向维度的 3 倍。机理得到确认；A/B 合理。

## 阶段 1 — 训练 A/B：2026-07-30 运行中（仅 identity 分支）

三个策略都在 **`sroiv2_strawberry_picking_lab_1302_occlusion`** 上训练，在 **100 个 episode 的验证集**上验证，使用 `--policy.umi_rot6d_identity_norm=true` 和 `--val_freq=10000`。在单张 RTX 4090 上并发（~16/24 GB）。已验证：每个日志都包含 “UMI relative-EE stats … forced to identity”。Baseline（分支 A）是操作员已有的运行。

| 策略 | 入口 | Norm | bs | steps | save | ~速率 | ETA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | `train_relative_ee_processor.py --policy.type=act` | MIN_MAX | 8 | 2,500,000 | 100k | ~8–26/s | 数天 |
| SmolVLA | `train_relative_ee_processor.py --policy.path=lerobot/smolvla_base` | MIN_MAX | 8 | 2,500,000 | 100k | ~7/s | 数天 |
| π0.5 | `train_pi05_lora.py --policy.pretrained_path=lerobot/pi05_base`（LoRA **r16**，bf16，grad-ckpt） | QUANTILES | 2 | 500,000 | 100k | ~2.5/s | ~55h |

输出目录：`outputs/train/{act_umi_identity_rot6d_1302, smolvla_umi_identity_rot6d_1302,
pi05_lora_r16_umi_identity_rot6d_1302}`。日志：`logs/{act,smolvla,pi05}_identity_1302.log`。
每 100k 步在 `.../checkpoints/<step>/pretrained_model` 保存 checkpoint → 首个评估点约在 100k。

### 确切命令（实际运行时）

ACT：
```bash
PYTHONPATH=src python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=10000 \
  --policy.type=act --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
  --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.repo_id=zfff/act_umi_identity_rot6d_1302 --policy.push_to_hub=false \
  --seed=1000 --save_freq=100000 --steps=2500000 --batch_size=8 --num_workers=4 \
  --log_freq=200 --eval_freq=0 \
  --output_dir=outputs/train/act_umi_identity_rot6d_1302 --job_name=act_umi_identity_rot6d_1302 \
  --wandb.enable=true --wandb.project=lerobot
```

SmolVLA（`HF_HUB_OFFLINE=1` — `smolvla_base` 和 `SmolVLM2-500M-Video-Instruct` 均已缓存）：
```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=10000 \
  --policy.path=lerobot/smolvla_base --policy.input_features=null \
  --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
  --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 --policy.train_state_proj=true \
  --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 --policy.scheduler_decay_lr=0.0000025 \
  --policy.repo_id=zfff/smolvla_umi_identity_rot6d_1302 --policy.push_to_hub=false \
  --seed=1000 --batch_size=8 --num_workers=4 --steps=2500000 --save_freq=100000 \
  --log_freq=200 --eval_freq=0 \
  --output_dir=outputs/train/smolvla_umi_identity_rot6d_1302 --job_name=smolvla_umi_identity_rot6d_1302 \
  --wandb.enable=true --wandb.project=lerobot
```

π0.5（LoRA **rank 16**，`scheduler_decay_steps=500000`）：
```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src python examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=10000 \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true --policy.umi_rot6d_identity_norm=true \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.0001 --policy.scheduler_decay_lr=0.00001 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=500000 \
  --policy.repo_id=zfff/pi05_lora_r16_umi_identity_rot6d_1302 --policy.push_to_hub=false \
  --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=16 \
  --batch_size=2 --num_workers=8 --prefetch_factor=2 \
  --seed=1000 --steps=500000 --save_freq=100000 --log_freq=50 --eval_freq=0 \
  --output_dir=outputs/train/pi05_lora_r16_umi_identity_rot6d_1302 --job_name=pi05_lora_r16_umi_identity_rot6d_1302 \
  --wandb.enable=true --wandb.project=lerobot
```

启动脚本（复现全部三个运行）：`bash examples/umi_relative_ee/shell_scripts/run_identity_ab.sh {act|smolvla|pi05}`。

### 评估（事后，主指标 — 所有策略，全部 100 个 episode）
```bash
PYTHONPATH=src python examples/umi_relative_ee/eval_open_loop_dataset.py \
  --pretrained_path=outputs/train/<policy>_umi_identity_rot6d_1302/checkpoints/<step>/pretrained_model \
  --dataset_root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --samples_per_episode=10 --seed=1000 --device=cuda \
  --output_dir=outputs/debug/open_loop_identity_<policy>_<step>
```

省略 `--episode_indices` 会有意选择数据集中的每个 episode。
对于这个验证数据集，报告必须显示 `"dataset_total_episodes": 100`、
`"summary.num_episodes": 100`，并且通常还有 `"summary.num_samples": 1000`。
判定值是 `"summary.episode_balanced.rot_jerk_deg"`（chunk 内旋转加速度代理的旧版键，越低越平滑；参见“指标”一节）。
输出还会记录 `rotation_end_deg` / `rotation_chunk_mean_deg`（次要精度）、XYZ、gripper、旧版 `gt_*_jerk` 真实运动参照键、逐 episode、逐样本以及 padded-action 推理模式元数据。Flow 策略默认使用 masked 填充维度；评估现有 checkpoint 无需重新训练。要复现历史采样器，请添加 `--legacy_full_action_noise`，并将该结果标记为 legacy。

可选的 RTC 评估，用于观察 SmolVLA/π0.5 的部署行为（不适用于 ACT，也不是主 A/B 指标）：
```bash
PYTHONPATH=src python examples/umi_relative_ee/eval_rtc_dataset.py \
  --pretrained_path=outputs/train/<policy>_umi_identity_rot6d_1302/checkpoints/<step>/pretrained_model \
  --dataset_root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --transitions_per_episode=3 \
  --query_stride=5 --inference_delay=4 --execution_horizon=10 \
  --max_guidance_weight=10.0 --device=cuda \
  --output_dir=outputs/debug/rtc_identity_<policy>_<step>
```

### identity 分支全验证集精度结果（2026-07-31）

使用 `eval_open_loop_dataset.py` 进行 GPU 评估，覆盖全部 100 个验证 episode，每个 episode 取 10 个均匀间隔的有效查询（共 1,000 个查询），seed 1000。数值按 episode 平衡；下方的厘米值由报告中的米换算而来。

| 策略 | Ckpt | rot chunk mean | rot chunk end（次要） | xyz chunk mean | xyz chunk end | gripper chunk/end |
| --- | --- | --- | --- | --- | --- | --- |
| ACT | 0300000 | 2.793° | 4.857° | 1.48 cm | 2.30 cm | 0.096 / 0.141 |
| SmolVLA | 0200000 | 2.958° | 5.059° | 1.53 cm | 2.73 cm | 0.108 / 0.172 |

报告：

- `outputs/debug/open_loop_identity_act_0300000/act_umi_identity_rot6d_1302_0300000_open_loop_metrics.json`
- `outputs/debug/open_loop_identity_smolvla_0200000/smolvla_umi_identity_rot6d_1302_0200000_open_loop_metrics.json`

这些结果仅确立了 identity 分支的数值。在相同的 step 上对匹配的非 identity checkpoint 运行相同的评估器之前，它们**不能**支持或否定 H1。π0.5 有意暂缓，因为并发使用的 GPU 没有足够的空闲显存来完成该评估。

### 监控
每小时的会话 cron 检查这三个运行，并更新下方的实时状态；标记崩溃；一旦三个运行全部完成，就运行事后评估并记录结果。

### 实时状态
_由 babysit cron（e9f9fe49）检查。2026-07-31 ~14:45。_
- ACT — RUNNING（运行中），identity 已启用（1302）。~518.6K / 2.5M（21%）；val 0.0357 @510K（持平）。无错误。
- SmolVLA — RUNNING（运行中），identity 已启用（1302）。~295K / 2.5M（12%）；val 0.0271 @290K（持平 ~0.027）。无错误。
- π0.5 — RUNNING（运行中），identity 已启用（1302，r16）。~130.4K / 500K（26%）；val 0.0260 @130K（已恢复；最佳 0.0254 @90K）。无错误。
GPU 16.1 / 24 GB，利用率 100%。（val/loss 仅用于反映分支内部健康状态——不能与缩放 rot6d 的 baseline 分支比较；判定指标是事后的 chunk 内旋转加速度代理。）

### 可视化（开环预测，验证集 ep 0–2）

按照 `prediction_visualization.md`，使用最近的 checkpoint → 投影轨迹视频 + `prediction_metrics.json`（平均 **chunk-end** 误差，3 个验证 episode 共 161 帧）。任务 `"pick the strawberry"`，D405 装置 `--project`。

| 策略 | Ckpt | rot end-err | xyz end-err | gripper err | 输出目录 |
| --- | --- | --- | --- | --- | --- |
| ACT | 0400000 | **5.8°**（0.101 rad）↑较 200k 的 5.2° | 2.9 cm | 0.238 | `outputs/debug/viz_act_identity_1302_0400000/…/pred_episode_{0,1,2}.mp4` |
| SmolVLA | 0200000 | **4.3°** masked（unmasked 4.3°） | 3.6 cm | 0.241 | `…/viz_smolvla_identity_1302_0200000_masked/…`（unmasked：`…_0200000`） |
| π0.5 | 100000 | **4.8°** masked（unmasked 5.3°，−9%） | 4.4 cm | 0.268 | `…/viz_pi05_identity_1302_100000_masked/…`（unmasked：`…/viz_pi05_identity_1302`） |

这些是 3 个 episode 样本上的**早期** checkpoint——仅具方向性，不是策略中立的完整验证结果（后者由 `eval_open_loop_dataset.py` 给出；其报告列在本节上方）。样本趋势：ACT 上升（轻微过拟合；val 持平 ~0.035），SmolVLA 略有下降（在改善）。在匹配的 step 上分别与操作员 baseline 的 `prediction_metrics.json` 比较。更早 checkpoint 的可视化保留在按 step 区分的目录中（`viz_act_identity_1302_0200000`、`viz_smolvla_identity_1302_0100000`）。

## 阶段 2 — 确认：_等待阶段 1 结果_
