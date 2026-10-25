# Padded-noise strategies for UMI flow matching

SmolVLA and π0.5 support two coherent padded-coordinate strategies in this
repository: OpenPI-compatible full-width flow (Option A, the default) and
masked-subspace flow (Option B). This page defines both strategies and explains
how they affect old checkpoints, training, and inference.

The implementation follows the flow head in the local OpenPI checkout at
commit `03893832f2002a31016586874658662c669081ac`.

## Scope

For UMI relative-EE data:

- the real action has 10 values: XYZ, rotation-6D, and gripper;
- the model action width remains 32;
- normalization is policy-specific and unchanged;
- normalized actions are zero-padded from 10 to 32;
- SmolVLA's causal action attention is unchanged;
- π0 is unchanged.

The fixed-width coordinates 10 through 31 are not robot commands. Under the
default Option A strategy, they remain part of the learned flow and are
supervised to end at zero.

## Training equation

Let `a_real` be the normalized real action chunk and let `a` be its
zero-padded 32D representation:

```python
a_real = normalize(raw_action)        # [batch, chunk, 10]
a = pad_with_zeros(a_real, 32)        # [batch, chunk, 32]
epsilon = torch.randn_like(a)         # Gaussian noise in every dimension
t = Beta(1.5, 1.0) * 0.999 + 0.001
x_t = t * epsilon + (1.0 - t) * a
u_t = epsilon - a
v_t = model(observation, x_t, t)
loss_elementwise = (v_t - u_t).square()
```

The optimized loss averages all 32 coordinates. Consequently, the padded
coordinates learn the valid transport `epsilon -> 0`. They must not be
discarded from the loss.

`action_is_pad` has a different meaning: it marks invalid chunk timesteps at
episode boundaries. It removes complete timesteps from the loss but never
removes action coordinates.

Training reports:

- `loss`: optimization loss over valid timesteps and all 32 dimensions;
- `loss_per_dim`: the 32 individual coordinate losses;
- `flow_loss_real_dims`: mean over the real robot coordinates;
- `flow_loss_padded_dims`: mean over coordinates 10 through 31.

## Inference equation

Inference starts from full-width Gaussian noise and integrates from noise at
`t=1` to actions at `t=0`:

```python
x_t = torch.randn(batch, chunk, 32)
dt = -1.0 / num_steps

for step in range(num_steps):
    t = 1.0 + step * dt
    v_t = model.denoise_step(observation, x_t, t)
    x_t = x_t + dt * v_t

robot_actions = x_t[..., :10]
```

The default is 10 Euler steps. Noise, predicted velocity, RTC output, and Euler
state are never clamped or masked by action dimension. Cropping happens only
after the complete 32D integration.

`mask_padded_action_dims_at_inference` remains in serialized configs so old
checkpoints load, but SmolVLA and π0.5 ignore it. Its default is now `false`.
There is no CLI flag required to enable full-width inference.

## Existing checkpoints

A checkpoint trained by the previous mixed formulation saw 32D Gaussian noise
but optimized only the real 10 dimensions. Switching its inference to
full-width makes inference consistent with OpenPI, but cannot retroactively
teach its padded outputs to flow to zero.

Retrain SmolVLA and π0.5 to obtain a checkpoint that is consistent end to end.
Do not use masked inference for the retrained checkpoint. Do not use zero-noise
decoding as the deployment default; it remains a diagnostic for stochastic
sensitivity.

Existing normalization statistics and UMI processors are reused. Recompute
statistics only when the dataset or action representation changes.

## Training and deployment

Use the normal UMI training entrypoints:

- SmolVLA: [smolvla_relative_ee_training.md](./smolvla_relative_ee_training.md)
- π0.5: [README.md](./README.md#training-baseline) and
  [train_pi05_lora.sh](../shell_scripts/train_pi05_lora.sh)

Activate the existing environment before running commands:

```bash
conda activate py312
```

New checkpoints serialize `mask_padded_action_dims_at_inference=false`.
Async and synchronous deployment both use the same policy sampler, so the
server/client setup requires no special flow-matching option.

## Launch log (2026-08-01)

All runs below use the full-width contract above, dataset
`sroiv2_strawberry_picking_lab_1302_occlusion` (train) and
`sroiv2_strawberry_picking_lab_validation` (val), chunk 30, seed 1000,
`mask_padded_action_dims_at_inference=false`.

### 100K baselines (this host, RTX 4090)

Launcher [`run_openpi_fullwidth_100k.sh`](../shell_scripts/run_openpi_fullwidth_100k.sh)
(ACT/SmolVLA/pi0.5, 100K steps, val/10K, save/20K), launched as three parallel
background drivers. Results and jitter-eval table:
[`openpi_fullwidth_100k_results.md`](./openpi_fullwidth_100k_results.md).

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_openpi_fullwidth_100k.sh act    > examples/umi_relative_ee/logs/act_openpi_fullwidth_100k_driver.log    2>&1 < /dev/null &
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_openpi_fullwidth_100k.sh smolvla > examples/umi_relative_ee/logs/openpi_fullwidth_smolvla_driver.log  2>&1 < /dev/null &
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_openpi_fullwidth_100k.sh pi05   > examples/umi_relative_ee/logs/openpi_fullwidth_pi05_driver.log    2>&1 < /dev/null &
```

### 1M runs (launched 2026-08-01 23:45)

Launchers: [`run_pi05_fullwidth_1m.sh`](../shell_scripts/run_pi05_fullwidth_1m.sh) (this host)
and [`run_smolvla_fullwidth_1m_kiwi.sh`](../shell_scripts/run_smolvla_fullwidth_1m_kiwi.sh)
(kiwi). Both use `steps=1000000`, `save_freq=100000`, `val_freq=50000`,
`scheduler_decay_steps=1000000`.

#### π0.5 LoRA 1M — this host (RTX 4090), driver PID 1838849

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_pi05_fullwidth_1m.sh \
  > examples/umi_relative_ee/logs/pi05_1m_driver.log 2>&1 < /dev/null &
```

Inside the script (training log `examples/umi_relative_ee/logs/pi05_openpi_fullwidth_1M.log`,
output `outputs/train/pi05_lora_openpi_fullwidth_1302_1M`):

```bash
env HF_HUB_OFFLINE=1 /home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=50000 \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.0001 --policy.scheduler_decay_lr=0.00001 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=1000000 \
  --policy.repo_id=zfff/pi05_lora_openpi_fullwidth_1302_1M --policy.push_to_hub=false \
  --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=16 \
  --batch_size=2 --num_workers=8 --prefetch_factor=2 \
  --seed=1000 --steps=1000000 --save_freq=100000 --log_freq=50 --eval_freq=0 \
  --output_dir=outputs/train/pi05_lora_openpi_fullwidth_1302_1M \
  --job_name=pi05_lora_openpi_fullwidth_1302_1M \
  --wandb.enable=true --wandb.project=lerobot
```

#### SmolVLA 1M — kiwi (RTX 5080), driver PID 1182893

kiwi alias: `kiwiz='ssh zfei@10.98.19.22 -p 2203'`. Code + datasets were
rsynced to kiwi first (worktree to `/home/zfei/code/lerobot-fei-v5.0-umi-unified`,
datasets to `/home/zfei/data/`).

```bash
ssh -o BatchMode=yes zfei@10.98.19.22 -p 2203 '
  setsid nohup bash /home/zfei/code/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/shell_scripts/run_smolvla_fullwidth_1m_kiwi.sh \
    > /home/zfei/code/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/logs/smolvla_1m_driver.log 2>&1 < /dev/null &
'
```

Inside the script (training log `.../examples/umi_relative_ee/logs/smolvla_openpi_fullwidth_1M.log`,
output `.../outputs/train/smolvla_openpi_fullwidth_1302_1M`):

```bash
env HF_HUB_OFFLINE=1 /home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/home/zfei/data/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/home/zfei/data/sroiv2_strawberry_picking_lab_validation \
  --val_freq=50000 \
  --policy.path=lerobot/smolvla_base --policy.input_features=null \
  --policy.use_umi_relative_ee=true \
  --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=1000000 --policy.scheduler_decay_lr=0.0000025 \
  --policy.repo_id=zfff/smolvla_openpi_fullwidth_1302_1M --policy.push_to_hub=false \
  --seed=1000 --steps=1000000 --save_freq=100000 --log_freq=200 --eval_freq=0 \
  --batch_size=8 --num_workers=4 \
  --output_dir=outputs/train/smolvla_openpi_fullwidth_1302_1M \
  --job_name=smolvla_openpi_fullwidth_1302_1M \
  --wandb.enable=true --wandb.project=lerobot
```

## Option B: masked-subspace flow (`flow_matching_padding_mode = "masked_subspace"`)

Option A (OpenPI full-width, above) is the default. Option B is the other
coherent formulation: the flow lives entirely on the real `d`-dimensional
action subspace, and the `D - d` padded coordinates are forced to zero
throughout training and inference. Switch per run with a policy flag:

```bash
--policy.flow_matching_padding_mode=openpi_full_width  # Option A (default)
--policy.flow_matching_padding_mode=masked_subspace    # Option B
```

The field lives on `SmolVLAConfig` and `PI05Config`. π0 keeps its legacy masked
default and is unaffected.

### Training equation (Option B)

```python
a_real = normalize(raw_action)          # [batch, chunk, 10]
a = pad_with_zeros(a_real, 32)          # [batch, chunk, 32]
epsilon = randn_like(a)
epsilon[..., 10:] = 0                   # zero noise in padded coords 10..31
t = Beta(1.5, 1.0) * 0.999 + 0.001
x_t = t * epsilon + (1.0 - t) * a       # padded x_t stays 0
u_t = epsilon - a                        # padded target velocity 0
v_t = model(observation, x_t, t)
loss_elementwise = (v_t - u_t).square()
loss = mean over real dims 0..9 only     # padded coordinates excluded
```

Only the real coordinates are optimized. Because padded noise, state, and
target are all zero, padded coordinates never move. Implemented as
`mask_padded_action_dims(noise, active_action_dim, enabled=True)` at the
training noise sample site, and `reduce_flow_matching_loss(...,
active_action_dim=d)` for the loss.

### Inference equation (Option B)

The masked integration that was the pre-Option-A default:
`integrate_flow_matching(..., mask_padded_dims=True)` zeros padded coordinates
in the initial latent, masks the denoiser and RTC velocity, and clamps the
Euler state after every step.

```python
x_t = randn(batch, chunk, 32)
x_t[..., 10:] = 0        # masked at init
for step in range(num_steps):
    v_t = model.denoise_step(observation, x_t, t)
    v_t[..., 10:] = 0    # masked velocity
    x_t = x_t + dt * v_t
    x_t[..., 10:] = 0    # clamped each step
robot_actions = x_t[..., :10]
```

### When to use which

- **Option A (default)** matches OpenPI/π checkpoints exactly: the model spends
  capacity learning to transport padded noise to zero, and stochastic sampling
  is the intended decode. Use unless you have a reason not to.
- **Option B** never spends capacity on padding and keeps the learned flow on
  the real action manifold — the cheaper targeted design for a fixed real action
  width. Compare empirically against A on the same data/seed.

Both are mathematically coherent. For SmolVLA and π0.5, the previous mixed
formulation (full-width noise + real-dim-only loss) no longer exists. π0 retains
its legacy behavior and is outside this switch. Existing mixed-formulation
SmolVLA and π0.5 checkpoints load under either mode but were not trained
coherently — retrain.

### 1M run (Option B) — SmolVLA on kiwi

Launcher [`run_smolvla_masked_subspace_1m_kiwi.sh`](../shell_scripts/run_smolvla_masked_subspace_1m_kiwi.sh),
identical to the Option A kiwi launcher except
`--policy.flow_matching_padding_mode=masked_subspace`; output
`outputs/train/smolvla_masked_subspace_1302_1M`, repo
`zfff/smolvla_masked_subspace_1302_1M`.

## What this does not do

This change does not add a trajectory-smoothness loss, correlated noise,
zero-noise decoding, rotation-specific loss, or different action attention.
Those are separate experiments. First train and deploy this coherent OpenPI
baseline so any remaining XYZ or rotation jitter is not caused by mismatched
padding behavior.

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# UMI flow matching 的 padded-noise 策略

SmolVLA 与 π0.5 在本仓库中支持两种自洽的 padded 坐标策略：兼容 OpenPI 的全宽 flow（选项 A，默认）与 masked-subspace flow（选项 B）。本页定义这两种策略，并解释它们如何影响旧 checkpoint、训练与推理。

实现遵循本地 OpenPI 检出中 commit `03893832f2002a31016586874658662c669081ac` 的 flow head。

## 适用范围

对于 UMI 相对 EE 数据：

- 真实动作有 10 个值：XYZ、rotation-6D 与 gripper；
- 模型动作宽度仍为 32；
- 归一化是策略特定的，且保持不变；
- 归一化动作从 10 零填充到 32；
- SmolVLA 的 causal action attention 不变；
- π0 不变。

固定宽度坐标 10 到 31 不是机器人命令。在默认的选项 A 策略下，它们仍是学习到的 flow 的一部分，并被监督以零结束。

## 训练方程

设 `a_real` 为归一化后的真实动作 chunk，`a` 为其零填充的 32D 表示：

```python
a_real = normalize(raw_action)        # [batch, chunk, 10]
a = pad_with_zeros(a_real, 32)        # [batch, chunk, 32]
epsilon = torch.randn_like(a)         # Gaussian noise in every dimension
t = Beta(1.5, 1.0) * 0.999 + 0.001
x_t = t * epsilon + (1.0 - t) * a
u_t = epsilon - a
v_t = model(observation, x_t, t)
loss_elementwise = (v_t - u_t).square()
```

优化损失对所有 32 个坐标取平均。因此，填充坐标学到有效的传输 `epsilon -> 0`。它们不得从损失中丢弃。

`action_is_pad` 含义不同：它标记回合边界处无效的 chunk 时间步。它会从损失中移除完整时间步，但从不移除动作坐标。

训练报告：

- `loss`：在有效时间步和全部 32 个维度上的优化损失；
- `loss_per_dim`：32 个单独的坐标损失；
- `flow_loss_real_dims`：真实机器人坐标上的均值；
- `flow_loss_padded_dims`：坐标 10 到 31 上的均值。

## 推理方程

推理从全宽高斯噪声开始，并从 `t=1` 的噪声积分到 `t=0` 的动作：

```python
x_t = torch.randn(batch, chunk, 32)
dt = -1.0 / num_steps

for step in range(num_steps):
    t = 1.0 + step * dt
    v_t = model.denoise_step(observation, x_t, t)
    x_t = x_t + dt * v_t

robot_actions = x_t[..., :10]
```

默认是 10 个 Euler 步。噪声、预测速度、RTC 输出与 Euler 状态从不由动作维度进行钳制或掩码。裁剪只发生在完整的 32D 积分之后。

`mask_padded_action_dims_at_inference` 仍保留在序列化 config 中，以便旧 checkpoint 加载，但 SmolVLA 与 π0.5 会忽略它。其默认值现在是 `false`。启用全宽推理不需要任何 CLI flag。

## 现有 checkpoint

由先前混合公式训练的 checkpoint 看到 32D 高斯噪声，但只优化真实 10 个维度。将其推理切换为全宽会使推理与 OpenPI 一致，但无法追溯地教会其填充输出流向零。

重新训练 SmolVLA 与 π0.5，以获得端到端一致的 checkpoint。不要对重新训练的 checkpoint 使用 masked 推理。不要将零噪声解码作为部署默认；它仍是用于诊断随机敏感性的手段。

现有归一化统计与 UMI processor 会被复用。仅在数据集或动作表示变化时重新计算统计。

## 训练与部署

使用常规 UMI 训练入口：

- SmolVLA：[smolvla_relative_ee_training.md](./smolvla_relative_ee_training.md)
- π0.5：[README.md](./README.md#training-baseline) 与
  [train_pi05_lora.sh](../shell_scripts/train_pi05_lora.sh)

运行命令前激活现有环境：

```bash
conda activate py312
```

新 checkpoint 会序列化 `mask_padded_action_dims_at_inference=false`。异步与同步部署都使用相同的 policy sampler，因此 server/client 设置不需要特殊的 flow-matching 选项。

## 启动日志（2026-08-01）

以下所有运行都使用上述全宽契约，数据集 `sroiv2_strawberry_picking_lab_1302_occlusion`（训练）与 `sroiv2_strawberry_picking_lab_validation`（验证），chunk 30，seed 1000，`mask_padded_action_dims_at_inference=false`。

### 100K 基线（本机，RTX 4090）

启动器 [`run_openpi_fullwidth_100k.sh`](../shell_scripts/run_openpi_fullwidth_100k.sh)（ACT/SmolVLA/pi0.5，100K 步，val/10K，save/20K），以三个并行后台驱动方式启动。结果与 jitter-eval 表格：[`openpi_fullwidth_100k_results.md`](./openpi_fullwidth_100k_results.md)。

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_openpi_fullwidth_100k.sh act    > examples/umi_relative_ee/logs/act_openpi_fullwidth_100k_driver.log    2>&1 < /dev/null &
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_openpi_fullwidth_100k.sh smolvla > examples/umi_relative_ee/logs/openpi_fullwidth_smolvla_driver.log  2>&1 < /dev/null &
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_openpi_fullwidth_100k.sh pi05   > examples/umi_relative_ee/logs/openpi_fullwidth_pi05_driver.log    2>&1 < /dev/null &
```

### 1M 运行（2026-08-01 23:45 启动）

启动器：[`run_pi05_fullwidth_1m.sh`](../shell_scripts/run_pi05_fullwidth_1m.sh)（本机）与 [`run_smolvla_fullwidth_1m_kiwi.sh`](../shell_scripts/run_smolvla_fullwidth_1m_kiwi.sh)（kiwi）。二者都使用 `steps=1000000`、`save_freq=100000`、`val_freq=50000`、`scheduler_decay_steps=1000000`。

#### π0.5 LoRA 1M — 本机（RTX 4090），驱动 PID 1838849

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_pi05_fullwidth_1m.sh \
  > examples/umi_relative_ee/logs/pi05_1m_driver.log 2>&1 < /dev/null &
```

脚本内部（训练日志 `examples/umi_relative_ee/logs/pi05_openpi_fullwidth_1M.log`，输出 `outputs/train/pi05_lora_openpi_fullwidth_1302_1M`）：

```bash
env HF_HUB_OFFLINE=1 /home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/train_pi05_lora.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=50000 \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_umi_relative_ee=true \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.optimizer_lr=0.0001 --policy.scheduler_decay_lr=0.00001 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=1000000 \
  --policy.repo_id=zfff/pi05_lora_openpi_fullwidth_1302_1M --policy.push_to_hub=false \
  --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=16 \
  --batch_size=2 --num_workers=8 --prefetch_factor=2 \
  --seed=1000 --steps=1000000 --save_freq=100000 --log_freq=50 --eval_freq=0 \
  --output_dir=outputs/train/pi05_lora_openpi_fullwidth_1302_1M \
  --job_name=pi05_lora_openpi_fullwidth_1302_1M \
  --wandb.enable=true --wandb.project=lerobot
```

#### SmolVLA 1M — kiwi（RTX 5080），驱动 PID 1182893

kiwi 别名：`kiwiz='ssh zfei@10.98.19.22 -p 2203'`。代码 + 数据集先 rsync 到 kiwi（worktree 到 `/home/zfei/code/lerobot-fei-v5.0-umi-unified`，数据集到 `/home/zfei/data/`）。

```bash
ssh -o BatchMode=yes zfei@10.98.19.22 -p 2203 '
  setsid nohup bash /home/zfei/code/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/shell_scripts/run_smolvla_fullwidth_1m_kiwi.sh \
    > /home/zfei/code/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/logs/smolvla_1m_driver.log 2>&1 < /dev/null &
'
```

脚本内部（训练日志 `.../examples/umi_relative_ee/logs/smolvla_openpi_fullwidth_1M.log`，输出 `.../outputs/train/smolvla_openpi_fullwidth_1302_1M`）：

```bash
env HF_HUB_OFFLINE=1 /home/zfei/code/lerobot-fei-v5.0-umi-unified/.venv/bin/python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/home/zfei/data/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/home/zfei/data/sroiv2_strawberry_picking_lab_validation \
  --val_freq=50000 \
  --policy.path=lerobot/smolvla_base --policy.input_features=null \
  --policy.use_umi_relative_ee=true \
  --policy.device=cuda --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=1000000 --policy.scheduler_decay_lr=0.0000025 \
  --policy.repo_id=zfff/smolvla_openpi_fullwidth_1302_1M --policy.push_to_hub=false \
  --seed=1000 --steps=1000000 --save_freq=100000 --log_freq=200 --eval_freq=0 \
  --batch_size=8 --num_workers=4 \
  --output_dir=outputs/train/smolvla_openpi_fullwidth_1302_1M \
  --job_name=smolvla_openpi_fullwidth_1302_1M \
  --wandb.enable=true --wandb.project=lerobot
```

## 选项 B：masked-subspace flow（`flow_matching_padding_mode = "masked_subspace"`）

选项 A（OpenPI 全宽，见上）是默认。选项 B 是另一种自洽公式：flow 完全存在于真实的 `d` 维动作子空间上，`D - d` 个填充坐标在训练和推理全程被强制为零。按运行通过 policy flag 切换：

```bash
--policy.flow_matching_padding_mode=openpi_full_width  # Option A (default)
--policy.flow_matching_padding_mode=masked_subspace    # Option B
```

该字段位于 `SmolVLAConfig` 与 `PI05Config` 上。π0 保留其旧式 masked 默认值，不受影响。

### 训练方程（选项 B）

```python
a_real = normalize(raw_action)          # [batch, chunk, 10]
a = pad_with_zeros(a_real, 32)          # [batch, chunk, 32]
epsilon = randn_like(a)
epsilon[..., 10:] = 0                   # zero noise in padded coords 10..31
t = Beta(1.5, 1.0) * 0.999 + 0.001
x_t = t * epsilon + (1.0 - t) * a       # padded x_t stays 0
u_t = epsilon - a                        # padded target velocity 0
v_t = model(observation, x_t, t)
loss_elementwise = (v_t - u_t).square()
loss = mean over real dims 0..9 only     # padded coordinates excluded
```

只优化真实坐标。由于填充噪声、state 与目标都为零，填充坐标从不移动。实现为训练噪声采样处的 `mask_padded_action_dims(noise, active_action_dim, enabled=True)`，以及损失处的 `reduce_flow_matching_loss(..., active_action_dim=d)`。

### 推理方程（选项 B）

选项 A 之前默认的 masked 积分：`integrate_flow_matching(..., mask_padded_dims=True)` 在初始 latent 中将填充坐标置零，对 denoiser 与 RTC 速度做掩码，并在每一步后钳制 Euler 状态。

```python
x_t = randn(batch, chunk, 32)
x_t[..., 10:] = 0        # masked at init
for step in range(num_steps):
    v_t = model.denoise_step(observation, x_t, t)
    v_t[..., 10:] = 0    # masked velocity
    x_t = x_t + dt * v_t
    x_t[..., 10:] = 0    # clamped each step
robot_actions = x_t[..., :10]
```

### 何时使用哪种

- **选项 A（默认）** 与 OpenPI/π checkpoint 完全匹配：模型花费容量学习将填充噪声传输到零，随机采样是预期的解码方式。除非有理由不用，否则使用它。
- **选项 B** 从不在填充上花费容量，并将学习到的 flow 保持在真实动作流形上——对于固定真实动作宽度而言是更省成本的有针对性设计。在同一数据/seed 上与 A 做经验比较。

二者在数学上都是自洽的。对于 SmolVLA 与 π0.5，先前的混合公式（全宽噪声 + 仅真实维度损失）已不存在。π0 保留其旧式行为，不在此开关范围内。现有的混合公式 SmolVLA 与 π0.5 checkpoint 在任一模式下都能加载，但训练并不自洽——请重新训练。

### 1M 运行（选项 B）— kiwi 上的 SmolVLA

启动器 [`run_smolvla_masked_subspace_1m_kiwi.sh`](../shell_scripts/run_smolvla_masked_subspace_1m_kiwi.sh)，与选项 A 的 kiwi 启动器完全相同，只是使用 `--policy.flow_matching_padding_mode=masked_subspace`；输出 `outputs/train/smolvla_masked_subspace_1302_1M`，repo `zfff/smolvla_masked_subspace_1302_1M`。

## 本改动不包含什么

本改动不添加轨迹平滑损失、相关噪声、零噪声解码、旋转专用损失或不同的 action attention。这些是单独的实验。先训练并部署这个自洽的 OpenPI 基线，以便任何剩余的 XYZ 或旋转抖动都不是由填充行为不匹配造成的。

