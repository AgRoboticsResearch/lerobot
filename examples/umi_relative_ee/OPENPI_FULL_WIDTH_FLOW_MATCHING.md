# OpenPI full-width flow matching for UMI

SmolVLA and π0.5 use the same OpenPI-compatible flow-matching contract in this
repository. This page defines that contract and explains how it affects old
checkpoints, training, and inference.

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

The fixed-width coordinates 10 through 31 are not robot commands. They are
still part of the learned flow and are supervised to end at zero.

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
  [train_pi05_lora.sh](./train_pi05_lora.sh)

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

Launcher [`run_openpi_fullwidth_100k.sh`](./run_openpi_fullwidth_100k.sh)
(ACT/SmolVLA/pi0.5, 100K steps, val/10K, save/20K), launched as three parallel
background drivers. Results and jitter-eval table:
[`openpi_fullwidth_100k_results.md`](./openpi_fullwidth_100k_results.md).

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
setsid nohup bash examples/umi_relative_ee/run_openpi_fullwidth_100k.sh act    > examples/umi_relative_ee/logs/act_openpi_fullwidth_100k_driver.log    2>&1 < /dev/null &
setsid nohup bash examples/umi_relative_ee/run_openpi_fullwidth_100k.sh smolvla > examples/umi_relative_ee/logs/openpi_fullwidth_smolvla_driver.log  2>&1 < /dev/null &
setsid nohup bash examples/umi_relative_ee/run_openpi_fullwidth_100k.sh pi05   > examples/umi_relative_ee/logs/openpi_fullwidth_pi05_driver.log    2>&1 < /dev/null &
```

### 1M runs (launched 2026-08-01 23:45)

Launchers: [`run_pi05_fullwidth_1m.sh`](./run_pi05_fullwidth_1m.sh) (this host)
and [`run_smolvla_fullwidth_1m_kiwi.sh`](./run_smolvla_fullwidth_1m_kiwi.sh)
(kiwi). Both use `steps=1000000`, `save_freq=100000`, `val_freq=50000`,
`scheduler_decay_steps=1000000`.

#### π0.5 LoRA 1M — this host (RTX 4090), driver PID 1838849

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
setsid nohup bash examples/umi_relative_ee/run_pi05_fullwidth_1m.sh \
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
  setsid nohup bash /home/zfei/code/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/run_smolvla_fullwidth_1m_kiwi.sh \
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

## What this does not do

This change does not add a trajectory-smoothness loss, correlated noise,
zero-noise decoding, rotation-specific loss, or different action attention.
Those are separate experiments. First train and deploy this coherent OpenPI
baseline so any remaining XYZ or rotation jitter is not caused by mismatched
padding behavior.

