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
