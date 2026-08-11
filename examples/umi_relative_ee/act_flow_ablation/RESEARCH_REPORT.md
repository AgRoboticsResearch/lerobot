# ACT capacity and flow-objective investigation

**Status:** in progress  
**Started:** 2026-08-11  
**Branch:** `research/umi-act-flowmatching-ablation-20260811`  
**Source baseline:** `3feb3f3e`  
**Full-run artifacts:** `/media/zfei/Glowat512/projects/lerobot-arch-exp`  
**Smoke artifacts:** `/mnt/data1/sroi/lerobot_policy_ablation_20260811/smoke`

This is the single research record for the investigation. It intentionally
contains the question decomposition, evidence, experimental design, code
changes, exact controls, execution incidents, results, interpretation,
limitations, and lessons. Raw multi-gigabyte checkpoints remain outside Git;
all code, compact metrics, and conclusions belong on the research branch.

## 1. Questions and decision criteria

### Q1: Can the 1459 ACT be improved by scaling the backbone or architecture?

The useful answer is not merely whether a larger model gets a lower training
loss. An improvement must appear on held-out decoded physical trajectories and
must justify its memory, latency, and training cost. The staged test is:

1. reproduce the existing ResNet-18 ACT at an early common budget;
2. change only ResNet-18 → ResNet-34 → ResNet-50;
3. test a wider/deeper transformer only if backbone scaling is promising;
4. promote promising candidates to longer training and multiple seeds;
5. compare decoded xyz, rotation, gripper, and within-chunk jerk, not only ACT
   validation L1.

### Q2: Is weaker flow-policy behavior caused by flow matching itself?

The existing ACT, SmolVLA, and π0.5 runs confound at least five variables:
action objective, vision encoder, language/VLM backbone, pretrained weights,
trainable parameter subset, and optimizer. Comparing their raw validation
losses is invalid because ACT logs L1+KL while flow policies log velocity MSE.

The decisive control changes only the objective:

- `act_r18_l1`: ACT inference architecture, no VAE, direct L1 regression;
- `act_r18_flow_*`: the same ResNet-18, state/image encoder, transformer
  encoder/decoder, data, and chunk geometry, but rectified-flow velocity
  regression with noisy action and time inputs.

The VAE is removed from both because the 1459 ACT's KL has collapsed to nearly
zero, and the VAE encoder is absent at inference. This makes the L1/flow pair
much closer in trainable and inference architecture than ACT-VAE versus a VLM.
The repository's conventional ResNet + temporal U-Net Diffusion Policy is a
second non-VLM generative control. Together these distinguish:

- flow loses in ACT-flow and Diffusion Policy → objective/data representation
  is a plausible bottleneck;
- ACT-flow matches ACT-L1 but VLAs lag → VLM/fine-tuning path is the likely
  bottleneck;
- ACT-flow works but Diffusion Policy loses → denoiser architecture or
  optimization matters more than the generic objective;
- both non-VLM generative controls work → flow/diffusion itself is not the
  explanation.

## 2. Literature evidence used in the design

- [ACT](https://arxiv.org/abs/2304.13705) motivates action-chunk prediction as
  a way to reduce compounding error and model non-stationary demonstrations.
  The local implementation uses a ResNet visual encoder and a transformer over
  image/state conditioning.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) reports strong robot
  performance from conditional action diffusion, including multimodal action
  distributions, receding-horizon control, and visual conditioning. Therefore
  a weak local VLA is not by itself evidence that iterative generative action
  objectives are intrinsically inferior.
- [Flow Matching](https://arxiv.org/abs/2210.02747) defines simulation-free
  vector-field regression over conditional probability paths and reports that
  optimal-transport displacement paths can train and sample efficiently. The
  ACT-flow control uses the simplest straight path between action data and
  Gaussian noise.
- [OpenPI's reference implementation](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)
  transports Gaussian noise to actions by Euler integration of a learned
  velocity field. The local shared `integrate_flow_matching` implementation
  uses the same time direction: training interpolates
  `x_t = t * noise + (1-t) * action`, predicts `noise-action`, and inference
  integrates from `t=1` to `t=0` with negative steps.

These papers establish plausibility and implementation conventions; none can
answer the dataset-specific questions without controlled local experiments.

## 3. Audit of the existing 1459 ACT

Recovered from
`outputs/train/act_umi_identity_rot6d_1459/checkpoints/0800000/pretrained_model/train_config.json`
and the two console logs:

| Property | Existing 1459 ACT |
| --- | --- |
| train data | 1459 episodes / 140,522 frames |
| validation | 100 episodes / 9,274 frames |
| representation | UMI relative EE, 10D rot6d action, derived 20D state |
| normalization | MIN_MAX, rot6d coordinates forced to identity stats |
| chunk / executed | 30 / 30 |
| backbone | ImageNet ResNet-18 |
| transformer | width 512, 8 heads, FFN 3200, encoder 4, decoder 1 |
| VAE | latent 32, encoder 4, KL weight 10 |
| optimizer | AdamW, LR 1e-5, weight decay 1e-4 |
| batch / seed | 8 / 1000 |
| parameters | 51,579,786 (52M) |
| total training | 3,000,000 steps |

Parsing all 300 logged validation points gives a lowest stochastic validation
loss of **0.032595 at 2.48M** (L1 0.032592, KL effectively zero). At the common
early budgets it recorded 0.054203 at 10k, 0.043292 at 20k, 0.039702 at 30k,
0.036201 at 50k, and 0.036900 at 100k. The best saved-checkpoint behavior is
not the same as minimum validation loss.

The pre-existing all-validation open-loop audit evaluated 500 fixed query
frames per checkpoint. The 1459 ACT improves slowly from 25.1 mm xyz endpoint
error at 100k to about 22.9 mm at 2.3M; rotation endpoint improves from 4.91°
to about 4.40°. Rotation jerk is best around 700k (0.036°), then becomes a bit
worse with more fitting. Thus longer training buys modest endpoint accuracy
and can trade away smoothness.

An important correction to the question's premise: flow policies have not been
uniformly worse offline. In the existing report, π0.5 reaches 21.7–22.0 mm xyz
endpoint error at 100k, slightly better than ACT, while SmolVLA is much worse
at 35.6–35.9 mm and far jerkier. This mixed result is exactly why objective,
architecture, and VLM adaptation must be isolated. It does not overrule any
observed closed-loop deployment gap.

## 4. Implementation added for the controlled tests

### 4.1 Architecture-matched ACT flow

`ACTConfig.action_objective` now selects canonical `l1` (default, backward
compatible) or `flow_matching`. Flow mode requires `use_vae=false` and adds:

- a linear projection of the noisy 10D action at each of the 30 decoder tokens;
- a continuous sine/cosine time embedding and two-layer time MLP;
- masked velocity MSE for training;
- fixed-step Euler integration through the shared flow helper for inference;
- configurable uniform/Beta time sampling and inference steps.

Everything upstream of decoder inputs—ResNet, image tokens, state token,
transformer encoder, decoder blocks, normalization, and UMI processors—is the
same as no-VAE ACT-L1. Uniform time is the vanilla default. A Beta(1.5, 1.0)
variant mirrors the time bias used by local OpenPI-style VLA configs.

### 4.2 Conventional non-VLM Diffusion Policy

The existing Diffusion Policy now accepts the canonical UMI processors and
representation. Because its U-Net requires a horizon divisible by its temporal
downsampling factor, it trains at internal horizon 32 and returns/executes the
first 30 actions. It uses one current observation, canonical derived 20D state,
10D relative actions, padding-aware loss, and direct offline chunk inference.
The planned control uses ResNet-18, a `(256,512,1024)` 1D U-Net, 100 DDIM
training timesteps, and 10 inference steps.

### 4.3 Common evaluator

`eval_open_loop_dataset.py` now supports Diffusion Policy and correctly applies
runtime inference-step overrides to the nested diffusion model. All objectives
will be compared only after postprocessing back to absolute 7D physical poses.

## 5. Experiment matrix

All full experiments use the same 1459 train set, 100-episode validation set,
PyAV decoder, no image augmentation, ImageNet image statistics, identity rot6d
normalization, chunk 30, batch 8, seed 1000, and host RTX 4090. The first stage
uses a common optimizer-step budget and fixed evaluation queries. The flow LR
sweep is deliberate: equal LR isolates the objective, while a tuned LR avoids
mistaking an ACT-specific optimizer for an intrinsic flow failure.

| Variant | Purpose | Parameters | Status |
| --- | --- | ---: | --- |
| `act_r18_vae` | exact 1459 early-budget replication | 52M | pending |
| `act_r34_vae` | backbone-only scale | TBD | pending |
| `act_r50_vae` | backbone-only scale | TBD | pending |
| `act_r50_large` | ResNet-50 + 768-wide, 6e/3d transformer | 145M | smoke passed |
| `act_r18_l1` | no-VAE deterministic objective control | TBD | pending |
| `act_r18_flow_u_lr1e5` | exact-LR, uniform-time flow control | 35M | smoke passed |
| `act_r18_flow_u_lr1e4` | flow optimizer sensitivity | 35M | pending |
| `act_r18_flow_beta_lr1e4` | OpenPI-like time bias | 35M | pending |
| `diffusion_r18` | standard non-VLM diffusion control | 75M | smoke passed |

Promotion rules will be based on confidence intervals over decoded physical
metrics, not the lowest model-specific validation loss. At least the leading
ACT-capacity model, ACT-L1, and leading ACT-flow configuration should receive
multiple seeds before a final claim.

Stage-one evaluation uses five evenly spaced valid query frames in every one of
the 100 validation episodes. Reports use episode-balanced means and deterministic
95% nonparametric bootstrap intervals (10,000 resamples, seed 0), with episodes
as the resampling unit. ACT-flow and Diffusion Policy are additionally evaluated
with inference seeds 1000, 2000, and 3000 to expose sampling variance. Training
seeds are varied only after this screen selects configurations worth promoting.

## 6. Smoke experiments and resource observations

All training smoke tests ran on the host GPU, not in the sandbox, against the
real 1459 dataset. The workstation has one RTX 4090 (24,564 MiB). The original
source filesystem was 95% full, so new full artifacts use the external project
directory. At inspection time the external mount had 345 GB free.

| Run | Result | Parameters | Cold first step | Notes |
| --- | --- | ---: | ---: | --- |
| ACT-flow R18 | passed 2 updates + checkpoint | 34,728,266 | 4.65 s | canonical 31 raw poses became 30 relative targets |
| Diffusion R18 | passed 2 updates + checkpoint | 75,396,650 | 10.56 s | 33 raw poses became internal horizon 32 |
| ACT R50 large | passed 2 updates + checkpoint | 144,946,762 | 10.61 s | batch 8 fits; downloaded official ResNet-50 V2 weights |

Cold-step numbers include worker/video/model warm-up and are not steady-state
throughput. Dedicated timed runs are required before latency conclusions.

## 7. Validation and test evidence so far

- Ruff and whitespace checks pass on all changed files.
- 46 targeted CPU tests pass, 9 hardware/optional tests skip. These cover the
  new ACT-flow training/inference path, shared flow integrator, legacy ACT VAE
  behavior, ACT/Diffusion processors, and canonical UMI processor behavior.
- ACT-flow produces finite differentiable loss, gradients in its noisy-action
  projection, deterministic outputs under fixed input noise, and rejects bad
  noise shapes.
- Diffusion's UMI config produces `[-1, 0, ..., 31]`, strips the leading action
  into the two-pose derived state, and reconnects the same relative-action step
  to postprocessing.
- One-sample host-GPU checkpoint reload and physical-pose decoding passed for
  both ACT-flow and Diffusion Policy. Their very large errors after only two
  optimizer updates are expected and are not performance evidence.

## 8. Execution incidents and lessons already learned

1. The repository is a linked worktree whose Git metadata is outside the
   workspace write sandbox. The baseline commit required explicit host access.
   The original branch was pushed before creating the research branch, as
   requested.
2. `uv` could not create cache files under the sandboxed home cache. Lightweight
   sandbox checks use a task-specific cache under `/tmp`; GPU work uses host
   execution.
3. The globally installed ROS pytest plugins caused an unrelated module-level
   `deepdiff` skip to abort targeted collection. Disabling plugin autoload made
   the intended tests collect normally. This was a test-runner environment
   issue, not a policy failure.
4. `diffusers` was not installed. Running `uv sync --extra diffusion` alone
   removed previously selected optional dataset/training packages, because sync
   reconciles the complete requested environment. The environment was corrected
   with all three locked extras together: `training`, `dataset`, and
   `diffusion`. Future setup must request the union of required extras.
5. Installing the dataset extra made TorchCodec available and silently changed
   the default video backend from the original run's PyAV. The launcher now
   pins PyAV explicitly so decode backend does not become an experimental
   confound. The first ACT-flow smoke used TorchCodec before this was noticed;
   no scientific metric will be taken from that smoke.
6. The common evaluator had the same dynamic-backend issue: its first one-query
   ACT-flow inference attempt selected an incompatible TorchCodec installation
   and failed before model inference. The evaluator now defaults to and records
   PyAV, matching training and the baseline. This also makes failed evaluations
   retryable without leaving a directory that looks complete.
7. Raw validation losses cannot compare objectives. The existing report already
   demonstrates that validation-loss-best and decoded-metric-best disagree for
   every evaluated model. The experiment protocol therefore fixes decoded
   metrics as the cross-model endpoint.

## 9. Results

The controlled sweep is in progress. The exact ResNet-18 ACT replication gives
the first scientific sanity check:

| Run | 10k validation loss | Difference from historical |
| --- | ---: | ---: |
| historical 1459 ACT | 0.054203 | — |
| controlled `act_r18_vae`, seed 1000 | 0.054130 | -0.000073 (-0.13%) |

This close agreement makes major dataset, representation, decoder, or software
drift unlikely at the screening scale. It does not establish reproducibility of
decoded metrics until the 30k checkpoint is evaluated.

The final version of this section will contain run hashes, parameter counts,
throughput, peak memory, per-step validation curves, decoded metrics, seed means
or bootstrap confidence intervals, and explicit answers to Q1 and Q2. Smoke
success remains feasibility evidence only.

## 10. Reproduction

The variant launcher is `run_one.sh` in this directory. `run_stage1.sh` executes
the fixed matrix sequentially so models do not contend for the single GPU, and
`evaluate_one.sh` resolves exactly one final checkpoint and evaluates it without
manual path selection, while `evaluate_stage1.sh` fixes the deterministic and
three-seed generative matrix. `collect_results.py` extracts parameter counts, wall
times, complete validation curves, decoded metrics, and confidence intervals
into compact external CSV/JSON files without creating a second narrative doc.
Example:

```bash
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_vae 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_flow_u_lr1e4 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh diffusion_r18 30000 1000
```

The launcher refuses to overwrite an existing run. Full command-line configs
are also saved inside every checkpoint's `train_config.json`, and stdout/stderr
is retained in the artifact workspace.
