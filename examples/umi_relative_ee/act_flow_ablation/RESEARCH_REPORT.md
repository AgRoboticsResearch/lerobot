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
2. scale ResNet-18 → ResNet-34 → ResNet-50 and explicitly control the
   torchvision ImageNet-V1/V2 initialization choice;
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
- `act_r18_diffusion_lr1e5`: exactly the same learned time-conditioned ACT
  architecture as the matched flow model, but epsilon prediction with the
  standard squared-cosine DDIM schedule.

The VAE is removed from both because the 1459 ACT's KL has collapsed to nearly
zero, and the VAE encoder is absent at inference. This makes the L1/flow pair
much closer in trainable and inference architecture than ACT-VAE versus a VLM.
The repository's conventional ResNet + temporal U-Net Diffusion Policy is a
second non-VLM generative control. The exact ACT-flow/ACT-DP pair is needed
because a U-Net comparison still changes denoiser architecture. Together these
distinguish:

- flow loses in ACT-flow and Diffusion Policy → objective/data representation
  is a plausible bottleneck;
- ACT-flow matches ACT-L1 but VLAs lag → VLM/fine-tuning path is the likely
  bottleneck;
- ACT-flow works but Diffusion Policy loses → denoiser architecture or
  optimization matters more than the generic objective;
- both non-VLM generative controls work → flow/diffusion itself is not the
  explanation.
- ACT-DP beats ACT-flow at fixed learned architecture → path, target, or sampler
  is the likely cause; both lose similarly → shared iterative conditioning or
  optimization is more suspect than flow matching specifically.

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
same as no-VAE ACT-L1, as are the learned decoder positional queries and final
action head. Flow necessarily replaces the L1 decoder's zero content vectors
with projected noisy actions plus time embeddings; a structural regression test
proves that these six projection/MLP parameters are the only flow-only learned
parameter tensors and that every shared tensor name and shape matches ACT-L1.
Uniform time is the vanilla default. A Beta(1.5, 1.0) variant mirrors the time
bias used by local OpenPI-style VLA configs.

### 4.2 Architecture-identical ACT epsilon diffusion

The matched flow result alone cannot separate a weakness of straight-path flow
from a weakness of the time-conditioned ACT transformer. A new explicit
`diffusion` ACT objective therefore reuses the exact noisy-action projection,
continuous sinusoidal time embedding, two-layer time MLP, ResNet, observation
encoder, ACT decoder, positional embeddings, and action head used by
`flow_matching`. The two policies have exactly the same learned parameter names
and tensor shapes. Only the training target/path and sampler change:

- ACT-flow regresses `noise - action` along the linear interpolation and uses
  ten fixed Euler steps from noise to action;
- ACT-DP samples one of 100 squared-cosine diffusion timesteps, regresses
  epsilon, and uses a clipped ten-step DDIM sampler.

Both use batch 8, AdamW LR 1e-5 (including the backbone), identical UMI
processors, action-padding mask, training seeds, and fixed evaluation queries.
This is the strictest practical answer to “vanilla DP with the same architecture
as ACT.” Scheduler state has no learned parameters. A structural test proves
exact learned-architecture equality between ACT-flow and ACT-DP; finite-gradient,
fixed-noise determinism, scheduler-recipe, and bad-shape tests also pass.

### 4.3 Conventional non-VLM Diffusion Policy

The existing Diffusion Policy now accepts the canonical UMI processors and
representation. Because its U-Net requires a horizon divisible by its temporal
downsampling factor, it trains at internal horizon 32 and returns/executes the
first 30 actions. It uses one current observation, canonical derived 20D state,
10D relative actions, padding-aware loss, and direct offline chunk inference.
The planned control uses ResNet-18, a `(256,512,1024)` 1D U-Net, 100 DDIM
training timesteps, and 10 inference steps.

### 4.4 Common evaluator

`eval_open_loop_dataset.py` now supports Diffusion Policy and correctly applies
runtime inference-step overrides to the nested diffusion model. Its override
resolver now also distinguishes ACT-flow's `flow_num_inference_steps` from
ACT-DP's `diffusion_num_inference_steps`; this prevents a requested ACT sampler
override from creating an unused generic attribute. The controlled launchers
use each checkpoint's saved ten-step default, but the explicit path is covered
for reproducible sensitivity studies. All objectives will be compared only
after postprocessing back to absolute 7D physical poses.

### 4.5 Official UMI diffusion architecture ports

The upstream checkout at `/home/zfei/code/universal_manipulation_interface`
was audited at commit `d095ba9590df789df5189eea5ee7e431689038a6`. The documented
single-GPU command uses `train_diffusion_unet_timm_umi_workspace`, not the
transformer-denoiser configuration. Two separately registered policies were
therefore added:

- `umi_official_dp` ports the documented CLIP ViT-B/16 observation encoder,
  CLS-token global conditioning, `(256,512,1024)` conditional 1D U-Net,
  FiLM scale modulation, DDIM with 50 train/16 inference steps, epsilon
  prediction, 0.1 input perturbation, AdamW 3e-4, 2k-step cosine warmup, and
  the released EMA warmup (`power=0.75`, cap 0.9999). Its training-only image
  transforms are 95% random crop/resize and the exact color-jitter settings in
  the released YAML.
- `umi_official_transformer_dp` ports the companion
  `train_diffusion_transformer_umi_workspace`: all 197 ViT tokens are retained,
  the low-dimensional state becomes a learned token, and a 7-layer, 768-wide,
  8-head pre-norm transformer decoder denoises the action sequence. It adds the
  released ±5° rotation augmentation and uses a 3e-5 backbone LR versus 3e-4
  for the denoiser/observation projections.

The CLS detail follows executed upstream code rather than a literal reading of
one YAML field: the U-Net YAML requests `feature_aggregation: attention_pool_2d`,
but `TimmObsEncoder` warns that aggregation is ignored for
ViTs, resets it to `None`, and returns token 0. Likewise, its
`use_group_norm: True` conversion is guarded by `not pretrained`, so it does
not alter the pretrained CLIP ViT used here. These apparent YAML/code
discrepancies therefore do not require extra layers in the port.

The scheduler and EMA code paths were checked separately. The installed
`DDIMScheduler` defaults are `set_alpha_to_one=True` and `steps_offset=0`,
matching the values written explicitly in the upstream YAML. The EMA decay
equation, first-update behavior, parameter update, and optimization-step
increment match upstream. The port additionally copies buffers on every EMA
update; that is inert for this pretrained ViT plus GroupNorm U-Net because
there are no BatchNorm running statistics to average.

Training/validation mode handling was traced through the actual LeRobot loop.
Each optimizer and scheduler step completes before `policy.update()` advances
EMA. Training loss uses online weights; deterministic held-out validation calls
`policy.eval()`, which selects `ema_diffusion`, and forks/reset RNG to seed 0
for reproducible diffusion timesteps and noise. Thus validation and deployment
both measure the averaged copy rather than accidentally mixing online and EMA
models.

Both policies keep independent online and EMA copies in checkpoints; training
updates only online weights and validation/inference use EMA weights. This is
implemented behind new policy types and a new `umi-official-dp` dependency
extra, so ACT, matched ACT-flow, and ordinary `diffusion` defaults and classes
remain unchanged.

Normalization was traced through both dataset implementations. Upstream range
normalizes translation and gripper to `[-1,1]` while assigning identity
scale/offset to rotation-6D for actions and low-dimensional observations. The
shared LeRobot processor applies MIN_MAX to translation/gripper and forces
identity statistics on action rot6d plus both rot6d blocks in the derived 20D
state. This preserves the released geometry instead of independently stretching
near-constant rotation-matrix coordinates.

This is an architecture/recipe port, not a claim of bit-for-bit reproduction.
The necessary dataset/control adaptations are material and will be carried into
every interpretation:

| Released UMI task | This controlled comparison | Reason |
| --- | --- | --- |
| two observations, stride 3 at about 60 Hz | one current image + canonical derived 20D two-pose state | keep exactly the observations already supplied to ACT/DP on the 10 Hz LeRobot dataset |
| 32D low-dimensional input, including episode-start-relative orientation | canonical 20D relative pose+gripper state | the shared evaluator/processor does not expose the extra episode-start token |
| 16 strided action slots; deploy 8 | internal horizon 32; decode the common 30 actions | evaluate identical offsets `[-1,31]` and the same endpoint as ACT |
| no padded action samples | shared padded sampling with padding-masked loss | preserve common query coverage without learning copied boundary actions |
| batch 64 | batch 64 | preserve the released optimization recipe; report seen samples and compute because this is 8× the main matrix batch |

Consequently these candidates answer whether the released UMI visual encoder
and denoisers transfer well under the common LeRobot representation. They do
not isolate the effect of observation history, temporal stride, or the missing
episode-start orientation. The architecture-matched ACT-L1/flow pair remains
the clean objective isolation.

## 5. Experiment matrix

All stage-one controlled experiments use the same 1459 train set, 100-episode
validation set, PyAV decoder, no image augmentation, ImageNet image statistics,
identity rot6d normalization, chunk 30, batch 8, seed 1000, and host RTX 4090.
The official-UMI supplements deliberately retain their released policy-side
crop/color augmentation and batch 64; those differences are part of the
released recipe and are not presented as an architecture-only ablation. The
first stage uses a common optimizer-step budget and fixed evaluation queries.
The flow LR sweep is deliberate: equal LR isolates the objective, while a tuned
LR avoids mistaking an ACT-specific optimizer for an intrinsic flow failure.

Equal optimizer steps do **not** mean equal sample exposure for the official
recipes. At 30k steps, batch 64 draws 1.92M training samples, versus 240k for a
batch-8 stage-one run (8x as many); it also exceeds a 100k-step batch-8 run's
800k samples by 2.4x. Wall time and FLOPs differ further because the official
models use a ViT-B encoder and substantially larger denoisers. The official
candidates are therefore judged as end-to-end recipe candidates using decoded
accuracy together with latency and parameter count. A win cannot be attributed
to architecture alone; conversely, a loss despite the extra exposure is strong
negative evidence for this adapted recipe. The ACT-L1/ACT-flow pair remains the
controlled architecture-and-exposure comparison.

| Variant | Purpose | Parameters | Status |
| --- | --- | ---: | --- |
| `act_r18_vae` | exact 1459 early-budget replication | 52M | 30k + eval complete; 100k + eval complete |
| `act_r34_vae` | larger backbone, ImageNet-V1 initialization | 62M | 30k + eval complete |
| `act_r50_vae` | larger backbone + torchvision-recommended ImageNet-V2 initialization | 65M | 30k + eval complete; 100k + eval complete |
| `act_r50_v1_vae` | strict R18/R34-aligned ImageNet-V1 initialization control | 65M | 30k + 100k queued live before seed-1000 evaluation |
| `act_r50_large` | ResNet-50 + 768-wide, 6e/3d transformer | 145M | 30k + eval complete; not promoted |
| `act_r18_l1` | no-VAE deterministic objective control | 34M | 30k + eval complete; 100k + corrected exact-step eval complete |
| `act_r18_flow_u_lr1e5` | exact-LR, uniform-time flow control | 35M | 30k + eval complete; 100k queued |
| `act_r18_flow_u_lr1e4` | flow optimizer sensitivity | 35M | 30k + eval complete; rejected |
| `act_r18_flow_beta_lr1e4` | OpenPI-like time bias | 35M | 30k + eval complete; rejected |
| `act_r18_diffusion_lr1e5` | epsilon/DDIM objective at exact ACT-flow learned architecture and LR | 35M | 30k + 100k queued live before seed-1000 evaluation |
| `diffusion_r18` | standard non-VLM diffusion control | 75M | 30k + eval complete; 100k queued |
| `umi_official_dp` | released ViT-B + U-Net recipe port | 160M online / 320M with EMA | implementation/tests complete; supervised 30k retry active |
| `umi_official_transformer_dp` | released ViT-token + transformer denoiser port | 152M online / 304M with EMA | implementation/tests complete; queued behind U-Net retry |

Promotion rules will be based on confidence intervals over decoded physical
metrics, not the lowest model-specific validation loss. At least the leading
ACT-capacity model, ACT-L1, and leading ACT-flow configuration should receive
multiple seeds before a final claim.

Stage-one evaluation uses five evenly spaced query frames in the common valid
action-offset intersection `[-1, 31]` in every one of the 100 validation
episodes. Reports use episode-balanced means and deterministic
95% nonparametric bootstrap intervals (10,000 resamples, seed 0), with episodes
as the resampling unit. ACT-flow and Diffusion Policy are additionally evaluated
with inference seeds 1000, 2000, and 3000 to expose sampling variance. Training
seeds are varied only after this screen selects configurations worth promoting.
The collector keeps inference-seed averaging and training-seed variability as
two distinct statistical levels. It emits per-training-run episode bootstrap
intervals, then separate cross-training-seed mean/SD/min/max tables and a
deterministic hierarchical 95% interval that resamples training seeds before
episodes within each selected seed. Repeated sampler draws are averaged within
episodes and never treated as independently trained models. The hierarchical
interval is reported with the caveat that three training seeds still give a
coarse empirical distribution. The collector requires
exactly matching episode-ID sets across inference seeds and across each paired
policy comparison, failing loudly instead of silently intersecting or
shape-matching different episodes. It also checks that the seed encoded in each
directory matches the JSON, that inference seeds are unique within a run, and
that every fixed evaluation contains the expected 100 episodes and 500 queries.
Across confirmations it rejects duplicate training seeds and requires every
candidate/baseline paired comparison to have the same training-seed set.
Synchronized policy-only GPU latency is measured on the same queries, excluding
the first cold call; mean, median, p95, and peak allocated inference memory are
recorded alongside accuracy. This makes the cost of larger backbones and
iterative samplers explicit. Parameter-efficiency figures use online learnable
parameters; for EMA policies the separately reported total parameter state is
roughly twice as large but does not represent a second model executed during
inference.

The final report will include reproducibly generated figures in addition to
tables: validation learning curves, decoded physical-error bars with episode
bootstrap intervals, paired percentage-improvement intervals, and
accuracy-versus-parameter/latency trade-off plots. Figure inputs will be the
same compact CSV/JSON evidence produced by `collect_results.py`; the plotting
script and rendered figures will be versioned in this directory. SVG generation
uses a fixed element-ID salt and omits volatile timestamps, making repeated
renders from identical inputs byte-for-byte reproducible. Once confirmation
runs exist, curves average equal-step validation values across training seeds
with an SD band; endpoint, paired-improvement, and efficiency figures use the
training-seed mean and hierarchical interval instead of selecting an arbitrary
equal-budget run. Single-training-seed error bars reduce to episode-bootstrap
intervals and are labelled separately from multi-seed uncertainty.
Learning-curve objective panels use explicit, complete, disjoint variant groups
rather than substring inference. A regression test specifically requires
`act_r18_diffusion_lr1e5` to appear in the diffusion noise-MSE panel, so its
result cannot silently disappear into the ACT-L1 group when artifacts arrive.

## 6. Smoke experiments and resource observations

All training smoke tests ran on the host GPU, not in the sandbox, against the
real 1459 dataset. The workstation has one RTX 4090 (24,564 MiB). The original
source filesystem was 95% full, so new full artifacts use the external project
directory. At the initial inspection the external mount had 345 GB free. After
the screen, interrupted-run preservation, and queued long-run setup, artifacts
occupied only 7.4 GB and the mount still had 337 GB free; the 100k confirmation
matrix therefore has ample checkpoint headroom. The repository filesystem had
only 27 GB free, reinforcing the decision not to place model artifacts there.

| Run | Result | Parameters | Cold first step | Notes |
| --- | --- | ---: | ---: | --- |
| ACT-flow R18 | passed 2 updates + checkpoint | 34,728,266 | 4.65 s | canonical 31 raw poses became 30 relative targets |
| Diffusion R18 | passed 2 updates + checkpoint | 75,396,650 | 10.56 s | 33 raw poses became internal horizon 32 |
| ACT R50 large | passed 2 updates + checkpoint | 144,946,762 | 10.61 s | batch 8 fits; downloaded official ResNet-50 V2 weights |

Cold-step numbers include worker/video/model warm-up and are not steady-state
throughput. Dedicated timed runs are required before latency conclusions.

## 7. Validation and test evidence so far

- Ruff and whitespace checks pass on all changed files.
- Twelve focused collector/statistics tests pass in 0.04 s. They cover online
  versus EMA parameter accounting, exact inference-seed episode matching,
  fixed-query provenance, ratio-of-means paired improvement, single-seed
  reduction, training-seed cluster resampling, preservation of paired seed and
  episode indices, duplicate training seeds, and asymmetric candidate/baseline
  seed sets. The verified host command sets
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; otherwise an unrelated installed ROS/ament
  pytest plugin imports the optional hardware test module and skips collection
  on missing `deepdiff` before the requested pure-NumPy test file is collected.
- 48 targeted CPU tests pass, 9 hardware/optional tests skip. These cover the
  new ACT-flow training/inference path, shared flow integrator, legacy ACT VAE
  behavior, ACT/Diffusion processors, and canonical UMI processor behavior.
- ACT-flow produces finite differentiable loss, gradients in its noisy-action
  projection, deterministic outputs under fixed input noise, and rejects bad
  noise shapes.
- Twelve focused ACT generative-objective tests pass. In addition to the flow
  checks, they prove ACT-flow and ACT-DP have identical learned parameter names
  and shapes **and identical initialization under the same seed**, and verify
  ACT-DP's finite differentiable epsilon loss,
  squared-cosine scheduler, deterministic fixed-noise DDIM output, strict
  noise-shape rejection, and save/reload equality of fixed-noise output plus
  diffusion configuration.
- CPU construction with the exact queued feature geometry (20D derived state,
  10D action, image input, chunk 30) succeeds for both newly queued controls.
  R50-V1 ACT has exactly 64,654,218 parameters and resolves the cached V1
  weights; ACT-DP has exactly 34,728,266 finite parameters and constructs a
  100-training-step epsilon-prediction `DDIMScheduler`. Both remained on CPU,
  so this launch-fidelity check did not contend with the active host GPU job.
- Diffusion's UMI config produces `[-1, 0, ..., 31]`, strips the leading action
  into the two-pose derived state, and reconnects the same relative-action step
  to postprocessing.
- One-sample host-GPU checkpoint reload and physical-pose decoding passed for
  both ACT-flow and Diffusion Policy. Their very large errors after only two
  optimizer updates are expected and are not performance evidence.
- Seven focused tests for the new official UMI candidates pass, together with
  the legacy UMI Diffusion processor test. They verify factory registration,
  canonical relative-action processor wiring, finite differentiable losses,
  online-to-EMA updates, fixed-noise sampling shape, checkpoint/EMA round-trip,
  the transformer's lower-LR backbone parameter group, and the distinct
  released image paths (U-Net restores ImageNet normalization after pixel-space
  augmentation; transformer retains pixel-space values) using a tiny
  timm-compatible test encoder.
- A real cached `vit_base_patch16_clip_224.openai` CPU load and forward pass
  produced finite `[1,197,768]` tokens with 85,799,424 encoder parameters.
  Full construction gives 160,091,530 online trainable parameters for the
  U-Net candidate and 152,173,066 for the transformer candidate; checkpointed
  totals double to 320,183,060 and 304,346,132 because EMA is an explicit copy.
  Launchers pin Hugging Face offline mode for these cached weights, avoiding a
  metadata network request during queued training.

### 7.1 Compatibility and isolation audit

The research changes do not rewrite or silently opt existing policies into a
new objective. The original work was first committed/pushed on
`fei-v5.0-umi-unified`; all experiment code lives on the independent
`research/umi-act-flowmatching-ablation-20260811` branch. Training launchers
refuse existing output/log paths, and every new checkpoint is under the
external Glowat512 artifact root, so historical checkpoint directories remain
read-only.

Legacy ACT retains `action_objective="l1"` and `use_vae=true` defaults. Its old
forward/inference code is selected unless flow matching or ACT diffusion is
explicitly requested.
As a direct compatibility experiment, the historical 800k checkpoint at
`outputs/train/act_umi_identity_rot6d_1459` loads on the research branch as
L1+VAE with exactly 51,579,786 parameters and no missing/unexpected-weight
failure. Diffusion's UMI processors, delta indices, and normalization changes
are likewise gated by the new `use_umi_relative_ee=false` default; ordinary
Diffusion Policy keeps its prior path. The dataset factory change only adds
Diffusion to the allow-list when that flag is explicitly true. Evaluator and
launcher changes are confined to `examples/` and do not enter normal training.
Seventeen focused legacy/new ACT and Diffusion processor tests pass, with nine
optional tests skipped.

There is one temporary runtime interaction: stage-two training deliberately
occupies the only host RTX 4090. At the compatibility check it used about
4.45 GiB and 93% GPU utilization. Another simultaneous GPU training process
would contend for compute and could run out of memory; wait for or stop tmux
session `umi_arch_stage2_20260811` before launching unrelated GPU training.
This resource contention does not alter previous checkpoints or configs.

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
8. The first evaluation pass chose valid queries from `chunk_size=30`. That is
   correct for ACT's offsets `[-1, 0, ..., 29]`, but Diffusion Policy requests
   `[-1, 0, ..., 31]`; its first late query was therefore padded and the launcher
   stopped after 14 reports. The evaluator now accepts and records explicit
   offset bounds, the fixed launcher uses the common `[-1, 31]` intersection,
   and a regression test checks exact frame indices. The 14 earlier reports are
   preserved under `eval/` as superseded evidence. All scientific results use
   the clean rerun under `eval_common_h32/`.
9. Adding timm as a new optional extra made the lockfile stale while stage two
   was active. The lockfile was refreshed before the next sequential subprocess
   could launch, and the host environment was synced with the union of
   `training`, `dataset`, `diffusion`, `umi-official-dp`, and `test`; the running
   R50 process was not restarted or modified. An offline lock attempt first
   failed because the local cache lacked packages needed for other supported
   Python/platform resolution splits, so the successful lock used the registry.

## 9. Results

The exact ResNet-18 ACT replication gives
the first scientific sanity check:

| Run | 10k validation | 20k validation | 30k validation |
| --- | ---: | ---: | ---: |
| historical 1459 ACT | 0.054203 | 0.043292 | 0.039702 |
| controlled `act_r18_vae`, seed 1000 | 0.054130 (-0.13%) | 0.043604 (+0.72%) | 0.041139 (+3.62%) |

The close early agreement makes major dataset, representation, decoder, or
software drift unlikely at the screening scale. The modest 30k divergence is
large enough that the fresh run, not the historical scalar, is the strict
contemporary control for architecture comparisons.

The first larger-backbone recipe comparison is provisionally favorable to ResNet-34:

| ACT backbone | Parameters | Median update | 10k val | 20k val | 30k val |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 51,579,786 | 0.036 s | 0.054130 | 0.043604 | 0.041139 |
| ResNet-34 | 61,680,522 | 0.048–0.049 s | 0.051064 | 0.043164 | 0.039170 |
| ResNet-50 V2 | 64,654,218 | 0.075 s | 0.042517 | 0.037207 | 0.036259 |

This completed screen originally changed two coupled choices: R18/R34 use
torchvision ImageNet-V1 weights, whereas the recommended R50 launcher used
ImageNet-V2. The decoded gain below therefore supports the larger-R50-V2
**recipe**, but cannot yet allocate the gain entirely to architecture. The
added `act_r50_v1_vae` control holds the initialization family at V1, runs at
30k and 100k for seed 1000, and joins the 100k seeds 2000/3000 confirmation.
R50-V1 versus R18-V1 isolates capacity more strictly; R50-V2 versus R50-V1
isolates initialization at fixed architecture. Static torchvision resolution
confirms `ResNet50_Weights.IMAGENET1K_V1` is a valid distinct enum. The official
97.8 MiB V1 checkpoint was then prefetched without CUDA and verified on CPU
before the control released: its SHA-256 is
`0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`
(matching the torchvision URL prefix), it constructs a 25,557,032-parameter
finite ResNet-50 on CPU, and it now coexists with the cached V2 `11ad3fa6`
checkpoint. The queued run therefore has no network dependency at launch.
This load verifies the initialization artifact, not the scientific training
result; completion is still withheld until the real run and evaluation finish.

ResNet-34 reduces total validation loss by 5.7%, 1.0%, and 4.8% at the three
budgets; its 30k L1 is 0.037265 versus ResNet-18's 0.039285 (5.1% lower). It
reduces update throughput by roughly 25%. The decoded results below confirm the
capacity signal.

ResNet-50 V2 is a stronger recipe signal: its 10k total is 16.7% below ResNet-34
and 21.5% below ResNet-18, while its L1 (0.035436) is lower by 13.3% and 15.1%
respectively. At 20k its total (0.037207) remains 13.8% below ResNet-34 and
14.7% below ResNet-18, with L1 (0.034470) about 14% lower than both. It is also
2.1× slower per update than ResNet-18. At 30k its total (0.036259) is 7.4%
below ResNet-34 and 11.9% below ResNet-18; L1 (0.034574) is 7.2% and 12.0%
lower. The decoded metrics confirm that the gain survives in physical units.

The first transformer-scaling point is unfavorable at the baseline optimizer:
the 145M ResNet-50 + 768-wide 6e/3d model records 10k total 0.053834 and L1
0.037560, versus 0.042517 and 0.035436 for standard-width ResNet-50 V2. It is also
about 1.3× slower than that model and 2.6× slower than ResNet-18. Because the
larger model's training curve descends more slowly, this result tests equal-LR
architecture scaling; it does not rule out a higher-LR large-model variant.
By 30k the large model nearly catches up but still does not win: total 0.036617
versus 0.036259, and L1 0.035273 versus 0.034574 for standard-width ResNet-50 V2.
At the fixed budget it adds 80.3M parameters and about 32% update time without
a validation benefit.

Removing the ACT VAE gives a simpler early winner but not a final-budget winner.
Direct ACT-L1 records held-out L1 0.039505 / 0.039920 / 0.039448 at 10k/20k/30k,
versus VAE ACT 0.041715 / 0.040570 / 0.039285. Thus it is 5.3% better at 10k,
then plateaus and is effectively tied (0.4% worse) at 30k. It uses 17.4M fewer
training parameters and updates about 14% faster. The VAE is not needed for
30k L1 quality, while earlier stopping matters for the direct model.

The uniform-time matched-flow LR control rejects a simple optimizer explanation:

| Uniform ACT-flow LR | 10k flow MSE | 20k flow MSE | 30k flow MSE |
| --- | ---: | ---: | ---: |
| 1e-5 (ACT-matched) | 0.052217 | 0.046969 | 0.039829 |
| 1e-4 (flow-tuned candidate) | 0.081311 | 0.073134 | 0.076679 |

At 30k, 1e-4 is 92.5% worse in its own held-out velocity-MSE objective and has
regressed from its 20k value. Training remained finite, but isolated gradient
norm spikes above 100 accompanied the otherwise roughly unit-scale norms. Thus
the tenfold LR increase is not a rescue for this ACT-sized flow model. This is
an optimizer-sensitivity result only: neither row can yet be ranked against ACT
L1 or VAE losses until their decoded physical trajectories are evaluated.

The Beta(1.5, 1.0)-time, 1e-4 flow run records held-out velocity MSE
0.064493 / 0.060377 / 0.064477 at 10k/20k/30k. It trains cleanly in 18m24s,
with late gradient norms near 1.0, but also has a 20k optimum followed by mild
regression. Those numbers are not directly comparable with uniform-time MSE
because the validation sampler follows each run's training-time distribution.
The common decoded-trajectory evaluation is therefore required to determine
whether beta sampling actually improves the policy.

Standard non-VLM Diffusion Policy records held-out noise-prediction MSE
0.014588 / 0.009324 / 0.008077 at 10k/20k/30k, a monotonic 44.6% reduction.
Its cosine learning-rate schedule reaches the configured 1e-6 floor near the
end, while updates remain stable at about 0.033 s. This scalar is meaningful
only within Diffusion Policy and is not evidence that it beats ACT: decoded
physical errors from the common queries remain the cross-objective endpoint.

![Validation learning curves](figures/validation_learning_curves.png)

### 9.1 Decoded physical metrics at 30k

All rows below use the corrected common 500-query set. Generative rows average
inference seeds 1000/2000/3000 within each episode before averaging episodes.
Latency is synchronized policy-only median latency; memory is peak allocated
CUDA memory. Lower is better throughout.

![Decoded endpoint errors](figures/decoded_endpoint_errors.png)

| Variant | XYZ chunk (mm) | XYZ end (mm) | Rot chunk (deg) | Rot end (deg) | Median (ms) | Peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACT R18 VAE | 18.30 | 27.50 | 3.249 | 5.516 | 7.13 | 267 |
| ACT R34 VAE | 17.36 | 25.65 | 3.147 | 4.947 | 8.55 | 305 |
| ACT R50 V2 VAE | 14.90 | **23.65** | 2.677 | **4.390** | 9.89 | 341 |
| ACT R50 large | **14.57** | 23.83 | **2.650** | 4.462 | 11.51 | 653 |
| ACT R18 L1 | 17.91 | 28.18 | 3.143 | 5.117 | **6.70** | 200 |
| ACT-flow uniform, 1e-5 | 18.75 | 30.86 | 3.767 | 6.290 | 29.90 | 203 |
| ACT-flow uniform, 1e-4 | 37.12 | 57.94 | 5.184 | 7.070 | 29.87 | 203 |
| ACT-flow beta, 1e-4 | 34.59 | 59.13 | 3.810 | 5.689 | 29.62 | 203 |
| ACT-DP, 1e-5 | 24.50 | 38.70 | 5.025 | 8.319 | 137.04 | 203 |
| Diffusion R18 | 15.71 | 27.27 | 3.391 | 5.838 | 23.23 | 345 |
| Official UMI U-Net | 16.14 | 28.76 | 3.239 | 5.838 | 47.28 | 1,277 |

Paired episode bootstrap comparisons (10,000 resamples) establish:

- ResNet-34 versus R18 improves XYZ endpoint by 6.7% (95% CI 1.5–11.7%)
  and rotation endpoint by 10.3% (4.5–15.9%).
- ResNet-50 V2 versus R18 V1 improves XYZ endpoint by 14.0% (9.6–18.2%), rotation
  endpoint by 20.4% (15.3–25.3%), XYZ chunk mean by 18.6% (14.9–22.0%), and
  rotation chunk mean by 17.6% (13.3–21.7%). All four paired difference
  intervals exclude zero.

![Paired endpoint improvements](figures/paired_endpoint_improvements.png)
- R50-large versus standard-width R50 V2 is tied on all four pose metrics: for
  endpoint XYZ its improvement is -0.8% (CI -6.0–4.2%), and for endpoint
  rotation -1.6% (-8.0–4.2%). It adds 80.3M parameters, 16% inference latency,
  and 312 MiB peak memory without a supported accuracy gain.
- ACT-L1 versus ACT-VAE is tied in XYZ but improves endpoint rotation by 7.2%
  (2.1–12.0%). It is the fastest and smallest ACT control.
- Official UMI U-Net versus ACT-L1 improves XYZ chunk mean by 9.9% (paired
  episode CI 5.7–13.9%), but its XYZ endpoint is tied (-2.1% improvement, CI
  -8.0–3.6%), endpoint rotation is 14.1% worse (7.5–21.2% worse), and both XYZ
  and rotational jerk are substantially worse. These figures average its
  three inference seeds; only one training seed exists so far.

The R50-V2 result is therefore not merely a lower training loss: it is a
sizable, statistically supported decoded-pose improvement for the combined
backbone-plus-initialization recipe. Scaling the already-large transformer at
the same optimizer is not supported. Attribution of the R50 gain specifically
to backbone capacity remains provisional until the queued R50-V1 control is
decoded across training seeds.

### 9.1.1 Fresh 100k deterministic checkpoints

The recovered fresh 100k checkpoints have now both passed a host-GPU decoded
smoke test and the complete fixed 500-query evaluation. These rows use training
seed 1000 and deterministic inference seed 1000; their intervals resample the
same 100 validation episodes and therefore do not include training-seed
uncertainty.

| Variant | XYZ chunk (mm) | XYZ end (mm) | Rot chunk (deg) | Rot end (deg) | Gripper end | Median (ms) | Peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACT R18 VAE, 100k | 16.39 | 24.74 | 2.873 | 4.892 | 0.1603 | **8.63** | 267 |
| ACT R50 V2 VAE, 100k | **14.28** | 24.44 | 2.793 | 4.875 | **0.1366** | 10.79 | 341 |
| ACT R18 L1, 100k | 14.34 | **23.69** | **2.769** | **4.850** | 0.1451 | 9.03 | **200** |

Paired episode differences make this a narrower result than the validation-loss
gap suggests. R50 improves chunk-mean XYZ by **12.83%** (95% CI
8.90--16.57%) and gripper endpoint by **14.83%** (7.14--21.98%). Its endpoint
XYZ improvement is only 1.23% (-4.38--6.64%), endpoint rotation 0.33%
(-5.53--5.93%), and chunk rotation 2.77% (-2.07--7.62%); none of those three
intervals excludes no improvement. R50 also raises median inference latency by
25% and worsens rotational jerk by 11.56% (7.44--15.68% worse), while XYZ jerk
is tied. Thus the larger R50-V2 recipe has a repeatable chunk-translation and
gripper benefit at 100k, but not a demonstrated endpoint-pose benefit at this
seed. The strict R50-V1 and multi-training-seed controls remain essential before
attributing the improvement to backbone capacity or recommending R50
unconditionally.

The corrected exact-step ACT-L1 result changes the practical deterministic
recommendation. Relative to R18 VAE, direct L1 improves chunk XYZ by **12.52%**
(paired episode CI 9.05--15.88%), gripper endpoint by **9.53%**
(4.24--14.67%), rotational jerk by **5.12%** (0.84--9.13%), and XYZ jerk by
**13.92%** (10.81--16.97%). Endpoint XYZ improves 4.25% but remains tied
(-0.43--8.82%), and both rotation errors are tied. It is therefore the current
low-cost default: essentially the best pose accuracy in this 100k deterministic
set, the smallest online allocation, and smoother trajectories than either VAE
control. R50-V2 retains only a small chunk-XYZ edge over L1 (14.28 versus
14.34 mm) and a better gripper endpoint, insufficient by itself to justify its
extra inference cost without the strict V1/multi-seed controls.

### 9.2 Flow matching and Diffusion Policy isolation

The closest objective isolation is ACT-flow uniform 1e-5 versus ACT-L1. Both
use the same ResNet-18 encoder, 512-wide ACT transformer, action/state
representation, no VAE, data, optimizer LR, and queries; only the direct L1
head versus time-conditioned velocity field and iterative sampler differ.
Matched flow is worse on every pose metric: endpoint XYZ error is 9.5% higher
(paired CI 3.7–15.4%), endpoint rotation 22.9% higher (15.9–30.5%), chunk XYZ
4.7% higher (0.2–9.2%), and chunk rotation 19.9% higher (14.9–25.0%). The 1e-4
and beta-time variants fail much more severely in translation. Thus this
particular vanilla conditional-flow formulation has a real deficit at 30k; it
is not explained by a VLM.

Vanilla non-VLM Diffusion Policy gives a different answer. Relative to ACT-L1,
it improves chunk XYZ by 12.3% (8.0–16.3%), is tied on endpoint XYZ at +3.2%
(-2.3–8.5%), but worsens chunk rotation by 7.9% (3.4–12.6%) and endpoint
rotation by 14.1% (8.2–20.3%). Relative to ACT-VAE it improves chunk XYZ by
14.2% (9.8–18.4%), ties endpoint XYZ, and ties both rotation metrics at 95%.
Therefore flow/diffusion objectives are **not intrinsically always worse than
ACT**: standard DP is competitive and materially better on average translation
through the chunk. The matched straight-line flow implementation is the weak
case, while the earlier 100k π0.5 result being slightly better than ACT in XYZ
also rules out a blanket “all flow models fail” statement.

The completed architecture-matched ACT-DP control closes that residual
confound at seed 1000. Across inference seeds 1000/2000/3000 it obtains endpoint
XYZ 38.70 mm (95% episode CI 36.63--40.69) and rotation 8.319 deg
(7.931--8.723). Relative to ACT-L1 on paired identical queries, this is 37.36%
worse in endpoint translation (paired improvement CI -45.46% to -29.81%) and
62.60% worse in endpoint rotation (-72.89% to -53.09%). It is also worse than
matched ACT-flow 1e-5 by 25.40% in endpoint translation (-29.09% to -21.93%)
and 32.26% in endpoint rotation (-37.49% to -26.94%). Conversely, standard
temporal-U-Net DP remains near ACT-L1 in endpoint translation and far better
than ACT-DP. Therefore changing flow to epsilon diffusion inside this ACT
denoiser does not rescue performance; denoiser/conditioning/optimization recipe
matters at least as much as the broad generative objective. These intervals
capture episodes and inference seeds but not training-seed variability, so the
queued 100k and seed-2000/3000 confirmations remain necessary.

There is nevertheless an important control-quality cost. ACT-L1 rotation/XYZ
jerk is 0.091 deg / 0.00073 m, matched flow is 1.093 deg / 0.00466 m, and DP is
0.481 deg / 0.00186 m; the ground-truth values are 0.158 deg / 0.00067 m.
Iterative generative samples are substantially less smooth at 30k. Flow is
4.5×, standard DP 3.5×, and ACT-DP 20.4× slower than ACT-L1 at inference;
ACT-DP's ten transformer denoising passes cost 137 ms median even though its
online peak allocation is only about 203 MiB.

![Accuracy and latency trade-off](figures/accuracy_latency_tradeoff.png)

### 9.3 Answers and promotion decision after stage one

**Q1:** the completed screen shows that the ResNet-50-V2 recipe is the strongest
tested ACT improvement over the fresh 1459 control. ResNet-34-V1 is a smaller
positive step; the 145M widened transformer is not worthwhile at the tested
LR/budget. Because the R50 comparison also changed ImageNet initialization,
“capacity alone improves ACT” remains a hypothesis rather than a completed
attribution. R50-V1, R50-V2, and R18-V1 are promoted to the longer/multi-seed
comparison before recommending replacement of the multi-million-step
historical checkpoint.

**Q2:** no single explanation fits. Matched ACT-flow and same-architecture
ACT-DP are both significantly worse than ACT-L1, so simply swapping velocity
flow for epsilon diffusion does not fix this ACT-transformer generative path.
But vanilla temporal-U-Net DP without a VLM is competitive with ACT and better
on chunk translation, so flow/diffusion itself is not the fundamental problem.
Denoiser architecture, conditioning, optimizer/sampler design, VLM fine-tuning,
and trajectory smoothness are separate axes; the existing π0.5 result further
indicates that the VLM path can work. ACT-L1, uniform flow 1e-5,
architecture-matched ACT-DP, and standard DP are promoted with R18, R50-V1,
and R50-V2 to determine whether these conclusions persist at 100k and across
training seeds.

Stage two therefore trains fresh 100k runs (not scheduler-incompatible resumes)
for ACT R18 VAE, ACT R50 V2 VAE, ACT R18 L1, uniform ACT-flow 1e-5, and
Diffusion R18; the newly identified R50-V1 and architecture-matched ACT-DP
controls are inserted as separate 30k/100k successors before evaluation. Fresh
runs are required because Diffusion Policy's cosine scheduler was
constructed for 30k steps and had already reached its floor; extending that
optimizer state to 100k would not be equivalent to a 100k schedule. After the
100k screen, the surviving comparison will be repeated with training seeds
2000 and 3000 so that training-seed variability, which episode bootstrap cannot
measure, is included in the final recommendation.

The stage-two sequence was launched on the host RTX 4090 at 2026-08-11 20:41
Asia/Taipei in tmux session `umi_arch_stage2_20260811`. Its first run,
`act_r18_vae_seed1000_100000steps`, initialized successfully at about 26.7
steps/s. Checkpoints/logs remain under
`/media/zfei/Glowat512/projects/lerobot-arch-exp`. This was an active long-run
confirmation at launch and is not mixed into the completed 30k table above;
its later completion is recorded below.

At the 2026-08-11 22:40 progress check, fresh ACT R18 had completed 100k in
1h08m20s. Its validation total/L1 at 90k was the run-best
0.035114/0.034751; the final 100k values were 0.035413/0.035243, showing a
small late fluctuation rather than continued monotonic improvement. Fresh ACT
R50 was then at 36.6k/100k (about 12.6–13.0 steps/s). Its 10k/20k/30k total
validation losses were 0.043859/0.038979/0.034967, with corresponding L1
0.036001/0.036323/0.033306. Thus R50 at 30k is already below R18's best 100k
total and L1 validation values. This is encouraging convergence evidence only;
the common decoded evaluation is deferred until the final 100k checkpoint.

At the 2026-08-11 23:21 check, R50 reached 67k without NaNs, OOMs, or dataloader
failures, sustaining about 12.6–13.0 steps/s. Its 40k, 50k, and 60k validation
total/L1 values were 0.033985/0.033025, 0.032902/0.032421, and
0.033020/0.032475. The small 50k→60k fluctuation does not erase the large gap
to R18. The two official UMI candidates did not contend with this job. They
were initially scheduled after the existing five-run stage-two sequence; the
wrapper interruption described below changed the realized ordering while
preserving single-GPU execution.

At the 2026-08-12 00:06 check, R50 had in fact completed all 100k updates and
saved a valid final checkpoint. Its validation total/L1 values at 70k, 80k,
90k, and 100k were 0.031453/0.030956, 0.031671/0.031365,
0.031066/0.030817, and 0.031239/0.031060. The 90k point is the minimum on
both measures, and the final checkpoint remains substantially below the R18
100k control. The original `set -e` stage-two wrapper nevertheless terminated
between the child's `End of training` message and its wrapper completion
marker, so the remaining three promoted runs did not start. The final R50
checkpoint was independently verified on disk and is retained as complete. It
was later reloaded for a successful host-GPU decoded query and a complete common
500-query evaluation, after which the log received a clearly labelled
recovered-complete marker rather than pretending the original wrapper returned
success. The matching R18 100k checkpoint was evaluated the same way; their
decoded comparison is reported in Section 9.1.1.

The first official U-Net attempt then passed a real two-update batch-64 smoke
test and trained normally through update 2,195 (approximately 4.9 updates/s).
At 2026-08-12 00:15:09 the host kernel recorded a segmentation fault for that
Python process in `libc.so.6`. There was no preceding traceback, non-finite
loss, CUDA error, NVIDIA Xid, or out-of-memory event; host memory still had
about 50 GiB available. The machine does not provide `coredumpctl`, so a native
stack trace could not be recovered. Because PyAV decoding and multiprocessing
are the principal native CPU path that differs from pure model computation,
the conservative retry sets `num_workers=0` and disables persistent workers.
This is a reliability intervention, not a model/hyperparameter change, and the
interrupted attempt is timestamp-archived rather than overwritten.

The single-process retry subsequently passed the exact earlier failure point:
the same Python PID that started at 08:05:21 reached update 2,232 without a
restart, exceeding the interrupted run's update 2,195. A contemporaneous kernel
check contained no new segfault, OOM, NVIDIA Xid, or CUDA event. This is direct
evidence that removing multiprocessing changed the observed failure behavior;
it does not by itself prove that the entire 30k run will complete, so the
supervisor and bounded retries remain active.

At the 2026-08-12 10:10 check, the same retry PID had advanced to update
8,518/30,000 (28.4%) at approximately 1.14 updates/s, with a log age below one
second. The independent five-minute monitor reported 12,405 MiB allocated GPU
memory, 64°C, 97% utilization at its latest sample, and about 337 GiB free on
the artifact mount. No restart or new failure signature occurred. The first
held-out validation is scheduled at update 10,000; until it is emitted, this is
health/progress evidence only, not model-quality evidence.

The retry crossed that first validation boundary without restarting. At update
10,000 it reported held-out epsilon-prediction loss **0.018457**; the nearby
training windows were approximately 0.021–0.022, gradient norm had declined to
about 0.109, and throughput remained approximately 1.14 updates/s. At the
2026-08-12 10:38 check it had reached 10,273/30,000. This is favorable
within-run convergence/stability evidence. It is deliberately not compared
numerically with ACT L1, ACT flow velocity MSE, or the other DP recipes because
their objectives, timestep distributions, and loss scales differ. Decoded
physical metrics remain the cross-policy decision endpoint.

To prevent a second child failure from silently stopping the study,
`supervise_remaining.sh` now records every child exit status, preserves failed
attempts under the external artifact root's `interrupted/` directory, retries
each full job up to three times, smoke-tests official candidates at batch sizes
64/32/16/8, and advances past an exhausted job. It queues the two official
30k candidates first, followed by the still-missing ACT-L1, matched ACT-flow,
and ResNet-18 Diffusion Policy 100k runs. Only one foreground child uses the
host GPU at a time.

A non-contending `supervise_capacity_control.sh` successor first waits for that
training tmux session, then trains R50-V1 and architecture-matched ACT-DP at
30k and 100k for seed 1000 with the
same bounded retry/single-process fallback. A refreshed
`supervise_evaluations.sh` watchdog waits for this capacity-control session
before touching the GPU. It evaluates completed 100k ACT R18/R50-V2/R50-V1/
ACT-L1 checkpoints once, evaluates R50-V1 at 30k, and evaluates stochastic
ACT-flow, ACT-DP, and Diffusion Policy checkpoints with seeds 1000/2000/3000.
It applies the same three-seed
protocol to both official 30k UMI candidates, and archives/retries interrupted
evaluation seeds independently. It then runs result collection and figure
generation. Thus an evaluation failure cannot discard completed training or
prevent later candidates from being measured.

`insert_capacity_control_chain.sh` performs the live insertion defensively. It
requires the main training supervisor to exist, requires evaluation and both
confirmation sessions to contain their exact expected waiting messages, and
refuses to touch any non-idle successor. Only then does it remove successors in
reverse dependency order, rebuild capacity-control → evaluation → confirmation
training → confirmation evaluation, refresh the monitor, and verify all six
sessions. This prevents a rewire from accidentally killing active GPU work.

At 2026-08-12 10:11, the three successor panes were rechecked and still
contained the exact expected idle wait messages. The guarded insertion was then
requested, but the execution environment rejected host mutation *before the
script started* because the workspace had exhausted approval credits. No tmux
session or GPU process changed. The original main-training → evaluation →
confirmation-training → confirmation-evaluation chain therefore remained live;
the additional-control scripts were staged and tested but were not yet in that
live dependency graph. Using an indirect command path to evade that boundary
would have made the safety audit meaningless and was not attempted.

After explicit continuation approval, the same guarded script succeeded at
2026-08-12 10:38:25. It did not touch the active main-training session. A
read-back verified all six live sessions and every dependency message:
capacity control waits for main training, evaluation waits for capacity
control, confirmation training waits for evaluation, and confirmation
evaluation waits for confirmation training. The refreshed monitor also saw all
five workload/dependency sessions on its first heartbeat. Its lifecycle list
now includes the later extended-candidate session, so telemetry does not stop
when the core confirmation chain exits while SmolVLA or LingBot is still
training. R50-V1 and ACT-DP are
therefore genuinely in the live queue before evaluation, not merely represented
in launcher source.

Independent training-seed confirmation is also encoded as a non-contending
successor rather than mixed into the screen. After all seed-1000 evaluations,
`supervise_confirmation_training.sh` trains the seven promoted controlled
variants at seeds 2000 and 3000 with the same 100k budget, preserving and
retrying incomplete attempts. `supervise_confirmation_evaluations.sh` then
applies one inference seed to deterministic ACT variants and three inference
seeds to each generative variant. This yields three independent training seeds
for the Q1 R18/R50-V1/R50-V2 comparison and for the Q2
ACT-L1/ACT-flow/ACT-DP/standard-DP
comparison while keeping sampler variability nested inside training runs. A
first attempt uses the established four-worker path; after any child failure,
later attempts switch to single-process decoding (`num_workers=0`, no persistent
workers), so a repeat does not exercise the same PyAV multiprocessing boundary
that caused the official U-Net native crash.

`monitor_experiment_chain.sh` independently records five-minute heartbeats for
the six-session chain: live dependency sessions, relevant process count,
latest-log age, artifact-disk headroom, and GPU temperature/power/memory/use.
It warns after 20 minutes without a training/evaluation log update or below
50 GiB free, but deliberately never kills a process; recovery and forward
progress remain owned by the bounded-retry supervisors above.

## 10. Throughput intervention and concurrent scheduling

At 2026-08-12 11:53, the first official UMI U-Net run was at step 15,465/30,000.
It was definitely training on CUDA with PyAV, but used `num_workers=0` after the
earlier four-worker PyAV attempt segfaulted at step 2,195. Its stable timing was
approximately `data_s=0.685--0.690` and `updt_s=0.190`, or only 1.14--1.16
steps/s. The long idle intervals were therefore serialized video decoding, not
CPU model execution and not TorchCodec: both the failed and surviving commands
explicitly selected `dataset.video_backend=pyav`. Merely uninstalling
TorchCodec would not alter that selected code path and would remove a useful
future decoder control, so it was retained.

Because this experiment did not save intermediate weights, restarting it
necessarily discarded about 15.5k optimization steps (roughly 3.8 hours). The
10k validation observation, loss 0.018457, remains in the archived log as
provenance but is not treated as a checkpoint result. Before restarting, the
launcher was changed to permit an explicit save interval and the official
queue was configured to save at 10k, 20k, and 30k. Two PyAV workers were chosen
as the conservative middle point between the unstable four-worker setting and
the slow zero-worker fallback.

The first restart smoke exposed an in-progress LingBot integration error:
`ConstantWithWarmupSchedulerConfig` was referenced but absent from this
branch. This happened before any optimization step. A registered linear-warmup
then constant scheduler was added, its LingBot import was tested, and the
supervisor was relaunched. This is an important operational lesson: optional
policy modules must remain import-safe because factory registration can affect
unrelated policies even when that candidate is not selected.

The repaired real run began at 11:56:40 with batch 64, CUDA, PyAV, and two
workers. At step 200 it measured `data_s=0.063`, `updt_s=0.227`, and 3.73
steps/s: data latency fell by about 90.8% and end-to-end throughput improved
about 3.3x. This directly isolates loader concurrency as the utilization fix.
With the GPU still having about 12 GiB free, the missing architecture-matched
ACT diffusion control (`act_r18_diffusion_lr1e5`, seed 1000, 30k) was started
concurrently. Combined allocation stabilized near 15.5/24.6 GB and 100% GPU
utilization. Under contention the official run remained near 2.97 steps/s,
while ACT-DP ran near 11.4--12.2 steps/s; aggregate progress increased without
OOM. This companion is expendable if memory pressure appears, while the
official run remains the protected primary job.

Retry policy now follows a measured 4 -> 2 -> 0 worker ladder where relevant,
rather than jumping directly from maximum multiprocessing to serialized
decoding. The scientific and operational conclusions are distinct: decoder
workers change throughput, not the policy objective, while concurrent training
changes wall-clock scheduling and must never be used to compare raw per-step
speed between models.

### SmolVLA rotation-notation control

The SmolVLA comparison holds the pretrained checkpoint, action expert, flow
objective, optimizer, state representation, chunk length, and padded-action
strategy fixed. The only intended action-space difference is `rot6d` (10D:
xyz + two rotation-matrix rows + gripper) versus axis-angle (7D: xyz + a
rotation vector + gripper). Both use the same chunk-start transform
`inverse(T_start) @ T_target`; both retain the shared 20D rot6d state bridge.

Axis-angle cannot be normalized independently per coordinate without changing
its geometry: the vector direction is the rotation axis, so three different
MIN_MAX slopes shear that axis. The implementation therefore computes one
scalar bound equal to the largest absolute training-set rotation-vector
coordinate and assigns `[-bound, +bound]` to all three rotation dimensions.
Normalization is then a scalar multiplication and preserves both the axis and
relative component ratios. Position and gripper retain dataset-derived
per-coordinate statistics. Focused tests cover the 7D SE(3) round trip,
processor selection and chunk reference, serialized horizons, and the shared
symmetric bound; the combined SmolVLA/LingBot/UMI processor suite currently has
32 passing CPU tests after also covering evaluator sampler-field selection and
the non-UMI LingBot identity-denormalization path.

A later host audit re-ran the 21 tests most directly tied to this extension:
the reversible UMI processors, symmetric axis-angle statistics, SmolVLA
notation configuration, LingBot 7D-to-30D bridge/configuration, and resumable
asset-integrity gate all passed. The ordinary repository invocation initially
reported zero collected tests because globally installed ROS Kilted pytest
entry-point plugins imported an unrelated hardware module whose optional
`deepdiff` dependency was absent. Re-running with third-party plugin autoload
disabled collected the intended files and produced `21 passed`; the earlier
zero-test exit is not counted as validation evidence. This invocation isolates
the test harness only and does not modify the shared training environment.

The guarded extension queue trains both SmolVLA notations at seed 1000 for 30k
and at seeds 1000/2000/3000 for 100k, followed by three inference seeds per
checkpoint. This is deliberately paired: same dataset queries, initialization,
training budget, and model width allow the notation effect to be estimated
without attributing a VLM change to rotation representation.

### LingBot-VA candidate and interpretation boundary

LingBot-VA is included as a pretrained VLA/world-model candidate, not as the
answer to the architecture-matched flow-isolation question. Its fixed latent
action tensor has 30 channels, but this single-arm dataset maps exactly seven
active channels `[0..6]` to relative xyz, relative axis-angle, and gripper;
unused channels remain zero and are action-masked in the loss. Raw dataset
actions remain absolute 7D poses and are reversibly converted by the same
chunk-start bridge used in the SmolVLA axis-angle condition.

Small Hub-manifest inspection changed the initialization choice before the
10.2 GB transformer download. `lerobot/lingbot_va_base` is a 3-camera,
14-active-action, 256x320 checkpoint. `lerobot/lingbot_va_libero_long` is the
released 2-camera single-arm checkpoint with exactly channels 0..6,
128x128 images, four actions per video frame, and four latent frames per chunk.
The latter is therefore the closest defensible initialization for one-camera
strawberry adaptation. Training overrides only the camera list to
`observation.images.camera`, disables the LIBERO-specific horizontal flip,
uses flex attention, and applies rank-8 LoRA; frozen Wan VAE and UMT5 weights
remain outside the optimizer. The first predicted chunk contains 12 executable
actions because LingBot treats latent frame zero as observed conditioning;
subsequent chunks contain 16. The postprocessor serializes separate 12-step
initial and 16-step subsequent chunk references so queued actions are composed
against the correct absolute pose.

This candidate has major confounds relative to ACT: a 5B pretrained video-action
DiT, pretrained text/video features, a different executable horizon, and
world-model latent loss. A better result would demonstrate useful transfer, not
that generic flow matching is intrinsically superior; a worse result could be
caused by the one-camera/domain shift, optimization or memory constraints, or
the objective. The ACT-L1/ACT-flow/ACT-DP trio remains the causal Q2 control.

Because this is the largest and least certain candidate, its supervised stage
now runs before the eight long SmolVLA notation jobs rather than after them.
That changes latency to evidence, not the comparison matrix: both SmolVLA
representations and all planned seeds remain queued. LingBot first waits for a
shared structural predicate to verify the >9 GB trainable file, VAE and text
encoder configs plus materialized weights, tokenizer, and absence of every
`.incomplete` shard. It then runs a two-update, batch-1, worker-0 host-GPU
preflight with checkpoint saving disabled. Only a successful flex-attention,
LoRA, camera/action-layout, forward/backward/optimizer pass promotes the 30k
run; failed smoke artifacts are retained and evaluation is skipped. Tests cover
both a complete fake asset set and rejection after adding an incomplete shard.

### Shared-environment dependency incident

At 12:12 on 2026-08-12, an attempt to add LingBot dependencies with a narrow
`uv sync --extra lingbot_va --extra umi-official-dp` pruned 134 packages not in
that selected environment, including PyAV, datasets, and test tools. Already
loaded training workers continued temporarily, but both live runs failed when
validation spawned fresh workers: the partially removed PyAV package raised
`ModuleNotFoundError: av.subtitles`. ACT-DP reached exactly step 10,000 before
the failure; at that time its launcher did not yet save until 30k, so those
weights were not recoverable. The official run also had no completed recovery
checkpoint before its validation boundary.

The full dataset/training/SmolVLA/LingBot/test/development environment was
restored, `import av, av.subtitles` was explicitly verified, and both runs were
restarted with two PyAV workers and 10k save intervals. Broken TorchCodec 0.11.1
was then removed from the repository virtual environment: it could not load
against PyTorch 2.11 and the host FFmpeg/libgcc combination, while every
experiment command explicitly selects PyAV. Post-recovery measurements returned
to roughly 2.94 steps/s for the contended official job and 11.5--12.1 steps/s
for ACT-DP at about 15.3 GB combined VRAM.

The recovered ACT-DP run entered its step-10,000 validation at 12:41:40 with
two PyAV workers and completed at 12:42:34 with
`loss=diffusion_loss=0.020295`. The 139,010,496-byte model file plus optimizer,
RNG, processor, and config state were independently verified under checkpoint
`010000`, and training resumed beyond step 11k. This demonstrates that fresh
validation workers now import and decode correctly and, unlike the failed
attempt, leaves a durable recovery point. To reduce the blast radius of any
later host or validation failure, `run_one.sh` now defaults every run longer
than 10k steps to 10k recovery checkpoints (while retaining an explicit
environment override).

The same recovered ACT-DP run completed its second validation at step 20,000
with `loss=diffusion_loss=0.018327`, improving 9.70% from the 10k value. A
second complete model/optimizer/RNG checkpoint was verified under `020000`.
This is still an intermediate training-curve observation; the causal comparison
uses the common decoded evaluation of the final promoted checkpoint.

At step 30,000 the recovered ACT-DP run completed validation with
`loss=diffusion_loss=0.014094`, a 30.55% reduction from its 10k value and
23.09% from 20k. The trainer then wrote full checkpoint `030000` and logged
`End of training`. The outer shell pipeline nevertheless returned status 2
without a traceback and therefore omitted its usual completion marker. This
was not treated as success by assertion: the complete checkpoint inventory was
verified and the saved policy was independently loaded for a real host-GPU
decoded validation smoke (one episode/query), which succeeded. The original
status-2 line remains in the log, followed by an explicit recovered-complete
provenance marker; the model was not wastefully retrained. Full fixed-query
evaluation was then launched as a low-memory companion to the continuing
official UMI run.

The recovered official UMI U-Net run likewise crossed all three durable boundaries.
Step-10,000 validation completed with `loss=0.018488`; step 20,000 improved to
`loss=0.017670`, 4.42% lower, before the final value rose to `0.018583` at
30k. At each boundary a ~1.28 GB model plus ~1.28 GB
optimizer state, exact training-step record, RNG, scheduler, processors, and
configs were independently verified under `010000`, `020000`, and `030000`.
The common evaluation completed 500 queries for each of inference seeds 1000,
2000, and 3000. Their averaged episode-balanced errors were 16.14 mm XYZ chunk,
28.76 mm XYZ endpoint, 3.239° rotation chunk, and 5.838° rotation endpoint;
mean synchronized inference latency was 48.05 ms. This closes the specific
PyAV/fresh-worker recovery question and provides the released U-Net comparison,
while the architecture-matched transformer denoiser remains in training.

The operational lesson is stronger than merely “install PyAV”: never run a
pruning environment synchronization against a virtual environment used by live
training. Large candidate dependencies must be installed additively or in a
separate environment, and imports used by future DataLoader workers must be
tested before the next validation/checkpoint boundary. Failed logs were kept
with incident-specific names; they are provenance, not scientific endpoints.

A second reliability lesson came from the successful ACT-DP and R50 runs: a
shell pipeline can return nonzero after the trainer has emitted `End of
training` and written its complete final checkpoint, but before the launcher
appends its own completion marker. Treating the wrapper marker as the only
truth would archive and retrain valid work. All training supervisors now share
a conservative durable-completion check requiring the exact requested
`training_step`, nonempty model, optimizer, RNG, config, and processor state,
plus the trainer's terminal message. When only the outer marker is missing,
the supervisor appends an explicit `recovered-complete` provenance line and
advances; partial checkpoints cannot pass. A sidecar applies the same rule to
the already-running official U-Net process, whose shell functions predate the
patch. Focused tests cover exact-step mismatch, missing optimizer state, and
idempotent recovery marking.

After the official U-Net's 20k checkpoint, a bounded concurrency experiment
used otherwise idle memory for the missing seed-1000 ACT-L1 100k control. The
previous dependency-era failed log was moved intact under `interrupted/`; no
checkpoint or live owner existed. ACT-L1 then launched with two PyAV workers
and its own durable-completion guard. In steady state the official model used
about 12.0 GiB and ACT-L1 2.7 GiB, leaving 9.2 GiB free. Aggregate GPU
utilization reached 100%; ACT-L1 added roughly 12 steps/s. The first short
sample put the official model at 3.5--3.8 steps/s, but the longer steady-state
sample settled near 3.0 steps/s versus approximately 3.8 alone, with median
update time rising from ~0.192 to ~0.284 s. Thus concurrency costs about 21%
of official-model throughput rather than being free, while advancing ACT-L1 by
roughly four updates for every official update. Both models remained finite
without OOM. The chain monitor now
tracks this companion session, and the main queue will skip its ACT-L1 slot if
the independently guarded run has completed rather than duplicate it.

The same two-job allocation remained healthy through the next supervised retry:
the host GPU reported 99% utilization, 15.82 GiB allocated, 8.29 GiB free, and
only the two intended trainer PIDs in the CUDA process table. ACT-L1 completed
and passed the full durability check for checkpoint `070000` (model, optimizer,
RNG/config/processor state, and exact step) before resuming beyond 71k. Its 70k
held-out L1 was 0.032481, above the 60k value 0.031042 and therefore evidence
that final-step selection should not be conflated with best validation-step
selection. Concurrently, the architecture-matched official transformer
denoiser crossed 7k with finite training loss near 0.029. Its update/data times
were approximately 0.36--0.38/0.02 s, confirming that the GPU computation—not
video decoding—is the limiting stage under contention. PyAV remained the
explicit backend; the transient child processes were its DataLoader workers,
not CPU trainers. Both large LingBot frozen-weight partials continued to grow
during this interval, so the asset supervisor was retained rather than
restarted and losing resumable download progress.

ACT-L1 subsequently completed validation and a fully durable checkpoint at
80k. Held-out L1 was 0.032604, essentially unchanged from 70k (0.032481) and
5.03% above the current 60k minimum (0.031042). Training resumed after the
checkpoint. Two consecutive later validations therefore support preserving and
reporting the best-validation checkpoint alongside the final 100k checkpoint;
they do not justify stopping the fixed-budget run early, because decoded common
queries—not validation loss alone—are the comparison endpoint.

The last pre-terminal recovery boundary at 90k also passed the full durability
predicate. Validation L1 recovered to 0.031907, better than 70k/80k but still
2.79% above the 60k minimum, while recent training L1 was near 0.017. The late
trajectory is therefore non-monotonic rather than a simple continuously
worsening overfit curve. The fixed 100k run continues under its exact-completion
guard; final and best-validation decoded checkpoints should both be retained in
the analysis if their common-query behavior differs materially.

The final 100k validation then reached 0.030966, the lowest held-out L1 in the
run and 0.24% below the previous 60k minimum. The final checkpoint passed the
full completion predicate (136,884,976-byte model, 273,604,264-byte optimizer,
exact step 100000, RNG/config/processor state, terminal trainer message, and
wrapper marker), and its completion guard released the canonical evaluator.
The apparent late-regression concern therefore resolved by the fixed endpoint:
the primary 100k checkpoint is also the validation-selected checkpoint. The
60k decoded sensitivity run remains useful for measuring how much physical
trajectory quality can vary across nearly tied validation losses, not for
replacing the pre-registered final endpoint.

Completion exposed an evaluation-provenance bug before the final table was
frozen. An early evaluator had run at 15:03 while this nominal 100k run held
only checkpoint `030000`; its report filename correctly contained `030000`, but
the generic completion marker named only the run and inference seed. At final
completion the old supervisor therefore mistook that 30k result for a completed
100k evaluation. The report and output were preserved under
`interrupted_evaluations/..._stale030000_*`, not deleted. Canonical completion
now requires exactly one nonempty metrics filename containing the requested
zero-padded checkpoint step, and the collector independently rejects a report
whose suffix differs from the run budget. Focused tests cover wrong-step,
missing-marker, and duplicate-report cases. The corrected evaluator command
was then observed loading `checkpoints/100000/pretrained_model` on the host GPU.
The corrected 500-query result then completed with the required `100000`
suffix. Canonical collection (15 runs, 64 validation points, 26 evaluations)
and all four PNG/SVG figure families were regenerated from it; the exact-step
collector would now reject reintroduction of the archived stale report.
Sleeping seed-1000, confirmation, extended-candidate, and transformer early
evaluation supervisors were audited; only processes that had sourced the old
predicate were restarted with their original dependency waits. No active
trainer or completion guard was interrupted.

To test that question without changing the primary endpoint, a separate
checkpoint-selection evaluator is queued after the canonical final ACT
evaluation. It explicitly resolves recovery step `060000` and writes only to
`eval_checkpoint_h32`, which the canonical collector does not scan. It uses the
same 100 episodes, five fixed queries per episode, horizon bounds, PyAV backend,
and deterministic seed as the final-checkpoint evaluation. Thus it can diagnose
whether the validation minimum selects a better decoded policy while the main
tables and figures remain an unbiased fixed-100k comparison.

That sensitivity evaluation also completed. Step 60k versus 100k is tied in
chunk XYZ (-0.82% final-checkpoint improvement, CI -2.82--1.26%), endpoint XYZ
(-0.26%, -2.76--2.22%), and endpoint rotation (-1.52%, -4.70--1.38%). The 60k
checkpoint has 3.21% lower chunk rotation error (0.43--6.30% supported), but
100k improves gripper endpoint by 6.32% (1.98--10.50%), rotational jerk by
18.94% (16.53--21.24%), and XYZ jerk by 26.30% (24.51--27.97%). The fixed final
checkpoint is therefore the more balanced control policy despite nearly tied
pose errors, and scalar validation loss alone would not reveal its smoothness
gain.

Two canonical early-evaluation waiters advance analysis without forking the
protocol: official U-Net inference seed 1000 starts after its exact 30k durable
completion, and deterministic ACT-L1 seed 1000 starts after its 100k companion
completion. Each uses the same 100-episode/500-query evaluator, canonical
output path, completion marker, and bounded archive/retry behavior as the main
evaluation supervisor. Later matrix evaluation therefore skips successful
early results and still fills every missing generative seed.

There is a subtle single-GPU handoff race at this boundary: the main queue can
start the next transformer's batch-size smoke test at the same moment an early
evaluator observes the previous trainer's completion. The early-evaluation
wrapper now waits five minutes for allocations to settle and then requires at
least 4 GiB of reported free VRAM before loading a checkpoint. This guard was
loaded by restarting only the two sleeping waiter sessions; neither active
trainer was interrupted. It reduces avoidable OOM risk without serializing the
entire experiment chain.

The official U-Net terminal transition then exposed two independent recovery
checker assumptions. LeRobot saved the final directory as `030000`, while the
checker first looked only for `30000`; its `training_step.json` was also
pretty-printed across lines, while the exact-step regular expression assumed
the closing brace was on the same line. The false negative let the legacy
wrapper archive a valid run and start a duplicate. Live supervision stopped
that duplicate after roughly 90 updates and stopped a prematurely released
R50-V1 successor before useful training. The canonical artifact was restored
and accepted only after independently confirming the 1.28 GB model, 1.28 GB
optimizer, RNG/scheduler and processor state, exact step 30000, validation
loss 0.018583, and `End of training`. The checker now resolves either padded or
unpadded directories and removes JSON newlines before exact-step matching; a
real-format zero-padded/pretty-JSON regression test accompanies the fix. The
official evaluation seeds that had safely begun were retained, while sleeping
downstream wrappers were removed for reconstruction in the intended order.
The architecture-matched official transformer denoiser now has its own exact
30k completion sidecar and canonical three-inference-seed early evaluator.
This avoids delaying the central U-Net-versus-transformer isolation result
behind the remaining 100k training queue; the later matrix evaluator reuses
these outputs. Its evaluator retains the same five-minute allocation-settle
window and 4 GiB free-VRAM gate.

That transformer-denoiser control completed its first validation and durable
checkpoint at 10k. Held-out diffusion loss was 0.020186; the checkpoint contains
a 1,217,458,704-byte model, 1,217,460,564-byte optimizer state, exact step,
RNG/scheduler, processor states, and configs. For context, the official U-Net
recorded 0.018488 at the same budget, but this small scalar gap is not yet an
architecture ranking: parameterization and loss aggregation differ, and the
pre-registered common decoded trajectories at final 30k remain the endpoint.

At 20k, transformer-denoiser validation loss rose to 0.023143, 14.65% above
its 10k value even though recent training loss continued downward near 0.020.
The full 20k recovery state again passed the durability predicate with the same
1.217 GB model/optimizer pair plus exact step, RNG, scheduler, processors, and
configs. This is evidence of a widening train/validation gap or related
scheduler/EMA dynamics, not yet evidence that the 10k policy decodes better.
The pre-registered 30k run therefore continues, and final common-query metrics
remain primary; retaining 10k/20k recovery checkpoints permits a later
checkpoint-sensitivity check if the final decoded result is unexpectedly poor.

The transformer-denoiser control subsequently completed all 30k updates and
passed the exact-final-step durability gate. Its 30k validation loss was
0.029431: 27.17% above 20k and 45.80% above the 10k minimum, while terminal
training loss had fallen to approximately 0.017. The final checkpoint contains
the 1,217,458,704-byte model, 1,217,460,564-byte optimizer state, exact step,
RNG/scheduler state, processors, and configs, followed by the explicit `End of
training` marker. This monotonic held-out degradation (0.020186 -> 0.023143 ->
0.029431) is strong evidence that selecting this candidate by terminal training
loss would be unsafe. It still does not by itself establish that transformer
denoising is worse than the official U-Net: the pre-registered common-query
decoded evaluation at exact step 30k remains the architecture comparison, and
the retained 10k checkpoint is a distinct checkpoint-selection question.

LingBot asset prefetch has a related but distinct integrity guard. During a
transient Hub SSL EOF, `hf download --local-dir` reported success by returning
the existing directory even though its 10.2 GB `model.safetensors` was still a
9.02 GB `.incomplete` cache blob. The supervised prefetch now requests the
trainable file explicitly, retains resumable partial data, and accepts it only
after the final >9 GB file is materialized and no incomplete blob remains. It
downloads only the frozen `vae/`, `text_encoder/`, and `tokenizer/` subtrees
from the source repository (not a redundant transformer), validates their
configs and weight files, and retries transient network failures. A CLI success
code is therefore no longer mistaken for model readiness.

The supervised download subsequently finished and the shared structural gate
returned success. The final frozen inventory contains all three UMT5 shards
(4,935,812,536, 4,983,103,192, and 1,442,935,480 bytes; 11,361,820,672 bytes
declared by the index), the 2,818,777,808-byte Wan VAE, tokenizer/configuration
files, and no `.incomplete` blobs. Together with the separately verified
10,177,841,732-byte trainable checkpoint, this closes asset acquisition. The
downloader exited and was not restarted. The extension supervisor remains
deliberately behind confirmation evaluation and will apply the same predicate
again before its two-update host-GPU smoke, preventing asset completion from
causing an unsafe third concurrent CUDA allocation.

## 11. Reproduction

The variant launcher is `run_one.sh` in this directory. `run_stage1.sh` executes
the fixed matrix sequentially so models do not contend for the single GPU, and
`evaluate_one.sh` resolves exactly one final checkpoint and evaluates it without
manual path selection, while `evaluate_stage1.sh` fixes the deterministic and
three-seed generative matrix. `collect_results.py` extracts parameter counts, wall
times, complete validation curves, decoded metrics, and confidence intervals
into compact external CSV/JSON files without creating a second narrative doc.
`run_stage2.sh` and `evaluate_stage2.sh` encode the seven promoted 100k controls.
`run_official_umi_dp.sh` and `evaluate_official_umi_dp.sh` encode the two
supplemental released-UMI recipe candidates. `supervise_remaining.sh` is the
fault-tolerant single-GPU queue used for the remaining long runs;
`supervise_capacity_control.sh` inserts the strict R50-V1 initialization and
architecture-matched ACT-DP controls before evaluation; and
`supervise_evaluations.sh` is its non-contending evaluation successor. Figures
label the training budget explicitly; when both 30k and 100k results exist,
endpoint/efficiency charts select the highest completed budget per variant and
the learning-curve chart retains each independent budget as a separate line.
The two `supervise_confirmation_*.sh` launchers encode the subsequent
seed-2000/3000 training and evaluation chain.
`plot_results.py` renders both SVG and high-resolution PNG figures from the
collector outputs. Use a writable Matplotlib cache, for example:

```bash
MPLCONFIGDIR=/tmp/lerobot-matplotlib uv run python \
  examples/umi_relative_ee/act_flow_ablation/plot_results.py
```
Example:

```bash
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_vae 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_flow_u_lr1e4 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh diffusion_r18 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_official_umi_dp.sh 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/supervise_remaining.sh 100000 30000 1000
```

The launcher refuses to overwrite an existing run. Full command-line configs
are also saved inside every checkpoint's `train_config.json`, and stdout/stderr
is retained in the artifact workspace.
