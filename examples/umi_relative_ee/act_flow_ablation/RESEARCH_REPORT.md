# ACT capacity and flow-objective investigation

**Status:** complete — seed-1000 controlled matrix (§9.1–9.2), π0.5 650K/700K flow-VLM reference (§9.2.2), SmolVLA rotation-notation ablation (§9.2.3), and the official-openpi rot6d-vs-rotvec replication (§9.2.4). §9.2.4 includes a horizon-matched correction: at equal 10-step scoring, SmolVLA / π0.5 port / official openpi are all statistically tied at 9–10 mm endpoint — earlier cross-model endpoint spreads were a horizon artifact; the real differentiators are smoothness and sample efficiency. The multi-seed (seed 2000/3000) confirmation was dropped for compute efficiency after two artifact-disk failures stranded the checkpoints (§8, incident 12); a 2026-08-17 salvage audit later found six seed-2000/3000 companion checkpoints alive at partial budgets and evaluated them (§9.2.6) — variant rank order replicates across training seeds — and then scored the fully-intact historical production ACT across its entire 100k–3M budget range on the same metric set (§9.2.7). Conclusions otherwise rest on a single training seed with per-episode bootstrap intervals, supplemented by the well-trained π0.5 references. A π0.5-port 700K→1M continuation is in flight on kiwi (resumed 2026-08-16, ETA ≈ 64 h; §9.2.2) — its evaluation will extend the §9.2.2 and horizon-10 tables when it lands. A horizon-30 openpi arm plus a JAX-vs-PyTorch matched-recipe stack A/B are in flight on the host (§9.2.5, sequential chain, first results ≈ 2026-08-17).
**Started:** 2026-08-11  
**Branch:** `research/umi-act-flowmatching-ablation-20260811`  
**Source baseline:** `3feb3f3e`  
**Full-run artifacts:** `/mnt/data1/projects/lerobot-arch-exp` (moved off the failed external disk, §8 incident 12)  
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
- [π0.5](https://arxiv.org/abs/2504.16054) reports scaling-law results on a
  metric quartet designed to be robust to metric choice: MSE and L1 averaged
  over action dimensions and the chunk horizon, plus thresholded accuracies
  accuracy@τ — the fraction of action dimensions within τ of ground truth in
  normalized action units — at τ=0.5 (motion-intent level, suited to
  human-to-robot transfer) and τ=0.1 (movement-precision level, informative
  in-domain). The 2026-08-16/17 evaluator extensions mirror exactly this
  quartet on decoded physical poses.

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
as the resampling unit. The decoded metric set per query covers per-step
rotation geodesic angle, xyz-norm, and gripper errors (chunk mean/RMSE/MSE plus
endpoint), within-chunk jerk, and — added 2026-08-16, before the §9.2.5 and
π0.5-1M evaluations — component-wise **L1** and **per-dimension MSE** for xyz
(m) and the axis-angle vector (deg): the action-space sense the training
objectives optimize, implemented as `per_component_l1_mse` in
`eval_open_loop_dataset.py` and mirrored in `eval_openpi_open_loop.py` (whose
summary registry previously computed but omitted the chunk-MSE keys — fixed in
the same change; both evaluators now emit the identical metric set, and the
openpi one takes `--action_horizon` for the h30 arm). A π0.5-style thresholded
**accuracy@τ** (added 2026-08-17; τ = 0.5 "motion-intent level" and τ = 0.1
"movement-precision level" — see the π0.5 reference in §2) completes the set:
the fraction of action
dimensions (over steps × dims, inclusive ≤) whose error falls within τ in
normalized action units. The normalization is protocol-fixed, not per-model:
per-dim errors are divided by (q99 − q01)/2 half-ranges pooled over this
evaluation's own GT chunks (q01 → −1, q99 → +1, the π0.5 quantile convention),
so the metric stays comparable across the MIN_MAX-trained ACT matrix and the
quantile-trained flow models; the scales are recorded in every report JSON.
Overall (`action_`) accuracy spans all seven decoded dims; `xyz_` and
`rotvec_` views follow the L1/MSE component split. Evaluations produced
before 2026-08-16 predate the L1/per-dim-MSE keys and before 2026-08-17 the
accuracy@τ keys (their JSONs already contain
the norm-based chunk MSE/RMSE); the kiwi/openpi checkpoints retain weights and
can be re-scored on demand, whereas the ACT seed-1000 matrix proved
unrecoverable after the disk failures (§9.2.6), so its tables keep the
norm-based metrics as the surviving record. ACT-flow and Diffusion Policy are additionally evaluated
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

10. The environment synced for stage two (incident 9) omits the `matplotlib-dep`
    extra, so a later `plot_results.py` invocation failed with
    `ModuleNotFoundError: No module named 'matplotlib'` even though figures had
    rendered earlier under a broader environment (`collect_status=0,
    plot_status=1`). Rather than re-sync — which risks the pruning incidents
    above while live training is running — plotting now runs through
    `uv run --with matplotlib`, an ephemeral overlay that leaves the shared
    training venv and the live training processes untouched. The evaluation
    supervisor's plot line was patched to use this path so confirmation
    auto-plotting does not silently skip figure regeneration, and the seed-1000
    figures were regenerated this way after the fix.

11. On 2026-08-13 at 13:36:26 the artifact disk `/dev/sdb1`
    (`/media/zfei/Glowat512`, ext4, all full-run checkpoints and decoded
    metrics) began returning `Input/output error` on writes and then on reads.
    This single event killed all four concurrent confirmation training jobs at
    the same timestamp (they all I/O on that mount) and took down every
    companion/supervisor tmux session whose logs live there; the host did not
    reboot (2-week uptime) and the tmux server itself survived. The failure is a
    hardware/disk-level fault, not a software or OOM event. As a result 9 of 14
    multi-seed confirmation runs were trained but only the seed-1000 matrix is
    fully evaluated; 5 confirmation runs (act_r50_vae s3000 at 70k,
    act_r50_v1_vae s2000 at 50k, act_r50_v1_vae s3000 unstarted,
    act_r18_flow_u_lr1e5 s3000 at 30k, diffusion_r18 s3000 at 30k) did not
    finish. The scientific seed-1000 results are **not** at risk: they were
    integrated into this report (Sections 9.1--9.2) and the figures/CSVs on the
    separate repository disk before the fault progressed. No multi-seed
    *evaluation* results existed yet (only training), so the only at-risk items
    are the large retrainable training checkpoints and the per-eval metric JSONs
    on the failing mount. Resume support (`--resume=true` via `UMI_RESUME=true`
    in `run_one.sh`; resume-aware `run_companion.sh`) was added so the surviving
    partial checkpoints can be continued once the disk is restored rather than
    retrained from scratch. **This incident is unresolved and blocks the
    queue** — advancing training/evaluation requires the disk fault to be
    repaired (fsck / remount / replacement) by the operator; it is not
    recoverable from software alone.

12. On 2026-08-14 at ~23:50 the same artifact disk (`/dev/sdc1` after its
    first recovery) failed a **second time** with block-layer `Input/output
    error` on both writes and reads, killing all four concurrently-resumed
    multi-seed confirmation jobs and stranding their checkpoints (reads failed,
    so no salvage copy was possible — the same mode as incident 11's nadir).
    Repeated failure shows the external disk is unreliable. Per operator
    directive the artifact root was **permanently moved** to the healthy internal
    `/mnt/data1/projects/lerobot-arch-exp` (`/dev/sda1`, ext4 rw, 280 GB free),
    and — for compute efficiency and to stop depending on the flaky disk — the
    multi-seed (seed 2000/3000) confirmation was **dropped** rather than
    retrained a third time. The ablation therefore finalizes on the seed-1000
    controlled matrix plus the independently-trained π0.5 reference (§9.2.2).
    Lesson: an external/USB disk is the wrong place for long-running checkpoint
    writes; the internal `/mnt/data1` root is now canonical, and all
    supervisors/companions read `UMI_ABLATION_ROOT` so a future retrain lands
    there directly.

13. On 2026-08-14, standing up the official-openpi replication (§9.2.4) hit a
    chain of host-network and data-compatibility faults, each recovered in
    software: (a) the pinned openpi LeRobot is format **v2.1** while the
    strawberry datasets are **v3.0** (chunked multi-episode parquet/mp4 + parquet
    metadata), so `LeRobotDatasetMetadata` fell back to a Hub lookup and 404'd —
    fixed by a one-time v3.0→v2.1 reshard (per-episode parquet + ffmpeg-split
    per-episode mp4 + jsonl metadata); (b) the host's Tailscale intercepted DNS
    for `storage.googleapis.com` (synthetic `198.18.0.80`, unreachable), blocking
    the 11.6 GiB `pi05_base` orbax checkpoint download — bypassed non-invasively
    by pinning real Google front-end IPs with `curl --resolve`; (c) resuming the
    partially-downloaded checkpoint across different downloader generations
    produced **size-correct but corrupt** shards (orbax later failed with
    `ZSTD_decompressStream` corruption) — caught, pinpointed with crc32c checks
    against GCS's JSON-API hashes, and re-downloaded cleanly with per-file
    verification; (d) LoRA training at the default `batch_size=32` OOMed the
    24 GB RTX 4090 — reduced to 16 with the step count right-sized (§9.2.4).
    Lessons: size checks are not integrity checks (verify a real checksum —
    GCS's XML listing ETags are only MD5 for small objects; the JSON API's
    crc32c covers everything); never resume a file across different writers;
    `pkill -f <script>` neither matches its spawned `curl` children nor avoids
    matching your own supervising shell — kill children explicitly and by PID.

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

### 9.1.2 Strict R50-V1 initialization control

The strict control was completed at 30k before its 100k continuation was
started. R50-V1 uses the same ResNet-50 width as the strong R50-V2 recipe but
keeps the V1/ImageNet initialization and optimizer configuration, isolating the
initialization-plus-capacity change that was confounded in the first screen.
The checkpoint has 65.0M learnable parameters and was evaluated on the same
500 fixed queries with inference seeds 1000/2000/3000. The three reports are
identical in decoded pose metrics (the seed only changes stochastic-sampler
bookkeeping), providing a useful reproducibility check.

| Variant | XYZ chunk (mm) | XYZ end (mm) | Rot chunk (deg) | Rot end (deg) | Gripper end | Rot jerk (deg) | Median (ms) | Peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACT R50 V1 VAE, 30k | 14.39 | **23.19** | **2.603** | **4.359** | 0.1671 | **0.122** | **10.77** | 341 |
| ACT R50 V2 VAE, 30k | 14.90 | 23.65 | 2.677 | 4.390 | 0.1602 | 0.098 | 9.89 | 341 |
| ACT R18 VAE, 30k | 18.30 | 27.50 | 3.249 | 5.516 | 0.1662 | 0.126 | 7.13 | 267 |

Relative to the R18 VAE control, R50-V1 improves chunk XYZ by 21.4% (paired
episode bootstrap 95% CI 17.7--24.8%), endpoint XYZ by 15.7% (11.1--19.8%),
chunk rotation by 19.9% (15.9--23.7%), and endpoint rotation by 21.0%
(16.4--25.3%). Relative to R50-V2, however, the differences are small and
their episode intervals cross zero for endpoint and chunk pose errors. V1 is
slightly better on all four pose means at this budget, but has higher
rotational jerk and a slightly worse gripper endpoint. This means the earlier
R50-V2 gain cannot yet be credited to width alone: both R50 recipes beat R18,
while V1 versus V2 is effectively tied at 30k and still mixes initialization,
VAE details, and finite-budget optimization.

The R50-V1 100k continuation has since completed training and its exact 100k
decoded evaluation. It is the strongest deterministic ACT pose accuracy at the
full seed-1000 budget: 13.72 mm XYZ chunk, **22.33 mm XYZ endpoint**, 2.623°
rotation chunk, **4.584° rotation endpoint**, 0.1435 gripper endpoint, 0.056°
rotational jerk, and 0.441 mm XYZ jerk. Relative to the R18-VAE 100k control it
improves endpoint XYZ by **9.8%** (paired episode CI 5.1--14.3%) and endpoint
rotation by **6.3%** (1.5--11.1%), and chunk XYZ by 16.3%. Crucially it also
edges the R50-V2 recipe at the same budget (22.33 vs 24.44 mm endpoint XYZ,
4.584 vs 4.875° endpoint rotation; chunk XYZ 13.72 vs 14.28 mm). At 30k V1 and
V2 had been tied (Section 9.1.2); at 100k the V1 initialization is at least as
good as V2 on every pose metric. Because V1 holds the ImageNet-initialization
family fixed against the R18/R34 controls, this means the ResNet-50 capacity
gain is **not** an artifact of the torchvision-recommended V2 weights — the
capacity attribution survives the strict initialization control at full budget.
(Latency is omitted here because this checkpoint was evaluated under heavy
multi-job GPU contention; the contention-independent decoded accuracy above is
the valid comparison.) The V1-vs-V2 paired interval is not emitted by the
summary's R18-anchored pairing, so the small V1 edge over V2 is reported
descriptively pending the multi-seed hierarchical interval.

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

The architecture-matched ACT-DP control has since completed its fixed 100k
budget (seed 1000, three inference seeds) and confirms that the 30k deficit does
not close with longer training. At 100k it decodes to 18.43 mm XYZ chunk, 28.08
mm XYZ endpoint, 3.092° rotation chunk, 5.207° rotation endpoint, 0.442°
rotational jerk, and 2.187 mm XYZ jerk — better than its own 30k policy
(38.70 mm endpoint XYZ) but still the weakest objective. Relative to ACT-L1 at
the same budget it is **18.5% worse in endpoint XYZ** (paired episode CI −24.4
to −12.8%), **28.5% worse in chunk XYZ** (−33.7 to −23.5), and 7.4% worse in
endpoint rotation (−12.9 to −2.0); all three intervals exclude zero. Relative to
matched ACT-flow 100k it is 7.7% worse in endpoint XYZ (−12.1 to −3.5) and tied
in endpoint rotation (−6.5 to +2.2, interval crossing zero). Thus at the full
seed-1000 budget, replacing rectified-flow velocity regression with epsilon/DDIM
diffusion inside the identical learned ACT denoiser does not rescue the
generative path; if anything ACT-DP remains slightly worse than even matched
flow on translation. This sharpens the Section 9.3 conclusion: the weakness is
specific to the time-conditioned ACT-transformer denoiser/conditioning path, not
to the flow objective itself (standard temporal-U-Net DP and the released UMI
transformer denoiser remain competitive).

There is nevertheless an important control-quality cost. ACT-L1 rotation/XYZ
jerk is 0.091 deg / 0.00073 m, matched flow is 1.093 deg / 0.00466 m, and DP is
0.481 deg / 0.00186 m; the ground-truth values are 0.158 deg / 0.00067 m.
Iterative generative samples are substantially less smooth at 30k. Flow is
4.5×, standard DP 3.5×, and ACT-DP 20.4× slower than ACT-L1 at inference;
ACT-DP's ten transformer denoising passes cost 137 ms median even though its
online peak allocation is only about 203 MiB.

![Accuracy and latency trade-off](figures/accuracy_latency_tradeoff.png)

### 9.2.1 100k vanilla flow and Diffusion Policy follow-up

The matched ACT-flow run then completed its fixed 100k budget. Its validation
velocity MSE was non-monotonic (0.052486, 0.047565, 0.040275, 0.040449,
0.039067, 0.046577, 0.045642, 0.039433, 0.042094, 0.044822 at 10k--100k).
The 50k checkpoint was the scalar minimum, but the exact 100k decoded policy
was better on the same queries: translation chunk/endpoint improved 9.58% and
12.26% (95% paired episode-bootstrap CIs 6.62--12.44 and 7.90--16.34),
rotation chunk improved 3.43% (0.49--6.21), rotational jerk 20.82%
(19.58--22.06), and XYZ jerk 7.89% (6.52--9.21); endpoint rotation and
gripper endpoint were tied. The final values were 15.655 mm / 25.995 mm XYZ,
2.965 degrees / 5.105 degrees rotation, 0.666 degrees rotational jerk, and
0.002963 m XYZ jerk. This is a second independent demonstration that held-out
velocity MSE alone is not a physical checkpoint selector.

The vanilla non-VLM ResNet18 Diffusion Policy completed its 100k budget as
well. Its within-family noise-MSE curve reached 0.007998 at 70k, then rose to
0.008607, 0.008592, and 0.008481 at 80k, 90k, and 100k. Nevertheless, exact
100k decoded control was better than the best-validation 70k checkpoint on
translation chunk/endpoint (5.84% and 2.31%, CIs 4.27--7.39 and 0.34--4.22),
rotation chunk/endpoint (11.37% and 6.31%, CIs 9.17--13.63 and 4.00--8.60),
rotational jerk (44.76%, 42.98--46.44), and XYZ jerk (44.09%, 42.25--45.85).
Gripper endpoint was tied. The final 100k policy measured 14.069 mm / 24.598
mm XYZ, 2.953 degrees / 5.251 degrees rotation, 0.187 degrees rotational
jerk, 0.000743 m XYZ jerk, 31.47 ms mean inference, and 346 MiB peak CUDA
allocation. Thus standard temporal-U-Net diffusion remains competitive with
ACT and improves substantially with the longer fixed budget, while its scalar
best checkpoint is not automatically its best decoded controller.

Together with the exact 30k transformer-versus-U-Net comparison above, the
100k follow-ups sharpen Q2: flow matching is not intrinsically the culprit,
and iterative diffusion is not intrinsically inferior to direct ACT. The
weakness is specific to some denoiser/conditioning/optimization combinations;
trajectory decoding and smoothness must be measured alongside objective loss.

### 9.2.2 Well-trained π0.5 flow-VLM reference (650K / 700K)

To test whether the matched ACT-flow deficit reflects flow matching itself or
only the small ACT-transformer denoiser, a well-trained flow-VLM — the π0.5 LoRA
run `pi05_openpi_split_lora_masked_1459_bs4_1m` — was evaluated on the same fixed
500-query common protocol at two checkpoints (650K and 700K of a 1M schedule),
each with three inference seeds (1000/2000/3000). Inference-seed variability is
negligible (±0.17 mm endpoint XYZ, ±0.01° rotation), so the means are tight.

**Configuration record (verified against the checkpoint's saved
`train_config.json`).** The port is the LeRobot/PyTorch `pi05` policy initialized
from the official `lerobot/pi05_base` weights (4.18B total parameters, "direct
port of the OpenPI implementation"). Its flow loss runs in **masked-subspace
mode** (`flow_matching_padding_mode=masked_subspace`: the velocity MSE is masked
to the active 10D action dims rather than taken full-width over the padded model
dim) — per the SmolVLA 1M padding A/B
(`examples/umi_relative_ee/OPENPI_FULL_WIDTH_FLOW_MATCHING.md`), the masked and
`openpi_full_width` modes are statistically equivalent, so this choice is not a
confound. Fine-tuning is **split-rank LoRA**: PaliGemma backbone r/α 16 +
`gemma_expert` action expert r/α 32 — the same rank split as official openpi's
`gemma_2b_lora` + `gemma_300m_lora` — with `action_in_proj`/`action_out_proj`/
`time_mlp` fully trainable and the vision tower frozen
(`freeze_vision_encoder=true`, `train_expert_only=false`): 38.6M trainable
(0.9%). Chunk/executed horizon 30/30; 10D UMI rot6d relative-EE actions with the
processor-derived 20D state; **quantile normalization for state and actions**
(the same family openpi uses, so normalization is matched across stacks, unlike
the ACT matrix's MIN_MAX). Optimization: LR 5e-5 cosine-decaying to 5e-6 over the
full 1M steps (1k warmup), batch 4, seed 1000, bf16 + gradient checkpointing, on
the same 1459_occlusion train set. At the evaluated 700K point the model had seen
2.8M samples ≈ 19.9 epochs. **2026-08-16:** training was resumed from the 700K
checkpoint to complete the full 1M schedule (300k steps at ~1.3 steps/s on the
RTX 5080, ETA ≈ 64 h; step/optimizer/scheduler state and the same W&B run
`ud42a4qb` restored via `resume_pi05_split_lora_kiwi_1459_1m.sh`); the 1M
checkpoint will be evaluated under the identical protocol when it lands.

| Checkpoint | XYZ end (mm) | Rot end (deg) | Rot jerk (deg) | Gripper end |
| --- | ---: | ---: | ---: | ---: |
| π0.5 LoRA 700K | **21.77 ± 0.17** | **4.25 ± 0.01** | **0.07** | 0.14 |
| π0.5 LoRA 650K | 21.97 ± 0.21 | 4.32 ± 0.01 | 0.08 | 0.14 |

Both π0.5 checkpoints beat every ACT and diffusion-policy variant in the matrix
on endpoint pose accuracy (next best: ACT R50-V1 100k at 22.33 mm / 4.58°), and
both are smoother than the ground-truth trajectory (rotational jerk 0.07–0.08°
versus GT 0.158°). The 650K→700K gain is small (0.2 mm, 0.07°), indicating the
flow-VLM has largely plateaued by 650K. This sharpens the Q2 conclusion: flow
matching is emphatically not the bottleneck — a well-trained flow-VLM is the
strongest and smoothest controller here, so the deficit of the matched ACT-flow
run (Section 9.2, ~26 mm) is attributable to the ACT-transformer
denoiser/conditioning/optimization recipe and its 100k budget, not to velocity
flow matching itself. Training budget is a first-order variable: the π0.5
reference used 6.5–7× the ACT variants' 100k steps, so the head-to-head endpoint
comparison must be read with that budget asymmetry in mind, and any future
matched-budget flow comparison should train the ACT-flow path substantially
longer before concluding flow is weak. (π0.5 700K prediction videos are under
`outputs/debug/viz_pi05_700k/`; raw metrics under
`eval_common_h32/pi05_openpi_split_lora_1459_{650k,700k}/`.)

### 9.2.3 Rotation-notation ablation: rot6d vs axis-angle (SmolVLA)

A separate question from the ACT/flow matrix is whether the **rotation
parameterization** of the relative-EE action matters. The UMI convention stores
rotation as a continuous 6D rep (two rows of the rotation matrix, "rot6d"), but
axis-angle (3D rotvec) is the native storage and is cheaper. SmolVLA was trained
to 100k steps at seed 1000 in two conditions that differ **only** in
`umi_rotation_representation` — `rot6d` (10D action: xyz + rot6d + gripper) versus
`axis_angle` (7D: xyz + rotvec + gripper) — holding the pretrained checkpoint,
action expert, flow objective, batch size, and 20D rot6d state bridge fixed. Both
were open-loop evaluated on the fixed 100-episode / 500-query validation protocol.

| Notation | XYZ end (mm) | Rot end (deg) | Rot jerk (deg) | XYZ jerk (mm) |
| --- | ---: | ---: | ---: | ---: |
| rot6d | 26.87 [25.28, 28.49] | 4.60 [4.36, 4.85] | 0.91 [0.89, 0.93] | 4.09 [4.00, 4.19] |
| axis-angle | 27.00 [25.44, 28.58] | 4.76 [4.49, 5.04] | **0.83 [0.81, 0.85]** | 4.12 [4.02, 4.21] |
| ground truth | — | — | 0.158 | 0.66 |

![Rotation notation across both stacks — endpoint accuracy ties, jitter effects
flip sign](figures/notation_cross_stack.png)

Endpoint accuracy is **statistically tied**: the 95% bootstrap intervals overlap
heavily on both XYZ (±1.6 mm around ~27 mm) and rotation (±0.25° around ~4.7°), so
rot6d is not measurably more accurate than axis-angle for SmolVLA here. The one
significant difference is **rotational jitter**: axis-angle is smoother
(0.83° vs 0.91°, disjoint intervals), although both remain ~5× the ground-truth
jerk (0.158°), so neither notation resolves SmolVLA's well-known jitter. The
result is mildly counter-intuitive — rot6d's continuity is often expected to
*reduce* jitter — and is most plausibly explained by the relative-EE actions being
near-identity (small rotations), where axis-angle's singularity at 180° never
manifests and the extra rot6d→rotvec decode step injects a small amount of noise.
Practical takeaway: **rotation parameterization is not a meaningful accuracy or
deployment lever for this task**; axis-angle is preferred for its smaller action
dimension and marginally smoother output. (An independent replication on the
official openpi π0.5 LoRA path — rot6d vs rotvec, JAX — is in progress to test
whether the conclusion transfers to a flow-VLM; see §9.2.4. Raw
SmolVLA metrics under
`outputs/research_report/smolvla_notation_eval_20260814/`.)

### 9.2.4 Independent replication on official openpi π0.5 (rot6d vs rotvec) — awaiting results

The SmolVLA tie above leaves open whether the conclusion transfers to a larger
flow-VLM trained with the **official openpi stack** (JAX/Flax, Physical
Intelligence's own π0.5 LoRA recipe) rather than our LeRobot ports. A controlled
replication is therefore running as an independent experiment: strawberry-1459
fine-tuned on **official openpi** with LoRA (`gemma_2b_lora` rank 16 on the
PaliGemma backbone + `gemma_300m_lora` rank 32 on the action expert, initialized
from the official `pi05_base` orbax checkpoint), in two arms differing **only** in
rotation notation:

- **rotvec arm** — 7D action (xyz + rotvec + gripper), matching the native storage;
- **rot6d arm** — 10D action (xyz + first-two-rows rot6d + gripper), matching the
  UMI convention used everywhere else in this report (same row convention as the
  LeRobot port, verified by an exact rot6d↔rotvec round-trip test).

Both arms share: identical data (episodes, frames, video), per-frame
`observation.state` derived from the action, no delta transform (the on-disk data
is already start-anchored relative), quantile normalization computed on the same
30k-frame sample, action horizon 10, batch size 16, 20 000 steps (~320k samples ≈
2.3 epochs, seed fixed by the JAX default), prompt "pick the strawberry",
checkpoints at 10k/20k. Feeding the data required converting the v3.0 LeRobot
layout to the v2.1 per-episode layout that openpi's pinned LeRobot reads
(per-episode parquet + mp4, jsonl metadata — script
`reshard_openpi_datasets_v21.py`); batch size is 16 because 32 exhausts the 24 GB
RTX 4090, and openpi's vision tower processes the zero-filled wrist-camera slots
(its standard single-camera pattern, as in LiberoInputs), which is accepted as the
honest openpi-method cost (2.4 s/step).

Evaluation will use the same decoded-metric protocol as §9.2.3 (fixed
100-episode / 500-query validation set, episode-balanced means with 95% bootstrap
intervals) via a purpose-built `eval_openpi_open_loop.py` whose rotation/jerk
math was verified to match `eval_open_loop_dataset.py` exactly; rot6d-arm outputs
are decoded to rotvec before scoring so both arms are compared in the same
units.

**Status (2026-08-16): complete.** Both arms trained to 20 000 steps (bs 16,
~13.4 h each) and were evaluated on the fixed 100-episode / 500-query protocol.

| Notation | XYZ end (mm) | Rot end (deg) | Rot jerk (deg) | XYZ jerk (mm) | Latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| rotvec (7D) | 10.05 [9.44, 10.70] | 1.66 [1.57, 1.75] | 0.20 [0.20, 0.21] | **0.92 [0.89, 0.94]** | 0.11 |
| rot6d (10D) | 9.41 [8.89, 9.94] | 1.69 [1.61, 1.78] | **0.16 [0.16, 0.17]** | 0.97 [0.95, 1.00] | 0.11 |
| ground truth | — | — | 0.153 | 0.65 | — |

![Notation comparison across both stacks](figures/notation_cross_stack.png)

![Horizon-matched (10-step) endpoint vs samples seen — all stacks tied; openpi ~9× more sample-efficient](figures/openpi_budget_context.png)

**Notation read-out (preregistered).** Endpoint accuracy is again
**statistically tied** — the position intervals overlap heavily (rot6d's point
estimate is 6% lower but well inside rotvec's interval) and rotation is
essentially identical — replicating §9.2.3 and making "rotation parameterization
is not a meaningful accuracy lever for this near-identity relative-EE task" a
two-stack finding (PyTorch/SmolVLA and JAX/openpi-π0.5). The jitter effects are
small, significant, and **stack-specific in direction**: here rot6d is smoother
in rotation (0.16° vs 0.20°, disjoint intervals — and essentially at the
ground-truth 0.153°) while rotvec is smoother in translation (0.92 vs 0.97 mm);
SmolVLA showed the opposite rotation direction (axis-angle smoother). With
opposite signs across two stacks, neither notation's jitter advantage is a
robust property — it is an interaction with the surrounding training stack, not
an intrinsic effect of the representation.

**Absolute-performance comparison — corrected after a horizon confound.** The
openpi arms' raw endpoints (9.4–10.1 mm at horizon 10) initially appeared 2.2–2.7×
better than ACT, the lerobot-port π0.5, and SmolVLA (all horizon 30). **That
comparison was confounded**: endpoint error is evaluated at t+10 versus t+30, and
error grows with look-ahead. Prompted by the prediction videos (whose qualitative
quality looked comparable to the port's), the port 700K checkpoint was re-scored
with chunks truncated to the first 10 steps (`--eval_horizon 10`, three inference
seeds, SD 0.025 mm):

| Horizon-10 endpoint | XYZ end (mm) | Rot end (deg) | Rot chunk-mean (deg) | Rot jerk (deg) |
| --- | ---: | ---: | ---: | ---: |
| SmolVLA rot6d 100k (bs8, 800k samples) | **8.97 [8.52, 9.45]** | 1.69 [1.60, 1.78] | 0.92 | 0.55 |
| π0.5 port 700K (bs4, 2.8M samples; masked-subspace flow, split-LoRA r16/r32) | **8.98 [8.40, 9.57]** | **1.61 [1.54, 1.71]** | **0.85** | **0.072** |
| openpi rot6d 20k (bs16, 320k samples) | 9.41 [8.89, 9.94] | 1.70 [1.61, 1.78] | 1.00 | 0.161 |
| openpi rotvec 20k (bs16, 320k samples) | 10.06 [9.44, 10.70] | 1.66 [1.57, 1.75] | 1.00 | 0.202 |

(ACT R50-V1 could not be re-scored at horizon 10: its weights were stranded by
the second artifact-disk failure — §8, incident 12 — and only its metric JSONs
survived. All its §9.1 numbers remain horizon-30.)

At matched horizon the port and both openpi arms are **statistically tied on
endpoint accuracy** (intervals overlap) — and so is SmolVLA. An earlier draft of
this section argued via per-step chunk-means that the gap was "not primarily a
horizon artifact"; that argument was wrong — averaging over a longer horizon
necessarily includes the growing-error tail, and at matched horizon the port's
chunk-mean (4.42 mm) is in fact *better* than openpi's (5.38 mm). The corrected
conclusions are:

1. **There is no cross-stack endpoint-accuracy gap at all.** SmolVLA (450M), the
   π0.5 port (3B, 2.8M samples), and official openpi π0.5 (3B, 320k samples) all
   land at 9–10 mm / 1.6–1.7° when scored at the same horizon. The earlier
   "2.3× at 1/35 steps" openpi headline — and by extension every cross-horizon
   endpoint comparison in this report's earlier tables — was the horizon
   artifact, and is retracted.
2. **What actually separates the stacks is smoothness and efficiency.** Rotation
   jitter: port 0.072° < GT 0.153° ≈ openpi rot6d 0.161° ≪ SmolVLA 0.55°. And
   the official recipe reaches the shared accuracy point with **~9× fewer
   samples** than the port (320k ≈ 2.3 epochs vs 2.8M ≈ 20 epochs) and ~2.5×
   fewer than SmolVLA — a real sample-efficiency win for compute-constrained
   fine-tuning, neutral for final accuracy.
3. Cross-horizon endpoint numbers must not be compared directly anywhere in this
   report; within-family comparisons (§9.1–9.2.3) share a horizon by
   construction and remain valid.
4. Verified recipe differences that remain untested as *accuracy or efficiency*
   levers: peak LR (2.5e-5 cosine vs the port's 5e-5), batch (16 vs 4), state
   construction (current-frame relative pose vs the port's processor-derived
   state), and the training stack itself (JAX bf16 vs PyTorch). The padded-dim
   loss treatment is **ruled out** by the `flow_matching_padding_mode` A/B (1M
   steps, statistically equivalent): it is a real difference — the port trains
   with masked-subspace loss (active 10D only; §9.2.2 config record) while
   official openpi effectively trains full-width over its padded action dim —
   but the A/B showed the two modes tie, so it cannot explain the ~9×
   sample-efficiency gap. The port's PEFT adapter coverage was
   *broader* than the official recipe's (vision tower included), so adapter
   capacity does not explain the efficiency gap either. (Raw metrics under
   `outputs/research_report/openpi_sroi_eval/`, `eval_common_h32/pi05_port_700k_h10/`,
   and `eval_common_h32/smolvla_rot6d_h10/`; checkpoints
   `~/codes/openpi/checkpoints/pi05_lora_sroi_{rotvec,rot6d}/run1/19999/`;
   prediction videos for validation episodes 0–2 of both arms under
   `outputs/debug/viz_openpi/{rot6d,rotvec}/`.)

### 9.2.5 Horizon-30 openpi arm + JAX-vs-PyTorch stack A/B — in flight

The horizon-matched correction (§9.2.4) left two attribution questions open: does
official openpi keep its behavior at the port's native 30-step horizon, and is the
remaining port-vs-openpi recipe difference (LR, batch, schedule shape) or the
training stack itself (JAX vs PyTorch)? Two arms were launched on the host RTX
4090 on 2026-08-16 to answer both:

- **Arm A — `pi05_lora_sroi_rot6d_h30` (official openpi, JAX):** identical to the
  §9.2.4 rot6d arm in every respect (same resharded v2.1 dataset and norm-stats
  file — per-dim quantiles are horizon-independent — same split LoRA r16/r32,
  bs16, 20k steps, default 2.5e-5 cosine-over-30k recipe, EMA off) except
  `action_horizon=30`. This is simultaneously (a) a like-for-like full-chunk
  comparison against every horizon-30 model in this report, and (b) the
  JAX half of the stack A/B.
- **Arm B — lerobot-port π0.5 LoRA with the openpi recipe (PyTorch):** the port
  trained with Arm A's hyperparameters, changing only the stack. Matched
  explicitly: rot6d 10D actions, chunk/execute 30/30, split-rank LoRA
  (PaliGemma r/α 16 + `gemma_expert` r/α 32, identical module regexes and
  full-training projections), `pi05_base` init with frozen vision tower, bs16,
  20k steps (= 320k samples), save 5k/keep 10k+20k, AdamW lr 2.5e-5 peak /
  betas (0.9, 0.95) / eps 1e-8 / **wd 1e-10** (overridden from the port's 0.01
  default) / grad-clip 1.0, warmup 1k + cosine over **30k steps to 2.5e-6
  stopping mid-cosine at 20k exactly like openpi**, full-width flow loss
  (`flow_matching_padding_mode=openpi_full_width`), Beta(1.5, 1) flow-time
  sampling, 224 px, quantile normalization.

Matching the schedule required one small code change: the port's
`CosineDecayWithWarmupSchedulerConfig` auto-scales warmup/decay into the
training length when `--steps < decay_steps` (it would have squeezed the 30k
cosine into 20k and ended at the LR floor, unlike openpi which stops
mid-cosine). A new `scheduler_auto_scale=false` knob (with a regression test
proving the no-scale path stays mid-cosine while the default reaches the floor)
restores verbatim openpi behavior; the smoke run confirmed warmup proceeding at
the full 1000-step scale in the real trainer.

Three stack-native differences remain and are recorded rather than removed
(they are part of what "the stack" means here): the port's processor-derived
20D two-pose state vs openpi's current-frame 10D state; norm stats over the
full 140k train frames vs openpi's 30k-frame sample (both q01/q99 quantiles on
the same distribution); and JAX-bf16 vs PyTorch-bf16 numerics — the last being
the object of the A/B. The port trains on the native v3.0 dataset layout,
openpi on the v2.1 reshard of the same frames.

**Smoke evidence (both arms, 12 steps, 2026-08-16):** openpi h30 fits bs16 in
~17.5 GiB after rematerialization (XLA reports it cannot go below this; the h10
arms were smaller) with a clean orbax save; the PyTorch port runs bs16/h30 with
gradient checkpointing without OOM. Steady-state throughput: JAX ~2.4 s/it,
PyTorch ~2.15 s/it → ETAs ≈ 13.5 h and ≈ 12 h, run sequentially by
`run_h30_stack_ablation_chain.sh` (openpi first, port behind it; both logs
under `/mnt/data1/projects/lerobot-arch-exp/logs/`). Checkpoints land in
`~/codes/openpi/checkpoints/pi05_lora_sroi_rot6d_h30/run1/19999/` (10k
intermediate deleted for disk) and
`/mnt/data1/projects/lerobot-arch-exp/outputs/train/pi05_port_openpi_args_rot6d_h30_bs16_20k/`.

**Analysis plan (preregistered).** Both final checkpoints will be evaluated on
the fixed 100-episode / 500-query protocol (`eval_openpi_open_loop.py` for the
JAX arm, `eval_open_loop_dataset.py` for the port, full 30-step chunks — no
horizon matching needed within this pair). A-vs-§9.2.4-rot6d isolates the
horizon effect inside official openpi at fixed budget; B-vs-A isolates
JAX-vs-PyTorch at matched recipe; B-vs-the-kiwi-1M-port isolates the recipe
(LR/batch/schedule/padding-mode) inside the PyTorch stack at matched horizon
— the 1M continuation (§9.2.2) also gives B a same-stack long-budget reference.

### 9.2.6 Partial-budget seed-2000/3000 salvage check (new L1/MSE metrics)

The dropped multi-seed confirmation (§8, incident 12) was partially recovered on
2026-08-17. A weights-level audit (checking for actual `.safetensors`, not
directory names) of the canonical artifact root found that the **entire seed-1000
matrix is unrecoverable** — every `train/<run>_seed1000_*/checkpoints/<step>/pretrained_model/`
directory is an empty skeleton (the failed disk's directory tree was copied, the
file contents were not), and the canonical `eval_common_h32/` metric JSONs for
the ACT matrix suffered the same fate (only the six kiwi π0.5-port JSONs
survive). The §9.2.4 statement that ACT R50-V1 "cannot be re-scored at horizon
10" therefore stands, the seed-1000 tables above are the surviving record of
those evaluations, and none of them can be extended with the L1/per-dim-MSE
metrics. (An initial name-level listing had suggested seed-1000 weights were
alive; the safetensors-level check overruled it. Loading a husk fails with
`draccus.ParsingError: Expected a dict with a 'type' key for PreTrainedConfig,
got {}`.)

Six seed-2000/3000 companion retrains — started on the healthy internal root
before the multi-seed phase was dropped — did retain real checkpoints: ACT-L1
seeds 2000/3000 at the full 100k budget, ACT R50-VAE seeds 2000/3000 stopped at
80k, and matched ACT-flow seeds 2000/3000 stopped at 50k. All six were evaluated
on 2026-08-17 with the updated evaluator (per-component L1 / per-dim MSE /
accuracy@τ, commits 51ff19f5 + this change) under the identical fixed
100-episode / 500-query protocol;
deterministic ACT at inference seed 1000, ACT-flow at inference seeds
1000/2000/3000 (inference-seed spread ≤0.7 mm endpoint, similar to π0.5's
±0.17 mm). Outputs live under `reeval_v2metrics/eval_common_h32/` (a shadow
artifact root that symlinks `train/`; the legacy tree was left untouched).

| Training seed | Budget | XYZ end (mm) | Rot end (°) | XYZ L1/dim (mm) | XYZ MSE/dim (µm²) | Rotvec L1/dim (°) | Rotvec MSE/dim (°²) | acc@0.5 | acc@0.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACT-L1 s2000 | 100k | 24.33 | 4.833 | 7.10 | 144.5 | 1.387 | 4.62 | 0.975 | 0.719 |
| ACT-L1 s3000 | 100k | 23.73 | 4.868 | 7.11 | 141.1 | 1.424 | 4.84 | 0.972 | 0.719 |
| ACT R50-VAE s2000 | 80k | 22.21 | 4.518 | 6.61 | 121.9 | 1.330 | 4.10 | 0.976 | 0.736 |
| ACT R50-VAE s3000 | 80k | 22.10 | 4.230 | 6.62 | 121.8 | 1.252 | 3.70 | 0.978 | 0.744 |
| ACT-flow s2000 | 50k | 31.42 | 5.70 | 9.96 | 236.6 | 1.74 | 6.49 | 0.963 | 0.634 |
| ACT-flow s3000 | 50k | 31.97 | 5.45 | 10.21 | 273.6 | 1.62 | 5.81 | 0.961 | 0.654 |

Read-outs, restricted to matched-step comparisons (budgets differ across the
set, so these rows are not comparable with the seed-1000 100k tables except
where noted):

1. **Variant rank order replicates across training seeds.** In both seeds,
   R50-VAE < ACT-L1 < ACT-flow on every one of the seven columns. The Q1
   capacity conclusion and the Q2 matched-flow deficit are not seed-1000
   artifacts. ACT-L1 at 100k lands at 23.7–24.3 mm across three seeds
   (seed-1000: 23.69 mm) — cross-seed SD ≈ 0.35 mm against between-variant
   gaps of 8–9 mm.
2. **The dropped multi-seed confirmation would not have changed any
   conclusion.** Training-seed spread (≤0.6 mm L1, ≤0.6 mm flow, ≤0.11 mm R50
   within pairs) is an order of magnitude smaller than every paired variant gap
   the report rests on — retroactively validating the compute-efficiency
   decision and the use of episode-bootstrap intervals.
3. **R50-VAE at only 80k (22.1–22.2 mm) already matches the best seed-1000
   100k ACT endpoints** (R50-V1 22.33 mm, ACT-L1 23.69 mm) — consistent with
   the §9.1.2 capacity attribution, though this is a cross-budget observation.
4. **ACT-flow at 50k sits at 31.4–32.0 mm in both seeds** versus ≈29.6 mm
   derived for seed-1000 at 50k (from the §9.2.1 paired 12.26% endpoint
   improvement to 25.995 mm at 100k): seed-1000 was the favorable draw, and the
   flow-vs-L1 deficit is, if anything, larger in the recovered seeds.
5. **The new metrics reorder nothing**: L1 and per-dim MSE rank the six runs
   identically on both translation and rotation. The MSE:L1 ratio separates the
   families (flow 24–27 vs L1 ≈ 20 vs R50 ≈ 18.4 µm/mm), i.e. flow's errors
   have a heavier tail, consistent with its measured roughness (§9.2).
6. **accuracy@τ sharpens the same picture**: at τ=0.5 (motion intent) all
   six runs cluster within 1.7 pp (0.961–0.978) — every variant captures the
   coarse motion — while at τ=0.1 (movement precision) the families separate by
   up to 11 pp (flow 0.634–0.654 vs ACT-L1 0.719 vs R50-VAE 0.736–0.744).
   Precision, not motion intent, is where the objectives differ — mirroring
   how π0.5 uses the two thresholds for transfer vs in-domain scaling trends.

Driver: `reeval_seed23k_v2metrics.sh` (idempotent, husk-guarded, VRAM-gated at
≥4 GiB free so it ran concurrently with the in-flight h30 chain); per-eval logs
under `reeval_v2metrics/logs/`. Pre-accuracy@τ copies of the same ten reports
are preserved under `reeval_v2metrics/eval_common_h32_pre_tau/`.

### 9.2.7 Historical production ACT: 30-point budget curve on the v2 metrics

A follow-up question — why accuracy@τ covered only three ACT variants — led to
a fresh weights-level audit on 2026-08-17 that re-confirmed the §9.2.6 husk
inventory (27 of 33 `train/` directories empty; exactly six real runs, all
already scored) but found that the **original production run**
`outputs/train/act_umi_identity_rot6d_1459` — the 3M-step R18-VAE model whose
audit motivated this investigation (§3) — is fully intact on the repository
disk, outside the failed artifact mount: all thirty 100k-spaced checkpoints
retain real weights. Every checkpoint from 100k to 3M was therefore evaluated
with the full v2 metric set under the identical fixed 100-episode / 500-query
protocol (driver `eval_historical_act_curve.sh`, launched as a VRAM-gated
backfill alongside the in-flight h30 chain; outputs
`reeval_v2metrics/eval_common_h32/act_umi_identity_rot6d_1459_<step>steps/seed1000/`).
This is the only ACT-family budget curve on the new metrics and the only
surviving model trained past 100k; at 3M steps × batch 8 it has seen 24M
samples (≈171 epochs of the 140,522-frame train set).

| Steps | XYZ end (mm) | Rot end (°) | XYZ L1/dim (mm) | acc@0.5 | acc@0.1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100k | 25.57 [23.64, 27.56] | 5.019 | 8.47 | 0.969 | 0.681 [0.667, 0.695] |
| 200k | 24.35 [22.60, 26.19] | 4.977 | 8.06 | 0.972 | 0.693 [0.679, 0.707] |
| 400k | 23.89 [22.19, 25.64] | 4.854 | 7.95 | 0.972 | 0.697 [0.682, 0.712] |
| 600k | 23.87 [22.07, 25.73] | 4.810 | 7.93 | 0.973 | 0.699 [0.685, 0.713] |
| 800k | 23.83 [21.99, 25.70] | 4.761 | 7.82 | 0.973 | 0.705 [0.691, 0.720] |
| 1M | 23.31 [21.53, 25.13] | 4.711 | 7.72 | 0.973 | 0.707 [0.693, 0.722] |
| 1.5M | 23.44 [21.69, 25.24] | 4.579 | 7.68 | 0.974 | 0.713 [0.698, 0.727] |
| 2M | 23.42 [21.62, 25.27] | 4.498 | 7.66 | 0.973 | 0.716 [0.701, 0.731] |
| 2.5M | 23.40 [21.52, 25.30] | 4.511 | 7.71 | 0.974 | 0.715 [0.700, 0.729] |
| 3M | 23.31 [21.45, 25.20] | 4.498 | 7.66 | 0.974 | 0.715 [0.700, 0.731] |

(The table shows ten milestones; the figure below plots all thirty points.
Rotvec L1/dim declines 1.469 → 1.320° over the same range; XYZ MSE/dim
189 → 160 µm².)

![Historical ACT 30-point budget curve](figures/historical_act_budget_curve.png)

Read-outs:

1. **acc@0.5 is budget-blind; acc@0.1 is the budget-sensitive metric.** Across
   a 30× step range acc@0.5 moves only 0.969 → 0.975 — less than its own
   interval half-width — while acc@0.1 rises 0.681 → 0.718 with the 100k and
   ≥1.4M intervals fully disjoint. This is a local, within-recipe confirmation
   of exactly the π0.5 doctrine (§2): τ=0.5 saturates at the motion-intent
   level, τ=0.1 resolves in-domain precision trends.
2. **Precision gains are front-loaded and bounded.** Roughly half of the total
   +3.4pp acc@0.1 gain arrives by 400k; from 1M on the curve is flat within
   its intervals (0.707–0.718). XYZ endpoint error plateaus at 23.3–23.9 mm
   from 400k — the 100k-vs-3M endpoint intervals even overlap slightly,
   whereas acc@0.1 separates them cleanly, making accuracy@0.1 the more
   sensitive early budget indicator in this family.
3. **Late training trades smoothness for marginal precision.** Within-chunk
   rotation jerk improves to a 0.7M minimum of 0.037° and then degrades ~45%
   to 0.054° at 3M, replicating the §3 pre-existing audit (best ≈0.036° around
   700k) and reinforcing its warning against training ACT far past the
   endpoint plateau.
4. **Protocol cross-validation.** The new fixed-protocol numbers agree with
   the §3 audit within 0.3–0.6 mm (25.57 vs 25.1 mm at 100k; 23.2 vs 22.9 mm
   at 2.3M), tying the v2-metric series to the historical record.
5. **Capacity beats a 30× budget on precision.** At 100k the historical
   R18-VAE scores acc@0.1 = 0.681, and even at 3M it never exceeds 0.718 —
   below the R50-VAE seed-2000/3000 companions at only 80k (0.736–0.744) and
   the ACT-L1 companions at 100k (0.719). A backbone/objective change at
   ≤1× the common budget outperforms a 30× budget range of the original
   recipe — the sharpest single line of evidence that the §9.1 capacity and
   objective results dominate longer training on this task.

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
and R50-V2 to determine whether these conclusions persist at the full 100k
budget (single training seed; variability quantified by per-episode bootstrap).

Stage two therefore trains fresh 100k runs (not scheduler-incompatible resumes)
for ACT R18 VAE, ACT R50 V2 VAE, ACT R18 L1, uniform ACT-flow 1e-5, and
Diffusion R18; the newly identified R50-V1 and architecture-matched ACT-DP
controls are inserted as separate 30k/100k successors before evaluation. Fresh
runs are required because Diffusion Policy's cosine scheduler was
constructed for 30k steps and had already reached its floor; extending that
optimizer state to 100k would not be equivalent to a 100k schedule. After the
100k screen, the surviving comparison was to be repeated at training seeds 2000
and 3000 to capture training-seed variability that the per-episode bootstrap
cannot measure. That multi-seed confirmation was started but ultimately **dropped
for compute efficiency** after two artifact-disk failures (§8, incident 12)
stranded the seed-2000/3000 checkpoints; the final recommendation therefore
rests on the single seed-1000 matrix with per-episode bootstrap intervals,
strengthened by the independently-trained π0.5 650K/700K flow-VLM reference
(§9.2.2), which is stable to ±0.17 mm endpoint XYZ across three inference seeds.
A future seed-2000/3000 iteration would tighten the intervals but is unlikely to
reverse the rank order given the size of the seed-1000 gaps and the consistency
of the π0.5 reference — a prediction the §9.2.6 partial-budget salvage check
subsequently confirmed directly: two recovered seeds each for ACT-L1, R50-VAE,
and matched flow replicated the rank order on every metric, with cross-seed
spreads an order of magnitude smaller than the variant gaps.

**Final recommendation (seed-1000 basis).** Endpoint-pose accuracy at matched
100k budgets ranks: ACT-L1 ≈ ACT R50-V1 (best of the ACT/diffusion family,
~22 mm endpoint; R50-V1's gain over R18 survives the strict V1-initialization
control, so ResNet-50 capacity — not the torchvision V2 weights — is the cause)
> ACT R50-V2 > standard temporal-U-Net Diffusion Policy > matched ACT-flow >
ACT-DP (worst). The matched ACT-flow and ACT-DP deficits are attributable to
the ACT-transformer denoiser/conditioning recipe, **not** to flow/diffusion per
se: standard DP is competitive with ACT, and a well-trained π0.5 flow-VLM
(§9.2.2, 21.8 mm endpoint, smoother than ground truth) is the strongest
controller of all — confirming flow matching is sound when paired with a
capable denoiser and sufficient training budget (π0.5 used 6.5–7× the ACT
budget, so training budget is a first-order variable the matched 100k
comparison does not control). Practical defaults: **ACT-L1** as the lightweight
deterministic controller (lowest inference cost, smoothest ACT trajectory);
**ACT R50-V1** when the extra ~25% latency is acceptable for a small pose
gain; and the **flow-VLM path (π0.5 / openpi)** when VLM inference cost is
justified by its clear accuracy and smoothness lead — with two corrections from
the horizon-matched re-scoring (§9.2.4): at equal 10-step scoring the π0.5 port,
official openpi, and even SmolVLA are **statistically tied at 9–10 mm endpoint**
(the earlier cross-model endpoint spreads were a horizon artifact), so the VLM
path's advantage on this task is **smoothness and sample efficiency, not final
endpoint accuracy** — the port is the smoothest controller measured (0.072°
rot-jerk vs GT 0.153°) and official openpi reaches the shared operating point
with ~9× fewer samples. Rotation parameterization,
by contrast, is **not** a lever worth spending on: the SmolVLA rot6d-vs-axis-angle
ablation (§9.2.3) found endpoint accuracy statistically tied, and the
official-openpi replication (§9.2.4) reproduced that tie on a second stack —
with the small jitter differences flipping sign between stacks, confirming they
are stack interactions rather than representation properties. rotvec/axis-angle
(7D) remains the practical default for its smaller action dimension. Note
finally that ACT
trajectory smoothness (rotational jerk 0.056°–0.091°) is comparable to or below
the ground-truth jerk (0.158°), so the iterative generative samplers' roughness
is not a reason to avoid ACT's deterministic path here.

The stage-two sequence was launched on the host RTX 4090 at 2026-08-11 20:41
Asia/Taipei in tmux session `umi_arch_stage2_20260811`. Its first run,
`act_r18_vae_seed1000_100000steps`, initialized successfully at about 26.7
steps/s. Checkpoints/logs were originally under
`/media/zfei/Glowat512/projects/lerobot-arch-exp` and, after that disk failed
twice, live under `/mnt/data1/projects/lerobot-arch-exp` (§8, incident 12). This was an active long-run
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

At 21:02 on 2026-08-12, the same measured-headroom rule was applied to the
capacity-control queue: R50-V1 100k was using about 4.5 GiB CUDA and the card
had over 19 GiB free, so an independent R18-VAE seed-2000 confirmation was
started rather than leaving the device mostly idle. After startup the two
trainers together allocated about 7.6 GiB, reached 100% device utilization at
about 363 W, and retained more than 16 GiB free. R50-V1 was at 13.4k/100k and
R18 seed-2000 at 1.8k/100k at the latest check; both had finite losses and no
CUDA, PyAV, or native-worker errors. R50 slowed from roughly 13 to 7 steps/s
under contention, while R18 remained near 12--13 steps/s. This is an
intentional throughput/wall-clock trade-off, not a change to either scientific
configuration. Completion predicates prevent the later confirmation supervisor
from retraining this seed.

At 22:23 on 2026-08-12, the measured-headroom rule was extended to front-run the
seed-2000 half of the confirmation matrix while the confirmation supervisor
remains gated hours behind capacity control and evaluation. With R50-V1 100k
(the protected primary) at ~46k/100k using ~4.6 GiB, the existing R18-VAE
seed-2000 companion at ~60k/100k using ~2.7 GiB, and ~14 GiB free, two
additional companion queues were launched through a new `run_companion.sh`
wrapper that mirrors `supervise_remaining.sh`'s recovery contract (bounded
4→2→0 worker retry, interrupted-attempt preservation, and
`recover_training_completion`). Queue A trains `act_r18_l1` then
`act_r18_flow_u_lr1e5`; queue B trains `act_r18_diffusion_lr1e5` then
`diffusion_r18` (the temporal U-Net retains its two-worker setting). Each writes
to the canonical `train/<variant>_seed2000_100000steps` path, so the
confirmation supervisor's completion predicate will skip whatever finishes
first. Combined allocation stabilized near 13.0/24.6 GiB at 100% utilization
with ~11 GiB free; the four jobs sustained finite losses with no CUDA or
native-worker errors. Under four-way contention the R18-class companions ran at
~6--7 step/s (versus ~12--13 solo) and the protected R50-V1 primary slowed from
~7 to ~3.9 step/s. This is an acceptable total-throughput trade-off because the
~41 GPU-hour confirmation phase is the dominant cost and is now overlapping the
~8 GPU-hour seed-1000 finish; the contention also eases naturally once the
R18-VAE seed-2000 companion completes (~60k/100k at launch). The R50
Q1 companions (`act_r50_vae`, `act_r50_v1_vae`) were deliberately deferred until
the R50-V1 primary frees the card, to avoid double-R50 contention on the
protected seed-1000 milestone. This front-running reorders wall-clock
scheduling only; it changes no scientific configuration, and the
~35-hour confirmation phase now overlaps the seed-1000 completion phase rather
than following it serially.

The front-running was then generalized to cover the entire confirmation matrix
through five self-backfilling companion queues (`run_companion.sh` with a new
`wait_for_slot` gate). The gate bounds total concurrent training jobs — counted
as distinct `job_name=` values, since PyAV dataloader workers inherit the
parent command line — to four (the GPU saturation point) and requires ≥4 GiB
free VRAM before starting another, so several waiting queues cannot
oversubscribe the card or OOM the protected primary. As soon as any in-flight
job finishes, the next waiting queue starts within 30 s, closing idle gaps
without manual launch timing. The five lanes partition all 14 confirmation runs
disjointly: existing R18-VAE s2000; queue A `act_r18_l1`/`act_r18_flow_u_lr1e5`
s2000; queue B `act_r18_diffusion_lr1e5`/`diffusion_r18` s2000; queue C the four
R50 runs (`act_r50_vae`, `act_r50_v1_vae` × seeds 2000/3000); queue D the five
R18 seed-3000 runs. Each writes the canonical path, so the confirmation
supervisor's completion predicate skips whatever a companion finishes first. To
prevent the eventual confirmation supervisor from archiving an in-progress
companion directory and starting a conflicting run, `supervise_confirmation_training.sh`
gained an `is_running` guard: before archiving or training any run it waits (up
to a 12 h valve) for any live process already training that exact run, then
re-checks completion — so the supervisor inherits a companion's result rather
than overwriting it. The parked confirmation-training session was restarted to
load this guarded script; its dependency wiring (waits for evaluation, is waited
on by confirmation-evaluation) is unchanged. This changes only wall-clock
scheduling and queue coordination; no scientific configuration is altered.

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

That exact-step comparison is now complete for inference seeds 1000, 2000, and
3000. All three reports identify `checkpoints/030000/pretrained_model` in both
their filename and payload. Averaged across inference seeds, the transformer
decoded to 14.499 mm translation chunk error, 25.069 mm translation endpoint
error, 2.727 degrees rotation chunk error, 4.883 degrees rotation endpoint
error, 0.1742 gripper endpoint error, 0.2373 degrees rotation jerk, and 0.000964
m translation jerk. In the registered episode-paired comparison against the
official U-Net, the transformer improved translation chunk error by 10.15%
(95% CI 6.21--13.95), translation endpoint by 12.83% (7.80--17.64), rotation
chunk by 15.80% (12.17--19.36), rotation endpoint by 16.36% (11.65--20.82),
rotation jerk by 20.89% (18.18--23.39), and translation jerk by 19.42%
(16.68--21.97). Gripper chunk and endpoint intervals crossed zero. Mean
inference latency ranged from 51.5--58.2 ms for the transformer versus
47.2--49.2 ms for the U-Net, while measured peak CUDA allocation was slightly
lower (1,186.8 versus 1,276.5 MiB).

This isolation falsifies a simple "transformer denoisers are the problem"
explanation for the earlier flow/diffusion deficit: under the released UMI
ViT-token recipe, the transformer is consistently more accurate and smoother
than the official U-Net, with only a modest latency penalty. It also falsifies
the converse claim that the stochastic family is uniformly superior to ACT:
against the matched-budget ACT-L1 control, the transformer improves translation
chunk and endpoint errors by 19.03% and 11.03%, but worsens rotation jerk by
161.68% and translation jerk by 31.45%; rotation endpoint and gripper endpoint
remain tied. The evidence therefore points to an objective/architecture/
optimization interaction and a real accuracy--smoothness--latency trade-off,
not an intrinsic failure of flow matching itself or a single VLM effect.

The architecture-matched ACT-flow extension also completed its pre-registered
100k budget. Its held-out velocity MSE followed a non-monotonic trajectory:
0.052486, 0.047565, 0.040275, 0.040449, 0.039067, 0.046577, 0.045642,
0.039433, 0.042094, and 0.044822 at 10k increments. Thus 50k was the scalar
validation minimum even though terminal training loss continued downward near
0.027--0.035. The exact 100k checkpoint passed the complete durability audit
(139,010,496-byte model, 277,853,520-byte optimizer, exact step, RNG,
processors/configs, and `End of training`). Its canonical seed-1000 evaluation
loaded the 100k checkpoint, while a provenance-separated sensitivity run under
`eval_checkpoint_h32` explicitly loaded step 50k; the latter remains excluded
from primary tables and figures.

On identical episodes and query offsets, decoded trajectory quality contradicted
the scalar validation ranking. Relative to 50k, the 100k checkpoint improved
translation chunk error by 9.58% (95% paired episode-bootstrap CI 6.62--12.44),
translation endpoint by 12.26% (7.90--16.34), rotation chunk by 3.43%
(0.49--6.21), rotation jerk by 20.82% (19.58--22.06), and translation jerk by
7.89% (6.52--9.21); rotation endpoint and gripper endpoint intervals crossed
zero. The final decoded values were 15.655 mm translation chunk, 25.995 mm
translation endpoint, 2.965 degrees rotation chunk, 5.105 degrees rotation
endpoint, 0.1492 gripper endpoint, 0.666 degrees rotation jerk, and 0.002963 m
translation jerk. This is direct evidence that held-out flow velocity MSE is
not a sufficient checkpoint-selection proxy for physical trajectory accuracy
or smoothness. Fixed-budget reporting plus decoded checkpoint sensitivity is
therefore retained for the remaining stochastic policies.

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
