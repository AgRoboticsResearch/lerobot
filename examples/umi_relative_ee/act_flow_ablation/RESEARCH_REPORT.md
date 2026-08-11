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
| `act_r18_vae` | exact 1459 early-budget replication | 52M | 30k + eval complete |
| `act_r34_vae` | backbone-only scale | 62M | 30k + eval complete |
| `act_r50_vae` | backbone-only scale | 65M | 30k + eval complete; promoted |
| `act_r50_large` | ResNet-50 + 768-wide, 6e/3d transformer | 145M | 30k + eval complete; not promoted |
| `act_r18_l1` | no-VAE deterministic objective control | 34M | 30k + eval complete; promoted |
| `act_r18_flow_u_lr1e5` | exact-LR, uniform-time flow control | 35M | 30k + eval complete; promoted |
| `act_r18_flow_u_lr1e4` | flow optimizer sensitivity | 35M | 30k + eval complete; rejected |
| `act_r18_flow_beta_lr1e4` | OpenPI-like time bias | 35M | 30k + eval complete; rejected |
| `diffusion_r18` | standard non-VLM diffusion control | 75M | 30k + eval complete; promoted |

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
Synchronized policy-only GPU latency is measured on the same queries, excluding
the first cold call; mean, median, p95, and peak allocated inference memory are
recorded alongside accuracy. This makes the cost of larger backbones and
iterative samplers explicit.

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
- 48 targeted CPU tests pass, 9 hardware/optional tests skip. These cover the
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
8. The first evaluation pass chose valid queries from `chunk_size=30`. That is
   correct for ACT's offsets `[-1, 0, ..., 29]`, but Diffusion Policy requests
   `[-1, 0, ..., 31]`; its first late query was therefore padded and the launcher
   stopped after 14 reports. The evaluator now accepts and records explicit
   offset bounds, the fixed launcher uses the common `[-1, 31]` intersection,
   and a regression test checks exact frame indices. The 14 earlier reports are
   preserved under `eval/` as superseded evidence. All scientific results use
   the clean rerun under `eval_common_h32/`.

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

The first backbone-only comparison is provisionally favorable to ResNet-34:

| ACT backbone | Parameters | Median update | 10k val | 20k val | 30k val |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 51,579,786 | 0.036 s | 0.054130 | 0.043604 | 0.041139 |
| ResNet-34 | 61,680,522 | 0.048–0.049 s | 0.051064 | 0.043164 | 0.039170 |
| ResNet-50 | 64,654,218 | 0.075 s | 0.042517 | 0.037207 | 0.036259 |

ResNet-34 reduces total validation loss by 5.7%, 1.0%, and 4.8% at the three
budgets; its 30k L1 is 0.037265 versus ResNet-18's 0.039285 (5.1% lower). It
reduces update throughput by roughly 25%. The decoded results below confirm the
capacity signal.

ResNet-50 is a stronger signal: its 10k total is 16.7% below ResNet-34
and 21.5% below ResNet-18, while its L1 (0.035436) is lower by 13.3% and 15.1%
respectively. At 20k its total (0.037207) remains 13.8% below ResNet-34 and
14.7% below ResNet-18, with L1 (0.034470) about 14% lower than both. It is also
2.1× slower per update than ResNet-18. At 30k its total (0.036259) is 7.4%
below ResNet-34 and 11.9% below ResNet-18; L1 (0.034574) is 7.2% and 12.0%
lower. The decoded metrics confirm that the gain survives in physical units.

The first transformer-scaling point is unfavorable at the baseline optimizer:
the 145M ResNet-50 + 768-wide 6e/3d model records 10k total 0.053834 and L1
0.037560, versus 0.042517 and 0.035436 for backbone-only ResNet-50. It is also
about 1.3× slower than that model and 2.6× slower than ResNet-18. Because the
larger model's training curve descends more slowly, this result tests equal-LR
architecture scaling; it does not rule out a higher-LR large-model variant.
By 30k the large model nearly catches up but still does not win: total 0.036617
versus 0.036259, and L1 0.035273 versus 0.034574 for backbone-only ResNet-50.
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

### 9.1 Decoded physical metrics at 30k

All rows below use the corrected common 500-query set. Generative rows average
inference seeds 1000/2000/3000 within each episode before averaging episodes.
Latency is synchronized policy-only median latency; memory is peak allocated
CUDA memory. Lower is better throughout.

| Variant | XYZ chunk (mm) | XYZ end (mm) | Rot chunk (deg) | Rot end (deg) | Median (ms) | Peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACT R18 VAE | 18.30 | 27.50 | 3.249 | 5.516 | 7.13 | 267 |
| ACT R34 VAE | 17.36 | 25.65 | 3.147 | 4.947 | 8.55 | 305 |
| ACT R50 VAE | 14.90 | **23.65** | 2.677 | **4.390** | 9.89 | 341 |
| ACT R50 large | **14.57** | 23.83 | **2.650** | 4.462 | 11.51 | 653 |
| ACT R18 L1 | 17.91 | 28.18 | 3.143 | 5.117 | **6.70** | 200 |
| ACT-flow uniform, 1e-5 | 18.75 | 30.86 | 3.767 | 6.290 | 29.90 | 203 |
| ACT-flow uniform, 1e-4 | 37.12 | 57.94 | 5.184 | 7.070 | 29.87 | 203 |
| ACT-flow beta, 1e-4 | 34.59 | 59.13 | 3.810 | 5.689 | 29.62 | 203 |
| Diffusion R18 | 15.71 | 27.27 | 3.391 | 5.838 | 23.23 | 345 |

Paired episode bootstrap comparisons (10,000 resamples) establish:

- ResNet-34 versus R18 improves XYZ endpoint by 6.7% (95% CI 1.5–11.7%)
  and rotation endpoint by 10.3% (4.5–15.9%).
- ResNet-50 versus R18 improves XYZ endpoint by 14.0% (9.6–18.2%), rotation
  endpoint by 20.4% (15.3–25.3%), XYZ chunk mean by 18.6% (14.9–22.0%), and
  rotation chunk mean by 17.6% (13.3–21.7%). All four paired difference
  intervals exclude zero.
- R50-large versus backbone-only R50 is tied on all four pose metrics: for
  endpoint XYZ its improvement is -0.8% (CI -6.0–4.2%), and for endpoint
  rotation -1.6% (-8.0–4.2%). It adds 80.3M parameters, 16% inference latency,
  and 312 MiB peak memory without a supported accuracy gain.
- ACT-L1 versus ACT-VAE is tied in XYZ but improves endpoint rotation by 7.2%
  (2.1–12.0%). It is the fastest and smallest ACT control.

The R50 result is therefore not merely a lower training loss: it is a sizable,
statistically supported decoded-pose improvement. Scaling the visual backbone
is the successful intervention; scaling the already-large transformer at the
same optimizer is not.

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

There is nevertheless an important control-quality cost. ACT-L1 rotation/XYZ
jerk is 0.091 deg / 0.00073 m, matched flow is 1.093 deg / 0.00466 m, and DP is
0.481 deg / 0.00186 m; the ground-truth values are 0.158 deg / 0.00067 m.
Iterative generative samples are substantially less smooth at 30k. Flow is
4.5× and DP 3.5× slower than ACT-L1 at inference, although both remain below
30 ms median on the RTX 4090.

### 9.3 Answers and promotion decision after stage one

**Q1:** yes. A ResNet-50 backbone is the strongest tested ACT improvement over
the fresh 1459 control. ResNet-34 is a smaller positive step; the 145M widened
transformer is not worthwhile at the tested LR/budget. The claim is currently
an offline 30k claim, so R18 and R50 are promoted to a longer budget before
recommending replacement of the multi-million-step historical checkpoint.

**Q2:** no single explanation fits. Matched ACT-flow is significantly worse
than the architecture-matched L1 policy, so that flow formulation/sampler needs
work independent of any VLM. But vanilla DP without a VLM is competitive with
ACT and better on chunk translation, so generative modeling itself is not the
fundamental problem. VLM fine-tuning, objective/sampler design, and trajectory
smoothness are separate axes; the existing π0.5 result further indicates that
the VLM path can work. ACT-L1, uniform flow 1e-5, and DP are promoted with R18
and R50 to determine whether these conclusions persist at 100k.

Stage two therefore trains fresh 100k runs (not scheduler-incompatible resumes)
for ACT R18 VAE, ACT R50 VAE, ACT R18 L1, uniform ACT-flow 1e-5, and Diffusion
R18. Fresh runs are required because Diffusion Policy's cosine scheduler was
constructed for 30k steps and had already reached its floor; extending that
optimizer state to 100k would not be equivalent to a 100k schedule. After the
100k screen, the surviving comparison will be repeated with training seeds
2000 and 3000 so that training-seed variability, which episode bootstrap cannot
measure, is included in the final recommendation.

The stage-two sequence was launched on the host RTX 4090 at 2026-08-11 20:41
Asia/Taipei in tmux session `umi_arch_stage2_20260811`. Its first run,
`act_r18_vae_seed1000_100000steps`, initialized successfully at about 26.7
steps/s. Checkpoints/logs remain under
`/media/zfei/Glowat512/projects/lerobot-arch-exp`; this is an active long-run
confirmation and is not included in the completed 30k table above.

## 10. Reproduction

The variant launcher is `run_one.sh` in this directory. `run_stage1.sh` executes
the fixed matrix sequentially so models do not contend for the single GPU, and
`evaluate_one.sh` resolves exactly one final checkpoint and evaluates it without
manual path selection, while `evaluate_stage1.sh` fixes the deterministic and
three-seed generative matrix. `collect_results.py` extracts parameter counts, wall
times, complete validation curves, decoded metrics, and confidence intervals
into compact external CSV/JSON files without creating a second narrative doc.
`run_stage2.sh` and `evaluate_stage2.sh` encode the five promoted 100k controls.
Example:

```bash
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_vae 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_flow_u_lr1e4 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh diffusion_r18 30000 1000
```

The launcher refuses to overwrite an existing run. Full command-line configs
are also saved inside every checkpoint's `train_config.json`, and stdout/stderr
is retained in the artifact workspace.
