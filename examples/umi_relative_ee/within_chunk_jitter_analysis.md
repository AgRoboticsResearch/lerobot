# Within-chunk action jitter: ACT vs flow matching

> **Status:** inference fix implemented after mathematical audit, updated 2026-07-31.
> Companion to [`rot6d_identity_norm_experiment.md`](./rot6d_identity_norm_experiment.md).
>
> **Superseded runtime:** the masked-inference experiment documented below is
> historical. SmolVLA and π0.5 now follow the full-width OpenPI contract in
> [`OPENPI_FULL_WIDTH_FLOW_MATCHING.md`](./OPENPI_FULL_WIDTH_FLOW_MATCHING.md).

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
