---
conversation_id: current-codex-session
created: 2026-07-29
tags:
  - lerobot
  - smolvla
  - act
  - umi-relative-ee
  - action-smoothness
  - debugging
---

# SmolVLA Action Smoothness Investigation

## Overview

This investigation compared SmolVLA and ACT action chunks on the same UMI
relative end-effector validation data. The observed SmolVLA zig-zag is present
in the raw model prediction: it is not introduced by visualization, absolute
pose reconstruction, or the robot controller.

The strongest finding is that independent Gaussian action noise is the primary
source of high-frequency variation. Supplying zero initial noise made both
tested SmolVLA checkpoints substantially smoother without materially degrading
their validation endpoint errors. Zero-noise inference is nevertheless outside
the model's trained Gaussian prior and must be validated on the robot before it
is considered a deployment solution. The current decision is **not to add zero-noise inference yet**; retain it only as diagnostic evidence.

No training process was interrupted and no model, dataset, or source file was
changed during the investigation.

## Experiment

### Data and checkpoints

- Validation dataset:
  `sroi/sroiv2_strawberry_picking_lab_validation`
- Sample count: 161 valid frames from 3 episodes
- Chunk size: 30
- ACT:
  `outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/0100000/pretrained_model`
- SmolVLA frozen/expert-only: kiwi checkpoint 500K
- SmolVLA unfrozen/full-model: kiwi checkpoint 200K
- Device: CUDA

The SmolVLA checkpoints were copied temporarily from kiwi for inference and
deleted locally when the comparison finished. The originals on kiwi were not
modified.

### Metrics

- **Roughness**: mean norm of the second finite difference of XYZ targets
  inside each 30-step chunk, in millimeters. Lower is smoother.
- **Path ratio**: predicted path length divided by endpoint displacement.
  A straight path is approximately 1; a large value indicates zig-zag.
- **First-step jump**: distance between the chunk base and target 0.
  The UMI target at index 0 should be the identity/current pose.
- **Endpoint error**: XYZ distance between predicted and ground-truth final
  targets.

| Model | Roughness (mm) | Path ratio | First-step jump (mm) | Endpoint error (mm) |
|---|---:|---:|---:|---:|
| Ground truth | 0.54 | 1.33 | approximately 0 | — |
| ACT 100K | 0.77 | 1.14 | 0.77 | 30.5 |
| SmolVLA frozen 500K, Gaussian noise | 6.27 | 3.51 | 2.46 | 43.2 |
| SmolVLA unfrozen 200K, Gaussian noise | 5.56 | 2.48 | 2.00 | 34.7 |
| SmolVLA frozen 500K, zero noise | 0.33 | 1.13 | 1.67 | 42.4 |
| SmolVLA unfrozen 200K, zero noise | 0.23 | 1.05 | 1.76 | 33.8 |

Additional observations:

- Repeated predictions for the same observation differed by about 8.5 mm on
  average for SmolVLA, with a p95 of about 20 mm.
- ACT returned exactly the same chunk for repeated copies of an observation.
- Increasing SmolVLA integration from 10 to 30 denoising steps changed
  roughness only from approximately 5.72 mm to 5.63 mm.
- Averaging eight independent SmolVLA samples helped, but remained rougher than
  zero-noise inference and costs approximately eight times as much inference.
- Standard SmolVLA rotation steps were approximately 0.71–0.79 degrees versus
  0.23 degrees for ground truth and 0.09 degrees for ACT.
- Unfreezing improved endpoint accuracy and roughness, but did not eliminate
  the Gaussian-sampling effect.

## Code Findings

### The visualization exposes model output

`examples/umi_relative_ee/visualize_predictions.py:552-555` calls
`predict_action_chunk`, unnormalizes the model-relative result, and only then
turns it into a trajectory. The visible zig-zag therefore exists before the
absolute action postprocessor or IK/controller.

### UMI relative-EE conversion is consistent

`src/lerobot/processor/umi_relative_ee_processor.py:72-125` performs the
expected SE(3) conversions:

- absolute to relative: `inverse(T_reference) @ T_target`
- relative to absolute: `T_reference @ T_relative`

Every target in a chunk is relative to the same chunk-start pose. Target 0 in
the training data is the current pose/identity transform. SmolVLA's roughly
2 mm first-target error is learned model error rather than a frame-composition
error.

### SmolVLA has no trajectory-continuity objective

Relevant code is in
`src/lerobot/policies/smolvla/modeling_smolvla.py:621` and
`src/lerobot/policies/smolvla/modeling_smolvla.py:777-876`.

Training constructs:

```python
x_t = t * noise + (1 - t) * actions
u_t = noise - actions
loss = mse(u_t, predicted_flow)
```

Inference initializes the entire action chunk from independent Gaussian noise
and performs Euler integration. The loss is pointwise across time/action
dimensions; it has no explicit velocity, acceleration, jerk, or target-0
identity term.

The clean-action estimate available during flow training is:

```python
action_hat = x_t - t * predicted_flow
```

This is the appropriate value on which to impose future trajectory continuity
losses.

### ACT is deterministic in this setup

ACT uses direct chunk reconstruction with elementwise L1 loss. The tested UMI
ACT configuration had temporal ensembling disabled, so its measured smoothness
was intrinsic to the predicted chunk rather than produced by an execution-time
ensemble.

### Current UMI RTC cannot be enabled directly

`examples/umi_relative_ee/rtc.md:14-50` explains that guided RTC is not
currently safe for UMI relative-EE policies. Leftover actions remain relative
to the old chunk base and must first be:

1. converted to absolute SE(3) targets;
2. re-expressed relative to the new chunk-start pose;
3. converted to row-based rot6d;
4. normalized using checkpoint statistics.

RTC primarily addresses cross-chunk seams. It would not by itself remove the
within-chunk zig-zag measured here.

## Root-Cause Assessment

1. **Independent Gaussian sampling is the dominant source.** The large
   standard-versus-zero-noise difference isolates this effect.
2. **The pointwise flow loss does not penalize temporal discontinuity.**
3. **The identity target at chunk index 0 is learned rather than enforced.**
4. **More Euler integration steps do not solve the issue.** It is not primarily
   numerical integration error.
5. **Training longer is unlikely to be sufficient.** Frozen 500K remained much
   rougher than ACT 100K, and unfreezing only partially helped.
6. **The processor is not the policy-specific cause.** ACT and ground-truth
   chunks use the same UMI transformation and are smooth.


## Why Public SmolVLA and Pi0.5 Demos Can Look Smooth

A smooth rollout video does not establish that the policy's raw action chunk is smooth. Public demonstrations and this experiment differ at several layers.

### Commanded targets versus realized motion

This investigation visualizes every raw Cartesian target. A robot video shows motion after motor position control, velocity and acceleration limiting, command interpolation, mechanical inertia, and possibly explicit low-pass filtering. Several millimeters of target variation can be hard to see in a video while still causing vibration, motor effort, or poor contact behavior.

### Chunk execution

The official SmolVLA system describes asynchronous inference with overlapping chunk fusion to reduce jitter. The current UMI deployment predicts 30 actions, queues them, and executes all 30 directly. It does not blend overlapping predictions, so within-chunk variation and chunk-boundary discontinuities are exposed.

### Representation and controller differences

Many public demonstrations use native joint-position actions on SO-100 or SO-101-class robots. This experiment uses relative Cartesian end-effector poses, row-based rot6d, a required identity target at index 0, and IK after prediction. Joint-position servos naturally suppress some variation, whereas Cartesian pose errors can be amplified by IK into joint movement.

### Pretraining and fine-tuning differences

SmolVLA was primarily pretrained with standard LeRobot action representations. The custom 10D UMI relative-EE representation requires fine-tuning to learn new geometric and temporal correlations. Freezing the vision encoder and training only the expert further limits adaptation. The unfrozen checkpoint's improved roughness and endpoint accuracy support this explanation, although unfreezing did not eliminate the sampling effect.

Flow matching is not inherently rough. A sufficiently trained vector field can map Gaussian samples to coherent chunks. These results diagnose this checkpoint, representation, and execution stack; they do not imply that all SmolVLA, Pi0.5, or flow-matching policies are rough. Larger-scale pretraining, different data, model capacity, and execution-time chunk handling can all produce smoother rollouts.

Published demos also normally show selected successful episodes at normal speed rather than raw trajectories, repeated stochastic samples, acceleration, jerk, motor-current oscillation, or failed samples.

### Comparison needed to isolate the difference

Use the same observations and metrics to compare:

1. SmolVLA with its native joint-action representation.
2. SmolVLA with UMI relative-EE actions.
3. Raw Cartesian chunks before IK and controller constraints.
4. Executed joint commands after IK, interpolation, and limiting.

If native joint predictions are smooth but UMI predictions are rough, representation adaptation is the main issue. If both are rough, model/training configuration is more likely. If raw predictions are rough but executed commands are smooth, the apparent demo difference mainly comes from the control stack.

Per-dimension normalization statistics should also be checked. Relative XYZ motion may have small physical variance while normalized flow noise has unit variance. Imbalanced XYZ, rot6d, and gripper scales can turn small normalized residuals into visible Cartesian oscillation.

## Recommended Work

### 1. Deferred zero-noise inference ablation

Zero-noise inference was useful diagnostically, but the current decision is not to add it. If revisited later, expose it only as an explicit experiment such as `--noise_mode=gaussian|zero|fixed`.

Flow matching is trained to transport samples from its Gaussian base distribution. Zero is a valid numerical input but jointly atypical for a high-dimensional Gaussian chunk; it is not guaranteed to map to the conditional mean, mode, or safest action. If reconsidered, test it offline and then in a conservative robot dry-run because it may suppress multimodal/contact behavior or over-smooth necessary corrections.

A fixed random seed only makes predictions repeatable; it does not remove the
within-chunk noise pattern.

### 2. Enforce the first target

After prediction, force target 0 to the identity/current pose, including the
current gripper value. This removes the chunk-start offset but does not fix
internal roughness.

### 3. Add UMI-safe overlapping chunk fusion

Query a new chunk before consuming all 30 actions and blend overlapping
predictions in absolute SE(3). Old predictions must be re-anchored to the new
chunk frame before any model-space guidance.

Translation can be blended directly. Rotation should use SO(3) interpolation,
not elementwise averaging of rot6d.

### 4. Add training-time continuity losses

Apply configurable losses to `action_hat`, with weights defaulting to zero:

- target-0 identity/current-pose loss;
- first-difference matching against the demonstrated trajectory;
- second-difference/acceleration matching;
- geodesic SO(3) rotation continuity;
- optionally jerk regularization if acceleration loss is insufficient.

Losses should be normalized by physical units or feature statistics so that
translation, rotation, and gripper terms have controlled relative weights.

### 5. More invasive research directions

- temporally correlated or warm-start action priors;
- continuation-conditioned flow training;
- seam/velocity/acceleration losses across overlapping chunks;
- UMI-aware RTC after SE(3) re-anchoring support and integration tests exist.

## Online References

- [Official SmolVLA blog](https://huggingface.co/blog/smolvla) — flow-matching
  noise and overlapping chunk fusion intended to avoid control jitter.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — standard transport from a base distribution to the data distribution.
- [Action-to-Action Flow Matching](https://arxiv.org/abs/2602.07322) — previous-action-informed initialization trained as part of the policy.
- [Latent action flow matching](https://arxiv.org/abs/2601.23087) — temporally regularized action latents for smoother flow-policy execution.
- [ACT paper](https://arxiv.org/abs/2304.13705) — action chunking and temporal
  ensembling for precise smooth motion.
- [LeRobot RTC documentation](https://huggingface.co/docs/lerobot/rtc) and
  [RTC paper](https://arxiv.org/abs/2506.07339) — overlapping-prefix guidance
  for chunk transitions.
- [LeRobot issue #1239](https://github.com/huggingface/lerobot/issues/1239) —
  analogous report of ACT working smoothly while SmolVLA shakes on the same
  data despite low training loss.
- [ChunkFlow](https://arxiv.org/abs/2607.12992) — overlap blending and explicit
  seam/first-/second-order continuity losses.
- [POTR](https://arxiv.org/abs/2605.24433) — RTC guidance aimed at reducing
  acceleration and jerk.
- [Legato](https://arxiv.org/abs/2602.12978) — continuation and prior-mixture
  ideas for chunked flow policies.

## Kiwi Training Snapshot

Final read-only status check:

| Run | Step | Throughput | Target steps |
|---|---:|---:|---:|
| Frozen vision/expert-only | 546,391 | approximately 6.2 steps/s | 2,500,000 |
| Unfrozen/full-model | 218,966 | approximately 3.7 steps/s | 2,500,000 |

Both jobs remained active. No signal, pause, restart, checkpoint write, or
configuration change was issued.

## Key Commands

### Read-only kiwi status

```bash
ssh -p 2203 zfei@10.98.19.22 \
  'ps -eo pid,etime,%cpu,%mem,args | grep -E "train_relative_ee_processor.py.*smolvla" | grep -v grep'
```

### Source inspection

```bash
rg -n "predict_action_chunk|unnormalize_actions|rel_actions_to_traj" \
  examples/umi_relative_ee/visualize_predictions.py

rg -n "sample_noise|mse_loss|x_t =|u_t =|num_steps" \
  src/lerobot/policies/smolvla/modeling_smolvla.py

rg -n "relative|reanchor|unsafe|RTC" \
  examples/umi_relative_ee/rtc.md
```

## Files Modified

| File | Change |
|---|---|
| `conversation_summary/2026-07-29_smolvla_action_smoothness_investigation.md` | Added this investigation report |

## Tags

#python #pytorch #lerobot #robotics #smolvla #act #umi #debugging
