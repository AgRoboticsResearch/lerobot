# Rotation normalization: UMI identity vs. our per-dim scaling

> **Status:** analysis + running A/B experiment (2026-07-31). This documents the
> one place our `umi_relative_ee` pipeline materially diverges from canonical UMI,
> why it is a candidate cause of full-data rotation "jumpiness," and the reversible
> experiment tracked in [`rot6d_identity_norm_experiment.md`](./rot6d_identity_norm_experiment.md).

## TL;DR

UMI leaves the 6D-rotation action components **unscaled (identity)** and only
normalizes position and gripper. We scale **all 10 dims uniformly** — including
the 6 rotation ones — with per-dimension MIN_MAX (ACT/SmolVLA) or QUANTILES (π0.5).
π0.5's quantile normalizer was designed for **joint-space** actions and has never
been validated on a 6D rotation representation. For *relative, near-identity*
rotations this per-dim scaling over-weights the near-constant rotation components
and is a plausible contributor to the rotation jumpiness observed on full-data
runs.

## The divergence

### Our pipeline — uniform per-dim scaling of all 10 action dims

The UMI stats are computed **flat over the whole 10D action** with no sub-field
split, so the 6 `rot6d` dims get their own statistics just like position:

- `src/lerobot/datasets/umi_relative_ee_stats.py:75-86` — one
  `RunningQuantileStats` over `[pos(3), rot6d(6), gripper(1)]`.
- `src/lerobot/processor/normalize_processor.py:329` — the normalizer selects a
  mode by **FeatureType** (`ACTION`), then applies that single mode to every dim.
- Per-policy mode:
  - π0.5: `src/lerobot/policies/pi05/configuration_pi05.py:80` → `QUANTILES`
    (q01/q99 → [-1,1]).
  - ACT: `src/lerobot/policies/act/configuration_act.py:172` → `MIN_MAX`.
  - SmolVLA: `src/lerobot/policies/smolvla/configuration_smolvla.py:142` → `MIN_MAX`.
- The 20D state is treated identically; its `rot6d` components live at dims
  `[3:9]` and `[13:19]`.

### Canonical UMI — rotation at identity

UMI scales **per sub-field**: position → range [-1,1], **rot6d → identity
(scale=1, offset=0)**, gripper → range [-1,1]:

- `detached-umi-policy/diffusion_policy/dataset/umi_dataset.py:221-227`
  (`get_range_normalizer_from_stat` for pos/gripper,
  `get_identity_normalizer_from_stat` for the 6D rotation).
- Rotation matrix entries are already bounded in [-1,1] (two orthonormal unit
  rows), so scaling them is unnecessary and distorts their geometry.

### openpi / π0.5 — no rotation in the action space at all

Every action space in openpi is **joint-space** (joint angles + a scalar gripper):
Aloha 14D, DROID 8D, Libero 7D, base π0.5 padded to 32D. There is no quaternion,
axis-angle, euler, or 6D rotation anywhere in the pipeline.

- π0.5 / π0-FAST default normalization is **quantile per-dim**:
  `(x - q01)/(q99 - q01) * 2 - 1` → [-1,1]
  (`openpi/src/openpi/transforms.py:137-145`, mode switch at
  `openpi/src/openpi/training/config.py:187`: `use_quantile_norm = model_type != PI0`).
- Stats are computed flat, keyed only by `"state"`/`"actions"`, one statistic per
  dimension (`openpi/scripts/compute_norm_stats.py:102`).

**Implication:** our LeRobot π0.5 path matches openpi's default mode, so we did
*not* pick the wrong mode — the mode is correct *for π0.5*. The mismatch is that
we feed a 6D rotation representation into a normalizer that was tuned for
well-spread, independent joint dimensions and never designed to handle rotation.

## Why per-dim scaling can hurt *relative* rot6d

`rot6d` is the first two rows of the rotation matrix. A relative action
`T_curr⁻¹ @ T_future` is **near identity** for most of a chunk (small per-frame
motion), so its `rot6d` clusters tightly around `[1, 0, 0, 0, 1, 0]`:

- diagonal dims (`rot6d[0]`, `rot6d[4]`) sit near **+1** with a **narrow** spread,
- off-diagonal dims sit near **0**.

Applying **per-dimension quantile/min-max** scaling to that:

- The diagonal "1" components have a thin `[q01, q99]` band (e.g. ~[0.7, 0.99]),
  which quantile scaling **stretches across all of [-1,1]** — a large multiplicative
  amplification of a near-constant signal.
- In the regression / flow-matching loss each dim's effective weight is
  `1/(q99 - q01)` (or `1/std` for mean/std). The near-constant diagonal dims get the
  **largest** weight; the off-diagonal dims that actually encode *which way* the
  gripper rotated get the **smallest**.

Net effect: the loss is dominated by predicting the near-constant `[1, 1]`
diagonal components to many significant figures, while the dimensions carrying the
real rotation direction are comparatively under-fit. The model snaps toward
identity and is noisy on the informative axes — i.e. rotation jumpiness. UMI's
identity normalization keeps every component in natural scale with balanced
weighting and never creates this imbalance.

This is consistent with existing debug results:

- Single-episode overfit reaches a ~0.9° rotation floor (memorization wins
  despite the bad weighting).
- Full-data generalization stalls at ~4.6–5.6°, and LoRA rank is **not** the
  bottleneck.
- A loss-weighting / target-distribution problem cripples generalization while
  leaving overfitting intact — exactly this signature.

> **Caveat:** the magnitude of the imbalance depends on how tightly the relative
> rotations cluster (offset-1 targets are ~identity, offset-30 less so, and stats
> are pooled across the whole chunk). The principle — per-dim affine scaling of a
> coupled rotation representation distorts component weighting, worst on
> near-identity data — holds regardless; the open question is how large the effect
> is for this dataset. It is cheap to measure.

## Do not bother with a mode swap

Switching π0.5 → π0 (quantiles → mean/std) does **not** fix this: mean/std has the
same per-dim weighting flaw (`1/std` amplifies the low-variance diagonal
components just as `1/(q99 - q01)` does). The issue is per-dim affine scaling of a
rotation representation, independent of which affine. The fix is identity on the
rotation sub-field, not a different global mode.

## Experiment (reversible, minimal)

Force the `rot6d` slice to identity in the stats **before** normalization. This
reproduces UMI's behavior exactly and requires no architecture change:

- **Action (10D):** dims `[3:9]` → `q01 = -1, q99 = 1` (π0.5) / `min = -1, max = 1`
  (ACT, SmolVLA).
- **State (20D):** dims `[3:9]` and `[13:19]` → same.

Cheapest hook: overwrite those slices in the return value of
`compute_umi_relative_ee_stats` (`src/lerobot/datasets/umi_relative_ee_stats.py`,
the `return` near line 93). Gate it behind a config flag so it can be toggled per
run without code edits.

Decision criterion: compare matched checkpoints with the policy-neutral
`eval_open_loop_dataset.py` on all 100 validation episodes (10 evenly spaced valid
queries per episode). The primary value is the episode-balanced within-chunk
rotation second-difference/acceleration proxy (currently stored under the legacy
key `rot_jerk_deg`); chunk-end rotation error is secondary accuracy. A meaningful
smoothness reduction (target: at least 20% relative, beyond seed noise) supports
the jumpiness hypothesis, provided the paired checkpoints used the same action
padding/noise objective. RTC evaluation remains a separate SmolVLA/π0.5 deployment
metric and does not support ACT.

The padding/noise qualification matters because a later implementation audit
found that SmolVLA historically sampled an unconstrained 32-D latent for a 10-D
UMI action but trained loss only on the ten real dimensions. See
[`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md) for the
OpenPI comparison, GPU ablations, and the two coherent fixed-width formulations.
Inference now masks the 22 unused coordinates throughout the flow ODE by default
for SmolVLA, π0, and π0.5, including the velocity exposed to RTC. This applies to
existing checkpoints without retraining. Use `--legacy_full_action_noise` in the
UMI evaluators only for historical reproduction; training-side padding
objectives remain a deferred retraining A/B.

### Long-term form

A per-sub-field normalizer step (position → range, **rotation → identity**,
gripper → range) so the identity treatment is explicit rather than a stats
override. This mirrors `detached-umi-policy`'s
`UmiDataset.get_normalizer` exactly.

## Evidence index

| Claim | Location |
| --- | --- |
| Our stats computed flat over 10D | `src/lerobot/datasets/umi_relative_ee_stats.py:75-86` |
| Our normalizer: one mode per FeatureType, per-dim | `src/lerobot/processor/normalize_processor.py:329` |
| π0.5 action mode = QUANTILES | `src/lerobot/policies/pi05/configuration_pi05.py:80` |
| ACT action mode = MIN_MAX (UMI path) | `src/lerobot/policies/act/configuration_act.py:172` |
| SmolVLA action mode = MIN_MAX (UMI path) | `src/lerobot/policies/smolvla/configuration_smolvla.py:142` |
| UMI: rot6d → identity, pos/gripper → range | `detached-umi-policy/diffusion_policy/dataset/umi_dataset.py:221-227` |
| openpi quantile formula | `openpi/src/openpi/transforms.py:137-145` |
| openpi mode switch (non-PI0 → quantile) | `openpi/src/openpi/training/config.py:187` |
| openpi stats keyed only by state/actions | `openpi/scripts/compute_norm_stats.py:102` |
