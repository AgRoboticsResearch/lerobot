# Within-chunk action jitter: ACT vs SmolVLA (frozen vs unfrozen VLM)

**Question:** Is there improvement, along training, in *action-to-action smoothness within a single
predicted chunk* (the 30 actions one prediction emits)? For ACT and SmolVLA, on train and val.

**Metric.** For every predicted chunk, convert the predicted actions and the GT actions to a 3D
gripper-tip point trajectory in the **chunk-start frame** (`rel_actions_to_traj` for predictions,
`gt_abs_to_rel_traj` for GT — same frame, same units). Within-chunk jitter = **mean L2 norm of the
second difference** (jerk) of that trajectory: `mean_t ‖p[t+1] − 2·p[t] + p[t−1]‖`. Lower = smoother.
Translation in mm, rotation in mrad. Reported as **predicted jerk** and the **pred/GT ratio**
(ratio < 1 means the prediction is smoother than the real motion).

**Setup.** Open-loop prediction, episodes 0–2, 60 valid frames each, `num_steps=10` (SmolVLA;
rectified flow, so 10 vs 30 is negligible). Same UMI relative-EE rot6d representation for all three
models (apples-to-apples). Reproduce with `examples/umi_relative_ee/measure_chunk_jitter.py`.

**GT baselines (constant):** val TRANS 0.538 mm, val ROT 8.95 mrad; train TRANS 0.709 mm, train ROT 9.42 mrad.

## Translation jitter (mm) — pred (pred/GT ratio)
| model · ckpt | val | train |
|---|---|---|
| **ACT** 100K | 0.88 (1.6×) | 1.05 (1.5×) |
| **ACT** 500K | 0.56 (1.0×) | 0.61 (0.9×) |
| **ACT** 1100K | 0.34 (0.6×) | 0.53 (0.7×) |
| **SmolVLA frozen** 100K | 7.16 (13.3×) | 6.92 (9.8×) |
| **SmolVLA frozen** 500K | 6.50 (12.1×) | 6.48 (9.1×) |
| **SmolVLA frozen** 900K | 5.65 (10.5×) | 5.67 (8.0×) |
| **SmolVLA unfrozen** 100K | 5.40 (10.0×) | 6.94 (9.8×) |
| **SmolVLA unfrozen** 200K | 5.11 (9.5×) | 6.83 (9.6×) |
| **SmolVLA unfrozen** 400K | 4.75 (8.8×) | 5.98 (8.4×) |

## Rotation jitter (mrad) — pred (pred/GT ratio)
| model · ckpt | val | train |
|---|---|---|
| **ACT** 100K | 4.45 (0.50×) | 5.67 (0.60×) |
| **ACT** 500K | 4.22 (0.47×) | 3.22 (0.34×) |
| **ACT** 1100K | 2.99 (0.33×) | 2.71 (0.29×) |
| **SmolVLA frozen** 100K | 91.0 (10.2×) | 76.3 (8.1×) |
| **SmolVLA frozen** 500K | 83.1 (9.3×) | 69.1 (7.3×) |
| **SmolVLA frozen** 900K | 77.3 (8.6×) | 61.5 (6.5×) |
| **SmolVLA unfrozen** 100K | 63.4 (7.1×) | 77.8 (8.3×) |
| **SmolVLA unfrozen** 200K | 61.9 (6.9×) | 75.8 (8.0×) |
| **SmolVLA unfrozen** 400K | 58.3 (6.5×) | 64.7 (6.9×) |

## Findings

1. **ACT is ~10–20× smoother within-chunk than SmolVLA, and reaches GT-level smoothness.**
   ACT's pred/GT ratio is **≤1.6× and falling** (translation hits **0.6× at 1100K** — smoother than
   the real motion; rotation is **0.3–0.6×**, i.e. ACT over-smooths rotation). SmolVLA sits at
   **~7–13× GT** for translation and rotation throughout. At the latest checkpoints: ACT val
   translation **0.34 mm** vs SmolVLA unfrozen **4.75 mm** vs frozen **5.65 mm** — a ~14–17× gap.

2. **All three improve along training, but ACT converges to GT while SmolVLA plateaus at a noise floor.**
   - ACT val translation: 0.88 → 0.56 → 0.34 mm (ratio 1.6× → 0.6×) — approaches/drops below GT.
   - SmolVLA frozen val translation: 7.16 → 6.50 → 5.65 mm (ratio 13× → 10×) — ~21% lower, but the
     ~10× ratio barely moves.
   - SmolVLA unfrozen val translation: 5.40 → 5.11 → 4.75 mm (ratio 10× → 9×) — same gentle decline.
   SmolVLA's floor is the **flow-matching per-frame sampling noise** (each `predict_action_chunk`
   draws fresh `torch.normal` noise); training reduces it only marginally.

3. **Unfreezing the VLM makes SmolVLA smoother**, most clearly on val (trans 4.75 vs 5.65 mm,
   rot 58 vs 77 mrad at the latest) — consistent with its lower val loss. On train they're closer.

4. **Train jitter > val jitter** for every model because the training episodes themselves are jerkier
   (GT train 0.71 vs val 0.54 mm). The pred/GT ratio — the model-noise measure — is similar on both.

## Root cause & levers
- **ACT** decodes its chunk from a CVAE latent → near-deterministic, temporally coherent by
  construction (smooth chunks are ACT's design goal).
- **SmolVLA** samples its chunk via flow-matching from noise → residual sampling noise smeared
  across the 30 steps. This is the within-chunk jitter, and it's why `num_steps` (10 vs 30) made no
  visible difference (rectified flow converges by ~10 steps — the noise, not the integrator, is the
  issue).
- **To get GT-smooth SmolVLA chunks** the levers are: (a) **sample-averaging** (run `sample_actions`
  N times, average the chunk — variance ↓ by √N), or (b) a **deterministic head**. More training
  steps won't close the ~10× gap.

## Bottom line
Within a single prediction, **ACT is at/below GT smoothness; SmolVLA is ~10× jerkier** (flow-matching
sampling-noise floor). Both SmolVLA variants smooth slowly with training and never approach ACT/GT;
unfreezing the VLM helps modestly. If smooth on-robot chunks matter, ACT wins on this axis outright,
and SmolVLA would need sample-averaging or a deterministic decoder to compete.
