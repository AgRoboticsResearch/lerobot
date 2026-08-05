# Transient chunk-start offset (~300K) — not a masked-subspace bug

**Date:** 2026-08-03. Companion to [`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)
(the ACT-vs-flow / padding A/B audit). This documents a separate observation from the A/B viz.

## Observation
In the A/B viz videos, `viz_ab_masked_*` (Option **B**, masked_subspace, ckpt **300K**) showed an
**almost-fixed "move to the left" segment at the start of every predicted chunk**, on every frame —
i.e. an input-independent initial jump. `viz_ab_openpi_*` (Option **A**, ckpt **600K**) did not.

## Method
Open-loop prediction on validation episodes 0–2, 40 frames. Metric: the mean predicted first action of
the chunk (`pred_rel[0]`, the relative translation that becomes the first trajectory segment), in the
chunk-start frame, averaged across frames. A real motion has `a[0] ≈ 0` (no instantaneous jump);
**GT `a[0]` y = 0.00 mm**. Measured for A and B across checkpoints, plus GT.

## Finding: it is a transient training artifact, identical in A and B
Mean `a[0]` **y** (mm) per checkpoint (GT = 0.0; the artifact axis is +y, with a smaller −x ≈ −1.7 mm):

| step | A (openpi_full_width) | B (masked_subspace) |
|------|------|------|
| 100K | +0.82 | −1.31 |
| 200K | −2.22 | −3.04 |
| **300K** | **+5.77** | **+5.78** |
| 400K | +0.66 | *(not trained yet)* |
| 500K | −2.05 | |
| 600K | −0.41 | |

(GT overall chunk-mean y = +1.34 mm; biased checkpoints also inflate this — A@300K +4.62, B@300K +8.86.)

- The offset is **wrong** (GT `a[0]` = 0; it is not real gripper motion) and **input-independent**
  (constant across frames), so it is a genuine model artifact.
- **It is NOT caused by `masked_subspace`.** Option A (openpi_full_width) has the **identical +5.77 mm
  spike at 300K**. Both formulations pass through it.
- **It is transient.** Absent at 100K/200K, **peaks right at ~300K**, then **resolves by 400–600K**
  (A@400K +0.66 → A@600K −0.41, i.e. back to ~0).

The reason it appeared only in the B videos: **B was at 300K (inside the spike) while A was viz'd at
600K (past it).**

## Root cause
A training-dynamics wobble: during the ~300K phase the model temporarily learns a near-constant **+y
velocity component at the chunk start**, which the flow-matching Euler integrator turns into a fixed
endpoint offset on `a[0]` (a constant velocity bias `b` integrates to a constant output offset `−b`).
As training continues the model corrects it.

Both runs spike at the *same step* and *same magnitude* because they share the **same seed / data
order and the same LR schedule**, so their trajectories are correlated and hit the same transient
phase together. The flow-matching integration, the `masked_subspace` masking, and the loss are all
mathematically correct (verified in `src/lerobot/policies/flow_matching.py::integrate_flow_matching`
and `modeling_smolvla.py` `forward`/`sample_actions`) — this is not a code bug and not a property of
the padding mode. Note this is distinct from the within-chunk *jitter* (per-step noise) analyzed in
the companion doc; this is a *constant endpoint offset* on the first action.

## Recommendation
- It **self-resolves** — A is clean by 400–600K; B will be too past ~400K.
- For deployment or clean viz, **use a checkpoint ≥ 400K**. Re-render the B videos once B has a
  400K+ checkpoint and the "left move" will be gone.
- If many checkpoints are saved through this phase (save_freq=100K), prefer the ones on either side
  of the ~300K spike (e.g. 100–200K or ≥400K) when picking a deploy checkpoint by val loss.

## Reproduce
Mean `a[0]` translation per checkpoint (val eps 0–2, 40 frames), via the shared viz machinery:
```bash
# on kiwi (.venv) or host (py312), HF_HUB_OFFLINE=1
python - <<'PY'   # see measure script pattern in within_chunk_jitter_analysis.md
# load checkpoint -> predict_action_chunk on val eps 0,1,2 (delta-windowed) ->
# unnormalize -> mean of pred_rel[0, :3] across ~40 frames; compare to GT a[0] (≈0).
PY
```
Checkpoint roots on kiwi:
`outputs/train/smolvla_openpi_fullwidth_1302_1M/checkpoints/<step>/pretrained_model` (A) and
`outputs/train/smolvla_masked_subspace_1302_1M/checkpoints/<step>/pretrained_model` (B);
dataset `~/data/sroiv2_strawberry_picking_lab_validation`.
