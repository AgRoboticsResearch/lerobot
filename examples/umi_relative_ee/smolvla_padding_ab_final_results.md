# SmolVLA flow-matching padding A/B — final results (1M, complete)

**Date:** 2026-08-03. Both runs finished the full 1M steps.

Records the final validation-loss and endpoint-error numbers for the A/B
flow-matching-padding experiment, completing the picture sketched by the earlier
docs: formulation ([`padded_noise_strategy.md`](./padded_noise_strategy.md)),
early 100K ([`openpi_fullwidth_100k_results.md`](./openpi_fullwidth_100k_results.md)),
within-chunk jitter ([`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)),
and the transient offset ([`chunk_start_offset_transient.md`](./chunk_start_offset_transient.md)).

## Setup
- **Task:** SmolVLA UMI relative-EE, frozen VLM, on `sroiv2_strawberry_picking_lab_1302_occlusion`
  (train) + `sroiv2_strawberry_picking_lab_validation` (val, 100 eps).
- **Common config:** `policy.path=lerobot/smolvla_base`, chunk 30, batch 8, steps 1M,
  save_freq 100K, val_freq 10K, seed 1000, LR 1e-4 / warmup 1K / decay 30K → 2.5e-6.
  **Trainable params: 99.88 M of 450.05 M total (22%)** — action expert + state proj;
  VLM backbone (vision encoder + 16-layer SmolLM2) frozen.
- **Option A — `openpi_full_width`** (default): full-width 32-D Gaussian noise + loss over
  all 32 coords. `outputs/train/smolvla_openpi_fullwidth_1302_1M`.
- **Option B — `masked_subspace`**: noise/velocity/loss restricted to the real 10 dims;
  padded coords forced to 0 through train + inference. `outputs/train/smolvla_masked_subspace_1302_1M`.

## Validation loss (`flow_loss_real_dims` — the fair, full-val-set metric)
| run | best (step) | final @1M |
|-----|------|------|
| **A — openpi_full_width** | **0.0376** @50K | 0.0704 |
| **B — masked_subspace**   | **0.0400** @50K | 0.0785 |

A is marginally lower at both points but within ~6–10% → **essentially tied**. Both ~2× their
50K best by 1M (heavy overfit; best-val @50K is unsaved due to `save_freq=100K`, so the earliest
*saved* deployable ckpt is 100K).

## Endpoint error — open-loop prediction (val eps 0–2 only, 161 frames) @1M
| run | xyz (mm) | rot (°) | grip | …with `--zero_noise` (xyz / rot / grip) |
|-----|------|------|------|------|
| **A — openpi** val | 52.6 | 4.73 | 0.390 | 53.7 / **4.22** / 0.391 |
| **B — masked** val | 49.1 | 5.20 | 0.363 | 48.7 / **5.02** / 0.366 |

Train (eps 0–2, 215 frames) @1M for reference: A 15.8 mm / 1.60° (zn 16.5 / 1.44),
B 17.1 mm / 1.53° (zn 16.5 / 1.46).

Mixed at 1M — B better on val xyz, A better on val rot — so again **A ≈ B**.

## Conclusion
- **The masked-subspace formulation (B) gives no measurable benefit over openpi_full_width (A).**
  They tie on real-dim val loss, within-chunk jitter (see `within_chunk_jitter_analysis.md`),
  and endpoint accuracy. The original hypothesis (padded-noise coupling as the jitter root cause)
  is **not confirmed** — the within-chunk jitter is dominated by real-dimension flow-matching
  sampling noise, which both formulations share.
- **`--zero_noise` decoding** (all-zero flow latent) gives a small, consistent rotation-endpoint
  improvement (~0.3–0.5° val) with xyz unchanged — a diagnostic, not a deployment fix.
- **Both overfit by ~50K**; the 1M runs were ~20× past convergence. Deploy the earliest saved
  checkpoint (100K) or re-run with `save_freq=10000` to capture the ~50K best.
- The "fixed left move" once seen in B@300K viz was a **transient training artifact both A and B
  share** (~300K, resolves by 400–600K), not a masked-subspace bug — see
  `chunk_start_offset_transient.md`.

## Caveats
- Validation **loss** is the authoritative full-val-set number (computed every 10K by the trainer).
- The **endpoint errors** above are from open-loop viz on **val episodes 0–2 only** (161 frames),
  at the **final 1M (overfit)** checkpoint — a sample, not the full 100-episode val set, and not at
  the best-val region. A full-val endpoint eval (all 100 eps, at 100K or 1M) would be more rigorous.

## Checkpoints (on kiwi)
- A: `outputs/train/smolvla_openpi_fullwidth_1302_1M/checkpoints/{0100000…1000000,last}`
- B: `outputs/train/smolvla_masked_subspace_1302_1M/checkpoints/{0100000…1000000,last}`
