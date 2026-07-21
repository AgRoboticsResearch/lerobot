# Scaling Experiment — ACT, relative-EE, strawberry picking

**Status:** IN PROGRESS (1012-ep run done; 100-ep & 500-ep training in parallel).
**Started:** 2026-07-18. **Expected completion:** ~2026-07-21.
**Owner:** Zhenghao. **Machine:** pangolin (1× RTX 4090, 24 GB).

## 1. Goal / hypothesis

Measure how **validation loss** of an ACT policy (UMI-style relative end-effector
actions, rot6d, action-chunk 30) scales with **training dataset size**, holding
everything else fixed (model, optimizer, steps, validation set).

Hypothesis: validation loss follows an approximately power-law decay in dataset
size, `L(D) ≈ a·D^(−b) + c`, and the model overfits the smaller subsets well
before the 2.5M-step budget — so **best (early-stopped) val loss** is the fair
comparison metric, not final.

## 2. Method

**Policy / training (identical across all three runs):**
- Model: ACT, 52M params
- Action representation: UMI-style relative EE, **rot6d (10D)**, `chunk_size=30`, `n_action_steps=30`
- `derive_state_from_action=true`, `use_relative_actions=true`, `pose_dim=6`, `use_rot6d=true`
- Normalization: MIN_MAX (UMI-style); ImageNet stats for cameras
- Batch size 8, AdamW lr 1e-5, **2.5M steps**, `val_freq=10000`
- **No checkpoints saved** (`--save_checkpoint=false`) — disk conservation; the
  validation curve (console log + wandb) is the durable artifact.

**Only the training subset varies:** 100 / 500 / 1012 episodes (nested prefix slices —
100 ⊂ 500 ⊂ 1012, identical episode content). Each run recomputes its own
normalization stats over its own subset. The validation set is **the same 100
episodes / 9274 frames** for all three runs.

## 3. Datasets (all LeRobot v3, 30 fps, RealSense D405, task "pick the strawberry")

| train subset | episodes | frames | path (root) |
|---|---|---|---|
| 100-ep | 100 | 11,732 | `sroiv2_strawberry_picking_lab_1000onesb_100` |
| 500-ep | 500 | 46,640 | `sroiv2_strawberry_picking_lab_1000onesb_500` |
| 1012-ep (full) | 1012 | 88,218 | `sroiv2_strawberry_picking_lab_1000onesb` |
| **validation (shared)** | 100 | 9,274 | `sroiv2_strawberry_picking_lab_validation` |

All under `/mnt/data1/sroi/lerobot/`. The full set is a **masked** rebuild (trajectories
rebuilt 2026-07-14). Per-dataset lineage in each `DATA_SOURCES.md`.

**Caveat:** subsets are **leading-prefix slices** by session/episode order, so at small N
early sessions are over-represented (100-ep averages ~117 frames/ep vs ~87 for the full
set). Fine for a nested scaling sweep; acknowledged in the report.

## 4. Runs

| tag | status | best val loss (so far) | best-val step | final val loss |
|---|---|---|---|---|
| 100-ep | training (~76%) | 0.0833 | 380k | TBD |
| 500-ep | training (~75%) | 0.0572 | 70k | TBD |
| 1012-ep | **done** | 0.0456 | 180k | 0.0480 |

**Preliminary scaling signal (best val): 0.083 → 0.057 → 0.046** for 100→500→1012 ep —
monotonic, consistent with a power law. (Final-loss values are noisier due to overfit.)

**Key observation:** every run converges then overfits well before 2.5M steps — best val
at 70k/180k/380k for 500/1012/100-ep respectively. Extra steps add nothing to val loss
(and slightly hurt). Train loss keeps dropping (e.g. 100-ep: 0.014→0.006) while val is
flat → classic memorization of the limited demos.

## 5. Reproduce

```bash
# Env: /home/zfei/anaconda3/envs/py310/bin/python (lerobot editable install of this repo)
bash outputs/train/ee_vs_joints/scaling_analysis/run_scaling_subset.sh <100|500> <dataset_dirname>
```
`run_scaling_subset.sh` hard-codes the full config above (only the dataset + output dir
change). The 1012-run used the same config directly via
`examples/umi_relative_ee/train_relative_ee_processor.py`.

## 6. Automation / monitoring

- Both new runs are **detached** (`nohup setsid`); survive session restarts.
- A recurring cron (every 30 min, `:08`/`:38`, **session-only**) babysits and, when both
  runs hit 2.5M, auto-compiles the deliverable per `FIGURE_SPEC.md` then self-deletes.
- Logs: `run_100ep.console.log`, `run_500ep.console.log`. 1012-run curve in
  `val_curve_1000ep.csv` (val) and `_1012_history.csv` (train+val, 10k pts from cloud).
- **wandb:** all three runs grouped in project
  [`biorobotlab/lerobot_scaling_ee`](https://wandb.ai/biorobotlab/lerobot_scaling_ee):
  `act_scaling_ee_chunk30_{100,500,1012}ep`. The 1012 run is a **mirror/replay**
  (`5wkorw6c`, tagged `mirror-of:lerobot/mgnsa3nw`) of the original finished run
  (`biorobotlab/lerobot/mgnsa3nw`, logged under a different default project) — wandb
  can't move runs across projects, so the real metrics were re-logged at the same steps.

## 7. Deliverable

A standard scaling-report figure set + `SCALING_REPORT.md` (see **`FIGURE_SPEC.md`**):
val & train loss vs steps (3 dataset sizes), val loss vs dataset-size (log-log, power-law
fit + exponent), generalization gap vs steps, and loss components.

## 8. Caveats / known issues
- Leading-prefix subsets (§3) — small-N over-representation of early sessions.
- Large transient grad-norm spikes on sampled steps (up to ~900) but train loss stays
  flat — grad clipping contains them; no divergence.
- Single GPU, runs are parallel (each ~0.066 s/step under contention vs 0.034 solo).
