#!/usr/bin/env python
"""Overfitting verification — proves the full pipeline works end-to-end.

Loads the trained checkpoint, overfits a SINGLE batch for N steps, then
verifies predictions match ground truth nearly perfectly.

If any part of the pipeline is broken (stats, relative conversion, normalization,
postprocessor), the model CANNOT overfit — loss won't go to zero and predictions
won't converge to ground truth.

Usage:
  cd /home/hls/codes/lerobot_piper_sroi
  uv run --directory lerobot python lerobot/verify_overfit.py
  uv run --directory lerobot python lerobot/verify_overfit.py --steps 2000 --no-plot
"""

import logging
import sys

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/test_ee_dataset"
CHECKPOINT_DIR = "/home/hls/codes/lerobot_piper_sroi/outputs/smolvla_umi_ee_test/checkpoints/001000/pretrained_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def green(s):
    return f"\033[32m{s}\033[0m"


def red(s):
    return f"\033[31m{s}\033[0m"


def bold(s):
    return f"\033[1m{s}\033[0m"


def section(title):
    print(f"\n{'='*70}")
    print(f"  {bold(title)}")
    print(f"{'='*70}")


def check(condition, label):
    if condition:
        print(f"  {green('[PASS]')} {label}")
        return True
    else:
        print(f"  {red('[FAIL]')} {label}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════
section("SETUP — load checkpoint, prepare single batch")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies import make_policy, make_pre_post_processors

ds = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT)
ds = recompute_stats(
    ds, num_workers=2,
    relative_action=True, relative_exclude_joints=["gripper"],
    relative_state=True, relative_exclude_state_joints=["gripper"],
    state_obs_steps=2, derive_state_from_action=True,
)

cfg = SmolVLAConfig(
    derive_state_from_action=True,
    use_relative_actions=True,
    relative_exclude_joints=["gripper"],
    relative_exclude_state_joints=["gripper"],
    device=DEVICE,
    resize_imgs_with_padding=(512, 512),
    freeze_vision_encoder=True,
    train_expert_only=True,
    train_state_proj=True,
    load_vlm_weights=False,
    push_to_hub=False,
    optimizer_lr=3e-4,
    optimizer_weight_decay=1e-10,
    optimizer_grad_clip_norm=10,
    pretrained_path=str(CHECKPOINT_DIR),
)

dt = resolve_delta_timestamps(cfg, ds.meta)
ds_dt = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT, delta_timestamps=dt)

policy = make_policy(cfg=cfg, ds_meta=ds.meta)
policy = policy.to(DEVICE)

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=str(CHECKPOINT_DIR), dataset_stats=ds.meta.stats
)

# Fixed batch — we'll overfit this
batch = ds_dt[50]
batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v] for k, v in batch.items()}
batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

gt_action_raw = batch["action"].clone()  # [1, 51, 7] absolute
gt_action_eval = gt_action_raw[:, 1:, :]  # [1, 50, 7] — after DeriveState trim

processed = preprocessor(batch)
print(f"\n  Preprocessed action shape: {list(processed['action'].shape)}")
print(f"  Preprocessed state shape:  {list(processed['observation.state'].shape)}")
print(f"  GT action (abs) shape:     {list(gt_action_eval.shape)}")

# ══════════════════════════════════════════════════════════════════════════════
# Overfit single batch
# ══════════════════════════════════════════════════════════════════════════════
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--no-plot", action="store_true")
args, _ = parser.parse_known_args()

N_STEPS = args.steps

section(f"OVERFITTING — single batch, {N_STEPS} steps")

opt_cfg = cfg.get_optimizer_preset()
sched_cfg = cfg.get_scheduler_preset()
optimizer = opt_cfg.build(policy.parameters())
scheduler = sched_cfg.build(optimizer, N_STEPS)

losses = []
policy.train()
for step in range(N_STEPS):
    loss, loss_dict = policy.forward(processed)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    if step % 200 == 0 or step == N_STEPS - 1:
        total_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        logger.info(f"  step {step:4d}: loss={loss.item():.6f}, grad_norm={total_norm:.4f}")

    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer_grad_clip_norm)
    optimizer.step()
    scheduler.step()

loss_start = losses[0]
loss_end = losses[-1]
loss_min = min(losses)
loss_reduction = (loss_start - loss_end) / loss_start * 100

print(f"\n  Loss start:  {loss_start:.6f}")
print(f"  Loss end:    {loss_end:.6f}")
print(f"  Loss min:    {loss_min:.6f}")
print(f"  Reduction:   {loss_reduction:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# Verify predictions match ground truth
# ══════════════════════════════════════════════════════════════════════════════
section("VERIFY — model predictions vs ground truth")

policy.eval()
with torch.no_grad():
    pred_norm_rel = policy.predict_action_chunk(processed)
    pred_abs = postprocessor(pred_norm_rel)  # [1, 50, 7] absolute

dim_names = ds.meta.features["action"]["names"]
if isinstance(dim_names, dict):
    dim_names = dim_names.get("axes", list(dim_names.keys()))

# Per-dim error
error = (pred_abs[0] - gt_action_eval[0].cpu()).abs()
mean_err = error.mean().item()
max_err = error.max().item()

print(f"\n  Mean abs error: {mean_err:.6f}")
print(f"  Max abs error:  {max_err:.6f}")
print(f"\n  Per-dim MAE:")
print(f"  {'Dim':>12s}  {'MAE':>10s}  {'in mm/rad'}")
print(f"  {'-'*12}  {'-'*10}  {'-'*12}")
for d, name in enumerate(dim_names):
    dim_mae = error[:, d].mean().item()
    unit = "mm" if d < 3 else ("rad" if d < 6 else "")
    scale = 1000 if d < 3 else 1
    print(f"  {name:>12s}  {dim_mae:10.6f}  {dim_mae*scale:.4f} {unit}")

# Show t=0 comparison
print(f"\n  t=0 comparison:")
print(f"    GT:   {[f'{v:.4f}' for v in gt_action_eval[0, 0].cpu().tolist()]}")
print(f"    Pred: {[f'{v:.4f}' for v in pred_abs[0, 0].cpu().tolist()]}")
print(f"    Err:  {[f'{v:.4f}' for v in error[0].tolist()]}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
if not args.no_plot:
    section("PLOT — loss curve")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        ax = axes[0]
        ax.plot(losses, linewidth=0.5, color="#1f77b4")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Overfitting Loss ({N_STEPS} steps)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.annotate(f"Start: {loss_start:.4f}", xy=(0, loss_start), fontsize=8, color="red")
        ax.annotate(f"End: {loss_end:.6f}", xy=(N_STEPS - 1, loss_end), fontsize=8, color="green")

        # GT vs Pred per dim (first 20 timesteps)
        ax = axes[1]
        n_show = min(20, gt_action_eval.shape[1])
        t_axis = np.arange(n_show)
        for d, (name, color) in enumerate(zip(
            dim_names,
            ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#34495e"],
        )):
            ax.plot(t_axis, gt_action_eval[0, :n_show, d].cpu(), "--", color=color, alpha=0.5, linewidth=1)
            ax.plot(t_axis, pred_abs[0, :n_show, d].cpu(), "-", color=color, linewidth=1.5, label=name)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.set_title("GT (dashed) vs Pred (solid) — first 20 steps")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        plot_path = "/home/hls/codes/lerobot_piper_sroi/outputs/overfit_verification.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        print(f"\n  Plot saved to: {plot_path}")
    except ImportError:
        print("\n  matplotlib not installed, skipping plot.")

# ══════════════════════════════════════════════════════════════════════════════
# Verdict
# ══════════════════════════════════════════════════════════════════════════════
section("VERDICT")

checks = []
checks.append(check(loss_end < 0.01,
                    f"Final loss < 0.01: {loss_end:.6f}"))
checks.append(check(mean_err < 0.005,
                    f"Mean prediction error < 5mm: {mean_err:.6f}"))
checks.append(check(not torch.isnan(pred_abs).any(),
                    "No NaN in predictions"))
checks.append(check(not torch.isinf(pred_abs).any(),
                    "No Inf in predictions"))
checks.append(check(loss_reduction > 90,
                    f"Loss reduced >90%: {loss_reduction:.1f}%"))

all_pass = all(checks)
print(f"\n  {'ALL CHECKS PASSED ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")
if all_pass:
    print(f"\n  The full pipeline works end-to-end:")
    print(f"    raw data → DeriveState → RelativeActions → RelativeState")
    print(f"    → Normalize → Model → Unnormalize → AbsoluteActions → output")
    print(f"  Stats are correct, relative conversions are correct,")
    print(f"  normalization maps are consistent, model can learn the mapping.")
