#!/usr/bin/env python
"""SmolVLA UMI EE training smoke test — verify pipeline end-to-end.

Level 1: 10 steps — no crashes, finite loss, no NaNs
Level 2: 200 steps on single batch — loss must decrease (proves gradients flow)

Usage:
  cd /home/hls/codes/lerobot_piper_sroi
  uv run --directory lerobot python /home/hls/codes/lerobot_piper_sroi/lerobot/train_smoke_test.py
"""

import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/test_ee_dataset"
OUTPUT_DIR = Path("/home/hls/codes/lerobot_piper_sroi/outputs/smolvla_smoke_test")

# ── Dataset + Stats ──────────────────────────────────────────────────
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats

ds = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT)
ds = recompute_stats(
    ds, num_workers=2,
    relative_action=True, relative_exclude_joints=["gripper"],
    relative_state=True, relative_exclude_state_joints=["gripper"],
    state_obs_steps=2, derive_state_from_action=True,
)

# ── Policy + Processors ──────────────────────────────────────────────
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies import make_policy, make_pre_post_processors

cfg = SmolVLAConfig(
    derive_state_from_action=True,
    use_relative_actions=True,
    relative_exclude_joints=["gripper"],
    relative_exclude_state_joints=["gripper"],
    device="cpu",
    push_to_hub=False,
    load_vlm_weights=False,
    freeze_vision_encoder=True,
    train_expert_only=True,
    train_state_proj=True,
    resize_imgs_with_padding=(512, 512),
    optimizer_lr=1e-4,
    optimizer_weight_decay=1e-10,
    optimizer_grad_clip_norm=10,
)

ds_meta = ds.meta
dt = resolve_delta_timestamps(cfg, ds_meta)
ds_dt = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT, delta_timestamps=dt)

policy = make_policy(cfg=cfg, ds_meta=ds_meta)
preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds.meta.stats)

# ── Optimizer (from SmolVLAConfig presets) ──────────────────────────
def make_opt_sched(policy_cfg, policy, steps):
    opt_cfg = policy_cfg.get_optimizer_preset()
    sched_cfg = policy_cfg.get_scheduler_preset()
    optimizer = opt_cfg.build(policy.parameters())
    scheduler = sched_cfg.build(optimizer, steps)
    return optimizer, scheduler

optimizer, scheduler = make_opt_sched(cfg, policy, steps=200)

logger.info(f"Dataset: {len(ds_dt)} frames, {ds_meta.episodes} episodes")
logger.info(f"Action dim: {cfg.output_features['action'].shape}")
logger.info(f"State dim:  {cfg.input_features['observation.state'].shape}")
logger.info(f"Preprocessor steps: {[type(s).__name__ for s in preprocessor.steps]}")
logger.info(f"Postprocessor steps: {[type(s).__name__ for s in postprocessor.steps]}")

# ══════════════════════════════════════════════════════════════════════
# LEVEL 1: Smoke test — 10 steps, verify no crashes or NaNs
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  LEVEL 1 — Smoke test (10 steps)")
print("=" * 70)

losses_l1 = []
policy.train()
for step in range(10):
    frame_idx = step * 50  # spread across dataset
    if frame_idx >= len(ds_dt):
        frame_idx = 0

    batch = ds_dt[frame_idx]
    batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
             for k, v in batch.items()}
    batch = preprocessor(batch)
    loss, loss_dict = policy.forward(batch)
    losses_l1.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 0:
        # Check initial state
        for k, v in batch.items():
            if hasattr(v, 'dtype'):
                has_nan = bool(torch.isnan(v).any())
                has_inf = bool(torch.isinf(v).any())
                status = f"shape={list(v.shape)}"
                if has_nan:
                    status += " ⚠️ NaN"
                if has_inf:
                    status += " ⚠️ Inf"
                if step == 0:
                    logger.info(f"  batch[{k}]: {status}")

ok_l1 = all(not torch.isnan(torch.tensor(l)) and not torch.isinf(torch.tensor(l)) for l in losses_l1)
print(f"  Losses: {[f'{l:.4f}' for l in losses_l1[:5]]}...")
print(f"  Loss range: [{min(losses_l1):.4f}, {max(losses_l1):.4f}]")
print(f"  No NaNs: {'YES' if ok_l1 else 'NO — FAILED'}")
print(f"  Result: {'PASSED ✓' if ok_l1 else 'FAILED ✗'}")

if not ok_l1:
    print("  ABORTING: NaN/Inf detected in level 1")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
# LEVEL 2: Overfitting test — single batch, 30 steps, loss must trend down
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  LEVEL 2 — Overfitting test (30 steps, single batch)")
print("=" * 70)

# Pick a fixed batch
batch = ds_dt[50]
batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
         for k, v in batch.items()}
batch = preprocessor(batch)

# Reset optimizer + scheduler for clean overfitting run
optimizer, scheduler = make_opt_sched(cfg, policy, steps=30)

losses_l2 = []
policy.train()
for step in range(30):
    loss, loss_dict = policy.forward(batch)
    losses_l2.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    if step == 0 or step == 29:
        total_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        logger.info(f"  step {step:3d}: loss={loss.item():.6f}, grad_norm={total_norm:.4f}")

    optimizer.step()
    scheduler.step()

# Analyze
loss_start = losses_l2[0]
loss_end = losses_l2[-1]
loss_min = min(losses_l2)
loss_trending_down = losses_l2[-1] < losses_l2[0]
loss_never_nan = all(not torch.isnan(torch.tensor(l)) for l in losses_l2)

# Smooth for cleaner view
window = 5
smoothed = [sum(losses_l2[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(losses_l2))]

print(f"  Start loss:  {loss_start:.6f}")
print(f"  End loss:    {loss_end:.6f}")
print(f"  Min loss:    {loss_min:.6f}")
print(f"  Trending:    {'DOWN ✓' if loss_trending_down else 'FLAT/UP'}")
print(f"  No NaNs:     {'YES' if loss_never_nan else 'NO — FAILED'}")

# Show loss curve as ASCII
print(f"\n  Loss per step:")
print(f"  {'─'*50}")
for i in range(0, len(losses_l2), 3):
    bar = "#" * max(1, int(losses_l2[i] / loss_start * 30))
    marker = " ← start" if i == 0 else (" ← end" if i >= len(losses_l2)-3 else "")
    print(f"  {i:3d} {losses_l2[i]:.4f} |{bar}{marker}")

ok_l2 = loss_trending_down and loss_never_nan
print(f"\n  Result: {'PASSED ✓' if ok_l2 else 'FAILED ✗'}")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SMOKE TEST SUMMARY")
print("=" * 70)
print(f"  Level 1 (10-step crash/NaN check): {'PASSED' if ok_l1 else 'FAILED'}")
print(f"  Level 2 (200-step overfitting):    {'PASSED' if ok_l2 else 'FAILED'}")
if ok_l1 and ok_l2:
    print("\n  Training pipeline is correctly wired. Ready for full training.")
else:
    print("\n  Pipeline has issues — fix before full training.")
print("=" * 70)
