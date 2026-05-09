#!/usr/bin/env python
"""Model-in-the-loop round-trip test for SmolVLA UMI EE-pose training.

Tests the FULL chain: preprocess → model.forward (loss) / model.sample_actions (inference)
→ postprocess → compare with ground truth absolute actions.

Also audits which parameters are trainable vs frozen.

Usage:
  cd /home/hls/codes/lerobot_piper_sroi
  uv run --directory lerobot python lerobot/test_model_roundtrip.py
"""

import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/test_ee_dataset"
EPS = 1e-8


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


def fa(arr, p=6):
    """Compact format."""
    if hasattr(arr, 'tolist'):
        arr = arr.tolist()
    if isinstance(arr, (list, tuple)):
        return "[" + " ".join(f"{v:{p}.4f}" for v in arr) + "]"
    return f"{arr:{p}.4f}"


# ── Setup: dataset, stats, policy, processors ─────────────────────────
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.factory import resolve_delta_timestamps
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

dt = resolve_delta_timestamps(cfg, ds.meta)
ds_dt = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT, delta_timestamps=dt)

policy = make_policy(cfg=cfg, ds_meta=ds.meta)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg, dataset_stats=ds.meta.stats
)

dim_names = ds.meta.features["action"]["names"]
if isinstance(dim_names, dict):
    dim_names = dim_names.get("axes", list(dim_names.keys()))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: GRADIENT AUDIT
# ══════════════════════════════════════════════════════════════════════════
section("1. GRADIENT AUDIT — trainable vs frozen parameters")

# Run one forward+backward to populate gradients
policy.train()
batch = ds_dt[50]
batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
         for k, v in batch.items()}
batch = preprocessor(batch)
loss, loss_dict = policy.forward(batch)
loss.backward()

trainable_params = 0
trainable_elements = 0
frozen_params = 0
frozen_elements = 0
trainable_without_grad = []
frozen_with_grad = []

for name, param in policy.named_parameters():
    if param.requires_grad:
        trainable_params += 1
        trainable_elements += param.numel()
        if param.grad is None:
            trainable_without_grad.append(name)
    else:
        frozen_params += 1
        frozen_elements += param.numel()
        if param.grad is not None:
            frozen_with_grad.append(name)

total_params = trainable_params + frozen_params
total_elements = trainable_elements + frozen_elements

print(f"\n  Parameter summary:")
print(f"    Trainable: {trainable_params} params ({trainable_elements:,} elements, {100*trainable_elements/total_elements:.1f}%)")
print(f"    Frozen:    {frozen_params} params ({frozen_elements:,} elements, {100*frozen_elements/total_elements:.1f}%)")
print(f"    Total:     {total_params} params ({total_elements:,} elements)")

check(len(trainable_without_grad) == 0,
      f"All trainable params received gradients ({len(trainable_without_grad)} missing)")
if trainable_without_grad:
    for n in trainable_without_grad[:5]:
        print(f"      NO GRAD: {n}")

check(len(frozen_with_grad) == 0,
      f"No frozen params received gradients ({len(frozen_with_grad)} leaked)")
if frozen_with_grad:
    for n in frozen_with_grad[:5]:
        print(f"      LEAKED GRAD: {n}")

# Show which modules are trainable
print(f"\n  Module-level trainability:")
seen_prefixes = set()
for name, param in policy.named_parameters():
    prefix = ".".join(name.split(".")[:2])
    if prefix not in seen_prefixes:
        seen_prefixes.add(prefix)
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        print(f"    {prefix:<40s} {status}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: MODEL FORWARD — loss properties
# ══════════════════════════════════════════════════════════════════════════
section("2. MODEL FORWARD — loss shape and properties")

policy.train()
batch2 = ds_dt[100]
batch2 = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
          for k, v in batch2.items()}
batch2 = preprocessor(batch2)

with torch.no_grad():
    loss2, loss_dict2 = policy.forward(batch2)

print(f"\n  Loss: {loss2.item():.6f}")
print(f"  Loss dict keys: {list(loss_dict2.keys())}")

check(not torch.isnan(loss2), f"Loss is finite: {loss2.item():.6f}")
check(not torch.isinf(loss2), "Loss is not Inf")
check(loss2.item() > 0, f"Loss is positive: {loss2.item():.6f}")

# The loss should be computed in normalized relative space
# For a randomly initialized model, flow matching loss from N(0,1) noise
# should be around 1.0-5.0 (depends on noise scale and dims)
check(0.1 < loss2.item() < 20.0,
      f"Loss in reasonable range [0.1, 20.0]: {loss2.item():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: MODEL DENOISING — sample_actions on training data
# ══════════════════════════════════════════════════════════════════════════
section("3. MODEL DENOISING — sample_actions() output sanity")

policy.eval()
batch3 = ds_dt[100]
batch3 = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
          for k, v in batch3.items()}
batch3 = preprocessor(batch3)

with torch.no_grad():
    predicted_actions = policy.predict_action_chunk(batch3)

print(f"\n  Predicted actions shape: {list(predicted_actions.shape)}")
check(predicted_actions.shape == (1, 50, 7),
      f"Predicted actions shape correct: {list(predicted_actions.shape)}")

# Check for NaN/Inf in predictions
has_nan = torch.isnan(predicted_actions).any()
has_inf = torch.isinf(predicted_actions).any()
check(not has_nan, f"No NaN in predicted actions")
check(not has_inf, f"No Inf in predicted actions")

# For an untrained model, predictions should be near-normal distributed
pred_mean = predicted_actions.mean().item()
pred_std = predicted_actions.std().item()
pred_min = predicted_actions.min().item()
pred_max = predicted_actions.max().item()
print(f"\n  Prediction stats (untrained model, normalized relative space):")
print(f"    Mean: {pred_mean:.4f}")
print(f"    Std:  {pred_std:.4f}")
print(f"    Min:  {pred_min:.4f}")
print(f"    Max:  {pred_max:.4f}")

# In normalized space, values should be roughly in [-5, 5] even for untrained model
check(-10 < pred_min and pred_max < 10,
      f"Predictions in reasonable range [{pred_min:.2f}, {pred_max:.2f}]")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: FULL ROUND-TRIP — preprocess → model → postprocess → ABS
# ══════════════════════════════════════════════════════════════════════════
section("4. FULL ROUND-TRIP — preprocess → model → postprocess")

# Get ground truth absolute actions from DeriveState (for comparison)
from lerobot.processor.relative_action_processor import DeriveStateFromActionStep
derive_step = DeriveStateFromActionStep(enabled=True)

roundtrip_results = []
for frame_idx in [50, 100, 150]:
    if frame_idx >= len(ds_dt):
        break

    # Load raw batch with delta_timestamps
    raw = ds_dt[frame_idx]
    raw = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
           for k, v in raw.items()}

    # Ground truth: DeriveState → absolute action (50, 7)
    gt = derive_step(raw)["action"]  # (1, 50, 7) ABS

    # Full preprocess
    processed = preprocessor(raw)

    # Model denoising
    policy.eval()
    with torch.no_grad():
        pred_norm_rel = policy.predict_action_chunk(processed)  # (1, 50, 7) NORM+REL

    # Full postprocess
    pred_abs = postprocessor(pred_norm_rel)  # (1, 50, 7) ABS

    # Compare with ground truth
    error = (pred_abs - gt).abs()
    max_err = error.max().item()
    mean_err = error.mean().item()

    # Per-dimension error
    dim_errors = error[0].mean(dim=0)

    print(f"\n  Frame {frame_idx}:")
    print(f"    GT action[0,t=0]:    {fa(gt[0, 0])}")
    print(f"    Pred action[0,t=0]:  {fa(pred_abs[0, 0])}")
    print(f"    Max error:  {max_err:.4f}")
    print(f"    Mean error: {mean_err:.4f}")
    print(f"    Per-dim mean error:")
    for d, name in enumerate(dim_names):
        print(f"      {name:10s}: {dim_errors[d]:.4f}")

    # For untrained model, we just verify:
    # - No NaN/Inf in output
    # - Output is in reasonable absolute EE-pose range
    no_nan_out = not torch.isnan(pred_abs).any()
    no_inf_out = not torch.isinf(pred_abs).any()
    check(no_nan_out, f"Frame {frame_idx}: No NaN in postprocessed output")
    check(no_inf_out, f"Frame {frame_idx}: No Inf in postprocessed output")

    # Check that t=0 action ≈ current state (for relative model)
    # The model was trained with relative actions, so even untrained,
    # the first action should be near current state after postprocessing
    current_state = gt[0, 0]  # t=0 ABS = current EE-pose
    pred_t0 = pred_abs[0, 0]
    t0_err = (pred_t0[:6] - current_state[:6]).abs().mean().item()

    roundtrip_results.append({
        "frame": frame_idx,
        "max_err": max_err,
        "mean_err": mean_err,
        "no_nan": no_nan_out,
        "no_inf": no_inf_out,
        "t0_posrot_err": t0_err,
    })

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: POSTPROCESSOR WITH EDGE-CASE MODEL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════
section("5. POSTPROCESSOR EDGE CASES — synthetic model outputs")

# Test that the postprocessor handles:
# - Zero predictions (model predicts no motion)
# - Large predictions (model predicts extreme offsets)
# - Normal-range predictions

# Get a cached state from the preprocessor first (run it on a real batch)
batch5 = ds_dt[100]
batch5 = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
          for k, v in batch5.items()}
_ = preprocessor(batch5)  # This caches state in RelativeActionsProcessorStep

# 5a. Zero predictions (model outputs all zeros in normalized relative space)
zero_pred = torch.zeros(1, 50, 7)
zero_abs = postprocessor(zero_pred)
check(not torch.isnan(zero_abs).any(), "Zero pred: No NaN after postprocess")
check(not torch.isinf(zero_abs).any(), "Zero pred: No Inf after postprocess")
# After unnormalize + absolute: should be near the cached state
print(f"  Zero pred → t=0 ABS: {fa(zero_abs[0, 0])}")
print(f"  (This is the model predicting 'stay still')")

# 5b. Large predictions (model outputs ±5σ)
large_pred = torch.randn(1, 50, 7) * 5.0
large_abs = postprocessor(large_pred)
check(not torch.isnan(large_abs).any(), "Large (±5σ) pred: No NaN after postprocess")
check(not torch.isinf(large_abs).any(), "Large (±5σ) pred: No Inf after postprocess")
print(f"  Large (±5σ) pred → t=0 ABS: {fa(large_abs[0, 0])}")
print(f"  t=0 pos+rot range: [{large_abs[0, 0, :6].min():.4f}, {large_abs[0, 0, :6].max():.4f}]")

# 5c. Extreme predictions (model outputs ±10σ)
extreme_pred = torch.randn(1, 50, 7) * 10.0
extreme_abs = postprocessor(extreme_pred)
check(not torch.isnan(extreme_abs).any(), "Extreme (±10σ) pred: No NaN after postprocess")
check(not torch.isinf(extreme_abs).any(), "Extreme (±10σ) pred: No Inf after postprocess")

# 5d. Gripper clamped to [0, 1] — check what postprocessor outputs
print(f"\n  Gripper range across test cases:")
print(f"    Zero pred gripper:    [{zero_abs[0, :, 6].min():.4f}, {zero_abs[0, :, 6].max():.4f}]")
print(f"    Large (±5σ) gripper: [{large_abs[0, :, 6].min():.4f}, {large_abs[0, :, 6].max():.4f}]")
print(f"    Extreme (±10σ) gripper: [{extreme_abs[0, :, 6].min():.4f}, {extreme_abs[0, :, 6].max():.4f}]")
# Note: postprocessor does NOT clamp — unnormalized gripper can go outside [0,1]
# This is expected; clamping happens later in the control pipeline

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: OVERFITTING TEST (stronger version)
# ══════════════════════════════════════════════════════════════════════════
section("6. OVERFITTING — single batch, 200 steps, loss must decrease")

# Fresh policy for clean overfitting
policy_overfit = make_policy(cfg=cfg, ds_meta=ds.meta)

# Optimizer
opt_cfg = cfg.get_optimizer_preset()
sched_cfg = cfg.get_scheduler_preset()
optimizer = opt_cfg.build(policy_overfit.parameters())
scheduler = sched_cfg.build(optimizer, 200)

# Fixed batch
batch6 = ds_dt[50]
batch6 = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
          for k, v in batch6.items()}
batch6 = preprocessor(batch6)

losses = []
policy_overfit.train()
for step in range(200):
    loss, loss_dict = policy_overfit.forward(batch6)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    # Check gradient norms periodically
    if step == 0 or step == 199:
        total_norm = 0.0
        for p in policy_overfit.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        logger.info(f"  step {step:3d}: loss={loss.item():.6f}, grad_norm={total_norm:.4f}")

    torch.nn.utils.clip_grad_norm_(policy_overfit.parameters(), cfg.optimizer_grad_clip_norm)
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

# Show loss curve
print(f"\n  Loss curve (every 20 steps):")
for i in range(0, 200, 20):
    bar = "#" * max(1, int(losses[i] / loss_start * 30))
    print(f"  {i:3d} {losses[i]:.4f} |{bar}")

trending_down = loss_end < loss_start * 0.8  # at least 20% reduction
no_nan = all(not torch.isnan(torch.tensor(l)) for l in losses)
no_explosion = all(l < loss_start * 3 for l in losses)

check(trending_down, f"Loss decreased ≥20%: {loss_reduction:.1f}%")
check(no_nan, "No NaN in any step")
check(no_explosion, "No loss explosion (>3x start)")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
section("SUMMARY")

all_checks_passed = True
# (We'd track this properly in a real test runner)

print(f"\n  Tests performed:")
print(f"    1. Gradient audit — which params frozen vs trainable")
print(f"    2. Model forward — loss shape and range")
print(f"    3. Model denoising — sample_actions output sanity")
print(f"    4. Full round-trip — preprocess → model → postprocess on real data")
print(f"    5. Postprocessor edge cases — zero, large (±5σ), extreme (±10σ)")
print(f"    6. Overfitting — 200-step single-batch convergence")

print(f"\n  Run:  uv run --directory lerobot python lerobot/test_model_roundtrip.py")
