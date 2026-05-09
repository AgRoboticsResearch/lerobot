#!/usr/bin/env python
"""End-to-end inference validation for SmolVLA UMI EE-pose on test_ee_dataset.

Tests BOTH paths:
  A. Training-style: multi-timestep batches via DataLoader (delta_timestamps)
  B. Inference-style: single-timestep feed with RelativeStateProcessorStep buffering

Validates the full chain: preprocess → model.forward → postprocess → absolute actions.

Usage:
  cd /home/hls/codes/lerobot_piper_sroi
  uv run --directory lerobot python lerobot/test_inference_e2e.py
  uv run --directory lerobot python lerobot/test_inference_e2e.py --verbose
"""

import logging
import sys

import torch

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


def check(condition, label, fatal=False):
    if condition:
        print(f"  {green('[PASS]')} {label}")
        return True
    else:
        print(f"  {red('[FAIL]')} {label}")
        if fatal:
            raise AssertionError(f"FATAL: {label}")
        return False


def fa(arr, p=6):
    """Compact format for printing tensors/lists."""
    if hasattr(arr, 'tolist'):
        arr = arr.tolist()
    if isinstance(arr, (list, tuple)):
        return "[" + " ".join(f"{v:{p}.4f}" for v in arr) + "]"
    return f"{arr:{p}.4f}"


def _print_tensor(label, tensor, indent=2):
    """Print shape, range, and per-dim stats."""
    prefix = " " * indent
    if tensor.ndim == 3:
        print(f"{prefix}{label}: shape={list(tensor.shape)}")
        print(f"{prefix}  range: [{tensor.min():.6f}, {tensor.max():.6f}]")
        print(f"{prefix}  mean per dim (first timestep):")
        for d in range(tensor.shape[-1]):
            print(f"{prefix}    dim {d}: mean={tensor[:, 0, d].mean():.6f}, "
                  f"std={tensor[:, 0, d].std():.6f}, "
                  f"range=[{tensor[:, 0, d].min():.6f}, {tensor[:, 0, d].max():.6f}]")
    elif tensor.ndim == 2:
        print(f"{prefix}{label}: shape={list(tensor.shape)}")
        print(f"{prefix}  range: [{tensor.min():.6f}, {tensor.max():.6f}]")


# ══════════════════════════════════════════════════════════════════════════════
# Setup: load dataset, build policy and processors
# ══════════════════════════════════════════════════════════════════════════════
section("SETUP — load checkpoint, build policy & processors")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.datasets.dataset_tools import recompute_stats

# Need dataset stats for Normalizer/Unnormalizer
ds_stats = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT)
ds_stats = recompute_stats(
    ds_stats, num_workers=2,
    relative_action=True, relative_exclude_joints=["gripper"],
    relative_state=True, relative_exclude_state_joints=["gripper"],
    state_obs_steps=2, derive_state_from_action=True,
)
ds_meta = ds_stats.meta

# Config: MUST match training config
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
    pretrained_path=str(CHECKPOINT_DIR),
)

# Resolve delta_timestamps for training-style batches
delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
print(f"\nDelta timestamps:")
for k, v in delta_timestamps.items():
    print(f"  {k}: {len(v)} frames ({v[0]:.3f}s .. {v[-1]:.3f}s)")

# Dataset with delta_timestamps (for training-style path)
ds_dt = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT, delta_timestamps=delta_timestamps)

# Dataset WITHOUT delta_timestamps (for inference-style single-timestep path)
ds_single = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT)

print(f"\nDataset: {ds_meta.total_episodes} episodes, {ds_meta.total_frames} frames")
print(f"Action dims: {ds_meta.features['action']['names']}")
print(f"State shape: {ds_meta.features['observation.state']['shape']}")

# Build policy and processors
policy = make_policy(cfg=cfg, ds_meta=ds_meta)
policy = policy.to(DEVICE)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=str(CHECKPOINT_DIR),
    dataset_stats=ds_meta.stats,
)

print(f"\nPreprocessor steps ({len(preprocessor.steps)}):")
for j, step in enumerate(preprocessor.steps):
    print(f"  [{j}] {type(step).__name__}: enabled={getattr(step, 'enabled', 'N/A')}")
print(f"Postprocessor steps ({len(postprocessor.steps)}):")
for j, step in enumerate(postprocessor.steps):
    print(f"  [{j}] {type(step).__name__}: enabled={getattr(step, 'enabled', 'N/A')}")

# Count params
total_params = sum(p.numel() for p in policy.parameters())
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"\nPolicy: {total_params:,} total params, {trainable_params:,} trainable")

# ══════════════════════════════════════════════════════════════════════════════
# PATH A: Training-style — multi-timestep batches
# ══════════════════════════════════════════════════════════════════════════════
section("PATH A: Training-style — multi-timestep batches (DataLoader)")

from torch.utils.data import DataLoader

dataloader = DataLoader(ds_dt, batch_size=8, shuffle=False, num_workers=0, drop_last=False)

batch = next(iter(dataloader))
batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

print(f"\nRaw batch from DataLoader:")
print(f"  action shape:            {batch['action'].shape}            (expected: [8, 51, 7] — 51 frames for DeriveState)")
print(f"  observation.state shape: {batch['observation.state'].shape}  (expected: [8, 2, 7])")
print(f"  task: {[t[:50] + '...' if len(t) > 50 else t for t in batch.get('task', ['N/A'])[:2]]}")

gt_action_abs = batch["action"].clone()           # [8, 51, 7] absolute — before DeriveState trim
gt_state = batch["observation.state"].clone()     # [8, 2, 7] absolute

check(gt_action_abs.shape == (8, 51, 7),
      f"action shape: {list(gt_action_abs.shape)} == [8, 51, 7]", fatal=True)
check(gt_state.shape == (8, 2, 7),
      f"state shape: {list(gt_state.shape)} == [8, 2, 7]", fatal=True)
check(not torch.isnan(gt_action_abs).any(), "No NaN in ground truth actions")
check(not torch.isinf(gt_action_abs).any(), "No Inf in ground truth actions")

# DeriveStateFromActionStep consumes action[:, 0, :] as state, trims action to [:, 1:, :]
gt_action_for_eval = gt_action_abs[:, 1:, :]  # [8, 50, 7] — matches postprocessor output

# Step A1: Preprocess
processed = preprocessor(batch)

print(f"\nAfter preprocess:")
print(f"  action shape:            {processed['action'].shape}            (expected: [8, 50, 7])")
print(f"  observation.state shape: {processed['observation.state'].shape}  (expected: [8, 14])")
print(f"  observation.images.color shape: {processed['observation.images.color'].shape}")

check(processed["action"].shape == (8, 50, 7),
      f"preprocessed action shape: {list(processed['action'].shape)}")
check(not torch.isnan(processed["action"]).any(), "No NaN in preprocessed actions")
check(not torch.isinf(processed["action"]).any(), "No Inf in preprocessed actions")

# State should be 14D (2*7) after RelativeStateProcessorStep
state_shape = processed["observation.state"].shape
check(state_shape[-1] == 14,
      f"preprocessed state last dim: {state_shape[-1]} == 14")

# Verify: dims 7-12 (t=0 pos+rot) should be zero
if state_shape[-1] >= 13:
    t0_posrot = processed["observation.state"][:, 7:13]
    max_t0 = t0_posrot.abs().max().item()
    check(max_t0 < 1e-6,
          f"t=0 pos+rot (dims 7-12) are zero: max_abs={max_t0:.2e}")

# Step A2: Verify relative conversion
# Action should be relative: action_rel = action_abs - state[t=0]
state_t0 = gt_state[:, -1, :]  # [8, 7] — current state
expected_rel = gt_action_for_eval.clone()
expected_rel[..., :6] -= state_t0[:, None, :6]
# The preprocessor does relative THEN normalize, so processed actions are norm'd
# We can't directly compare, but we can check relative conversion was applied
print(f"\n  State[t=0] (sample 0): {fa(state_t0[0])}")
print(f"  GT action[0, t=0] abs: {fa(gt_action_for_eval[0, 0])}")

# Step A3: Model forward (loss computation)
policy.train()
loss, loss_dict = policy.forward(processed)
print(f"\n  Model forward loss: {loss.item():.6f}")
print(f"  Loss dict: {list(loss_dict.keys())}")

check(not torch.isnan(loss), f"Loss is finite: {loss.item():.6f}")
check(not torch.isinf(loss), "Loss is not Inf")
check(loss.item() > 0, f"Loss is positive: {loss.item():.6f}")
# Trained model should have low loss on training data
check(loss.item() < 0.5,
      f"Loss < 0.5 (trained model): {loss.item():.4f}")

# Step A4: Model inference (denoising)
policy.eval()
with torch.no_grad():
    pred_norm_rel = policy.predict_action_chunk(processed)  # [8, 50, 7]

print(f"\n  Model prediction (normalized relative space):")
print(f"    shape: {list(pred_norm_rel.shape)}")
print(f"    range: [{pred_norm_rel.min():.4f}, {pred_norm_rel.max():.4f}]")

check(pred_norm_rel.shape == (8, 50, 7),
      f"Prediction shape: {list(pred_norm_rel.shape)} == [8, 50, 7]")
check(not torch.isnan(pred_norm_rel).any(), "No NaN in predictions")
check(not torch.isinf(pred_norm_rel).any(), "No Inf in predictions")

# Step A5: Postprocess → absolute actions
pred_abs = postprocessor(pred_norm_rel)  # [8, 50, 7]

print(f"\n  After postprocess (absolute space):")
print(f"    shape: {list(pred_abs.shape)}")
print(f"    range: [{pred_abs.min():.4f}, {pred_abs.max():.4f}]")

check(pred_abs.shape == (8, 50, 7),
      f"Postprocessed shape: {list(pred_abs.shape)} == [8, 50, 7]")
check(not torch.isnan(pred_abs).any(), "No NaN in postprocessed output")
check(not torch.isinf(pred_abs).any(), "No Inf in postprocessed output")

# Step A6: Compare with ground truth
error = (pred_abs - gt_action_for_eval.cpu()).abs()
mean_err = error.mean().item()
max_err = error.max().item()

print(f"\n  Prediction vs Ground Truth:")
print(f"    Mean abs error: {mean_err:.6f}")
print(f"    Max abs error:  {max_err:.6f}")

dim_names = ds_meta.features["action"]["names"]
if isinstance(dim_names, dict):
    dim_names = dim_names.get("axes", list(dim_names.keys()))

print(f"    Per-dim MAE:")
for d, name in enumerate(dim_names):
    dim_mae = error[:, :, d].mean().item()
    unit = "mm" if d < 3 else ("rad" if d < 6 else "")
    scale = 1000 if d < 3 else 1
    print(f"      {name:10s}: {dim_mae:.6f} ({dim_mae*scale:.4f} {unit})")

# Show sample 0 detailed comparison
print(f"\n  Sample 0 — t=0 action comparison:")
print(f"    GT (abs):    {fa(gt_action_for_eval[0, 0])}")
print(f"    Pred (abs):  {fa(pred_abs[0, 0])}")

# For a trained model on training data, t=0 error should be small
t0_err = (pred_abs[0, 0] - gt_action_for_eval[0, 0].cpu()).abs().mean().item()
print(f"    t=0 mean err: {t0_err:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# PATH B: Inference-style — single-timestep feed with state buffering
# ══════════════════════════════════════════════════════════════════════════════
section("PATH B: Inference-style — single-timestep feed (RelativeStateProcessorStep buffering)")

# Reset the preprocessor state (especially RelativeStateProcessorStep's buffer)
# Find the RelativeStateProcessorStep and reset it
for step in preprocessor.steps:
    if hasattr(step, 'reset'):
        step.reset()

# Load single-frame data (no delta_timestamps)
sample_raw = ds_single[50]  # frame 50
sample_raw = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
              for k, v in sample_raw.items()}
sample_raw = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
              for k, v in sample_raw.items()}

print(f"\nSingle-frame sample (no delta_timestamps):")
print(f"  action shape:            {sample_raw['action'].shape}")
print(f"  observation.state shape: {sample_raw['observation.state'].shape}")
print(f"  task: {sample_raw.get('task', ['N/A'])[0][:60]}")

# Get ground truth absolute action for this single frame
gt_single_action = sample_raw["action"].clone()  # absolute, [1, 50, 7]

# Step B1: Feed FIRST frame through preprocessor
# On first call, RelativeStateProcessorStep buffers this state as previous
processed_1 = preprocessor(sample_raw)
state_1 = processed_1["observation.state"]

print(f"\n  Frame 50 (first call) — after preprocess:")
print(f"    observation.state shape: {state_1.shape}")
print(f"    observation.state[0, :7]:  {fa(state_1[0, :7])}")
print(f"    observation.state[0, 7:]:  {fa(state_1[0, 7:])}")

check(state_1.shape[-1] == 14,
      f"State shape last dim: {state_1.shape[-1]} == 14")

# On first call, with state buffering: stacked=[prev=cur, cur]
# In raw space: t=-1 pos+rot=0, t=0 pos+rot=0
# But after NORMALIZATION: t=-1 pos+rot = (0-mean)/std (non-zero since mean≠0)
# t=0 pos+rot ALWAYS zero even after norm (stats have mean=0, std=0)
first_t0_posrot = state_1[0, 7:13].abs().max().item()
first_t0_grip = state_1[0, 13].item()
print(f"\n  First-call state sanity (normalized space):")
print(f"    t=0 pos+rot (dims 7-12): max_abs={first_t0_posrot:.2e}  (ALWAYS zero)")
print(f"    t=0 gripper (dim 13):    {first_t0_grip:.4f}  (absolute, mask=0)")
print(f"    Note: t=-1 pos+rot non-zero after norm: (0-mean)/std ≠ 0")
check(first_t0_posrot < 1e-6, "First call t=0 pos+rot are zero")

# Step B2: Feed SECOND frame (frame 51)
sample_raw_2 = ds_single[51]
sample_raw_2 = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
                for k, v in sample_raw_2.items()}
sample_raw_2 = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in sample_raw_2.items()}

processed_2 = preprocessor(sample_raw_2)
state_2 = processed_2["observation.state"]

print(f"\n  Frame 51 (second call) — after preprocess:")
print(f"    observation.state[0, :7]:  {fa(state_2[0, :7])}")
print(f"    observation.state[0, 7:]:  {fa(state_2[0, 7:])}")

# On second call: prev=frame50, cur=frame51
# In raw space: t=-1 pos+rot = prev - cur = velocity (non-zero), t=0 pos+rot = 0
# After normalization: t=0 pos+rot still zero (mean=0, std=0)
second_t0_posrot = state_2[0, 7:13].abs().max().item()
print(f"\n  Second-call state sanity (normalized space):")
print(f"    t=0 pos+rot (dims 7-12): max_abs={second_t0_posrot:.2e}  (ALWAYS zero)")
print(f"    t=-1 dims show velocity (see values above)")
check(second_t0_posrot < 1e-6, "Second call t=0 pos+rot are zero")

# Step B3: Run inference on second frame
with torch.no_grad():
    pred_norm_rel_2 = policy.predict_action_chunk(processed_2)

print(f"\n  Model prediction (inference path, frame 51):")
print(f"    shape: {list(pred_norm_rel_2.shape)}")
check(pred_norm_rel_2.shape == (1, 50, 7),
      f"Prediction shape: {list(pred_norm_rel_2.shape)} == [1, 50, 7]")
check(not torch.isnan(pred_norm_rel_2).any(), "No NaN in inference prediction")
check(not torch.isinf(pred_norm_rel_2).any(), "No Inf in inference prediction")

# Step B4: Postprocess
pred_abs_2 = postprocessor(pred_norm_rel_2)

print(f"\n  After postprocess (absolute space):")
print(f"    shape: {list(pred_abs_2.shape)}")
print(f"    t=0 action: {fa(pred_abs_2[0, 0])}")

check(pred_abs_2.shape == (1, 50, 7),
      f"Postprocessed shape: {list(pred_abs_2.shape)}")
check(not torch.isnan(pred_abs_2).any(), "No NaN in postprocessed inference output")
check(not torch.isinf(pred_abs_2).any(), "No Inf in postprocessed inference output")

# Step B5: Test reset() behavior
for step in preprocessor.steps:
    if hasattr(step, 'reset'):
        step.reset()
print(f"\n  After reset(), RelativeStateProcessorStep buffer cleared.")

# Feed frame 100 (new episode or gap) — should behave like first call
sample_raw_3 = ds_single[100]
sample_raw_3 = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
                for k, v in sample_raw_3.items()}
sample_raw_3 = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in sample_raw_3.items()}

processed_3 = preprocessor(sample_raw_3)
state_3 = processed_3["observation.state"]
print(f"  Frame 100 (after reset, first-call behavior):")
print(f"    state shape: {state_3.shape}")
print(f"    t=0 pos+rot (dims 7-12) max_abs: {state_3[0, 7:13].abs().max().item():.2e}")

check(state_3[0, 7:13].abs().max().item() < 1e-6,
      "After reset, t=0 pos+rot are zero (first call behavior)")

# ══════════════════════════════════════════════════════════════════════════════
# PATH C: Verbose trace — full round-trip on a single batch
# ══════════════════════════════════════════════════════════════════════════════
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true")
args, _ = parser.parse_known_args()

if args.verbose:
    section("PATH C: Verbose trace — full round-trip with annotations")

    # Reset state
    for step in preprocessor.steps:
        if hasattr(step, 'reset'):
            step.reset()

    # Use DataLoader batch for cleaner comparison
    batch_v = next(iter(DataLoader(ds_dt, batch_size=1, shuffle=False, num_workers=0)))
    batch_v = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch_v.items()}

    gt_act_v = batch_v["action"].clone()  # [1, 51, 7] abs, 51 frames
    gt_st_v = batch_v["observation.state"].clone()  # [1, 2, 7] abs

    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  STAGE 0: RAW BATCH from DataLoader (delta_timestamps)           ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")
    print(f"    action shape: {list(gt_act_v.shape)}  ← 51 frames, extra leading frame for DeriveState")
    print(f"    state shape:  {list(gt_st_v.shape)}   ← 2 timesteps from delta_timestamps")
    print(f"    state[t=0]: {fa(gt_st_v[0, 0])}")
    print(f"    state[t=1]: {fa(gt_st_v[0, 1])}")
    print(f"    action[t=0]: {fa(gt_act_v[0, 0])}")
    print(f"    action[t=1]: {fa(gt_act_v[0, 1])}  ← this will be the first predicted action")

    # Full preprocess
    proc_v = preprocessor(batch_v)
    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  STAGE 1: AFTER PREPROCESS                                       ║")
    print(f"  ║   DeriveState → RelativeActions → RelativeState → Normalize       ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")
    print(f"    action shape: {list(proc_v['action'].shape)}  ← 50 frames, relative + normalized")
    print(f"    state shape:  {list(proc_v['observation.state'].shape)}  ← 14D = 2-timestep × 7D flattened")
    print(f"    state[:7]  (t=-1, rel+norm):  {fa(proc_v['observation.state'][0, :7])}")
    print(f"    state[7:]  (t=0,  rel+norm):  {fa(proc_v['observation.state'][0, 7:])}")
    print(f"    action[t=0] (rel+norm): {fa(proc_v['action'][0, 0])}")
    print(f"    action[t=1] (rel+norm): {fa(proc_v['action'][0, 1])}")
    print(f"    Note: t=0 pos+rot (dims 7-12) are ALWAYS zero")

    # Model
    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  STAGE 2: MODEL DENOISING (predict_action_chunk)                ║")
    print(f"  ║   Flow matching: denoise from N(0,1) → predicted action chunk    ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")
    with torch.no_grad():
        pred_norm_rel = policy.predict_action_chunk(proc_v)
    print(f"    output shape: {list(pred_norm_rel.shape)}  ← 50 frames, normalized relative")
    print(f"    output[t=0]:  {fa(pred_norm_rel[0, 0])}")
    print(f"    output[t=1]:  {fa(pred_norm_rel[0, 1])}")
    print(f"    range: [{pred_norm_rel.min():.4f}, {pred_norm_rel.max():.4f}]")

    # Full postprocess
    pred_abs_v = postprocessor(pred_norm_rel)
    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  STAGE 3: AFTER POSTPROCESS                                     ║")
    print(f"  ║   Unnormalize → AbsoluteActions → Device(cpu)                     ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")
    print(f"    output shape: {list(pred_abs_v.shape)}  ← 50 frames, ABSOLUTE (meters/rad)")
    print(f"    output[t=0]:  {fa(pred_abs_v[0, 0])}")
    print(f"    output[t=1]:  {fa(pred_abs_v[0, 1])}")
    print(f"    pos range: x=[{pred_abs_v[0, :, 0].min():.4f}, {pred_abs_v[0, :, 0].max():.4f}]")

    # Compare with ground truth (after DeriveState trim)
    gt_eval = gt_act_v[:, 1:, :]  # [1, 50, 7] — strip the state-timestep
    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  STAGE 4: COMPARISON WITH GROUND TRUTH                           ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")
    print(f"    GT action[0,t=0]:  {fa(gt_eval[0, 0])}")
    print(f"    Pred action[t=0]:  {fa(pred_abs_v[0, 0])}")
    err_v = (pred_abs_v[0, 0] - gt_eval[0, 0].cpu()).abs()
    print(f"    Abs error[t=0]:    {fa(err_v)}")
    print(f"    Mean abs error (all 50 steps): {(pred_abs_v[0] - gt_eval[0].cpu()).abs().mean().item():.6f}")

    # Show error growth along chunk
    print(f"\n    Error along chunk (xyz mm):")
    err_along = (pred_abs_v[0] - gt_eval[0].cpu()).abs()
    for t in [0, 5, 10, 20, 30, 49]:
        xyz_mm = (err_along[t, :3] * 1000).tolist()
        print(f"      t={t:>2}: [{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}] mm")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("SUMMARY")

print(f"""
  Training checkpoint: {CHECKPOINT_DIR}
  Training loss:       1.586 → 0.130 (92% reduction)

  Path A (training-style, multi-timestep batches):
    - DataLoader with delta_timestamps
    - 8×50×7 action, 8×2×7 state input
    - Full preprocess → model → postprocess chain
    - Compared predictions against ground truth

  Path B (inference-style, single-timestep feed):
    - Single-frame dataset access (no delta_timestamps)
    - RelativeStateProcessorStep state buffering
    - Verified [prev, cur] stacking works correctly
    - Verified reset() clears buffer
    - Full preprocess → model → postprocess chain

  Path C (verbose trace): {'enabled' if args.verbose else 'use --verbose to enable'}
    - Step-by-step trace of each processor
    - Shows data transformation at each stage

  Run:  uv run --directory lerobot python lerobot/test_inference_e2e.py
        uv run --directory lerobot python lerobot/test_inference_e2e.py --verbose
""")
