#!/usr/bin/env python
"""Manually recompute relative action + state stats from raw data and verify
against ``recompute_stats`` output.

This proves the stats computation is correct by deriving the same numbers
independently from first principles — no processor code, just numpy.

Usage:
  cd /home/hls/codes/lerobot_piper_sroi
  uv run --directory lerobot python lerobot/verify_relative_stats.py
"""

import numpy as np

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/test_ee_dataset"
EPS = 1e-8

# ── Load raw data ──────────────────────────────────────────────────────
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats

ds = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT)
ds = recompute_stats(
    ds, num_workers=2,
    relative_action=True, relative_exclude_joints=["gripper"],
    relative_state=True, relative_exclude_state_joints=["gripper"],
    state_obs_steps=2, derive_state_from_action=True,
)

official_action_mean = np.array(ds.meta.stats["action"]["mean"])
official_action_std  = np.array(ds.meta.stats["action"]["std"])
official_state_mean  = np.array(ds.meta.stats["observation.state"]["mean"])
official_state_std   = np.array(ds.meta.stats["observation.state"]["std"])

# Raw data from the HuggingFace dataset (parquet)
hf = ds.hf_dataset
# Use float32 — same as compute_relative_action_stats (line 714 of compute_stats.py)
raw_action = np.array(hf["action"], dtype=np.float32)          # [334, 7]
raw_state  = np.array(hf["observation.state"], dtype=np.float32)  # [334, 7]
episode_idx = np.array(hf["episode_index"])                    # [334]
chunk_size = 50
state_obs_steps = 2
mask = np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.float32)      # pos+rot=1, gripper=0
dim_names = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper"]

n_frames = len(raw_action)
print(f"Dataset: {n_frames} frames, {ds.meta.episodes} episodes")
print(f"Action shape: {raw_action.shape}")
print(f"State shape:  {raw_state.shape}")
print(f"Mask: {mask}")

# ══════════════════════════════════════════════════════════════════════════
# 1. MANUAL RELATIVE ACTION STATS
# ══════════════════════════════════════════════════════════════════════════

# Find all valid chunk starts (chunk of 50 within same episode)
max_start = n_frames - chunk_size
all_starts = np.arange(max_start + 1)
valid_starts = all_starts[
    episode_idx[all_starts] == episode_idx[all_starts + chunk_size - 1]
]
n_chunks = len(valid_starts)
print(f"\nValid chunk starts: {n_chunks} (need >=50 consecutive same-episode frames)")

# Build all relative action frames
# For each chunk start t, for each k in [0, 50):
#   rel[t, k, :] = action[t+k, :] - state[t, :] * mask
all_rel_actions = []
for t in valid_starts:
    chunk = raw_action[t:t+chunk_size].copy()          # (50, 7)
    state_t = raw_state[t]                              # (7,)
    # Subtract state from all 50 timesteps, masked
    chunk[:, :] -= state_t[None, :] * mask[None, :]
    all_rel_actions.append(chunk)

all_rel_actions = np.concatenate(all_rel_actions, axis=0)  # [n_chunks*50, 7]
print(f"Total relative action frames: {all_rel_actions.shape[0]} ({n_chunks} chunks × 50)")

# Compute manual stats
manual_action_mean = all_rel_actions.mean(axis=0)
manual_action_std  = all_rel_actions.std(axis=0)

print(f"\n{'Dim':>10s}  {'Official mean':>14s}  {'Manual mean':>14s}  {'Δ mean':>12s}  │  {'Official std':>14s}  {'Manual std':>14s}  {'Δ std':>12s}")
print(f"{'─'*10}  {'─'*14}  {'─'*14}  {'─'*12}  │  {'─'*14}  {'─'*14}  {'─'*12}")
for i, name in enumerate(dim_names):
    d_mean = abs(official_action_mean[i] - manual_action_mean[i])
    d_std  = abs(official_action_std[i] - manual_action_std[i])
    print(f"{name:>10s}  {official_action_mean[i]:14.6f}  {manual_action_mean[i]:14.6f}  {d_mean:12.2e}  │  {official_action_std[i]:14.6f}  {manual_action_std[i]:14.6f}  {d_std:12.2e}")

max_mean_err = abs(official_action_mean - manual_action_mean).max()
max_std_err  = abs(official_action_std - manual_action_std).max()
print(f"\n  Max mean error: {max_mean_err:.2e}")
print(f"  Max std error:  {max_std_err:.2e}  (Welford online vs numpy batch, float32 roundoff)")
# Float32 precision for 11800-element accumulation is ~1e-4
threshold = 1e-4
print(f"  Result: {'MATCH ✓' if max_mean_err < threshold and max_std_err < threshold else 'MISMATCH ✗'}")

# ══════════════════════════════════════════════════════════════════════════
# 2. MANUAL RELATIVE STATE STATS (14D)
# ══════════════════════════════════════════════════════════════════════════

# derive_state_from_action=True → source_key="action"
# state_obs_steps=2 → window of 2 consecutive action frames
# relative_exclude_state_joints=["gripper"] → mask for state

# Build the relative state mask from state names
# State names come from action names (source_key="action")
state_mask = np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.float64)  # same as action mask

# Find valid windows of 2 consecutive frames within same episode
max_start_state = n_frames - state_obs_steps
all_starts_state = np.arange(max_start_state + 1)
valid_state_starts = all_starts_state[
    episode_idx[all_starts_state] == episode_idx[all_starts_state + state_obs_steps - 1]
]
n_windows = len(valid_state_starts)
print(f"\n\nValid state windows: {n_windows} (need >=2 consecutive same-episode frames)")

# For each valid window [t, t+1]:
#   window = [action[t], action[t+1]]              # (2, 7)
#   current = action[t+1]                          # (7,)
#   rel_window = window - current * mask           # (2, 7)
#   flattened = rel_window.ravel()                 # (14,)
all_rel_states = []
for t in valid_state_starts:
    window = raw_action[t:t+state_obs_steps].copy()   # (2, 7)
    current = window[-1]                                # (7,) — action[t+1]
    # Apply relative conversion: subtract current from all timesteps, masked
    window -= current[None, :] * state_mask[None, :]
    flattened = window.ravel()                         # (14,)
    all_rel_states.append(flattened)

all_rel_states = np.array(all_rel_states)  # [n_windows, 14]
print(f"Total relative state vectors: {all_rel_states.shape[0]} ({n_windows} windows)")

manual_state_mean = all_rel_states.mean(axis=0)
manual_state_std  = all_rel_states.std(axis=0)

print(f"\n{'Idx':>4s}  {'Source':>20s}  {'Official mean':>14s}  {'Manual mean':>14s}  {'Δ mean':>12s}  │  {'Official std':>14s}  {'Manual std':>14s}  {'Δ std':>12s}")
print(f"{'─'*4}  {'─'*20}  {'─'*14}  {'─'*14}  {'─'*12}  │  {'─'*14}  {'─'*14}  {'─'*12}")
for i in range(14):
    timestep = "t=-1" if i < 7 else "t=0"
    dim = dim_names[i % 7]
    d_mean = abs(official_state_mean[i] - manual_state_mean[i])
    d_std  = abs(official_state_std[i] - manual_state_std[i])
    print(f"[{i:2d}]  {timestep + ' ' + dim:>20s}  {official_state_mean[i]:14.6f}  {manual_state_mean[i]:14.6f}  {d_mean:12.2e}  │  {official_state_std[i]:14.6f}  {manual_state_std[i]:14.6f}  {d_std:12.2e}")

max_state_mean_err = abs(official_state_mean - manual_state_mean).max()
max_state_std_err  = abs(official_state_std - manual_state_std).max()
print(f"\n  Max mean error: {max_state_mean_err:.2e}")
print(f"  Max std error:  {max_state_std_err:.2e}")
print(f"  Result: {'MATCH ✓' if max_state_mean_err < 1e-5 and max_state_std_err < 1e-5 else 'MISMATCH ✗'}")

# ══════════════════════════════════════════════════════════════════════════
# 3. VERIFY KEY PROPERTIES OF THE COMPUTED STATS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"  KEY PROPERTIES")
print(f"{'═'*70}")

# 3a. t=0 pos+rot should be EXACTLY zero in every window
t0_posrot = all_rel_states[:, 7:13]  # dims 7-12: t=0, pos+rot
max_t0 = np.abs(t0_posrot).max()
print(f"  t=0 pos+rot max abs value across all {n_windows} windows: {max_t0:.2e}")
print(f"  {'PASS: always zero ✓' if max_t0 < 1e-10 else 'FAIL: non-zero values exist ✗'}")

# 3b. t=-1 pos+rot should show velocity (non-zero in at least some windows)
t_neg1_posrot = all_rel_states[:, :6]  # dims 0-5: t=-1, pos+rot
t_neg1_max = np.abs(t_neg1_posrot).max()
print(f"  t=-1 pos+rot max abs value across all {n_windows} windows: {t_neg1_max:.6f}")
print(f"  {'PASS: velocity captured ✓' if t_neg1_max > 1e-6 else 'WARNING: zero velocity everywhere?'}")

# 3c. Gripper in state: NOT zero for either timestep (absolute values preserved)
t_neg1_grip = all_rel_states[:, 6]   # dim 6: t=-1 gripper
t_0_grip = all_rel_states[:, 13]      # dim 13: t=0 gripper
print(f"  t=-1 gripper mean: {t_neg1_grip.mean():.6f} (std={t_neg1_grip.std():.6f})")
print(f"  t=0  gripper mean: {t_0_grip.mean():.6f} (std={t_0_grip.std():.6f})")
print(f"  Gripper values are absolute (NOT zeroed by relative conversion)")

# 3d. Verify t=-1 gripper and t=0 gripper come from different frames
# t=-1 gripper = action[t, gripper], t=0 gripper = action[t+1, gripper]
# They should be highly correlated (consecutive frames) but not identical
grip_corr = np.corrcoef(t_neg1_grip, t_0_grip)[0, 1]
print(f"  t=-1 ↔ t=0 gripper correlation: {grip_corr:.6f}")
print(f"  {'PASS: consecutive frames highly correlated ✓' if grip_corr > 0.9 else 'Low correlation — investigate'}")

# 3e. Show a few example windows
print(f"\n  Example windows (first 3):")
print(f"  {'Window':>8s}  {'t=-1 ee.x':>12s}  {'t=-1 ee.y':>12s}  {'t=0 ee.x':>12s}  {'t=0 ee.y':>12s}  {'t=-1 grip':>12s}  {'t=0 grip':>12s}")
print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
for w in range(min(5, n_windows)):
    print(f"  {w:8d}  {all_rel_states[w, 0]:12.6f}  {all_rel_states[w, 1]:12.6f}  {all_rel_states[w, 7]:12.6f}  {all_rel_states[w, 8]:12.6f}  {all_rel_states[w, 6]:12.6f}  {all_rel_states[w, 13]:12.6f}")

print(f"\n{'═'*70}")
print(f"  VERIFICATION COMPLETE")
print(f"{'═'*70}")
