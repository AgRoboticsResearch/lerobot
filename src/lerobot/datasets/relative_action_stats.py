"""Stats computation for UMI-style relative rot6d actions.

Provides recompute_stats() and compute_relative_action_stats() for computing
normalization statistics on SE(3)-transformed relative actions with rot6d output.

This is a standalone module that does NOT modify the existing compute_stats.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lerobot.datasets.compute_stats import (
    RunningQuantileStats,
    aggregate_stats,
    compute_episode_stats,
)
from lerobot.datasets.utils import DATA_DIR, STATS_PATH, write_stats
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


# ============================================================================
# NumPy SE(3) rot6d Helpers
# ============================================================================


def _axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (..., 3) to rotation matrices (..., 3, 3) via Rodrigues."""
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True).clip(min=1e-7)
    k = axis_angle / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = np.zeros_like(kx)
    K = np.stack([zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], axis=-1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    I = np.eye(3, dtype=axis_angle.dtype)
    sin_t = np.sin(theta)[..., None]
    cos_t = np.cos(theta)[..., None]
    return I + sin_t * K + (1 - cos_t) * (K @ K)


def _matrix_to_rot6d_np(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices (..., 3, 3) to 6D rotation (first two rows)."""
    batch_dim = R.shape[:-2]
    return R[..., :2, :].copy().reshape(batch_dim + (6,))


def _pose_se3_relative_aa_to_rot6d_np(pose_from_aa: np.ndarray, pose_to_aa: np.ndarray) -> np.ndarray:
    """SE(3) relative: T_delta = inv(T_from) @ T_to.  7D aa input → 10D rot6d output.

    Input: (..., 7) [x,y,z,wx,wy,wz,gripper].
    Output: (..., 10) [dx,dy,dz, rot6d(6), gripper].
    """
    R_from = _axis_angle_to_matrix_np(pose_from_aa[..., 3:6])
    R_to = _axis_angle_to_matrix_np(pose_to_aa[..., 3:6])
    R_from_T = np.swapaxes(R_from, -2, -1)
    R_delta = R_from_T @ R_to

    dt = pose_to_aa[..., :3] - pose_from_aa[..., :3]
    t_delta = (R_from_T @ dt[..., None])[..., 0]
    rot6d_delta = _matrix_to_rot6d_np(R_delta)

    return np.concatenate([t_delta, rot6d_delta, pose_to_aa[..., 6:7]], axis=-1)


# ============================================================================
# Relative Action Stats
# ============================================================================


def _get_valid_chunk_starts(episode_indices: np.ndarray, chunk_size: int) -> np.ndarray:
    """Find start indices where chunk_size consecutive frames are from the same episode."""
    starts = []
    for ep_idx in np.unique(episode_indices):
        ep_mask = episode_indices == ep_idx
        ep_indices = np.where(ep_mask)[0]
        if len(ep_indices) < chunk_size:
            continue
        for i in range(len(ep_indices) - chunk_size + 1):
            if ep_indices[i + chunk_size - 1] - ep_indices[i] == chunk_size - 1:
                starts.append(ep_indices[i])
    return np.array(starts, dtype=np.int64)


def _compute_relative_chunk_batch(
    start_indices: np.ndarray,
    all_actions: np.ndarray,
    all_states: np.ndarray,
    chunk_size: int,
    relative_mask: np.ndarray,
) -> np.ndarray:
    """Vectorised relative-action computation for a batch of start indices.

    Converts 7D aa actions to 10D rot6d relative actions using SE(3) math.
    Returns (N * chunk_size, 10) float32 array.
    """
    if len(start_indices) == 0:
        return np.empty((0, 10), dtype=np.float32)

    offsets = np.arange(chunk_size)
    frame_idx = start_indices[:, None] + offsets[None, :]
    chunks = all_actions[frame_idx].copy()  # (N, chunk_size, 7)
    states = all_states[start_indices]       # (N, 7)

    # Broadcast state to match chunk timesteps
    state_expanded = np.broadcast_to(states[:, None, :], chunks.shape).copy()

    # SE(3) relative conversion: 7D aa → 10D rot6d
    chunks_rot6d = _pose_se3_relative_aa_to_rot6d_np(state_expanded, chunks)

    return chunks_rot6d.reshape(-1, 10)


def compute_relative_action_stats(
    hf_dataset,
    features: dict,
    chunk_size: int,
    exclude_joints: list[str] | None = None,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """Compute normalization statistics for relative rot6d actions.

    Iterates all valid action chunks within single episodes, converts to
    10D rot6d relative actions, and computes per-dimension statistics.
    """
    from lerobot.processor.relative_action_processor import RelativeRot6dActionsProcessorStep

    if exclude_joints is None:
        exclude_joints = ["gripper"]

    action_names = features.get(ACTION, {}).get("names")
    mask_step = RelativeRot6dActionsProcessorStep(
        enabled=True,
        exclude_joints=exclude_joints,
        action_names=action_names,
    )
    relative_mask = np.array(mask_step._build_mask(7), dtype=np.float32)

    logger.info("Loading action/state data for relative action stats...")
    all_actions = np.array(hf_dataset[ACTION], dtype=np.float32)
    episode_indices = np.array(hf_dataset["episode_index"])

    # Use action column as state source when observation.state doesn't exist
    if OBS_STATE in hf_dataset.features:
        all_states = np.array(hf_dataset[OBS_STATE], dtype=np.float32)
    else:
        logger.info("observation.state not in dataset — using action column as state source")
        all_states = all_actions

    valid_starts = _get_valid_chunk_starts(episode_indices, chunk_size)
    if len(valid_starts) == 0:
        raise RuntimeError(
            f"No valid chunks found (total_frames={len(episode_indices)}, chunk_size={chunk_size})"
        )

    logger.info(
        f"Computing relative rot6d action stats from {len(valid_starts)} chunks "
        f"(chunk_size={chunk_size})"
    )

    batch_size = 50_000
    batches = [valid_starts[i:i + batch_size] for i in range(0, len(valid_starts), batch_size)]

    running_stats = RunningQuantileStats()

    if num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(_compute_relative_chunk_batch, batch, all_actions, all_states, chunk_size, relative_mask)
                for batch in batches
            ]
            for future in as_completed(futures):
                running_stats.update(future.result())
    else:
        for batch in batches:
            running_stats.update(
                _compute_relative_chunk_batch(batch, all_actions, all_states, chunk_size, relative_mask)
            )

    stats = running_stats.get_statistics()
    logger.info(
        f"Relative rot6d action stats: mean_abs={np.abs(stats['mean']).mean():.4f}, "
        f"std={stats['std'].mean():.4f}"
    )
    return stats


# ============================================================================
# Full Stats Recomputation
# ============================================================================


def recompute_stats(
    dataset,
    skip_image_video: bool = True,
    relative_action: bool = False,
    relative_exclude_joints: list[str] | None = None,
    chunk_size: int = 50,
    num_workers: int = 0,
    derive_state_from_action: bool = False,
) -> dict:
    """Recompute stats with optional relative action support.

    When relative_action=True, computes action stats on 10D rot6d relative
    actions (converted from 7D aa via SE(3)).

    When derive_state_from_action=True, derives state from the action column
    for stats computation (no separate observation.state needed).

    Args:
        dataset: LeRobotDataset instance.
        skip_image_video: Skip image/video stats (keep existing).
        relative_action: Compute action stats in relative rot6d space.
        relative_exclude_joints: Joint names to keep absolute.
        chunk_size: Must match training chunk_size.
        num_workers: Parallel threads for stats computation.
        derive_state_from_action: Derive state from action column.

    Returns:
        Updated stats dict.
    """
    import pandas as pd

    features = dataset.meta.features
    meta_keys = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
    numeric_features = {
        k: v for k, v in features.items()
        if v["dtype"] not in ("image", "video", "string") and k not in meta_keys
    }

    features_to_compute = numeric_features if skip_image_video else {
        k: v for k, v in features.items() if v["dtype"] != "string" and k not in meta_keys
    }

    # Relative action stats
    if relative_action and ACTION in features:
        if relative_exclude_joints is None:
            relative_exclude_joints = ["gripper"]

        # When derive_state_from_action, use action column as state too
        if derive_state_from_action and OBS_STATE not in features:
            # Temporarily add OBS_STATE pointing to action for stats
            all_actions_for_state = np.array(dataset.hf_dataset[ACTION], dtype=np.float32)
            # Use action[t] as state for each chunk start
            relative_action_stats = _compute_relative_action_stats_derived(
                dataset.hf_dataset, features, chunk_size, relative_exclude_joints, num_workers,
            )
        else:
            relative_action_stats = compute_relative_action_stats(
                hf_dataset=dataset.hf_dataset,
                features=features,
                chunk_size=chunk_size,
                exclude_joints=relative_exclude_joints,
                num_workers=num_workers,
            )

        features_to_compute.pop(ACTION, None)
    else:
        relative_action_stats = None

    # Standard per-episode stats for remaining features
    if features_to_compute:
        logger.info(f"Computing standard stats for: {list(features_to_compute.keys())}")
        data_dir = dataset.root / DATA_DIR
        parquet_files = sorted(data_dir.glob("*/*.parquet"))
        numeric_keys = [k for k, v in features_to_compute.items() if v["dtype"] not in ("image", "video")]

        all_episode_stats = []
        for parquet_path in tqdm(parquet_files, desc="Computing stats"):
            df = pd.read_parquet(parquet_path)
            for ep_idx in sorted(df["episode_index"].unique()):
                ep_df = df[df["episode_index"] == ep_idx]
                episode_data = {}
                for key in numeric_keys:
                    if key in ep_df.columns:
                        values = ep_df[key].values
                        if hasattr(values[0], "__len__"):
                            episode_data[key] = np.stack(values)
                        else:
                            episode_data[key] = np.array(values)
                ep_stats = compute_episode_stats(episode_data, features_to_compute)
                all_episode_stats.append(ep_stats)

        new_stats = aggregate_stats(all_episode_stats) if all_episode_stats else {}
    else:
        new_stats = {}

    # Merge relative action stats
    if relative_action_stats is not None:
        new_stats[ACTION] = relative_action_stats

    # Also recompute state stats if derive_state_from_action
    if derive_state_from_action and relative_action:
        # State is derived from action, so use same relative stats approach
        # For 2-step relative state (20D), compute stats from derived state
        state_stats = _compute_relative_state_stats_derived(
            dataset.hf_dataset, features, chunk_size, relative_exclude_joints, num_workers,
        )
        if state_stats is not None:
            new_stats[OBS_STATE] = state_stats
            features_to_compute.pop(OBS_STATE, None)

    # Keep existing stats for features we didn't recompute
    if dataset.meta.stats:
        for key, value in dataset.meta.stats.items():
            if key not in new_stats:
                new_stats[key] = value

    # Write and update
    write_stats(new_stats, dataset.root)
    dataset.meta.stats = new_stats

    logger.info("Stats recomputed successfully")
    return new_stats


def _compute_relative_action_stats_derived(
    hf_dataset, features, chunk_size, exclude_joints, num_workers,
):
    """Compute relative action stats when state is derived from action column."""
    # Use action[t] as both state and action basis
    all_actions = np.array(hf_dataset[ACTION], dtype=np.float32)
    episode_indices = np.array(hf_dataset["episode_index"])

    valid_starts = _get_valid_chunk_starts(episode_indices, chunk_size + 1)  # +1 for state derivation
    if len(valid_starts) == 0:
        raise RuntimeError("No valid chunks for derived-state stats")

    # For derived state: state = action[t], future_actions = action[t+1:t+1+chunk_size]
    batch_size = 50_000
    batches = [valid_starts[i:i + batch_size] for i in range(0, len(valid_starts), batch_size)]
    running_stats = RunningQuantileStats()

    for batch in batches:
        chunks = []
        states = all_actions[batch]  # state at t
        for i, start in enumerate(batch):
            chunk = all_actions[start + 1:start + 1 + chunk_size]  # actions at t+1..t+chunk
            state_exp = np.broadcast_to(states[i], (chunk_size, 7)).copy()
            chunk_rel = _pose_se3_relative_aa_to_rot6d_np(state_exp, chunk)
            chunks.append(chunk_rel)
        running_stats.update(np.concatenate(chunks, axis=0))

    return running_stats.get_statistics()


def _compute_relative_state_stats_derived(
    hf_dataset, features, chunk_size, exclude_joints, num_workers,
):
    """Compute relative state stats (20D) derived from action column."""
    all_actions = np.array(hf_dataset[ACTION], dtype=np.float32)
    episode_indices = np.array(hf_dataset["episode_index"])

    valid_starts = _get_valid_chunk_starts(episode_indices, chunk_size + 1)
    if len(valid_starts) == 0:
        return None

    running_stats = RunningQuantileStats()

    for start in valid_starts[:10000]:  # Sample for speed
        state_prev = all_actions[start]       # action[t-1]
        state_curr = all_actions[start + 1]   # action[t]
        stacked = np.stack([state_prev, state_curr], axis=0)  # (2, 7)
        state_exp = np.broadcast_to(state_curr, (2, 7))
        relative = _pose_se3_relative_aa_to_rot6d_np(state_exp, stacked)  # (2, 10)
        running_stats.update(relative.flatten()[None])  # (1, 20)

    return running_stats.get_statistics()
