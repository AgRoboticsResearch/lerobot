"""Statistics for UMI-style, processor-derived EE actions and state."""

from __future__ import annotations

import logging

import numpy as np

from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True).clip(min=1e-7)
    axis = axis_angle / theta
    kx, ky, kz = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(kx)
    skew = np.stack(
        [zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], axis=-1
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    identity = np.eye(3, dtype=axis_angle.dtype)
    return identity + np.sin(theta)[..., None] * skew + (1 - np.cos(theta)[..., None]) * (
        skew @ skew
    )


def _absolute_aa_to_relative_rot6d(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    reference_rotation = _axis_angle_to_matrix(reference[..., 3:6])
    target_rotation = _axis_angle_to_matrix(target[..., 3:6])
    reference_rotation_t = np.swapaxes(reference_rotation, -2, -1)
    relative_rotation = reference_rotation_t @ target_rotation
    relative_translation = (
        reference_rotation_t @ (target[..., :3] - reference[..., :3])[..., None]
    )[..., 0]
    rot6d = relative_rotation[..., :2, :].copy().reshape(*relative_rotation.shape[:-2], 6)
    return np.concatenate([relative_translation, rot6d, target[..., 6:7]], axis=-1)


def _valid_query_starts(episode_indices: np.ndarray, query_length: int) -> np.ndarray:
    """Return starts for contiguous ``[t-1, t, ..., t+chunk-1]`` queries."""
    starts: list[np.ndarray] = []
    for episode in np.unique(episode_indices):
        indices = np.flatnonzero(episode_indices == episode)
        if len(indices) < query_length:
            continue
        candidates = indices[: len(indices) - query_length + 1]
        ends = indices[query_length - 1 :]
        starts.append(candidates[ends - candidates == query_length - 1])
    return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)


def compute_umi_relative_ee_stats(hf_dataset, chunk_size: int) -> dict[str, dict[str, np.ndarray]]:
    """Compute stats for the exact tensors produced by the UMI processors.

    The second queried action is the chunk base and first retained target. This
    intentionally includes the identity/current-pose target at action index 0.
    """
    actions = np.asarray(hf_dataset[ACTION], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            "UMI π0.5 training requires action shape [frames, 7] with "
            f"[xyz, axis-angle, gripper], got {actions.shape}"
        )
    episode_indices = np.asarray(hf_dataset["episode_index"])
    starts = _valid_query_starts(episode_indices, chunk_size + 1)
    if len(starts) == 0:
        raise ValueError(
            f"No episode contains the required {chunk_size + 1} contiguous frames "
            f"for chunk_size={chunk_size}"
        )

    logger.info("Computing UMI relative-EE stats from %d valid chunks", len(starts))
    action_stats = RunningQuantileStats()
    state_stats = RunningQuantileStats()
    offsets = np.arange(1, chunk_size + 1)

    # Keep temporary [num_chunks, chunk_size, 7] arrays bounded on large datasets.
    for batch_start in range(0, len(starts), 20_000):
        batch = starts[batch_start : batch_start + 20_000]
        base = actions[batch + 1]
        targets = actions[batch[:, None] + offsets[None, :]]
        expanded_base = np.broadcast_to(base[:, None, :], targets.shape)
        relative_actions = _absolute_aa_to_relative_rot6d(expanded_base, targets)
        action_stats.update(relative_actions)

        state_pair = np.stack([actions[batch], base], axis=1)
        state_base = np.broadcast_to(base[:, None, :], state_pair.shape)
        relative_state = _absolute_aa_to_relative_rot6d(state_base, state_pair).reshape(-1, 20)
        state_stats.update(relative_state)

    return {
        ACTION: action_stats.get_statistics(),
        "observation.state": state_stats.get_statistics(),
    }
