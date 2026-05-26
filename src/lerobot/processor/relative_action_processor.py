"""Processor steps for UMI-style relative EE actions with rot6d representation.

This module implements UMI-style processor-pipeline relative EE actions with
rot6d (10D) representation for rotation instead of axis-angle (3D).

Pipeline:
  Training:
    LeRobotDataset loads action [B, chunk+1, 7] (7D aa, extra timestep for state derivation)
      → DeriveStateFromActionStep: state=[action[t-1],action[t]] (7D×2), action=action[1:] (7D)
      → RelativeRot6dActionsProcessorStep: 7D aa → SE(3) relative → 10D rot6d
      → RelativeRot6dStateProcessorStep: [prev_aa, curr_aa] → relative rot6d → flatten 20D
      → NormalizerProcessorStep → model

  Inference:
    Robot FK → observation.state [7D aa]
      → RelativeRot6dActionsProcessorStep (caches state)
      → RelativeRot6dStateProcessorStep (buffers prev, converts to 20D rot6d)
      → Normalizer → model → 10D rot6d relative
      → Unnormalizer
      → AbsoluteRot6dActionsProcessorStep: 10D rot6d → 7D aa absolute
      → IK → joints

References:
  - lerobot_hls/src/lerobot/processor/relative_action_processor.py (UMI-style reference)
  - lerobot/src/lerobot/datasets/relative_ee_dataset.py (Pattern A rot6d math)
  - https://github.com/real-stanford/universal_manipulation_interface
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import ACTION, OBS_STATE


# ============================================================================
# SE(3) Helper Functions (PyTorch)
# ============================================================================


def _axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to rotation matrices via Rodrigues formula."""
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    k = axis_angle / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack([zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], dim=-1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    return I + sin_t * K + (1 - cos_t) * (K @ K)


def _matrix_to_axis_angle(R: Tensor) -> Tensor:
    """Convert rotation matrices to axis-angle vectors."""
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    vec = torch.stack([rx, ry, rz], dim=-1)
    sin_theta = vec.norm(dim=-1)
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    theta = torch.atan2(sin_theta / 2, cos_theta)
    near_zero = sin_theta.abs() < 1e-7
    result = vec / (sin_theta.unsqueeze(-1) + 1e-8) * theta.unsqueeze(-1)
    return torch.where(near_zero.unsqueeze(-1), vec / 2, result)


def _rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """Convert 6D rotation (first two rows of rotation matrix) to 3x3 rotation matrix.

    Follows UMI's row-based convention: rot6d = [row0(3), row1(3)].
    Reconstructs third row via Gram-Schmidt + cross product.
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]

    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-2)


def _matrix_to_rot6d(R: Tensor) -> Tensor:
    """Convert 3x3 rotation matrix to 6D rotation (first two rows).

    Follows UMI's row-based convention.
    """
    batch_dim = R.shape[:-2]
    return R[..., :2, :].clone().reshape(batch_dim + (6,))


def _pose_aa_to_matrix(aa_pose: Tensor) -> tuple[Tensor, Tensor]:
    """Convert 7D pose [x,y,z,wx,wy,wz,gripper] to (4,4) matrix + gripper.

    Returns (matrix, gripper) tuple. Gripper is not part of the SE(3) transform.
    """
    pos = aa_pose[..., :3]
    rotvec = aa_pose[..., 3:6]
    gripper = aa_pose[..., 6:7]

    R = _axis_angle_to_matrix(rotvec)
    T = torch.zeros(*aa_pose.shape[:-1], 4, 4, device=aa_pose.device, dtype=aa_pose.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T, gripper


def _pose_se3_relative_aa_to_rot6d(pose_from_aa: Tensor, pose_to_aa: Tensor) -> Tensor:
    """SE(3) relative: T_delta = inv(T_from) @ T_to.  Input/output: 7D aa → 10D rot6d.

    Args:
        pose_from_aa: (..., 7) reference pose [x,y,z,wx,wy,wz,gripper].
        pose_to_aa:   (..., 7) target pose [x,y,z,wx,wy,wz,gripper].

    Returns:
        (..., 10) relative pose [dx,dy,dz, rot6d(6), gripper].
    """
    T_from, _ = _pose_aa_to_matrix(pose_from_aa)
    T_to, gripper_to = _pose_aa_to_matrix(pose_to_aa)

    # T_delta = T_from^{-1} @ T_to
    R_from = T_from[..., :3, :3]
    t_from = T_from[..., :3, 3]
    R_from_T = R_from.transpose(-2, -1)

    R_delta = R_from_T @ T_to[..., :3, :3]
    dt = T_to[..., :3, 3] - t_from
    t_delta = (R_from_T @ dt.unsqueeze(-1)).squeeze(-1)

    rot6d_delta = _matrix_to_rot6d(R_delta)
    return torch.cat([t_delta, rot6d_delta, gripper_to], dim=-1)


def _pose_se3_absolute_rot6d_to_aa(pose_delta_rot6d: Tensor, pose_ref_aa: Tensor) -> Tensor:
    """SE(3) absolute: T = T_ref @ T_delta.  Input: 10D rot6d + 7D aa ref → 7D aa.

    Args:
        pose_delta_rot6d: (..., 10) relative pose [dx,dy,dz, rot6d(6), gripper].
        pose_ref_aa:      (..., 7) reference pose [x,y,z,wx,wy,wz,gripper].

    Returns:
        (..., 7) absolute pose [x,y,z,wx,wy,wz,gripper].
    """
    t_delta = pose_delta_rot6d[..., :3]
    rot6d_delta = pose_delta_rot6d[..., 3:9]
    gripper = pose_delta_rot6d[..., 9:10]

    T_ref, _ = _pose_aa_to_matrix(pose_ref_aa)
    R_ref = T_ref[..., :3, :3]
    t_ref = T_ref[..., :3, 3]

    R_delta = _rot6d_to_matrix(rot6d_delta)
    R_abs = R_ref @ R_delta
    t_abs = t_ref + (R_ref @ t_delta.unsqueeze(-1)).squeeze(-1)

    aa_abs = _matrix_to_axis_angle(R_abs)
    return torch.cat([t_abs, aa_abs, gripper], dim=-1)


# ============================================================================
# Batch Action/State Conversion Functions
# ============================================================================


def to_relative_actions_rot6d(
    actions_aa: Tensor,
    state_aa: Tensor,
    mask: Sequence[bool],
) -> Tensor:
    """Convert absolute 7D aa actions to relative 10D rot6d actions.

    Args:
        actions_aa: (B, T, 7) or (B, 7) absolute actions in axis-angle.
        state_aa: (B, 7) or (B, obs_steps, 7) current state. Last obs step used.
        mask: Which dims to convert. Length should cover action_dim.
              True = convert to relative, False = keep absolute.

    Returns:
        (B, T, 10) or (B, 10) relative actions in rot6d format.
    """
    mask_t = torch.tensor(mask, dtype=actions_aa.dtype, device=actions_aa.device)
    dims = mask_t.shape[0]

    if state_aa.device != actions_aa.device or state_aa.dtype != actions_aa.dtype:
        state_aa = state_aa.to(device=actions_aa.device, dtype=actions_aa.dtype)
    if state_aa.ndim == 3:
        state_aa = state_aa[:, -1, :]

    actions_aa = actions_aa.clone()

    # All first 6 dims (xyz + rotation) use SE(3), gripper uses element-wise
    # For rot6d output, we always need to expand from 7D aa to 10D rot6d
    if actions_aa.ndim == 3:
        # (B, T, 7) → (B, T, 10)
        B, T, _ = actions_aa.shape
        state_expanded = state_aa.unsqueeze(1).expand(B, T, 7)

        # Compute SE(3) relative for pose dims (first 6)
        relative_rot6d = _pose_se3_relative_aa_to_rot6d(state_expanded[..., :7], actions_aa[..., :7])
        # relative_rot6d is (B, T, 10) — already includes gripper from action

        # Apply mask for gripper: if mask[6] is False, keep original gripper
        if dims > 6 and not mask_t[6]:
            relative_rot6d[..., 9] = actions_aa[..., 6]

        return relative_rot6d
    else:
        # (B, 7) → (B, 10)
        relative_rot6d = _pose_se3_relative_aa_to_rot6d(state_aa[..., :7], actions_aa[..., :7])
        if dims > 6 and not mask_t[6]:
            relative_rot6d[..., 9] = actions_aa[..., 6]
        return relative_rot6d


def to_absolute_actions_rot6d(
    actions_rot6d: Tensor,
    state_aa: Tensor,
    mask: Sequence[bool],
) -> Tensor:
    """Convert relative 10D rot6d actions back to absolute 7D aa actions.

    Args:
        actions_rot6d: (B, T, 10) or (B, 10) relative rot6d actions.
        state_aa: (B, 7) or (B, obs_steps, 7) cached reference state.
        mask: Which dims were converted. Same semantics as to_relative_actions_rot6d.

    Returns:
        (B, T, 7) or (B, 7) absolute actions in axis-angle.
    """
    mask_t = torch.tensor(mask, dtype=actions_rot6d.dtype, device=actions_rot6d.device)
    dims = mask_t.shape[0]

    if state_aa.device != actions_rot6d.device or state_aa.dtype != actions_rot6d.dtype:
        state_aa = state_aa.to(device=actions_rot6d.device, dtype=actions_rot6d.dtype)
    if state_aa.ndim == 3:
        state_aa = state_aa[:, -1, :]

    if actions_rot6d.ndim == 3:
        B, T, _ = actions_rot6d.shape
        state_expanded = state_aa.unsqueeze(1).expand(B, T, 7)
        aa_abs = _pose_se3_absolute_rot6d_to_aa(actions_rot6d[..., :10], state_expanded[..., :7])
        if dims > 6 and not mask_t[6]:
            aa_abs[..., 6] = actions_rot6d[..., 9]
        return aa_abs
    else:
        aa_abs = _pose_se3_absolute_rot6d_to_aa(actions_rot6d[..., :10], state_aa[..., :7])
        if dims > 6 and not mask_t[6]:
            aa_abs[..., 6] = actions_rot6d[..., 9]
        return aa_abs


def to_relative_state_rot6d(
    state_aa: Tensor,
    mask: Sequence[bool],
) -> Tensor:
    """Convert 2-step 7D aa state to 20D relative rot6d state.

    Each timestep is expressed as an SE(3) offset from the current (last) timestep.

    Args:
        state_aa: (B, 2, 7) — 2 timesteps of 7D aa poses.
        mask: Which dims to convert.

    Returns:
        (B, 20) — flattened 2×10D relative rot6d state.
    """
    # Current state is the last timestep
    current = state_aa[:, -1:, :].expand_as(state_aa)  # (B, 2, 7)

    # For each timestep, compute SE(3) relative to current
    # This gives (B, 2, 10) — each step relative to current
    relative_rot6d = _pose_se3_relative_aa_to_rot6d(current, state_aa)  # (B, 2, 10)

    return relative_rot6d.flatten(start_dim=-2)  # (B, 20)


# ============================================================================
# Shared state cache (survives serialization — steps reference by key)
# ============================================================================

_state_cache: dict[str, tuple[Tensor, list[bool]]] = {}
_CACHE_KEY = "relative_rot6d"


def _cache_state(state: Tensor, mask: list[bool]) -> None:
    _state_cache[_CACHE_KEY] = (state.detach().cpu().clone(), mask)


def _get_cached_state() -> tuple[Tensor | None, list[bool] | None]:
    entry = _state_cache.get(_CACHE_KEY)
    if entry is None:
        return None, None
    return entry


# ============================================================================
# Processor Steps
# ============================================================================


@ProcessorStepRegistry.register("derive_state_from_action_rot6d")
@dataclass
class DeriveStateFromActionStep(ProcessorStep):
    """Derives 2-step observation.state from the action chunk (UMI-style).

    Expects action with one extra leading timestep: [B, chunk_size+1, D]
    from action_delta_indices = [-1, 0, 1, ..., chunk_size-1].
    Extracts [action[t-1], action[t]] as state and strips the extra timestep.
    No-op during inference (state comes from robot).
    """

    enabled: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        action = transition.get(TransitionKey.ACTION)
        if action is None or action.ndim < 3:
            return transition

        new_transition = deepcopy(transition)
        new_obs = dict(new_transition.get(TransitionKey.OBSERVATION, {}))

        # Extract 2-step state from first two action timesteps
        new_obs[OBS_STATE] = action[..., :2, :]
        # Strip leading timestep from action
        new_transition[TransitionKey.ACTION] = action[..., 1:, :]
        new_transition[TransitionKey.OBSERVATION] = new_obs

        # Strip extra leading timestep from padding masks
        comp_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        for pad_key in ("action_is_pad",):
            if pad_key in new_transition:
                pad_val = new_transition[pad_key]
                if isinstance(pad_val, torch.Tensor) and pad_val.ndim >= 2:
                    new_transition[pad_key] = pad_val[..., 1:]
            if pad_key in comp_data:
                pad_val = comp_data[pad_key]
                if isinstance(pad_val, torch.Tensor) and pad_val.ndim >= 2:
                    comp_data[pad_key] = pad_val[..., 1:]
        if comp_data:
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("relative_rot6d_actions_processor")
@dataclass
class RelativeRot6dActionsProcessorStep(ProcessorStep):
    """Converts 7D aa absolute actions to 10D rot6d relative actions via SE(3).

    During training: reads state (from DeriveStateFromActionStep or observation),
    computes T_rel = T_curr^{-1} @ T_future, outputs 10D [dx,dy,dz, rot6d(6), gripper].
    Caches state for the paired AbsoluteRot6dActionsProcessorStep.

    During inference: caches state from observation, no action conversion needed
    (model outputs 10D rot6d relative directly).
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_names: list[str] | None = None
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _normalize_names(names: list[str] | dict | None) -> list[str] | None:
        if names is None:
            return None
        if isinstance(names, dict):
            for key in ("axes", "names"):
                if key in names:
                    return list(names[key])
            return list(names.keys())
        return list(names)

    def _build_mask(self, action_dim: int) -> list[bool]:
        action_names = self._normalize_names(self.action_names)
        if not self.exclude_joints or action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))
        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        raw_state = observation.get(OBS_STATE) if observation else None

        # Use only the last timestep for relative action conversion
        if raw_state is not None:
            state = raw_state[..., -1, :] if raw_state.ndim >= 3 else raw_state
        else:
            state = None

        # Always cache state for the paired AbsoluteRot6dActionsProcessorStep
        if state is not None:
            self._last_state = state
            _cache_state(state, self._build_mask(7))

        if not self.enabled:
            return transition

        new_transition = deepcopy(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return transition

        mask = self._build_mask(7)  # 7D aa input
        new_transition[TransitionKey.ACTION] = to_relative_actions_rot6d(action, state, mask)
        return new_transition

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = deepcopy(features)
        # Action shape changes from [7] (aa) to [10] (rot6d)
        if PipelineFeatureType.ACTION in features and ACTION in features[PipelineFeatureType.ACTION]:
            feat = features[PipelineFeatureType.ACTION][ACTION]
            features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
                shape=(10,),
                type=feat.type,
            )
        return features


@ProcessorStepRegistry.register("absolute_rot6d_actions_processor")
@dataclass
class AbsoluteRot6dActionsProcessorStep(ProcessorStep):
    """Converts 10D rot6d relative actions back to 7D aa absolute actions.

    Reads cached state from the paired RelativeRot6dActionsProcessorStep
    and applies T_abs = T_ref @ T_delta.
    """

    enabled: bool = False
    relative_step: RelativeRot6dActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        # Get cached state: prefer paired step, fall back to shared cache
        if self.relative_step is not None:
            cached_state = self.relative_step.get_cached_state()
            mask = self.relative_step._build_mask(7)
        else:
            cached_state, mask = _get_cached_state()

        if cached_state is None:
            raise RuntimeError(
                "AbsoluteRot6dActionsProcessorStep: no cached state. "
                "Ensure the preprocessor runs before the postprocessor."
            )
        if mask is None:
            mask = [True] * 7

        new_transition = deepcopy(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        new_transition[TransitionKey.ACTION] = to_absolute_actions_rot6d(action, cached_state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("relative_rot6d_state_processor")
@dataclass
class RelativeRot6dStateProcessorStep(ProcessorStep):
    """Converts 2-step 7D aa state to 20D relative rot6d state.

    Training: state is (B, 2, 7) from DeriveStateFromActionStep.
    Converts each timestep to relative rot6d via SE(3) and flattens to (B, 20).

    Inference: state is (B, 7) single timestep. Buffers previous state,
    stacks [prev, curr], converts to relative rot6d, flattens to (B, 20).
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    state_names: list[str] | None = None
    _previous_state: Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, state_dim: int) -> list[bool]:
        state_names = RelativeRot6dActionsProcessorStep._normalize_names(self.state_names)
        if not self.exclude_joints or state_names is None:
            return [True] * state_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * state_dim

        mask = []
        for name in state_names[:state_dim]:
            state_name = str(name).lower()
            is_excluded = any(token == state_name or token in state_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < state_dim:
            mask.extend([True] * (state_dim - len(mask)))
        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is None:
            return transition

        new_transition = deepcopy(transition)
        new_obs = dict(new_transition.get(TransitionKey.OBSERVATION, {}))
        mask = self._build_mask(state.shape[-1])

        if state.ndim >= 3 and state.shape[-2] >= 2:
            # (B, 2, 7) — training: 2-step state from DeriveStateFromActionStep
            relative = to_relative_state_rot6d(state, mask)
            new_obs[OBS_STATE] = relative  # (B, 20)
        elif state.ndim == 2:
            # (B, 7) — inference: buffer previous and stack
            current = state
            if self._previous_state is None:
                self._previous_state = current.clone()
            prev = self._previous_state
            if prev.device != current.device or prev.dtype != current.dtype:
                prev = prev.to(device=current.device, dtype=current.dtype)
            stacked = torch.stack([prev, current], dim=-2)  # (B, 2, 7)
            relative = to_relative_state_rot6d(stacked, mask)
            new_obs[OBS_STATE] = relative  # (B, 20)
            self._previous_state = current.clone()
        else:
            return transition

        new_transition[TransitionKey.OBSERVATION] = new_obs
        return new_transition

    def reset(self) -> None:
        self._previous_state = None

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "state_names": self.state_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = deepcopy(features)
        # State shape changes from [7] (aa) to [20] (2×10 rot6d)
        if PipelineFeatureType.OBSERVATION in features and OBS_STATE in features[PipelineFeatureType.OBSERVATION]:
            feat = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                shape=(20,),
                type=feat.type,
            )
        return features
