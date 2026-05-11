# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "DeriveStateFromActionStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "RelativeStateProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
    "to_relative_state",
]


# --- SE(3) transformation helpers (torch, batched) ---


def _axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to rotation matrices via Rodrigues formula.

    Args:
        axis_angle: (..., 3) rotation vectors.
    Returns:
        (..., 3, 3) rotation matrices.
    """
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
    """Convert rotation matrices to axis-angle vectors.

    Args:
        R: (..., 3, 3) rotation matrices.
    Returns:
        (..., 3) axis-angle vectors.
    """
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


def _pose_se3_relative(pose_from: Tensor, pose_to: Tensor) -> Tensor:
    """SE(3) relative: T_delta = inv(T_from) @ T_to.

    Args:
        pose_from: (..., 6) reference pose [x, y, z, wx, wy, wz].
        pose_to:   (..., 6) target pose [x, y, z, wx, wy, wz].
    Returns:
        (..., 6) relative pose delta.
    """
    R_from = _axis_angle_to_matrix(pose_from[..., 3:6])
    R_to = _axis_angle_to_matrix(pose_to[..., 3:6])
    R_from_T = R_from.transpose(-2, -1)
    R_delta = R_from_T @ R_to
    dt = pose_to[..., :3] - pose_from[..., :3]
    t_delta = (R_from_T @ dt.unsqueeze(-1)).squeeze(-1)
    r_delta = _matrix_to_axis_angle(R_delta)
    return torch.cat([t_delta, r_delta], dim=-1)


def _pose_se3_absolute(pose_delta: Tensor, pose_ref: Tensor) -> Tensor:
    """SE(3) absolute: T = T_ref @ T_delta.

    Args:
        pose_delta: (..., 6) relative pose [x, y, z, wx, wy, wz].
        pose_ref:   (..., 6) reference pose [x, y, z, wx, wy, wz].
    Returns:
        (..., 6) absolute pose.
    """
    R_ref = _axis_angle_to_matrix(pose_ref[..., 3:6])
    R_delta = _axis_angle_to_matrix(pose_delta[..., 3:6])
    R = R_ref @ R_delta
    t = pose_ref[..., :3] + (R_ref @ pose_delta[..., :3].unsqueeze(-1)).squeeze(-1)
    r = _matrix_to_axis_angle(R)
    return torch.cat([t, r], dim=-1)


def _should_use_se3(mask_t: Tensor, dims: int, pose_dim: int) -> bool:
    """True when pose_dim >= 6, the mask covers at least pose_dim dims,
    and all of those dims are masked (the full 6-DoF pose is relative)."""
    return pose_dim >= 6 and dims >= pose_dim and mask_t[:pose_dim].all()


# --- Public conversion functions ---


def to_relative_actions(
    actions: Tensor, state: Tensor, mask: Sequence[bool], pose_dim: int = 0
) -> Tensor:
    """Convert absolute actions to relative (for masked dims).

    When *pose_dim* >= 6 and the mask covers all pose dims, the first *pose_dim*
    dimensions are treated as an SE(3) pose [x,y,z,wx,wy,wz] and converted via
    proper transformation-matrix math (T_delta = inv(T_state) @ T_action).
    Remaining masked dims use element-wise subtraction.

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim) or (B, obs_steps, state_dim). If 3D, the last obs step is used.
        mask: Which dims to convert. Can be shorter than action_dim.
        pose_dim: Number of leading dims that form an SE(3) pose (6: xyz + wxwywz).
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    if state.ndim == 3:
        state = state[:, -1, :]
    actions = actions.clone()

    if _should_use_se3(mask_t, dims, pose_dim):
        state_pose = state[..., :pose_dim]
        action_pose = actions[..., :pose_dim]
        if action_pose.ndim > state_pose.ndim:
            state_pose = state_pose.unsqueeze(-2)
        actions[..., :pose_dim] = _pose_se3_relative(state_pose, action_pose)
        remaining = mask_t[pose_dim:]
        if remaining.any():
            state_rem = state[..., pose_dim:dims] * remaining
            if actions.ndim == 3:
                state_rem = state_rem.unsqueeze(-2)
            actions[..., pose_dim:dims] -= state_rem
    else:
        state_offset = state[..., :dims] * mask_t
        if actions.ndim == 3:
            state_offset = state_offset.unsqueeze(-2)
        actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(
    actions: Tensor, state: Tensor, mask: Sequence[bool], pose_dim: int = 0
) -> Tensor:
    """Convert relative actions back to absolute (for masked dims).

    When *pose_dim* >= 6 and the mask covers all pose dims, the first *pose_dim*
    dimensions are treated as an SE(3) pose and converted via T = T_ref @ T_delta.
    Remaining masked dims use element-wise addition.

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim) or (B, obs_steps, state_dim). If 3D, the last obs step is used.
        mask: Which dims to convert. Can be shorter than action_dim.
        pose_dim: Number of leading dims that form an SE(3) pose (6: xyz + wxwywz).
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    if state.ndim == 3:
        state = state[:, -1, :]
    actions = actions.clone()

    if _should_use_se3(mask_t, dims, pose_dim):
        state_pose = state[..., :pose_dim]
        action_pose = actions[..., :pose_dim]
        if action_pose.ndim > state_pose.ndim:
            state_pose = state_pose.unsqueeze(-2)
        actions[..., :pose_dim] = _pose_se3_absolute(action_pose, state_pose)
        remaining = mask_t[pose_dim:]
        if remaining.any():
            state_rem = state[..., pose_dim:dims] * remaining
            if actions.ndim == 3:
                state_rem = state_rem.unsqueeze(-2)
            actions[..., pose_dim:dims] += state_rem
    else:
        state_offset = state[..., :dims] * mask_t
        if actions.ndim == 3:
            state_offset = state_offset.unsqueeze(-2)
        actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("derive_state_from_action_processor")
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
        new_transition = transition.copy()
        new_obs = dict(new_transition.get(TransitionKey.OBSERVATION, {}))
        new_obs[OBS_STATE] = action[..., :2, :]
        new_transition[TransitionKey.ACTION] = action[..., 1:, :]
        new_transition[TransitionKey.OBSERVATION] = new_obs
        # Strip extra leading timestep from *_is_pad masks (in complementary_data or top-level)
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


@ProcessorStepRegistry.register("delta_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    pose_dim: int = 0
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _normalize_names(names: list[str] | dict | None) -> list[str] | None:
        """Unwrap ``{"axes": [...]}`` or ``{"names": [...]}`` dict format to a plain list."""
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

        # When state_delta_indices loads multi-timestep state [B, n_obs, D],
        # use only the current (last) timestep for relative action conversion.
        if raw_state is not None:
            state = raw_state[..., -1, :] if raw_state.ndim >= 3 else raw_state
        else:
            state = None

        # Always cache state for the paired AbsoluteActionsProcessorStep
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask, self.pose_dim)
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        """Return the cached ``observation.state`` used as the reference point for relative/absolute action conversions."""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "pose_dim": self.pose_dim,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def to_relative_state(state: Tensor, mask: Sequence[bool], pose_dim: int = 0) -> Tensor:
    """Convert multi-timestep absolute state to relative (offset from current timestep).

    Each timestep becomes relative to the last (current) timestep.
    When *pose_dim* >= 6, uses SE(3) for the pose portion.

    Args:
        state: (..., n_obs, state_dim) — last timestep is the reference (current).
        mask: Which dims to convert. Can be shorter than state_dim.
        pose_dim: Number of leading dims that form an SE(3) pose (6: xyz + wxwywz).
    """
    mask_t = torch.tensor(mask, dtype=state.dtype, device=state.device)
    dims = mask_t.shape[0]
    current = state[..., -1:, :]
    state = state.clone()

    if _should_use_se3(mask_t, dims, pose_dim):
        current_pose = current[..., :pose_dim]
        state_pose = state[..., :pose_dim]
        state[..., :pose_dim] = _pose_se3_relative(current_pose, state_pose)
        remaining = mask_t[pose_dim:]
        if remaining.any():
            state[..., pose_dim:dims] -= current[..., pose_dim:dims] * remaining
    else:
        state[..., :dims] -= current[..., :dims] * mask_t
    return state


@ProcessorStepRegistry.register("relative_state_processor")
@dataclass
class RelativeStateProcessorStep(ProcessorStep):
    """Converts observation.state to relative (offset from current timestep).

    UMI-style relative proprioception: each state timestep is expressed as
    an offset from the current EE pose, providing velocity information.

    During training (multi-timestep input from ``state_delta_indices``):
        ``state[..., t, :] -= state[..., -1, :]`` — subtract current from all.

    During inference (single timestep): buffers the previous state and stacks
    ``[previous, current]`` before applying the relative conversion, producing
    the same ``[n_obs, D]`` shape the model expects.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint/dim names to keep absolute.
        state_names: State dimension names from dataset metadata.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    state_names: list[str] | None = None
    pose_dim: int = 0
    _previous_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, state_dim: int) -> list[bool]:
        state_names = RelativeActionsProcessorStep._normalize_names(self.state_names)
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

        new_transition = transition.copy()
        new_obs = dict(new_transition.get(TransitionKey.OBSERVATION, {}))
        mask = self._build_mask(state.shape[-1])

        if state.ndim >= 3:
            # [B, n_obs, D] — multi-timestep (training with state_delta_indices)
            relative = to_relative_state(state, mask, self.pose_dim)
            new_obs[OBS_STATE] = relative.flatten(start_dim=-2)  # [B, n_obs*D]
        elif state.ndim == 2:
            # [B, D] — single timestep (inference): buffer previous and stack
            current = state
            if self._previous_state is None:
                self._previous_state = current.clone()
            prev = self._previous_state
            if prev.device != current.device or prev.dtype != current.dtype:
                prev = prev.to(device=current.device, dtype=current.dtype)
            stacked = torch.stack([prev, current], dim=-2)  # [B, 2, D]
            relative = to_relative_state(stacked, mask, self.pose_dim)
            new_obs[OBS_STATE] = relative.flatten(start_dim=-2)  # [B, 2*D]
            self._previous_state = current.clone()

        new_transition[TransitionKey.OBSERVATION] = new_obs
        return new_transition

    def reset(self) -> None:
        """Reset the state buffer. Call at episode boundaries during inference."""
        self._previous_state = None

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "state_names": self.state_names,
            "pose_dim": self.pose_dim,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the cached state from its paired RelativeActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        relative_step: Reference to the paired RelativeActionsProcessorStep that caches state.
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        pose_dim = getattr(self.relative_step, "pose_dim", 0)
        new_transition[TransitionKey.ACTION] = to_absolute_actions(action, cached_state, mask, pose_dim)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
