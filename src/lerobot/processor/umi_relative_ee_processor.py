"""UMI-style relative end-effector processor steps.

The on-disk action is an absolute 7D pose
``[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]``.  The policy
uses a 10D relative pose ``[dx, dy, dz, rot6d(6), gripper]``.  Every target in
an action chunk is expressed from the same chunk-start pose.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to rotation matrices with Rodrigues' formula."""
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    axis = axis_angle / theta
    kx, ky, kz = axis.unbind(dim=-1)
    zeros = torch.zeros_like(kx)
    skew = torch.stack(
        [zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], dim=-1
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return identity + torch.sin(theta).unsqueeze(-1) * skew + (
        1 - torch.cos(theta).unsqueeze(-1)
    ) * (skew @ skew)


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to axis-angle vectors."""
    vector = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta_twice = vector.norm(dim=-1)
    cos_theta = (matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2
    theta = torch.atan2(sin_theta_twice / 2, cos_theta)
    result = vector / (sin_theta_twice.unsqueeze(-1) + 1e-8) * theta.unsqueeze(-1)
    return torch.where((sin_theta_twice < 1e-7).unsqueeze(-1), vector / 2, result)


def matrix_to_rot6d(matrix: Tensor) -> Tensor:
    """Return the first two matrix rows (the UMI row-based rot6d convention)."""
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """Reconstruct a rotation matrix from UMI's row-based rot6d representation."""
    row0 = torch.nn.functional.normalize(rot6d[..., :3], dim=-1)
    row1 = rot6d[..., 3:] - (row0 * rot6d[..., 3:]).sum(dim=-1, keepdim=True) * row0
    row1 = torch.nn.functional.normalize(row1, dim=-1)
    row2 = torch.cross(row0, row1, dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def absolute_aa_to_relative_rot6d(reference: Tensor, target: Tensor) -> Tensor:
    """Compute ``inverse(T_reference) @ T_target`` for 7D absolute poses."""
    reference_rotation = axis_angle_to_matrix(reference[..., 3:6])
    target_rotation = axis_angle_to_matrix(target[..., 3:6])
    reference_rotation_t = reference_rotation.transpose(-2, -1)
    relative_rotation = reference_rotation_t @ target_rotation
    relative_translation = (
        reference_rotation_t @ (target[..., :3] - reference[..., :3]).unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [relative_translation, matrix_to_rot6d(relative_rotation), target[..., 6:7]], dim=-1
    )


def relative_rot6d_to_absolute_aa(relative: Tensor, reference: Tensor) -> Tensor:
    """Compute ``T_reference @ T_relative`` and return a 7D absolute pose."""
    reference_rotation = axis_angle_to_matrix(reference[..., 3:6])
    absolute_rotation = reference_rotation @ rot6d_to_matrix(relative[..., 3:9])
    absolute_translation = reference[..., :3] + (
        reference_rotation @ relative[..., :3].unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [absolute_translation, matrix_to_axis_angle(absolute_rotation), relative[..., 9:10]], dim=-1
    )


def absolute_aa_to_relative_aa(reference: Tensor, target: Tensor) -> Tensor:
    """Compute ``inverse(T_reference) @ T_target`` as 7D xyz/axis-angle/gripper.

    LingBot-VA's released single-arm convention reserves exactly seven active
    channels inside its 30D multi-embodiment action tensor.  Unlike the canonical
    UMI policy representation, there is therefore no room for a 6D rotation.  This
    helper derives the same chunk-start-relative SE(3) transform as
    :func:`absolute_aa_to_relative_rot6d`, then records its rotation as a 3D
    axis-angle vector.
    """
    reference_rotation = axis_angle_to_matrix(reference[..., 3:6])
    target_rotation = axis_angle_to_matrix(target[..., 3:6])
    reference_rotation_t = reference_rotation.transpose(-2, -1)
    relative_rotation = reference_rotation_t @ target_rotation
    relative_translation = (
        reference_rotation_t @ (target[..., :3] - reference[..., :3]).unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [relative_translation, matrix_to_axis_angle(relative_rotation), target[..., 6:7]], dim=-1
    )


def relative_aa_to_absolute_aa(relative: Tensor, reference: Tensor) -> Tensor:
    """Compose a 7D relative xyz/axis-angle/gripper action with an absolute pose."""
    reference_rotation = axis_angle_to_matrix(reference[..., 3:6])
    absolute_rotation = reference_rotation @ axis_angle_to_matrix(relative[..., 3:6])
    absolute_translation = reference[..., :3] + (
        reference_rotation @ relative[..., :3].unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [absolute_translation, matrix_to_axis_angle(absolute_rotation), relative[..., 6:7]], dim=-1
    )


def to_umi_relative_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert a batch of absolute 7D action chunks to relative 10D chunks."""
    if actions.shape[-1] != 7 or state.shape[-1] != 7:
        raise ValueError(
            "UMI relative EE requires 7D [xyz, axis-angle, gripper] input; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    if state.ndim == 3:
        state = state[:, -1]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == 3:
        state = state.unsqueeze(1).expand(*actions.shape[:-1], 7)
    return absolute_aa_to_relative_rot6d(state, actions)


def to_umi_absolute_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert a batch of relative 10D action chunks to absolute 7D chunks."""
    if actions.shape[-1] != 10 or state.shape[-1] != 7:
        raise ValueError(
            "UMI relative EE requires 10D actions and a 7D reference state; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    if state.ndim == 3:
        state = state[:, -1]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == 3:
        state = state.unsqueeze(1).expand(*actions.shape[:-1], 7)
    return relative_rot6d_to_absolute_aa(actions, state)


def to_umi_relative_axis_angle_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert absolute 7D action chunks to LingBot-compatible relative 7D chunks."""
    if actions.shape[-1] != 7 or state.shape[-1] != 7:
        raise ValueError(
            "UMI relative axis-angle requires 7D [xyz, axis-angle, gripper] input; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    if state.ndim == 3:
        state = state[:, -1]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == 3:
        state = state.unsqueeze(1).expand(*actions.shape[:-1], 7)
    return absolute_aa_to_relative_aa(state, actions)


def to_umi_absolute_axis_angle_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert LingBot-compatible relative 7D chunks back to absolute 7D poses."""
    if actions.shape[-1] != 7 or state.shape[-1] != 7:
        raise ValueError(
            "UMI relative axis-angle requires 7D actions and a 7D reference state; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    if state.ndim == 3:
        state = state[:, -1]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == 3:
        state = state.unsqueeze(1).expand(*actions.shape[:-1], 7)
    return relative_aa_to_absolute_aa(actions, state)


def to_umi_relative_state(state: Tensor) -> Tensor:
    """Convert ``[previous, current]`` 7D poses to a flattened 20D state."""
    if state.ndim != 3 or state.shape[-2:] != (2, 7):
        raise ValueError(f"UMI state must have shape [batch, 2, 7], got {tuple(state.shape)}")
    current = state[:, -1:].expand_as(state)
    return absolute_aa_to_relative_rot6d(current, state).flatten(start_dim=-2)


_STATE_CACHE: dict[str, Tensor] = {}


def make_umi_cache_key() -> str:
    return f"umi_relative_ee_{uuid4().hex}"


@ProcessorStepRegistry.register("umi_derive_state_from_action")
@dataclass
class UmiDeriveStateFromActionStep(ProcessorStep):
    """Build state from ``action[t-1:t+1]`` and remove the leading action."""

    enabled: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not self.enabled or action is None or action.ndim < 3:
            return transition
        result = deepcopy(transition)
        observation = dict(result.get(TransitionKey.OBSERVATION) or {})
        observation[OBS_STATE] = action[..., :2, :]
        result[TransitionKey.OBSERVATION] = observation
        result[TransitionKey.ACTION] = action[..., 1:, :]

        complementary = dict(result.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        for container in (result, complementary):
            padding = container.get("action_is_pad")
            if isinstance(padding, Tensor) and padding.ndim >= 2:
                container["action_is_pad"] = padding[..., 1:]
        if complementary:
            result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("umi_relative_actions")
@dataclass
class UmiRelativeActionsStep(ProcessorStep):
    """Convert 7D absolute chunks to 10D UMI-relative chunks and cache the base."""

    enabled: bool = True
    cache_key: str = "umi_relative_ee"
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        raw_state = observation.get(OBS_STATE) if observation else None
        state = raw_state[..., -1, :] if raw_state is not None and raw_state.ndim >= 3 else raw_state
        if state is not None:
            self._last_state = state.detach().cpu().clone()
            _STATE_CACHE[self.cache_key] = self._last_state
        action = transition.get(TransitionKey.ACTION)
        if not self.enabled or action is None or state is None:
            return transition
        result = deepcopy(transition)
        result[TransitionKey.ACTION] = to_umi_relative_actions(action, state)
        return result

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def reset(self) -> None:
        self._last_state = None
        _STATE_CACHE.pop(self.cache_key, None)

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "cache_key": self.cache_key}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        action_features = result.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            feature = action_features[ACTION]
            action_features[ACTION] = PolicyFeature(type=feature.type, shape=(10,))
        return result


@ProcessorStepRegistry.register("umi_relative_axis_angle_actions")
@dataclass
class UmiRelativeAxisAngleActionsStep(ProcessorStep):
    """Convert absolute 7D chunks to relative 7D LingBot actions and cache the base."""

    enabled: bool = True
    cache_key: str = "umi_relative_ee_axis_angle"
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        raw_state = observation.get(OBS_STATE) if observation else None
        state = raw_state[..., -1, :] if raw_state is not None and raw_state.ndim >= 3 else raw_state
        if state is not None:
            self._last_state = state.detach().cpu().clone()
            _STATE_CACHE[self.cache_key] = self._last_state
        action = transition.get(TransitionKey.ACTION)
        if not self.enabled or action is None or state is None:
            return transition
        result = deepcopy(transition)
        result[TransitionKey.ACTION] = to_umi_relative_axis_angle_actions(action, state)
        return result

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def reset(self) -> None:
        self._last_state = None
        _STATE_CACHE.pop(self.cache_key, None)

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "cache_key": self.cache_key}

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("umi_drop_obs_state")
@dataclass
class UmiDropObsStateStep(ProcessorStep):
    """Remove the derived UMI state from the observation (image-only policies).

    ``UmiDeriveStateFromActionStep`` must still run (it strips the anchor frame
    from the action and provides the anchor that ``UmiRelativeActionsStep``
    converts the chunk against), but with `use_proprioception=False` the state
    must not reach the policy as an input. Placed AFTER the relative-actions
    step so the cached anchor (used by the postprocessor to convert
    predictions back to absolute poses) is already stored.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not observation or OBS_STATE not in observation:
            return transition
        result = deepcopy(transition)
        new_observation = dict(result[TransitionKey.OBSERVATION])
        new_observation.pop(OBS_STATE, None)
        result[TransitionKey.OBSERVATION] = new_observation
        return result

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("umi_relative_state")
@dataclass
class UmiRelativeStateStep(ProcessorStep):
    """Convert two absolute 7D poses to a flattened relative 20D state."""

    enabled: bool = True
    _previous_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is None:
            return transition
        if state.ndim == 2:
            previous = state if self._previous_state is None else self._previous_state.to(state)
            state_pair = torch.stack([previous, state], dim=1)
            self._previous_state = state.detach().clone()
        elif state.ndim == 3 and state.shape[1] == 2:
            state_pair = state
        else:
            raise ValueError(f"UMI state must be [B, 7] or [B, 2, 7], got {tuple(state.shape)}")
        result = deepcopy(transition)
        result_observation = dict(observation)
        result_observation[OBS_STATE] = to_umi_relative_state(state_pair)
        result[TransitionKey.OBSERVATION] = result_observation
        return result

    def reset(self) -> None:
        self._previous_state = None

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        observation_features = result.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            feature = observation_features[OBS_STATE]
            observation_features[OBS_STATE] = PolicyFeature(type=feature.type, shape=(20,))
        return result


@ProcessorStepRegistry.register("umi_absolute_actions")
@dataclass
class UmiAbsoluteActionsStep(ProcessorStep):
    """Convert model-relative 10D chunks back to absolute 7D EE targets."""

    enabled: bool = True
    cache_key: str = "umi_relative_ee"
    relative_step: UmiRelativeActionsStep | None = field(default=None, repr=False)
    single_action_reference_steps: int = 1
    _single_action_reference: Tensor | None = field(default=None, init=False, repr=False)
    _single_action_steps_remaining: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.single_action_reference_steps < 1:
            raise ValueError("single_action_reference_steps must be at least 1")

    def set_single_action_reference_steps(self, steps: int) -> None:
        """Set how many queued single actions share one chunk-start reference."""
        if steps < 1:
            raise ValueError("single_action_reference_steps must be at least 1")
        self.single_action_reference_steps = steps
        self._clear_single_action_reference()

    def _clear_single_action_reference(self) -> None:
        self._single_action_reference = None
        self._single_action_steps_remaining = 0

    def _reference_for_action(self, action: Tensor, current_state: Tensor) -> Tensor:
        # predict_action_chunk returns [batch, time, action_dim] and is
        # postprocessed immediately, so it always uses the latest observation.
        # select_action returns [batch, action_dim] one item at a time; all items
        # in its internal queue must instead retain the frame that generated the
        # chunk, even though preprocessing refreshes current_state every tick.
        if action.ndim >= 3 or self.single_action_reference_steps == 1:
            self._clear_single_action_reference()
            return current_state

        if self._single_action_steps_remaining == 0:
            self._single_action_reference = current_state.detach().clone()
            self._single_action_steps_remaining = self.single_action_reference_steps

        if self._single_action_reference is None:  # Defensive invariant for type checkers.
            raise RuntimeError("UMI single-action reference was not initialized")
        reference = self._single_action_reference
        self._single_action_steps_remaining -= 1
        if self._single_action_steps_remaining == 0:
            self._single_action_reference = None
        return reference

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        state = self.relative_step.get_cached_state() if self.relative_step is not None else None
        if state is None:
            state = _STATE_CACHE.get(self.cache_key)
        if state is None:
            raise RuntimeError("UMI postprocessor has no chunk-base state; run the preprocessor first")
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        state = self._reference_for_action(action, state)
        result = deepcopy(transition)
        result[TransitionKey.ACTION] = to_umi_absolute_actions(action, state)
        return result

    def reset(self) -> None:
        self._clear_single_action_reference()
        _STATE_CACHE.pop(self.cache_key, None)

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cache_key": self.cache_key,
            "single_action_reference_steps": self.single_action_reference_steps,
        }

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("umi_absolute_axis_angle_actions")
@dataclass
class UmiAbsoluteAxisAngleActionsStep(ProcessorStep):
    """Convert relative 7D LingBot chunks back to absolute 7D EE targets."""

    enabled: bool = True
    cache_key: str = "umi_relative_ee_axis_angle"
    relative_step: UmiRelativeAxisAngleActionsStep | None = field(default=None, repr=False)
    single_action_reference_steps: int = 1
    initial_single_action_reference_steps: int | None = None
    _single_action_reference: Tensor | None = field(default=None, init=False, repr=False)
    _single_action_steps_remaining: int = field(default=0, init=False, repr=False)
    _is_first_group: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.single_action_reference_steps < 1:
            raise ValueError("single_action_reference_steps must be at least 1")
        if (
            self.initial_single_action_reference_steps is not None
            and self.initial_single_action_reference_steps < 1
        ):
            raise ValueError("initial_single_action_reference_steps must be at least 1")

    def set_single_action_reference_steps(self, steps: int) -> None:
        if steps < 1:
            raise ValueError("single_action_reference_steps must be at least 1")
        self.single_action_reference_steps = steps
        self._clear_single_action_reference()

    def _clear_single_action_reference(self) -> None:
        self._single_action_reference = None
        self._single_action_steps_remaining = 0

    def _reference_for_action(self, action: Tensor, current_state: Tensor) -> Tensor:
        if action.ndim >= 3 or self.single_action_reference_steps == 1:
            self._clear_single_action_reference()
            return current_state
        if self._single_action_steps_remaining == 0:
            self._single_action_reference = current_state.detach().clone()
            self._single_action_steps_remaining = (
                self.initial_single_action_reference_steps
                if self._is_first_group and self.initial_single_action_reference_steps is not None
                else self.single_action_reference_steps
            )
            self._is_first_group = False
        if self._single_action_reference is None:
            raise RuntimeError("UMI axis-angle single-action reference was not initialized")
        reference = self._single_action_reference
        self._single_action_steps_remaining -= 1
        if self._single_action_steps_remaining == 0:
            self._single_action_reference = None
        return reference

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        state = self.relative_step.get_cached_state() if self.relative_step is not None else None
        if state is None:
            state = _STATE_CACHE.get(self.cache_key)
        if state is None:
            raise RuntimeError("UMI postprocessor has no chunk-base state; run the preprocessor first")
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        state = self._reference_for_action(action, state)
        result = deepcopy(transition)
        result[TransitionKey.ACTION] = to_umi_absolute_axis_angle_actions(action, state)
        return result

    def reset(self) -> None:
        self._clear_single_action_reference()
        self._is_first_group = True
        _STATE_CACHE.pop(self.cache_key, None)

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cache_key": self.cache_key,
            "single_action_reference_steps": self.single_action_reference_steps,
            "initial_single_action_reference_steps": self.initial_single_action_reference_steps,
        }

    def transform_features(self, features):
        return features
