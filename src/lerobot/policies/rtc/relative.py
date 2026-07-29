#!/usr/bin/env python

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

"""Relative-action helpers for Real-Time Chunking (RTC)."""

from __future__ import annotations

import torch

from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    to_relative_actions,
    to_umi_relative_actions,
)


def reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> torch.Tensor:
    """Convert absolute leftover actions into model-space for relative-action RTC policies.

    When using relative actions, the RTC prefix (previous chunk's unexecuted tail)
    is stored in absolute coordinates. Before feeding it back to the policy, this
    helper re-expresses those actions relative to the robot's current joint state
    and optionally normalizes them so the policy receives correctly scaled inputs.
    """
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)


def reanchor_umi_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> torch.Tensor:
    """Convert absolute UMI EE leftovers into the current chunk's model space.

    UMI actions in a chunk are all relative to the EE pose observed at that
    chunk's start. The processed RTC queue stores the same targets as absolute
    7D ``[xyz, axis-angle, gripper]`` poses. Rebuilding the prefix from that
    queue avoids comparing an old chunk frame with the latest chunk frame.

    The returned actions use UMI's 10D
    ``[relative xyz, row-based rot6d, gripper]`` representation and the loaded
    policy's action normalization statistics.
    """
    action_absolute = prev_actions_absolute.detach().cpu()
    state = current_state.detach().cpu()

    if action_absolute.ndim not in (2, 3):
        raise ValueError(
            f"UMI RTC leftovers must have shape [T, 7] or [B, T, 7], got {tuple(action_absolute.shape)}"
        )
    if action_absolute.shape[-1] != 7:
        raise ValueError(f"UMI RTC requires absolute 7D leftovers, got {action_absolute.shape[-1]}D")

    if action_absolute.ndim == 2 and state.ndim == 2:
        if state.shape[0] != 1:
            raise ValueError(
                "Unbatched UMI RTC leftovers require exactly one current state, "
                f"got shape {tuple(state.shape)}"
            )
        state = state[0]

    relative_actions = to_umi_relative_actions(action_absolute, state)
    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)
