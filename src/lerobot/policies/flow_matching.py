#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Shared flow-matching inference helpers."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def get_active_action_dim(config: Any) -> int:
    """Return the unpadded action dimension described by a policy config."""
    action_feature = config.action_feature
    if action_feature is None:
        return config.max_action_dim

    active_action_dim = action_feature.shape[0]
    if not 0 < active_action_dim <= config.max_action_dim:
        raise ValueError(
            "The action feature dimension must be in [1, max_action_dim], "
            f"got {active_action_dim} and max_action_dim={config.max_action_dim}."
        )
    return active_action_dim


def mask_padded_action_dims(
    tensor: Tensor,
    active_action_dim: int,
    enabled: bool,
) -> Tensor:
    """Zero fixed-width action coordinates beyond ``active_action_dim``.

    Multiplication is intentionally out-of-place so caller-owned noise tensors
    are never modified.
    """
    if not enabled or active_action_dim == tensor.shape[-1]:
        return tensor
    if not 0 < active_action_dim < tensor.shape[-1]:
        raise ValueError(
            "active_action_dim must be smaller than the tensor's last dimension when masking, "
            f"got {active_action_dim} for shape {tuple(tensor.shape)}."
        )
    mask = torch.arange(tensor.shape[-1], device=tensor.device) < active_action_dim
    return tensor * mask.to(dtype=tensor.dtype)


def integrate_flow_matching(
    noise: Tensor,
    num_steps: int,
    denoise_step: Callable[[Tensor, Tensor], Tensor],
    *,
    active_action_dim: int,
    mask_padded_dims: bool,
    rtc_processor: Any | None = None,
    rtc_enabled: bool = False,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
) -> Tensor:
    """Euler-integrate a flow field while keeping padded action dimensions zero.

    When masking is enabled, it is applied to the initial latent, the base
    denoiser output seen by RTC, RTC's guided velocity, and every Euler state.
    Masking the callback itself keeps RTC's endpoint estimate and error in the
    same subspace. It also prevents cross-coordinate leakage in RTC variants
    that differentiate the denoiser while computing guidance.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")

    x_t = mask_padded_action_dims(noise, active_action_dim, mask_padded_dims)
    batch_size = x_t.shape[0]
    dt = -1.0 / num_steps

    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=x_t.device).expand(batch_size)

        def masked_denoise_step(
            input_x_t: Tensor,
            current_timestep: Tensor = time_tensor,
        ) -> Tensor:
            velocity = denoise_step(input_x_t, current_timestep)
            return mask_padded_action_dims(velocity, active_action_dim, mask_padded_dims)

        if rtc_enabled:
            if rtc_processor is None:
                raise RuntimeError("RTC is enabled but no RTC processor was provided.")
            velocity = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step_partial=masked_denoise_step,
                execution_horizon=execution_horizon,
            )
        else:
            velocity = masked_denoise_step(x_t)

        velocity = mask_padded_action_dims(velocity, active_action_dim, mask_padded_dims)
        x_t = mask_padded_action_dims(x_t + dt * velocity, active_action_dim, mask_padded_dims)

        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=time, x_t=x_t, v_t=velocity)

    return x_t
