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

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.flow_matching import (
    get_active_action_dim,
    integrate_flow_matching,
    mask_padded_action_dims,
)
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@pytest.mark.parametrize("config_class", [SmolVLAConfig, PI0Config, PI05Config])
def test_flow_policy_configs_mask_padded_action_dims_by_default(config_class):
    field = config_class.__dataclass_fields__["mask_padded_action_dims_at_inference"]
    assert field.default is True


def test_get_active_action_dim_uses_action_feature():
    config = SimpleNamespace(
        action_feature=SimpleNamespace(shape=(10,)),
        max_action_dim=32,
    )
    assert get_active_action_dim(config) == 10


def test_mask_padded_action_dims_is_out_of_place():
    tensor = torch.randn(2, 3, 8)
    original = tensor.clone()

    masked = mask_padded_action_dims(tensor, active_action_dim=3, enabled=True)

    torch.testing.assert_close(tensor, original)
    torch.testing.assert_close(masked[..., :3], original[..., :3])
    torch.testing.assert_close(masked[..., 3:], torch.zeros_like(masked[..., 3:]))


def test_masked_flow_is_invariant_to_padded_input_noise():
    active_action_dim = 3
    first_noise = torch.randn(2, 4, 8)
    second_noise = first_noise.clone()
    second_noise[..., active_action_dim:] = torch.randn_like(second_noise[..., active_action_dim:])

    def coupled_denoiser(x_t, _time):
        padded_sum = x_t[..., active_action_dim:].sum(dim=-1, keepdim=True)
        active_velocity = x_t[..., :active_action_dim] + padded_sum
        padded_velocity = x_t[..., :1].expand(*x_t.shape[:-1], x_t.shape[-1] - active_action_dim)
        return torch.cat((active_velocity, padded_velocity), dim=-1)

    first = integrate_flow_matching(
        first_noise,
        4,
        coupled_denoiser,
        active_action_dim=active_action_dim,
        mask_padded_dims=True,
    )
    second = integrate_flow_matching(
        second_noise,
        4,
        coupled_denoiser,
        active_action_dim=active_action_dim,
        mask_padded_dims=True,
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        first[..., active_action_dim:], torch.zeros_like(first[..., active_action_dim:])
    )


def test_legacy_flow_preserves_full_width_euler_integration():
    noise = torch.randn(2, 4, 8)

    def denoiser(x_t, time):
        return x_t.square() + time[:, None, None]

    actual = integrate_flow_matching(
        noise,
        4,
        denoiser,
        active_action_dim=3,
        mask_padded_dims=False,
    )

    expected = noise
    dt = -0.25
    for step in range(4):
        time = torch.full((noise.shape[0],), 1.0 + step * dt)
        expected = expected + dt * denoiser(expected, time)

    torch.testing.assert_close(actual, expected)


def test_rtc_sees_masked_state_and_velocity_and_its_output_is_clamped():
    active_action_dim = 3

    class InspectingRTC:
        def __init__(self):
            self.calls = 0

        def denoise_step(self, *, x_t, original_denoise_step_partial, **_kwargs):
            self.calls += 1
            torch.testing.assert_close(
                x_t[..., active_action_dim:], torch.zeros_like(x_t[..., active_action_dim:])
            )
            velocity = original_denoise_step_partial(x_t)
            torch.testing.assert_close(
                velocity[..., active_action_dim:], torch.zeros_like(velocity[..., active_action_dim:])
            )
            guided_velocity = velocity.clone()
            guided_velocity[..., active_action_dim:] = 100.0
            return guided_velocity

        def is_debug_enabled(self):
            return False

    rtc = InspectingRTC()
    result = integrate_flow_matching(
        torch.randn(1, 4, 8),
        3,
        lambda x_t, _time: torch.ones_like(x_t),
        active_action_dim=active_action_dim,
        mask_padded_dims=True,
        rtc_processor=rtc,
        rtc_enabled=True,
        inference_delay=1,
        prev_chunk_left_over=torch.zeros(1, 2, active_action_dim),
        execution_horizon=2,
    )

    assert rtc.calls == 3
    torch.testing.assert_close(
        result[..., active_action_dim:], torch.zeros_like(result[..., active_action_dim:])
    )
