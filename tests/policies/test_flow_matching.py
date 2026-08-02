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
    reduce_flow_matching_loss,
)
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@pytest.mark.parametrize("config_class", [SmolVLAConfig, PI05Config])
def test_openpi_flow_policy_configs_do_not_mask_padded_action_dims(config_class):
    field = config_class.__dataclass_fields__["mask_padded_action_dims_at_inference"]
    assert field.default is False


def test_pi0_keeps_legacy_masked_inference_default():
    field = PI0Config.__dataclass_fields__["mask_padded_action_dims_at_inference"]
    assert field.default is True


@pytest.mark.parametrize("config_class", [SmolVLAConfig, PI05Config])
def test_flow_matching_padding_mode_defaults_to_openpi_full_width(config_class):
    field = config_class.__dataclass_fields__["flow_matching_padding_mode"]
    assert field.default == "openpi_full_width"


@pytest.mark.parametrize("config_class", [SmolVLAConfig, PI05Config])
def test_flow_matching_padding_mode_accepts_masked_subspace(config_class):
    config = config_class(flow_matching_padding_mode="masked_subspace")
    assert config.flow_matching_padding_mode == "masked_subspace"


@pytest.mark.parametrize("config_class", [SmolVLAConfig, PI05Config])
def test_flow_matching_padding_mode_rejects_unknown_value(config_class):
    with pytest.raises(ValueError, match="flow_matching_padding_mode"):
        config_class(flow_matching_padding_mode="masked_subsapce")


def test_get_active_action_dim_uses_action_feature():
    config = SimpleNamespace(
        action_feature=SimpleNamespace(shape=(10,)),
        max_action_dim=32,
    )
    assert get_active_action_dim(config) == 10


def test_full_width_loss_includes_padded_coordinates_and_masks_only_timesteps():
    losses = torch.zeros(2, 3, 4)
    losses[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    losses[0, 1] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    losses[0, 2] = 1000.0
    losses[1, 0] = torch.tensor([9.0, 10.0, 11.0, 12.0])
    losses[1, 1:] = 1000.0
    action_is_pad = torch.tensor([[False, False, True], [False, True, True]])

    loss, loss_per_dim = reduce_flow_matching_loss(losses, action_is_pad)
    per_sample_loss, per_sample_dims = reduce_flow_matching_loss(losses, action_is_pad, reduction="none")

    torch.testing.assert_close(loss, torch.tensor(6.5))
    torch.testing.assert_close(loss_per_dim, torch.tensor([5.0, 6.0, 7.0, 8.0]))
    torch.testing.assert_close(per_sample_loss, torch.tensor([4.5, 10.5]))
    torch.testing.assert_close(per_sample_dims, loss_per_dim)


def test_full_width_loss_without_temporal_padding():
    losses = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 4)

    loss, loss_per_dim = reduce_flow_matching_loss(losses)

    torch.testing.assert_close(loss, losses.mean())
    torch.testing.assert_close(loss_per_dim, losses.mean(dim=(0, 1)))


def test_masked_subspace_loss_restricts_to_real_action_dims():
    # Same construction as test_full_width_loss_includes_padded_coordinates_and_masks_only_timesteps
    losses = torch.zeros(2, 3, 4)
    losses[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    losses[0, 1] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    losses[0, 2] = 1000.0
    losses[1, 0] = torch.tensor([9.0, 10.0, 11.0, 12.0])
    losses[1, 1:] = 1000.0
    action_is_pad = torch.tensor([[False, False, True], [False, True, True]])

    # Option B: only the first 2 (real) coordinates participate; the 1000 padded
    # entries are excluded because they live in padded coordinates/timesteps.
    loss, loss_per_dim = reduce_flow_matching_loss(losses, action_is_pad, active_action_dim=2)
    per_sample_loss, _ = reduce_flow_matching_loss(
        losses, action_is_pad, reduction="none", active_action_dim=2
    )

    torch.testing.assert_close(loss, torch.tensor(5.5))
    torch.testing.assert_close(per_sample_loss, torch.tensor([3.5, 9.5]))
    torch.testing.assert_close(loss_per_dim, torch.tensor([5.0, 6.0]))
    assert loss_per_dim.shape[0] == 2


@pytest.mark.parametrize(
    ("losses", "action_is_pad", "reduction"),
    [
        (torch.zeros(2, 3), None, "mean"),
        (torch.zeros(2, 3, 4), torch.zeros(2, 2, dtype=torch.bool), "mean"),
        (torch.zeros(2, 3, 4), None, "sum"),
    ],
)
def test_full_width_loss_validates_inputs(losses, action_is_pad, reduction):
    with pytest.raises(ValueError):
        reduce_flow_matching_loss(losses, action_is_pad, reduction)


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


def test_openpi_flow_preserves_full_width_euler_integration():
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
