from __future__ import annotations

import torch

from lerobot.policies.lingbot_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id


def test_flow_match_scheduler_shapes_and_monotone_timesteps() -> None:
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(20)
    assert scheduler.timesteps.shape == (20,)
    assert torch.all(scheduler.timesteps[1:] <= scheduler.timesteps[:-1])

    sample = torch.zeros(1, 4, 2, 3, 3)
    assert scheduler.step(torch.ones_like(sample), scheduler.timesteps[0], sample).shape == sample.shape
    assert scheduler.add_noise(sample, torch.ones_like(sample), scheduler.timesteps[:2], t_dim=2).shape == sample.shape


def test_grid_and_patch_helpers_preserve_expected_layouts() -> None:
    assert get_mesh_id(4, 8, 16, 0).shape == (4, 4 * 8 * 16)
    action_grid = get_mesh_id(4, 4, 1, 1, action=True)
    assert action_grid.shape == (4, 16)
    assert torch.all(action_grid[1:3] < 0)

    sequence = torch.arange(1 * 4 * 8 * 16 * 48, dtype=torch.float32).reshape(1, 4 * 8 * 16, 48)
    assert data_seq_to_patch((1, 2, 2), sequence, 4, 8, 16).shape == (1, 48, 4, 8, 16)
