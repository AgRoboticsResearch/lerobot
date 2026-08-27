"""Tests for `ACTConfig.umi_state_window` (W-pose UMI proprio state window).

Covers the Q4 state-window experiment surface: the W-1 leading negative action
deltas, the `UmiDeriveStateFromActionStep` / `UmiRelativeStateStep` window
plumbing (state `[B, 10W]`, action still `[B, chunk, 10]`, pad-flag slicing of
the history prefix), the identity-current LAST state block, stats widths with
identity-rot6d slices, the checkpoint round-trip, and the W=2 default being
bit-identical to the historical two-pose formulas.
"""

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.datasets.umi_relative_ee_stats import (
    _absolute_aa_to_relative_rot6d,
    compute_umi_relative_ee_stats,
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor.converters import create_transition
from lerobot.processor.umi_relative_ee_processor import (
    UmiDeriveStateFromActionStep,
    UmiRelativeStateStep,
    to_umi_relative_state,
)
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERA = "observation.images.camera"
H = W = 64
CHUNK = 4


def make_config(window: int = 2, **overrides) -> ACTConfig:
    kwargs = {
        "input_features": {
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10 * window,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "latent_dim": 4,
        "n_vae_encoder_layers": 1,
        "pretrained_backbone_weights": None,
        "action_objective": "l1",
        "use_vae": False,
        "use_umi_relative_ee": True,
        "umi_state_window": window,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


def make_processed_batch(batch_size: int = 2, window: int = 2) -> dict[str, torch.Tensor]:
    return {
        CAMERA: torch.randn(batch_size, 3, H, W),
        OBS_STATE: torch.randn(batch_size, 10 * window),
        ACTION: torch.randn(batch_size, CHUNK, 10),
        "action_is_pad": torch.zeros(batch_size, CHUNK, dtype=torch.bool),
    }


# --- config: delta indices and validation -----------------------------------


def test_default_window_is_two_and_deltas_bit_identical():
    config = make_config()
    assert config.umi_state_window == 2
    assert config.action_delta_indices == [-1, 0, 1, 2, 3]


@pytest.mark.parametrize("window", [1, 0])
def test_window_below_two_raises(window):
    with pytest.raises(ValueError, match="umi_state_window"):
        make_config(window=window)


@pytest.mark.parametrize("window", [2, 5, 10])
def test_negative_delta_count_matches_window(window):
    config = make_config(window=window)
    deltas = config.action_delta_indices
    assert sum(1 for delta in deltas if delta < 0) == window - 1
    assert len(deltas) == window - 1 + CHUNK


def test_w5_and_w10_delta_lists():
    assert make_config(window=5).action_delta_indices == [-4, -3, -2, -1, 0, 1, 2, 3]
    assert make_config(window=10).action_delta_indices == list(range(-9, 4))


# --- processor steps ----------------------------------------------------------


def test_derive_step_slices_window_and_pad_flags():
    step = UmiDeriveStateFromActionStep(window=5)
    action = torch.rand(2, 5 - 1 + CHUNK, 7)
    pad = torch.zeros(2, 5 - 1 + CHUNK, dtype=torch.bool)
    pad[..., :4] = True  # clamped history prefix
    transition = create_transition(action=action)
    transition["action_is_pad"] = pad
    out = step(transition)
    assert out["observation"][OBS_STATE].shape == (2, 5, 7)
    assert out[ACTION].shape == (2, CHUNK, 7)
    assert out["action_is_pad"].shape == (2, CHUNK)
    assert not out["action_is_pad"].any()  # history prefix stripped from pads


def test_relative_state_step_shapes_and_live_guard():
    step = UmiRelativeStateStep(window=5)
    transition = create_transition(
        observation={OBS_STATE: torch.rand(2, 5, 7)}, action=torch.rand(2, CHUNK, 10)
    )
    out = step(transition)
    assert out["observation"][OBS_STATE].shape == (2, 50)

    live = UmiRelativeStateStep(window=5)
    live_transition = create_transition(observation={OBS_STATE: torch.rand(2, 7)})
    with pytest.raises(NotImplementedError, match="window=2"):
        live(live_transition)


def test_identity_current_block_is_last():
    window = torch.zeros(1, 5, 7)
    window[..., 6] = 0.37  # gripper
    state = to_umi_relative_state(window)
    assert state.shape == (1, 50)
    expected_last = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.37])
    torch.testing.assert_close(state[0, -10:], expected_last)
    with pytest.raises(ValueError, match="W>=2"):
        to_umi_relative_state(torch.zeros(1, 1, 7))


def test_processor_pipeline_constructs_windowed_steps():
    for window in (2, 5, 10):
        pre, _ = make_act_pre_post_processors(make_config(window=window))
        derive = next(s for s in pre.steps if isinstance(s, UmiDeriveStateFromActionStep))
        state_step = next(s for s in pre.steps if isinstance(s, UmiRelativeStateStep))
        assert derive.window == window
        assert state_step.window == window


def test_preprocessor_end_to_end_shapes():
    def run(window: int, batch_size: int = 2):
        config = make_config(window=window)
        stats = {
            CAMERA: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
            ACTION: {"min": torch.zeros(10), "max": torch.ones(10)},
            OBS_STATE: {
                "min": torch.zeros(10 * window),
                "max": torch.ones(10 * window),
            },
        }
        pre, _ = make_act_pre_post_processors(config, dataset_stats=stats)
        batch = {
            CAMERA: torch.rand(batch_size, 3, H, W),
            ACTION: torch.rand(batch_size, window - 1 + CHUNK, 7),
            "action_is_pad": torch.zeros(batch_size, window - 1 + CHUNK, dtype=torch.bool),
        }
        out = pre(batch)
        assert out[OBS_STATE].shape == (batch_size, 10 * window)
        assert out[ACTION].shape == (batch_size, CHUNK, 10)
        assert out[CAMERA].shape == (batch_size, 3, H, W)

    run(2)
    run(5)
    run(10)


# --- policy forward / round-trip ------------------------------------------------


def test_forward_loss_finite_at_w10():
    policy = ACTPolicy(make_config(window=10))
    policy.train()
    loss, _ = policy.forward(make_processed_batch(window=10))
    assert torch.isfinite(loss)


def test_checkpoint_roundtrip_preserves_window(tmp_path):
    torch.manual_seed(11)
    policy = ACTPolicy(make_config(window=5))
    policy.eval()
    batch = make_processed_batch(batch_size=1, window=5)
    expected = policy.predict_action_chunk(dict(batch))
    policy.save_pretrained(tmp_path)
    reloaded = ACTPolicy.from_pretrained(tmp_path)
    assert reloaded.config.umi_state_window == 5
    assert tuple(reloaded.config.robot_state_feature.shape) == (50,)
    reloaded.eval()
    assert torch.allclose(reloaded.predict_action_chunk(dict(batch)), expected)


# --- stats -----------------------------------------------------------------------


def _synthetic_dataset(lengths: list[int], seed: int = 3) -> dict:
    generator = np.random.default_rng(seed)
    actions, episodes = [], []
    for episode_index, length in enumerate(lengths):
        actions.append(generator.normal(size=(length, 7)).astype(np.float32))
        episodes.append(np.full(length, episode_index, dtype=np.int64))
    return {
        ACTION: np.concatenate(actions),
        "episode_index": np.concatenate(episodes),
    }


def test_stats_w2_matches_legacy_two_pose_formula():
    dataset = _synthetic_dataset([60, 50])
    chunk = 30
    stats = compute_umi_relative_ee_stats(dataset, chunk)

    # Legacy replication: starts with [s, s+chunk] contiguous, state = [pose(s), pose(s+1)].
    actions = dataset[ACTION]
    episodes = dataset["episode_index"]
    starts = np.concatenate(
        [
            episode_start + np.arange(len(np.flatnonzero(episodes == episode)) - chunk)
            for episode, episode_start in [(0, 0), (1, 60)]
        ]
    )
    assert len(starts) > 0
    base = actions[starts + 1]
    state_pair = np.stack([actions[starts], base], axis=1)
    state_base = np.broadcast_to(base[:, None, :], state_pair.shape)
    relative_state = _absolute_aa_to_relative_rot6d(state_base, state_pair).reshape(-1, 20)
    reference = RunningQuantileStats()
    reference.update(relative_state)
    expected = reference.get_statistics()
    for stat_name, array in expected.items():
        np.testing.assert_array_equal(stats["observation.state"][stat_name], array)


def test_stats_w5_width_and_identity_rot6d_slices():
    dataset = _synthetic_dataset([60, 50])
    chunk = 30
    stats = compute_umi_relative_ee_stats(dataset, chunk, identity_rot6d=True, state_window=5)
    state_stats = stats["observation.state"]
    for stat_name, array in state_stats.items():
        if stat_name == "count":
            continue
        assert array.shape == (50,), stat_name
    for pose in range(5):
        for stat_name, value in (("min", -1.0), ("max", 1.0)):
            block = state_stats[stat_name][10 * pose : 10 * pose + 10]
            np.testing.assert_array_equal(block[3:9], value)  # rot6d forced
            # translation and gripper dims untouched (not all ±1)
            assert not np.allclose(block[:3], value)
            assert block[9] != value
    # Action rot6d slice still forced at [3:9]; width unchanged.
    assert stats[ACTION]["min"].shape == (10,)
    np.testing.assert_array_equal(stats[ACTION]["min"][3:9], -1.0)


def test_stats_window_longer_than_episodes_raises():
    dataset = _synthetic_dataset([20])
    with pytest.raises(ValueError, match="contiguous"):
        compute_umi_relative_ee_stats(dataset, chunk_size=30, state_window=5)
