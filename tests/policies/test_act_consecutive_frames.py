"""Tests for `ACTConfig.consecutive_frames` (channel-stacked multi-frame input).

Covers the Q3 experiment surface: the observation delta indices, the widened backbone
conv1 with inflated pretrained filters, the `(B, T, C, H, W) -> (B, T*C, H, W)` merge at
both gather sites, and the checkpoint round-trip (guards against `strict=False` silently
dropping the widened conv1 if the config ever desynced from the weights).
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CAMERA = "observation.images.camera"
H = W = 64


def make_config(**overrides) -> ACTConfig:
    kwargs = {
        "input_features": {
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
        "chunk_size": 4,
        "n_action_steps": 4,
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
        "device": "cpu",
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


def make_batch(batch_size: int = 2, consecutive_frames: int = 1) -> dict[str, torch.Tensor]:
    frames = [torch.randn(batch_size, 3, H, W)]
    if consecutive_frames > 1:
        frames = [torch.randn(batch_size, consecutive_frames, 3, H, W)]
    return {
        CAMERA: frames[0],
        OBS_STATE: torch.randn(batch_size, 20),
        ACTION: torch.randn(batch_size, 4, 10),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def test_observation_delta_indices_default_is_none():
    assert make_config().observation_delta_indices is None


def test_observation_delta_indices_oldest_first_window():
    assert make_config(consecutive_frames=2).observation_delta_indices == [-1, 0]
    assert make_config(consecutive_frames=3).observation_delta_indices == [-2, -1, 0]


def test_consecutive_frames_must_be_positive():
    with pytest.raises(ValueError, match="consecutive_frames"):
        make_config(consecutive_frames=0)


def test_conv1_widened_and_inflated_from_pretrained_draw():
    torch.manual_seed(1234)
    single = ACTPolicy(make_config())
    torch.manual_seed(1234)
    stacked = ACTPolicy(make_config(consecutive_frames=2))

    ref_conv1 = single.model.backbone["conv1"]
    wide_conv1 = stacked.model.backbone["conv1"]
    assert wide_conv1.in_channels == 2 * ref_conv1.in_channels
    assert tuple(wide_conv1.weight.shape) == (64, 6, 7, 7)
    # Same RNG draw at construction => the widened filters must be the tiled
    # originals halved, so identical frames reproduce pretrained activations.
    assert torch.allclose(wide_conv1.weight[:, :3], ref_conv1.weight / 2)
    assert torch.allclose(wide_conv1.weight[:, 3:], ref_conv1.weight / 2)
    # Only conv1 grew: everything else matches the single-frame architecture.
    single_params = dict(single.model.named_parameters())
    stacked_params = dict(stacked.model.named_parameters())
    assert set(stacked_params) == set(single_params)
    for name, parameter in single_params.items():
        if name != "backbone.conv1.weight":
            assert stacked_params[name].shape == parameter.shape


def test_forward_and_predict_merge_time_into_channels():
    policy = ACTPolicy(make_config(consecutive_frames=2))
    policy.train()
    loss, _ = policy.forward(make_batch(consecutive_frames=2))
    assert torch.isfinite(loss)

    policy.eval()
    batch = make_batch(batch_size=1, consecutive_frames=2)
    policy._stack_consecutive_frames({OBS_IMAGES: [batch[CAMERA]]})  # exercised via gather below
    # The merge is what predict_action_chunk runs internally; verify shape flow
    # end to end.
    chunk = policy.predict_action_chunk(dict(batch))
    assert tuple(chunk.shape) == (1, 4, 10)


def test_single_frame_batch_is_untouched():
    policy = ACTPolicy(make_config())
    batch = make_batch()
    images = [batch[CAMERA]]
    merged = dict({OBS_IMAGES: images})
    policy._stack_consecutive_frames(merged)
    assert merged[OBS_IMAGES][0] is images[0]  # no-op at consecutive_frames == 1


def test_checkpoint_roundtrip_preserves_widened_conv1(tmp_path):
    torch.manual_seed(7)
    policy = ACTPolicy(make_config(consecutive_frames=2))
    policy.eval()
    batch = make_batch(batch_size=1, consecutive_frames=2)
    expected = policy.predict_action_chunk(dict(batch))

    policy.save_pretrained(tmp_path)
    reloaded = ACTPolicy.from_pretrained(tmp_path)
    assert reloaded.config.consecutive_frames == 2
    assert tuple(reloaded.model.backbone["conv1"].weight.shape) == (64, 6, 7, 7)
    assert torch.allclose(
        reloaded.model.backbone["conv1"].weight, policy.model.backbone["conv1"].weight
    )
    reloaded.eval()
    assert torch.allclose(reloaded.predict_action_chunk(dict(batch)), expected)


# --- Q4: image-only ACT (`use_proprioception=False`) -------------------------


def make_image_only_config(**overrides) -> ACTConfig:
    config = make_config(use_umi_relative_ee=True, **overrides)
    config.input_features = {
        CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)),
    }
    return config


def make_image_only_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        CAMERA: torch.randn(batch_size, 3, H, W),
        ACTION: torch.randn(batch_size, 4, 10),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def test_image_only_forward_and_predict():
    policy = ACTPolicy(make_image_only_config(use_proprioception=False))
    assert policy.config.robot_state_feature is None
    policy.train()
    loss, _ = policy.forward(make_image_only_batch())
    assert torch.isfinite(loss)
    policy.eval()
    chunk = policy.predict_action_chunk(make_image_only_batch(batch_size=1))
    assert tuple(chunk.shape) == (1, 4, 10)


def test_image_only_processor_skips_state_steps():
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.processor.umi_relative_ee_processor import (
        UmiDeriveStateFromActionStep,
        UmiDropObsStateStep,
        UmiRelativeActionsStep,
        UmiRelativeStateStep,
    )

    pre, _ = make_act_pre_post_processors(make_image_only_config(use_proprioception=False))
    step_types = [type(step) for step in pre.steps]
    # The derive step must stay (it strips the anchor frame from the action and
    # provides the anchor for the 7D->10D conversion); the state is dropped
    # right after the conversion and never made relative.
    assert UmiDeriveStateFromActionStep in step_types
    assert UmiRelativeActionsStep in step_types
    assert UmiDropObsStateStep in step_types
    assert UmiRelativeStateStep not in step_types
    drop_idx = step_types.index(UmiDropObsStateStep)
    assert step_types.index(UmiRelativeActionsStep) < drop_idx

    pre_with_state, _ = make_act_pre_post_processors(make_config(use_umi_relative_ee=True))
    with_state_types = [type(s) for s in pre_with_state.steps]
    assert UmiDeriveStateFromActionStep in with_state_types
    assert UmiRelativeStateStep in with_state_types
    assert UmiDropObsStateStep not in with_state_types


def test_image_only_processor_end_to_end_shapes():
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.utils.constants import OBS_STATE

    config = make_image_only_config(use_proprioception=False)
    pre, _ = make_act_pre_post_processors(
        config,
        dataset_stats={
            CAMERA: {
                "mean": torch.zeros(3, 1, 1),
                "std": torch.ones(3, 1, 1),
            },
            ACTION: {
                "min": torch.zeros(10),
                "max": torch.ones(10),
            },
        },
    )
    batch = {
        CAMERA: torch.rand(2, 3, H, W),
        # raw absolute action window: [t-1, t, ..., t+29] in 7D
        ACTION: torch.rand(2, 31, 7),
        "action_is_pad": torch.zeros(2, 31, dtype=torch.bool),
    }
    out = pre(batch)
    assert OBS_STATE not in out  # image-only: no state reaches the policy
    assert out[ACTION].shape == (2, 30, 10)  # anchor stripped + 7D -> 10D relative
    assert out[CAMERA].shape == (2, 3, H, W)


def test_image_only_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(11)
    policy = ACTPolicy(make_image_only_config(use_proprioception=False))
    policy.eval()
    batch = make_image_only_batch(batch_size=1)
    expected = policy.predict_action_chunk(dict(batch))
    policy.save_pretrained(tmp_path)
    reloaded = ACTPolicy.from_pretrained(tmp_path)
    assert reloaded.config.use_proprioception is False
    assert reloaded.config.robot_state_feature is None
    reloaded.eval()
    assert torch.allclose(reloaded.predict_action_chunk(dict(batch)), expected)
