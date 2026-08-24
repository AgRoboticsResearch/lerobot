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
