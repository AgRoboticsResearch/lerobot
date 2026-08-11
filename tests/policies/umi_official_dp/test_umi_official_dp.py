import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.umi_official_dp.configuration_umi_official_dp import (
    UmiOfficialDPConfig,
    UmiOfficialTransformerDPConfig,
)
from lerobot.policies.umi_official_dp.modeling_umi_official_dp import (
    UmiOfficialDPPolicy,
    UmiOfficialTransformerDPPolicy,
)
from lerobot.processor import UmiAbsoluteActionsStep, UmiRelativeActionsStep
from lerobot.utils.constants import ACTION, OBS_STATE


class _TinyViT(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.num_features = dim
        self.patch = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch(images).flatten(2).transpose(1, 2)
        return torch.cat((self.cls.expand(images.shape[0], -1, -1), patches), dim=1)


@pytest.fixture(autouse=True)
def fake_timm(monkeypatch):
    module = ModuleType("timm")
    module.create_model = lambda **kwargs: _TinyViT()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timm", module)


def _features():
    return {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
    }, {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))}


def _stats():
    return {
        OBS_STATE: {"min": torch.full((20,), -1.0), "max": torch.full((20,), 1.0)},
        "observation.images.front": {
            "mean": torch.tensor([0.485, 0.456, 0.406]),
            "std": torch.tensor([0.229, 0.224, 0.225]),
        },
        ACTION: {"min": torch.full((10,), -1.0), "max": torch.full((10,), 1.0)},
    }


@pytest.mark.parametrize(
    ("name", "config_class", "policy_class"),
    [
        ("umi_official_dp", UmiOfficialDPConfig, UmiOfficialDPPolicy),
        (
            "umi_official_transformer_dp",
            UmiOfficialTransformerDPConfig,
            UmiOfficialTransformerDPPolicy,
        ),
    ],
)
def test_factory_and_canonical_umi_processors(name, config_class, policy_class):
    inputs, outputs = _features()
    config = make_policy_config(name, device="cpu", input_features=inputs, output_features=outputs)

    assert isinstance(config, config_class)
    assert get_policy_class(name) is policy_class
    assert config.action_delta_indices == [-1] + list(range(32))
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=_stats())
    relative = next(step for step in preprocessor.steps if isinstance(step, UmiRelativeActionsStep))
    absolute = next(step for step in postprocessor.steps if isinstance(step, UmiAbsoluteActionsStep))
    assert absolute.relative_step is relative
    assert absolute.single_action_reference_steps == 30


def _batch(batch_size: int = 2):
    return {
        OBS_STATE: torch.randn(batch_size, 20),
        "observation.images.front": torch.randn(batch_size, 3, 32, 32),
        ACTION: torch.randn(batch_size, 32, 10).clamp(-1, 1),
        "action_is_pad": torch.zeros(batch_size, 32, dtype=torch.bool),
    }


def test_unet_candidate_forward_ema_and_sampling():
    inputs, outputs = _features()
    config = UmiOfficialDPConfig(
        device="cpu",
        input_features=inputs,
        output_features=outputs,
        vision_pretrained=False,
        image_size=(32, 32),
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy = UmiOfficialDPPolicy(config)
    loss, _ = policy(_batch())
    loss.backward()
    policy.update()
    policy.eval()
    actions = policy.predict_action_chunk(_batch(), noise=torch.zeros(2, 32, 10))

    assert torch.isfinite(loss)
    assert policy.ema_optimization_step.item() == 1
    assert actions.shape == (2, 30, 10)


def test_unet_candidate_checkpoint_round_trip_preserves_ema(tmp_path):
    inputs, outputs = _features()
    config = UmiOfficialDPConfig(
        device="cpu",
        input_features=inputs,
        output_features=outputs,
        vision_pretrained=False,
        image_size=(32, 32),
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy = UmiOfficialDPPolicy(config)
    policy.update()
    policy.save_pretrained(tmp_path)

    loaded = UmiOfficialDPPolicy.from_pretrained(tmp_path)

    assert isinstance(loaded.config, UmiOfficialDPConfig)
    assert loaded.ema_optimization_step.item() == 1
    assert torch.equal(next(policy.ema_diffusion.parameters()), next(loaded.ema_diffusion.parameters()))


def test_transformer_candidate_forward_sampling_and_backbone_lr_group():
    inputs, outputs = _features()
    config = UmiOfficialTransformerDPConfig(
        device="cpu",
        input_features=inputs,
        output_features=outputs,
        vision_pretrained=False,
        image_size=(32, 32),
        transformer_dim=32,
        transformer_num_heads=4,
        transformer_num_layers=1,
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy = UmiOfficialTransformerDPPolicy(config)
    loss, _ = policy(_batch())
    loss.backward()
    policy.update()
    policy.eval()
    actions = policy.predict_action_chunk(_batch(), noise=torch.zeros(2, 32, 10))
    optimizer_groups = policy.get_optim_params()

    assert torch.isfinite(loss)
    assert actions.shape == (2, 30, 10)
    assert optimizer_groups[1]["lr"] == pytest.approx(3e-5)
    assert sum(parameter.numel() for parameter in optimizer_groups[1]["params"]) > 0
