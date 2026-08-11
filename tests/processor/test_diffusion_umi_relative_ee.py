import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import (
    UmiAbsoluteActionsStep,
    UmiDeriveStateFromActionStep,
    UmiRelativeActionsStep,
    UmiRelativeStateStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_diffusion_umi_config_and_processors_use_canonical_pipeline():
    config = DiffusionConfig(
        use_umi_relative_ee=True,
        n_obs_steps=1,
        horizon=32,
        chunk_size=30,
        n_action_steps=30,
        down_dims=(128, 256),
        device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )
    stats = {
        OBS_STATE: {"min": torch.full((20,), -1.0), "max": torch.full((20,), 1.0)},
        ACTION: {"min": torch.full((10,), -1.0), "max": torch.full((10,), 1.0)},
    }

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, stats)

    assert config.action_delta_indices == [-1] + list(range(32))
    assert config.observation_delta_indices is None
    assert config.normalization_mapping["ACTION"] is NormalizationMode.MIN_MAX
    assert any(isinstance(step, UmiDeriveStateFromActionStep) for step in preprocessor.steps)
    relative = next(step for step in preprocessor.steps if isinstance(step, UmiRelativeActionsStep))
    assert any(isinstance(step, UmiRelativeStateStep) for step in preprocessor.steps)
    absolute = next(step for step in postprocessor.steps if isinstance(step, UmiAbsoluteActionsStep))
    assert absolute.relative_step is relative
    assert absolute.single_action_reference_steps == 30
