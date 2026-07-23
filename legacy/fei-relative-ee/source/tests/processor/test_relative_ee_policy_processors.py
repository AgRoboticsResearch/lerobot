"""Policy-generic tests for the UMI relative end-effector processor pipeline."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets import relative_action_stats
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, ProcessorStep
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.relative_action_processor import (
    AbsoluteRot6dActionsProcessorStep,
    RelativeRot6dActionsProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


class PassthroughTokenizer(ProcessorStep):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(self, features):
        return features


def _features():
    return (
        {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,)),
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )


def _stats():
    return {
        OBS_STATE: {"min": torch.full((20,), -2.0), "max": torch.full((20,), 2.0)},
        OBS_IMAGE: {},
        ACTION: {"min": torch.full((10,), -2.0), "max": torch.full((10,), 2.0)},
    }


def _raw_action_batch(batch_size: int, horizon_with_prefix: int) -> torch.Tensor:
    actions = torch.zeros(batch_size, horizon_with_prefix, 7)
    actions[..., 0] = torch.arange(horizon_with_prefix, dtype=torch.float32)
    actions[..., 6] = 0.5
    return actions


def _identity_relative_chunk(batch_size: int, horizon: int) -> torch.Tensor:
    actions = torch.zeros(batch_size, horizon, 10)
    actions[..., 3] = 1.0
    actions[..., 7] = 1.0
    actions[..., 9] = 0.5
    return actions


def _relative_steps(preprocessor, postprocessor):
    relative = next(
        step for step in preprocessor.steps if isinstance(step, RelativeRot6dActionsProcessorStep)
    )
    absolute = next(
        step for step in postprocessor.steps if isinstance(step, AbsoluteRot6dActionsProcessorStep)
    )
    return relative, absolute


def test_derived_action_stats_match_training_alignment():
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 0] = np.arange(4, dtype=np.float32)
    episode_indices = np.zeros(4, dtype=np.int64)
    stats = relative_action_stats._compute_relative_action_stats_derived(
        {ACTION: actions, "episode_index": episode_indices},
        {},
        chunk_size=2,
        exclude_joints=[],
        num_workers=0,
    )

    assert stats["min"][0] == pytest.approx(0.0)
    assert stats["max"][0] == pytest.approx(1.0)
    assert stats["mean"][0] == pytest.approx(0.5)


def test_recompute_stats_can_leave_source_dataset_untouched(monkeypatch, tmp_path):
    writes = []
    monkeypatch.setattr(relative_action_stats, "write_stats", lambda *args, **kwargs: writes.append(args))
    dataset = SimpleNamespace(
        meta=SimpleNamespace(features={}, stats={}),
        hf_dataset={},
        root=tmp_path,
    )

    relative_action_stats.recompute_stats(dataset, write_to_disk=False)

    assert writes == []
    assert dataset.meta.stats == {}


def test_smolvla_relative_processor_shapes_and_cache(monkeypatch):
    from lerobot.processor import relative_action_processor_smolvla as relative_smolvla

    monkeypatch.setattr(relative_smolvla, "TokenizerProcessorStep", PassthroughTokenizer)
    config = SmolVLAConfig(
        chunk_size=3,
        n_action_steps=3,
        derive_state_from_action=True,
        use_relative_actions=True,
        pose_dim=6,
        use_rot6d=True,
        device="cpu",
    )
    config.input_features, config.output_features = _features()
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }

    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, _stats())
    processed = preprocessor(
        {
            ACTION: _raw_action_batch(2, 4),
            OBS_IMAGE: torch.rand(2, 1, 3, 16, 16),
            "task": ["pick the strawberry", "pick the strawberry"],
        }
    )

    assert processed[OBS_STATE].shape == (2, 20)
    assert processed[ACTION].shape == (2, 3, 10)
    relative, absolute = _relative_steps(preprocessor, postprocessor)
    assert relative.cache_key == absolute.cache_key

    preprocessor.reset()
    postprocessor.reset()
    preprocessor({OBS_STATE: torch.zeros(2, 7), OBS_IMAGE: torch.rand(2, 3, 16, 16), "task": ["a", "b"]})
    output = postprocessor(_identity_relative_chunk(2, 3))
    assert output.shape == (2, 3, 7)


def test_diffusion_relative_processor_training_and_inference_shapes():
    config = DiffusionConfig(
        n_obs_steps=1,
        horizon=8,
        n_action_steps=4,
        derive_state_from_action=True,
        use_relative_actions=True,
        pose_dim=6,
        use_rot6d=True,
        device="cpu",
    )
    config.input_features, config.output_features = _features()
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, _stats())
    processed = preprocessor(
        {ACTION: _raw_action_batch(2, 9), OBS_IMAGE: torch.rand(2, 1, 3, 16, 16)}
    )
    assert processed[OBS_STATE].shape == (2, 1, 20)
    assert processed[ACTION].shape == (2, 8, 10)

    preprocessor.reset()
    postprocessor.reset()
    inference = preprocessor({OBS_STATE: torch.zeros(2, 7), OBS_IMAGE: torch.rand(2, 3, 16, 16)})
    assert inference[OBS_STATE].shape == (2, 20)
    output = postprocessor(_identity_relative_chunk(2, 4))
    assert output.shape == (2, 4, 7)


def test_relative_processors_round_trip_serialization_and_legacy_cache(tmp_path):
    config = DiffusionConfig(
        n_obs_steps=1,
        horizon=8,
        n_action_steps=4,
        derive_state_from_action=True,
        use_relative_actions=True,
        pose_dim=6,
        use_rot6d=True,
        device="cpu",
    )
    config.input_features, config.output_features = _features()
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }
    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, _stats())
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    def load_processors():
        loaded_preprocessor = PolicyProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        loaded_postprocessor = PolicyProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        return loaded_preprocessor, loaded_postprocessor

    loaded_preprocessor, loaded_postprocessor = load_processors()
    relative, absolute = _relative_steps(loaded_preprocessor, loaded_postprocessor)
    assert relative.cache_key == absolute.cache_key
    loaded_preprocessor({OBS_STATE: torch.zeros(1, 7), OBS_IMAGE: torch.rand(1, 3, 16, 16)})
    assert loaded_postprocessor(_identity_relative_chunk(1, 4)).shape == (1, 4, 7)

    # Processor files from existing ACT checkpoints predate cache_key. Removing it
    # exercises the default legacy cache shared by independently loaded pipelines.
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config_path = tmp_path / filename
        saved = json.loads(config_path.read_text())
        for step in saved["steps"]:
            step["config"].pop("cache_key", None)
        config_path.write_text(json.dumps(saved))

    legacy_preprocessor, legacy_postprocessor = load_processors()
    legacy_preprocessor({OBS_STATE: torch.zeros(1, 7), OBS_IMAGE: torch.rand(1, 3, 16, 16)})
    assert legacy_postprocessor(_identity_relative_chunk(1, 4)).shape == (1, 4, 7)


def test_relative_config_delta_indices_and_validation():
    smolvla = SmolVLAConfig(
        chunk_size=3,
        n_action_steps=3,
        derive_state_from_action=True,
        use_relative_actions=True,
        pose_dim=6,
        use_rot6d=True,
    )
    diffusion = DiffusionConfig(
        n_obs_steps=1,
        horizon=8,
        n_action_steps=4,
        derive_state_from_action=True,
        use_relative_actions=True,
        pose_dim=6,
        use_rot6d=True,
    )
    assert smolvla.action_delta_indices == [-1, 0, 1, 2]
    assert diffusion.action_delta_indices == [-1, 0, 1, 2, 3, 4, 5, 6, 7]

    with pytest.raises(ValueError, match="n_obs_steps=1"):
        DiffusionConfig(
            n_obs_steps=2,
            horizon=8,
            n_action_steps=4,
            use_relative_actions=True,
            pose_dim=6,
            use_rot6d=True,
        )


class FakeDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def generate_actions(self, batch, noise=None):
        self.inputs.append({key: value.clone() for key, value in batch.items()})
        batch_size = batch[OBS_STATE].shape[0]
        return torch.zeros(batch_size, 2, 3)


def _queue_only_diffusion_policy():
    policy = DiffusionPolicy.__new__(DiffusionPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        n_obs_steps=2,
        n_action_steps=2,
        image_features={},
        env_state_feature=None,
    )
    policy.diffusion = FakeDiffusionModel()
    policy.reset()
    return policy


def test_diffusion_predict_and_select_populate_observations_once():
    policy = _queue_only_diffusion_policy()
    first = torch.zeros(1, 20)
    second = torch.ones(1, 20)
    third = torch.full((1, 20), 2.0)

    predicted = policy.predict_action_chunk({OBS_STATE: first})
    assert predicted.shape == (1, 2, 3)
    assert policy.diffusion.inputs[-1][OBS_STATE].shape == (1, 2, 20)

    policy.reset()
    policy.diffusion.inputs.clear()
    policy.select_action({OBS_STATE: first})
    policy.select_action({OBS_STATE: second})
    policy.select_action({OBS_STATE: third})

    assert len(policy.diffusion.inputs) == 2
    latest_history = policy.diffusion.inputs[-1][OBS_STATE]
    assert torch.equal(latest_history[:, 0], second)
    assert torch.equal(latest_history[:, 1], third)
