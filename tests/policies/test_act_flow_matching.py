import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def make_flow_config(**overrides) -> ACTConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
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
        "action_objective": "flow_matching",
        "use_vae": False,
        "flow_num_inference_steps": 2,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


def make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, 3),
        OBS_ENV_STATE: torch.randn(batch_size, 2),
        ACTION: torch.randn(batch_size, 4, 2),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def test_flow_matching_requires_non_vae_act():
    with pytest.raises(ValueError, match="use_vae=false"):
        make_flow_config(use_vae=True)


def test_diffusion_requires_non_vae_act():
    with pytest.raises(ValueError, match="use_vae=false"):
        make_flow_config(action_objective="diffusion", use_vae=True)


def test_flow_matching_only_adds_objective_required_decoder_conditioning_parameters():
    l1_policy = ACTPolicy(make_flow_config(action_objective="l1"))
    flow_policy = ACTPolicy(make_flow_config())
    l1_parameters = dict(l1_policy.model.named_parameters())
    flow_parameters = dict(flow_policy.model.named_parameters())
    flow_only_parameters = {name for name in flow_parameters if name.startswith("flow_")}

    assert flow_only_parameters == {
        "flow_action_input_proj.weight",
        "flow_action_input_proj.bias",
        "flow_time_mlp.0.weight",
        "flow_time_mlp.0.bias",
        "flow_time_mlp.2.weight",
        "flow_time_mlp.2.bias",
    }
    assert set(flow_parameters) - flow_only_parameters == set(l1_parameters)
    for name, parameter in l1_parameters.items():
        assert flow_parameters[name].shape == parameter.shape


def test_flow_and_diffusion_have_exactly_the_same_learned_architecture():
    torch.manual_seed(123)
    flow_policy = ACTPolicy(make_flow_config())
    torch.manual_seed(123)
    diffusion_policy = ACTPolicy(make_flow_config(action_objective="diffusion"))

    flow_parameters = dict(flow_policy.model.named_parameters())
    diffusion_parameters = dict(diffusion_policy.model.named_parameters())

    assert set(flow_parameters) == set(diffusion_parameters)
    for name, parameter in flow_parameters.items():
        assert diffusion_parameters[name].shape == parameter.shape
        torch.testing.assert_close(diffusion_parameters[name], parameter)


def test_diffusion_rejects_invalid_inference_step_count():
    with pytest.raises(ValueError, match="diffusion_num_inference_steps"):
        make_flow_config(
            action_objective="diffusion",
            diffusion_num_train_timesteps=10,
            diffusion_num_inference_steps=11,
        )


def test_flow_matching_forward_is_finite_and_differentiable():
    policy = ACTPolicy(make_flow_config())
    loss, metrics = policy(make_batch())

    assert torch.isfinite(loss)
    assert metrics["flow_loss"] == pytest.approx(loss.item())
    assert len(metrics["flow_loss_per_dim"]) == 2
    loss.backward()
    assert policy.model.flow_action_input_proj.weight.grad is not None


def test_flow_matching_predict_action_chunk_shape_and_seeded_noise():
    policy = ACTPolicy(make_flow_config()).eval()
    batch = make_batch()
    batch.pop(ACTION)
    batch.pop("action_is_pad")
    noise = torch.randn(2, 4, 2)

    first = policy.predict_action_chunk(batch, noise=noise, num_steps=2)
    second = policy.predict_action_chunk(batch, noise=noise, num_steps=2)

    assert first.shape == (2, 4, 2)
    torch.testing.assert_close(first, second)


def test_flow_matching_rejects_wrong_noise_shape():
    policy = ACTPolicy(make_flow_config()).eval()
    batch = make_batch()
    batch.pop(ACTION)
    batch.pop("action_is_pad")

    with pytest.raises(ValueError, match="Flow noise must have shape"):
        policy.predict_action_chunk(batch, noise=torch.randn(2, 3, 2))


def test_diffusion_forward_is_finite_and_differentiable():
    policy = ACTPolicy(make_flow_config(action_objective="diffusion"))
    loss, metrics = policy(make_batch())

    assert torch.isfinite(loss)
    assert metrics["diffusion_loss"] == pytest.approx(loss.item())
    assert len(metrics["diffusion_loss_per_dim"]) == 2
    assert policy.noise_scheduler.config.prediction_type == "epsilon"
    assert policy.noise_scheduler.config.beta_schedule == "squaredcos_cap_v2"
    loss.backward()
    assert policy.model.flow_action_input_proj.weight.grad is not None


def test_diffusion_predict_action_chunk_shape_and_seeded_noise():
    policy = ACTPolicy(make_flow_config(action_objective="diffusion")).eval()
    batch = make_batch()
    batch.pop(ACTION)
    batch.pop("action_is_pad")
    noise = torch.randn(2, 4, 2)

    first = policy.predict_action_chunk(batch, noise=noise, num_steps=2)
    second = policy.predict_action_chunk(batch, noise=noise, num_steps=2)

    assert first.shape == (2, 4, 2)
    torch.testing.assert_close(first, second)


def test_diffusion_rejects_wrong_noise_shape():
    policy = ACTPolicy(make_flow_config(action_objective="diffusion")).eval()
    batch = make_batch()
    batch.pop(ACTION)
    batch.pop("action_is_pad")

    with pytest.raises(ValueError, match="Diffusion noise must have shape"):
        policy.predict_action_chunk(batch, noise=torch.randn(2, 3, 2))
