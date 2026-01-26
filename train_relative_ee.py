#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. All rights reserved.
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

"""
Training script for ACT policy with relative end-effector pose dataset (UMI-style).

This script uses RelativeEEDataset which transforms absolute ee poses to wrist-relative
poses using proper SE(3) transformation following UMI's approach:

UMI-Style Temporal Batching:
- Observations: (B, T, ...) where T is obs_state_horizon
- Each timestep is encoded independently: (B, T, ...) -> (B*T, ...) -> encode -> (B*T, F) -> (B, T*F)
- Images: (B, T, C, H, W) -> (B*T, C, H, W) - standard 3-channel backbone processes each frame
- State: (B, T, D) -> (B*T, D) - encode each timestep independently
- Features aggregated: (B*T, F) -> (B, T*F)

This preserves pretrained ResNet weights (trained on 3-channel RGB) and gives each
timestep a clean independent encoding before concatenation.

Actions:
- Future poses relative to current pose: T_rel = T_curr^{-1} @ T_future
- Shape: (action_horizon, 10)

Reference: https://github.com/real-stanford/universal_manipulation_interface
"""

import logging
import sys

import torch
import torch.nn as nn

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.relative_ee_dataset import RelativeEEDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import (
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TemporalFlattenProcessor,
    TemporalNormalizeProcessor,
)
from lerobot.scripts.lerobot_train import train


class TemporalACTWrapper(nn.Module):
    """Wrapper for ACT model that handles UMI-style temporal batching.

    This wrapper enables ACT (which expects n_obs_steps=1) to process temporal observations
    by:
    1. Flattening temporal dimension to batch: (B, T, ...) -> (B*T, ...)
    2. Encoding each timestep independently through standard backbone
    3. Aggregating features: (B*T, F) -> (B, T*F)

    This preserves pretrained weights and gives each timestep clean encoding.

    When obs_state_horizon=1, this wrapper is a no-op (standard ACT behavior).
    """

    def __init__(self, act_model: ACT, config: ACTConfig):
        super().__init__()
        self.model = act_model
        self.config = config
        self.obs_state_horizon = getattr(config, 'obs_state_horizon', 1)

    def forward(self, batch):
        """Forward pass with UMI-style temporal batching."""
        # If no temporal dimension, use standard forward
        if self.obs_state_horizon == 1:
            return self.model(batch)

        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_ENV_STATE, ACTION

        # Get batch size and original batch for later reference
        if OBS_IMAGES in batch and len(batch[OBS_IMAGES]) > 0:
            # Images have shape (B, T, C, H, W)
            img = batch[OBS_IMAGES][0]
            batch_size, T = img.shape[:2]
        elif OBS_STATE in batch:
            # State has shape (B, T, D)
            state = batch[OBS_STATE]
            if state.ndim == 3:
                batch_size, T = state.shape[:2]
            else:
                batch_size = state.shape[0]
                T = 1
        else:
            raise ValueError("No valid observation keys in batch")

        # Prepare the latent for input to the transformer encoder
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = torch.nn.functional.relu(
                self.model.vae_encoder_cls_embed.weight).unsqueeze(0).repeat(batch_size, 1, 1)

            # For VAE encoding, only use current timestep state (not aggregated temporal)
            if self.config.robot_state_feature:
                state = batch[OBS_STATE]  # (B, T, D)
                # Use the last timestep (current state) for VAE encoding
                state_current = state[:, -1, :]  # (B, D)
                state_embed = self.model.vae_encoder_robot_state_input_proj(state_current)  # (B, dim)
                state_embed = state_embed.unsqueeze(1)  # (B, 1, dim)

            action_embed = self.model.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.model.vae_encoder_pos_enc.clone().detach()

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )

            cls_token_out = self.model.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.model.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32
            ).to(batch[OBS_STATE].device)

        # Prepare transformer encoder inputs
        import einops
        encoder_in_tokens = [self.model.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = [self.model.encoder_1d_feature_pos_embed.weight[0:1]]  # (1, dim)

        # Handle state with temporal dimension - keep each timestep as separate token
        if self.config.robot_state_feature:
            state = batch[OBS_STATE]  # (B, T, D)
            B, T_state, D = state.shape
            state_flat = state.reshape(B * T_state, D)  # (B*T, D)
            state_embed_flat = self.model.encoder_robot_state_input_proj(state_flat)  # (B*T, dim)
            state_embed = state_embed_flat.reshape(B, T_state, -1)  # (B, T, dim)
            # Reshape to (T, B, dim) and add each timestep as separate token
            state_embed = state_embed.permute(1, 0, 2)  # (T, B, dim)
            # Extend encoder_in_tokens with each timestep's state
            for t in range(T_state):
                encoder_in_tokens.append(state_embed[t])
                encoder_in_pos_embed.append(self.model.encoder_1d_feature_pos_embed.weight[1:2])  # (1, dim)

        if self.config.env_state_feature:
            env_state_embed = self.model.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            encoder_in_tokens.append(env_state_embed)
            encoder_in_pos_embed.append(self.model.encoder_1d_feature_pos_embed.weight[2:3])  # (1, dim)

        if self.config.image_features:
            # Handle images with temporal dimension - UMI-style batching
            for img in batch[OBS_IMAGES]:
                # img shape: (B, T, C, H, W)
                B, T, C, H, W = img.shape
                img_flat = img.reshape(B * T, C, H, W)  # (B*T, C, H, W)

                # Process through backbone (standard 3-channel conv)
                cam_features = self.model.backbone(img_flat)["feature_map"]  # (B*T, C', H', W')

                # Reshape back to separate batch and temporal
                _, C_feat, H_feat, W_feat = cam_features.shape
                cam_features = cam_features.reshape(B, T, C_feat, H_feat, W_feat)  # (B, T, C', H', W')

                # Process each timestep independently for projection
                # Collect all spatial features from all timesteps
                all_cam_features = []
                all_cam_pos_embeds = []
                for t in range(T):
                    feat_t = cam_features[:, t]  # (B, C', H', W')
                    # Apply pos embedding (returns (1, C', H', W') for 2D pos embedder)
                    pos_t = self.model.encoder_cam_feat_pos_embed(feat_t).to(dtype=feat_t.dtype)
                    # The pos embedder returns shape (1, C', H', W') - the first dim is a "batch" dim
                    # We need to expand it to match actual batch size
                    pos_t = pos_t.expand(B, -1, -1, -1)  # (B, C', H', W')
                    # Apply projection
                    proj_t = self.model.encoder_img_feat_input_proj(feat_t)  # (B, dim, H', W')

                    # Rearrange to (H'*W', B, dim)
                    feat_t = einops.rearrange(proj_t, "b c h w -> (h w) b c")
                    pos_t = einops.rearrange(pos_t, "b c h w -> (h w) b c")

                    all_cam_features.append(feat_t)
                    all_cam_pos_embeds.append(pos_t)

                # Concatenate all timesteps along sequence dimension
                cam_features = torch.cat(all_cam_features, dim=0)  # (T*H'*W', B, dim)
                cam_pos_embed = torch.cat(all_cam_pos_embeds, dim=0)  # (T*H'*W', B, dim)

                # Add to encoder inputs - reshape spatial embeddings to (seq, 1, dim)
                encoder_in_tokens.extend(list(cam_features.unbind(0)))
                # For positional embeddings, take first batch element: (B, dim) -> (1, dim)
                # The stacked result should be (seq, 1, dim) where seq is number of tokens
                # and 1 is the batch dimension (we use first batch element as template)
                for x in cam_pos_embed.unbind(0):
                    encoder_in_pos_embed.append(x[:1])  # (B, dim) -> (1, dim)

        # Stack all tokens along the sequence dimension
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules
        encoder_out = self.model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.model.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.model.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C)
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.model.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


def make_relative_ee_dataset(cfg: TrainPipelineConfig, obs_state_horizon: int = 2, num_stat_samples: int = 1000):
    """
    Create RelativeEEDataset with proper SE(3) wrist-relative transformation (UMI-style).

    This replaces the standard make_dataset function to use RelativeEEDataset instead
    of LeRobotDataset. The dataset automatically updates its metadata to reflect the
    new action and observation shapes (10D instead of 7D).

    Args:
        cfg: Training pipeline configuration
        obs_state_horizon: Number of historical timesteps in observation state.
            Default is 2, matching UMI's low_dim_obs_horizon.
        num_stat_samples: Number of samples for normalization stats computation.
            0 = use all training samples (UMI-style, slow), default 1000 (fast).

    Returns:
        RelativeEEDataset instance
    """
    # Get dataset metadata
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision
    )

    # IMPORTANT: Store fps in policy config for inference use
    # During inference, the model needs to know what fps it was trained with
    # to correctly compute action horizons and delta timestamps.
    if not hasattr(cfg.policy, 'fps'):
        cfg.policy.fps = ds_meta.fps
        logging.info(f"Set policy.fps = {cfg.policy.fps} Hz (from dataset metadata)")

    # Resolve delta timestamps from policy config
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    # Create RelativeEEDataset with wrist-relative SE(3) transformation
    dataset = RelativeEEDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        obs_state_horizon=obs_state_horizon,  # UMI-style: historical observations
        num_stat_samples=num_stat_samples,  # Control stats computation speed
    )

    logging.info(f"Created RelativeEEDataset (UMI-style)")
    logging.info(f"  obs_state_horizon: {obs_state_horizon}")
    logging.info(f"  observation.state metadata shape: {dataset.meta.info['features']['observation.state']['shape']}")
    logging.info(f"  observation.images.camera metadata shape: {dataset.meta.info['features'].get('observation.images.camera', {}).get('shape', 'N/A')}")
    logging.info(f"  action metadata shape: {dataset.meta.info['features']['action']['shape']}")
    logging.info(f"  Action names: {dataset.meta.info['features']['action'].get('names', 'N/A')}")
    logging.info(f"  Note: Actual data includes temporal dimension (T={obs_state_horizon})")
    logging.info(f"        Data shapes: state (B,T,10), images (B,T,C,H,W)")
    logging.info("")
    logging.info(f"  UMI-style processing: (B,T,...) -> (B*T,...) -> encode -> (B*T,F) -> (B,T*F)")

    return dataset


# Override the make_dataset function in the train function's scope
import lerobot.scripts.lerobot_train as train_module
original_make_dataset = train_module.make_dataset


def _make_relative_ee_dataset_wrapper(cfg: TrainPipelineConfig):
    """Wrapper that creates RelativeEEDataset with configurable obs_state_horizon."""
    # Get obs_state_horizon from policy config or use default
    obs_state_horizon = getattr(cfg.policy, 'obs_state_horizon', 2)
    # Get num_stat_samples from training config
    num_stat_samples = getattr(cfg, 'num_stat_samples', 1000)
    return make_relative_ee_dataset(cfg, obs_state_horizon=obs_state_horizon, num_stat_samples=num_stat_samples)


train_module.make_dataset = _make_relative_ee_dataset_wrapper

# Also need to override in the factory module since it might be imported elsewhere
import lerobot.datasets.factory as factory_module
factory_module.make_dataset = _make_relative_ee_dataset_wrapper


# Override make_policy to wrap ACT with temporal batching
import lerobot.policies.factory as policy_factory
original_make_policy = policy_factory.make_policy


def _make_policy_wrapper(cfg, ds_meta=None, **kwargs):
    """Wrapper that wraps ACT with TemporalACTWrapper for UMI-style temporal batching."""
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = original_make_policy(cfg, ds_meta=ds_meta, **kwargs)

    # Wrap ACT model with temporal batching support
    if isinstance(cfg, ACTConfig) and isinstance(policy, ACTPolicy):
        obs_state_horizon = getattr(cfg, 'obs_state_horizon', 1)

        if obs_state_horizon > 1:
            # Wrap the ACT model with temporal batching
            original_model = policy.model
            policy.model = TemporalACTWrapper(original_model, cfg)
            logging.info(f"Wrapped ACT model with TemporalACTWrapper (obs_state_horizon={obs_state_horizon})")
            logging.info("  Using UMI-style temporal batching:")
            logging.info("    - Images: (B, T, C, H, W) -> (B*T, C, H, W) -> encode -> (B, T*F)")
            logging.info("    - State: (B, T, D) -> (B*T, D) -> encode -> (B, T*F)")
            logging.info("    - Preserves pretrained ResNet weights (3-channel)")

    return policy


policy_factory.make_policy = _make_policy_wrapper
train_module.make_policy = _make_policy_wrapper


def make_act_pre_post_processors_with_temporal(
    config: ACTConfig,
    dataset_stats: dict | None = None,
    obs_state_horizon: int = 2,
) -> tuple[
    PolicyProcessorPipeline[dict, dict],
    PolicyProcessorPipeline,
]:
    """Create ACT processors with temporal observation handling (UMI-style).

    This wraps the standard ACT processor pipeline but handles temporal observations
    by using temporal normalization and keeping temporal dimensions intact for batching.

    Args:
        config: ACT policy configuration
        dataset_stats: Normalization statistics from dataset
        obs_state_horizon: Number of temporal steps in observations

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    # Create custom preprocessor with temporal normalization
    # NOTE: Device step must come before normalization so stats are on same device as data
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        TemporalNormalizeProcessor(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
            obs_state_horizon=obs_state_horizon,
        ),
    ]

    # Standard postprocessor
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    preprocessor = PolicyProcessorPipeline[dict, dict](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[dict, dict](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
    )

    logging.info(f"Created temporal ACT processors with obs_state_horizon={obs_state_horizon}")
    logging.info(f"  Preprocessor steps: {[step.__class__.__name__ for step in preprocessor.steps]}")

    return preprocessor, postprocessor


# Override the make_pre_post_processors function for ACT
import lerobot.scripts.lerobot_train as train_module
original_make_pre_post_processors = policy_factory.make_pre_post_processors


def _make_pre_post_processors_wrapper(policy_cfg, pretrained_path=None, **kwargs):
    """Wrapper that adds temporal processor for ACT policies with RelativeEEDataset."""
    from lerobot.policies.act.configuration_act import ACTConfig

    # Only apply temporal processor for ACT policies
    if isinstance(policy_cfg, ACTConfig):
        obs_state_horizon = getattr(policy_cfg, 'obs_state_horizon', 2)
        # Extract dataset_stats from kwargs
        dataset_stats = kwargs.pop('dataset_stats', None)
        _ = pretrained_path  # Unused
        return make_act_pre_post_processors_with_temporal(
            config=policy_cfg,
            dataset_stats=dataset_stats,
            obs_state_horizon=obs_state_horizon,
        )
    else:
        # For other policies, use the original function
        return original_make_pre_post_processors(policy_cfg, pretrained_path=pretrained_path, **kwargs)


policy_factory.make_pre_post_processors = _make_pre_post_processors_wrapper
train_module.make_pre_post_processors = _make_pre_post_processors_wrapper

# Also need to override _make_processors_from_policy_config which is used internally by policies
original_make_processors_from_policy_config = policy_factory._make_processors_from_policy_config


def _make_processors_from_policy_config_wrapper(config, dataset_stats=None):
    """Wrapper that adds temporal processor for ACT policies."""
    from lerobot.policies.act.configuration_act import ACTConfig

    # Only apply temporal processor for ACT policies
    if isinstance(config, ACTConfig):
        obs_state_horizon = getattr(config, 'obs_state_horizon', 2)
        return make_act_pre_post_processors_with_temporal(
            config=config,
            dataset_stats=dataset_stats,
            obs_state_horizon=obs_state_horizon,
        )
    else:
        # For other policies, use the original function
        return original_make_processors_from_policy_config(config, dataset_stats=dataset_stats)


policy_factory._make_processors_from_policy_config = _make_processors_from_policy_config_wrapper


@parser.wrap()
def train_with_relative_ee(cfg: TrainPipelineConfig):
    """
    Train ACT policy using RelativeEEDataset (UMI-style).

    This function wraps the standard training pipeline but uses RelativeEEDataset
    instead of LeRobotDataset. The rest of the training process remains unchanged.

    Following UMI's design:
    - Observations: Historical poses relative to current (provides velocity info)
    - Images: Historical images provided (obs_state_horizon frames)
    - Actions: Future poses relative to current
    - Both use 6D rotation representation
    - Temporal observations use UMI-style batching (each timestep encoded independently)

    Args:
        cfg: Training pipeline configuration containing dataset and policy settings.
    """
    # Get obs_state_horizon from policy config or use default
    obs_state_horizon = getattr(cfg.policy, 'obs_state_horizon', 2)
    # Get num_stat_samples from training config
    num_stat_samples = getattr(cfg, 'num_stat_samples', 1000)

    # Log that we're using relative EE dataset with temporal observations
    logging.info("=" * 80)
    logging.info("Training with RelativeEEDataset (UMI-style: temporal obs + relative actions)")
    logging.info("=" * 80)
    logging.info(f"Dataset: {cfg.dataset.repo_id}")
    logging.info(f"Policy: {cfg.policy.type}")
    logging.info(f"obs_state_horizon: {obs_state_horizon}")
    logging.info(f"num_stat_samples: {num_stat_samples} ({'all samples' if num_stat_samples == 0 else f'{num_stat_samples} random samples'})")
    logging.info("")
    logging.info("UMI-style temporal batching:")
    logging.info("  1. Dataset output: (B, T, ...) where T={}".format(obs_state_horizon))
    logging.info("  2. Flatten to batch: (B, T, ...) -> (B*T, ...)")
    logging.info("  3. Encode independently: standard 3-channel backbone processes each frame")
    logging.info("  4. Aggregate features: (B*T, F) -> (B, T*F)")
    logging.info("")
    logging.info("This preserves pretrained ResNet weights and gives clean encoding per timestep.")
    logging.info("=" * 80)

    # IMPORTANT: Following UMI's normalization approach
    # UMI uses min/max normalization (map to [-1, 1]) for position and gripper,
    # and identity normalization for rotation 6D.
    # See: diffusion_policy/dataset/umi_dataset.py:223-225
    cfg.policy.normalization_mapping["ACTION"] = NormalizationMode.MIN_MAX
    cfg.policy.normalization_mapping["STATE"] = NormalizationMode.MIN_MAX
    logging.info("Normalization: MIN_MAX (UMI-style: pos/gripper to [-1,1], rot6d identity)")

    # Call the standard training function
    # It will use our overridden make_dataset function
    train(cfg)


if __name__ == "__main__":
    train_with_relative_ee()
