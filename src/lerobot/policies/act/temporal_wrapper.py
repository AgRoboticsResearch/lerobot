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

"""Temporal ACT wrapper for UMI-style temporal batching.

This module provides a wrapper for the ACT policy that enables it to process
temporal observations using UMI-style batching, where each timestep is encoded
independently through the standard backbone and features are concatenated.
"""

import torch
import torch.nn as nn

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT


class TemporalACTWrapper(nn.Module):
    """Wrapper for ACT model that handles UMI-style temporal batching.

    This wrapper enables ACT (which expects n_obs_steps=1) to process temporal observations
    by:
    1. Flattening temporal dimension to batch: (B, T, ...) -> (B*T, ...)
    2. Encoding each timestep independently through standard backbone
    3. Aggregating features: (B*T, F) -> (B, T*F)

    This preserves pretrained weights and gives each timestep clean encoding.

    Temporal positional encoding is applied via learnable scaling of base positional
    embeddings, allowing the transformer to distinguish between timesteps while
    maintaining compatibility with pretrained weights.

    All inputs (including obs_state_horizon=1) go through the unified temporal processing
    path for consistency.
    """

    def __init__(self, act_model: ACT, config: ACTConfig):
        super().__init__()
        self.model = act_model
        self.config = config
        self.obs_state_horizon = getattr(config, 'obs_state_horizon', 1)
        # Learnable temporal scale for positional encoding
        # This allows the model to learn the optimal temporal relationship between timesteps
        self.temporal_pos_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, batch):
        """Forward pass with UMI-style temporal batching."""

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

            # For VAE encoding, only use current timestep state/action (not aggregated temporal)
            if self.config.robot_state_feature:
                state = batch[OBS_STATE]  # (B, T, D)
                # Use the last timestep (current state) for VAE encoding
                state_current = state[:, -1, :]  # (B, D)
                state_embed = self.model.vae_encoder_robot_state_input_proj(state_current)  # (B, dim)
                state_embed = state_embed.unsqueeze(1)  # (B, 1, dim)

            # Extract current action for VAE encoding (handle temporal dimension)
            action = batch[ACTION]
            if action.ndim == 3:  # (B, T, D) - temporal dimension present
                action = action[:, -1, :]  # Use last timestep: (B, D)
            action_embed = self.model.vae_encoder_action_input_proj(action)  # (B, dim)
            action_embed = action_embed.unsqueeze(1)  # (B, 1, dim)

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
            # Add temporal offset to positional embeddings so the transformer can distinguish timesteps
            base_pos_embed = self.model.encoder_1d_feature_pos_embed.weight[1:2]  # (1, dim)
            for t in range(T_state):
                encoder_in_tokens.append(state_embed[t])
                # Apply temporal offset: timestep t gets scaled positional embedding
                # This preserves compatibility with pretrained weights while enabling temporal distinction
                temporal_pos = base_pos_embed * (1 + self.temporal_pos_scale * t)
                encoder_in_pos_embed.append(temporal_pos)

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
