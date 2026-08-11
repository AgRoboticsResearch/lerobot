import copy

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionPolicy,
    DiffusionSinusoidalPosEmb,
    _make_noise_scheduler,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_umi_official_dp import UmiOfficialDPConfig, UmiOfficialTransformerDPConfig


class UmiOfficialVisionEncoder(nn.Module):
    """The timm CLIP-ViT observation encoder used in the released UMI policies."""

    def __init__(self, config: UmiOfficialDPConfig):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "umi_official_dp requires timm. Install LeRobot with the "
                "`umi-official-dp` optional dependency."
            ) from exc

        self.config = config
        self.backbone = timm.create_model(
            model_name=config.vision_model_name,
            pretrained=config.vision_pretrained,
            global_pool="",
            num_classes=0,
        )
        if config.vision_frozen:
            self.backbone.requires_grad_(False)
        self.feature_dim = int(self.backbone.num_features)
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue,
        )
        self.register_buffer("imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def _preprocess(self, images: Tensor) -> Tensor:
        # The common LeRobot pipeline supplies ImageNet-normalized images. Undo it
        # before the upstream pixel-space augmentations.
        if self.config.input_is_imagenet_normalized:
            images = images * self.imagenet_std + self.imagenet_mean
        images = torchvision.transforms.functional.resize(images, self.config.image_size, antialias=True)
        if self.training:
            crop_h = int(self.config.image_size[0] * self.config.crop_ratio)
            crop_w = int(self.config.image_size[1] * self.config.crop_ratio)
            top, left, height, width = torchvision.transforms.RandomCrop.get_params(
                images, output_size=(crop_h, crop_w)
            )
            images = torchvision.transforms.functional.resized_crop(
                images,
                top,
                left,
                height,
                width,
                self.config.image_size,
                antialias=True,
            )
            if self.config.random_rotation_degrees > 0:
                angle = float(
                    torch.empty((), device=images.device).uniform_(
                        -self.config.random_rotation_degrees, self.config.random_rotation_degrees
                    )
                )
                images = torchvision.transforms.functional.rotate(images, angle)
            images = self.color_jitter(images)
        if self.config.apply_imagenet_normalization:
            images = (images - self.imagenet_mean) / self.imagenet_std
        return images

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.backbone(self._preprocess(images))
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected ViT tokens with shape [B,N,D], got {tuple(tokens.shape)}.")
        return tokens


class _OfficialEmaMixin:
    config: UmiOfficialDPConfig
    diffusion: nn.Module
    ema_diffusion: nn.Module
    ema_optimization_step: Tensor

    def _initialize_ema(self) -> None:
        self.ema_diffusion = copy.deepcopy(self.diffusion)
        self.ema_diffusion.eval()
        self.ema_diffusion.requires_grad_(False)
        self.register_buffer("ema_optimization_step", torch.zeros((), dtype=torch.long))

    def _ema_decay(self) -> float:
        optimization_step = int(self.ema_optimization_step.item())
        step = max(0, optimization_step - self.config.ema_update_after_step - 1)
        value = 1 - (1 + step / self.config.ema_inv_gamma) ** -self.config.ema_power
        if step <= 0:
            return 0.0
        return max(self.config.ema_min_value, min(value, self.config.ema_max_value))

    @torch.no_grad()
    def update(self) -> None:
        if not self.config.use_ema:
            return
        decay = self._ema_decay()
        for source, averaged in zip(
            self.diffusion.parameters(), self.ema_diffusion.parameters(), strict=True
        ):
            averaged.mul_(decay).add_(source.detach().to(dtype=averaged.dtype), alpha=1 - decay)
        for source, averaged in zip(self.diffusion.buffers(), self.ema_diffusion.buffers(), strict=True):
            averaged.copy_(source.detach().to(dtype=averaged.dtype))
        self.ema_optimization_step.add_(1)

    def _inference_model(self) -> nn.Module:
        return self.ema_diffusion if self.config.use_ema else self.diffusion


class UmiOfficialUnetDiffusionModel(nn.Module):
    def __init__(self, config: UmiOfficialDPConfig):
        super().__init__()
        self.config = config
        state_dim = config.robot_state_feature.shape[0]
        self.rgb_encoders = nn.ModuleList(
            [UmiOfficialVisionEncoder(config) for _ in range(len(config.image_features))]
        )
        global_cond_dim = state_dim + sum(encoder.feature_dim for encoder in self.rgb_encoders)
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = config.num_inference_steps or config.num_train_timesteps

    def _prepare_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        features = [batch[OBS_STATE]]
        images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
        image_features = [
            encoder(images)[:, 0]
            for encoder, images in zip(self.rgb_encoders, images_per_camera, strict=True)
        ]
        features.extend(feature.reshape(batch_size, n_obs_steps, -1) for feature in image_features)
        if self.config.env_state_feature:
            features.append(batch[OBS_ENV_STATE])
        return torch.cat(features, dim=-1).flatten(start_dim=1)

    def conditional_sample(self, batch_size: int, global_cond: Tensor, noise: Tensor | None = None) -> Tensor:
        device = global_cond.device
        dtype = global_cond.dtype
        sample = (
            noise
            if noise is not None
            else torch.randn(
                batch_size,
                self.config.horizon,
                self.config.action_feature.shape[0],
                device=device,
                dtype=dtype,
            )
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            timesteps = torch.full((batch_size,), timestep, dtype=torch.long, device=device)
            prediction = self.unet(sample, timesteps, global_cond=global_cond)
            sample = self.noise_scheduler.step(prediction, timestep, sample).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        conditioning = self._prepare_conditioning(batch)
        actions = self.conditional_sample(batch[OBS_STATE].shape[0], conditioning, noise=noise)
        start = self.config.n_obs_steps - 1
        return actions[:, start : start + self.config.n_action_steps]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        conditioning = self._prepare_conditioning(batch)
        trajectory = batch[ACTION]
        target_noise = torch.randn_like(trajectory)
        perturbed_noise = target_noise + self.config.input_perturbation * torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, perturbed_noise, timesteps)
        prediction = self.unet(noisy_trajectory, timesteps, global_cond=conditioning)
        target = target_noise if self.config.prediction_type == "epsilon" else trajectory
        loss = F.mse_loss(prediction, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            mask = (~batch["action_is_pad"]).unsqueeze(-1)
            return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1)
        return loss.mean()


class OfficialTransformerForActionDiffusion(nn.Module):
    def __init__(self, config: UmiOfficialTransformerDPConfig, max_condition_tokens: int):
        super().__init__()
        dim = config.transformer_dim
        self.input_embedding = nn.Linear(config.action_feature.shape[0], dim)
        self.action_position_embedding = nn.Parameter(torch.randn(1, config.horizon, dim))
        self.time_embedding = DiffusionSinusoidalPosEmb(dim)
        self.condition_position_embedding = nn.Parameter(torch.randn(1, max_condition_tokens + 1, dim))
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=config.transformer_num_heads,
            dim_feedforward=4 * dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.transformer_num_layers)
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, config.action_feature.shape[0])
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        nn.init.normal_(self.action_position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.condition_position_embedding, mean=0.0, std=0.02)

    def forward(self, sample: Tensor, timestep: Tensor | int, condition: Tensor) -> Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        time_token = self.time_embedding(timestep).unsqueeze(1)
        memory = torch.cat((condition, time_token), dim=1)
        memory = memory + self.condition_position_embedding[:, : memory.shape[1]]
        target = self.input_embedding(sample) + self.action_position_embedding[:, : sample.shape[1]]
        return self.head(self.final_norm(self.decoder(tgt=target, memory=memory)))


class UmiOfficialTransformerDiffusionModel(nn.Module):
    def __init__(self, config: UmiOfficialTransformerDPConfig):
        super().__init__()
        self.config = config
        self.rgb_encoders = nn.ModuleList(
            [UmiOfficialVisionEncoder(config) for _ in range(len(config.image_features))]
        )
        self.state_projection = nn.Linear(config.robot_state_feature.shape[0], config.transformer_dim)
        self.env_projection = (
            nn.Linear(config.env_state_feature.shape[0], config.transformer_dim)
            if config.env_state_feature
            else None
        )
        # ViT-B/16 at 224px emits 197 tokens per camera.
        tokens_per_image = (config.image_size[0] // 16) * (config.image_size[1] // 16) + 1
        max_tokens = tokens_per_image * len(config.image_features) + 1 + int(self.env_projection is not None)
        self.denoiser = OfficialTransformerForActionDiffusion(config, max_condition_tokens=max_tokens)
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = config.num_inference_steps or config.num_train_timesteps

    def _prepare_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        tokens = []
        images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
        for encoder, images in zip(self.rgb_encoders, images_per_camera, strict=True):
            image_tokens = encoder(images).reshape(batch_size, n_obs_steps, -1, self.config.transformer_dim)
            tokens.append(image_tokens.flatten(1, 2))
        tokens.append(self.state_projection(batch[OBS_STATE]))
        if self.env_projection is not None:
            tokens.append(self.env_projection(batch[OBS_ENV_STATE]))
        return torch.cat(tokens, dim=1)

    def conditional_sample(self, batch_size: int, condition: Tensor, noise: Tensor | None = None) -> Tensor:
        sample = (
            noise
            if noise is not None
            else torch.randn(
                batch_size,
                self.config.horizon,
                self.config.action_feature.shape[0],
                device=condition.device,
                dtype=condition.dtype,
            )
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.denoiser(sample, timestep, condition)
            sample = self.noise_scheduler.step(prediction, timestep, sample).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        condition = self._prepare_conditioning(batch)
        actions = self.conditional_sample(batch[OBS_STATE].shape[0], condition, noise=noise)
        start = self.config.n_obs_steps - 1
        return actions[:, start : start + self.config.n_action_steps]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        condition = self._prepare_conditioning(batch)
        trajectory = batch[ACTION]
        target_noise = torch.randn_like(trajectory)
        perturbed_noise = target_noise + self.config.input_perturbation * torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, perturbed_noise, timesteps)
        prediction = self.denoiser(noisy_trajectory, timesteps, condition)
        target = target_noise if self.config.prediction_type == "epsilon" else trajectory
        loss = F.mse_loss(prediction, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            mask = (~batch["action_is_pad"]).unsqueeze(-1)
            return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1)
        return loss.mean()


class _UmiOfficialPolicyBase(_OfficialEmaMixin, DiffusionPolicy):
    def __init__(self, config: UmiOfficialDPConfig):
        require_package("diffusers", extra="umi-official-dp")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.reset()

    def get_optim_params(self):
        return self.diffusion.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        # Upstream keeps the averaged copy permanently in inference mode.
        if hasattr(self, "ema_diffusion"):
            self.ema_diffusion.eval()
        return self

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        has_history = self._queues is not None and len(self._queues[OBS_STATE]) > 0
        if has_history:
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues}
        else:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
            for key in (OBS_STATE, OBS_IMAGES, OBS_ENV_STATE):
                if key in batch:
                    batch[key] = batch[key].unsqueeze(1)
        return self._inference_model().generate_actions(batch, noise=noise)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = dict(batch)
        if batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
        for key in self.config.image_features:
            if batch[key].ndim == 4:
                batch[key] = batch[key].unsqueeze(1)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        model = self.diffusion if self.training or not self.config.use_ema else self.ema_diffusion
        return model.compute_loss(batch), None


class UmiOfficialDPPolicy(_UmiOfficialPolicyBase):
    config_class = UmiOfficialDPConfig
    name = "umi_official_dp"

    def __init__(self, config: UmiOfficialDPConfig, **kwargs):
        super().__init__(config)
        self.diffusion = UmiOfficialUnetDiffusionModel(config)
        self._initialize_ema()


class UmiOfficialTransformerDPPolicy(_UmiOfficialPolicyBase):
    config_class = UmiOfficialTransformerDPConfig
    name = "umi_official_transformer_dp"

    def __init__(self, config: UmiOfficialTransformerDPConfig, **kwargs):
        super().__init__(config)
        self.diffusion = UmiOfficialTransformerDiffusionModel(config)
        self._initialize_ema()

    def get_optim_params(self):
        backbone_parameters = []
        other_parameters = []
        for name, parameter in self.diffusion.named_parameters():
            if ".backbone." in name:
                backbone_parameters.append(parameter)
            else:
                other_parameters.append(parameter)
        return [
            {"params": other_parameters},
            {"params": backbone_parameters, "lr": self.config.optimizer_backbone_lr},
        ]
