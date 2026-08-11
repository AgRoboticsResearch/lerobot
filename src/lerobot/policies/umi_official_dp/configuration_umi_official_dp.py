from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("umi_official_dp")
@dataclass
class UmiOfficialDPConfig(DiffusionConfig):
    """Official UMI ViT + conditional U-Net recipe adapted to LeRobot UMI data.

    The upstream UMI task predicts 16 strided actions from two strided observation
    frames. This port deliberately uses the ablation's canonical 32-slot model
    horizon and 30-action output so all policies answer exactly the same queries.
    """

    n_obs_steps: int = 1
    horizon: int = 32
    chunk_size: int | None = 30
    n_action_steps: int = 30
    use_umi_relative_ee: bool = True
    umi_rot6d_identity_norm: bool = True
    drop_n_last_frames: int = 0

    # ``vision_backbone`` is retained only because DiffusionConfig validates it;
    # the candidate below uses ``vision_model_name`` through timm.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    vision_model_name: str = "vit_base_patch16_clip_224.openai"
    vision_pretrained: bool = True
    vision_frozen: bool = False
    image_size: tuple[int, int] = (224, 224)
    crop_ratio: float = 0.95
    random_rotation_degrees: float = 0.0
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.5
    color_jitter_hue: float = 0.08
    # Upstream U-Net explicitly applies ImageNet normalization after augmentation.
    apply_imagenet_normalization: bool = True
    input_is_imagenet_normalized: bool = True

    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 50
    num_inference_steps: int | None = 16
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    input_perturbation: float = 0.1
    do_mask_loss_for_padding: bool = True

    use_ema: bool = True
    ema_update_after_step: int = 0
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999

    optimizer_lr: float = 3e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 2000

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.vision_model_name.startswith("vit"):
            raise ValueError(
                "The UMI official candidates currently require a timm ViT model; "
                f"got {self.vision_model_name}."
            )
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError(f"image_size must contain two positive values, got {self.image_size}.")
        if not 0 <= self.random_rotation_degrees <= 180:
            raise ValueError("random_rotation_degrees must be in [0, 180].")
        if self.input_perturbation < 0:
            raise ValueError("input_perturbation must be non-negative.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )


@PreTrainedConfig.register_subclass("umi_official_transformer_dp")
@dataclass
class UmiOfficialTransformerDPConfig(UmiOfficialDPConfig):
    """Official UMI ViT-token + transformer action denoiser recipe."""

    transformer_num_layers: int = 7
    transformer_num_heads: int = 8
    transformer_dim: int = 768
    transformer_dropout: float = 0.1
    optimizer_backbone_lr: float = 3e-5
    # The released transformer encoder does not apply ImageNet normalization.
    apply_imagenet_normalization: bool = False
    random_rotation_degrees: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transformer_dim % self.transformer_num_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_num_heads.")
        if self.optimizer_backbone_lr <= 0:
            raise ValueError("optimizer_backbone_lr must be positive.")
