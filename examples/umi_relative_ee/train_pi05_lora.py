#!/usr/bin/env python
"""Compatibility launcher for π0.5 LoRA UMI relative-EE training.

The example adapter loads generated Reach AnyPoint chunks before passing them
to the standard trainer. Ordinary UMI datasets use the standard dataset factory.
"""

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

if __package__:
    from .reach_anypoint_dataset import train_with_reach_anypoint
else:
    from reach_anypoint_dataset import train_with_reach_anypoint


@parser.wrap()
def train_pi05_lora(cfg: TrainPipelineConfig) -> None:
    train_with_reach_anypoint(cfg)


if __name__ == "__main__":
    train_pi05_lora()
