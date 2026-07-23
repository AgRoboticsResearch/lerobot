#!/usr/bin/env python
"""Compatibility launcher for π0.5 LoRA UMI relative-EE training.

Dataset preparation now lives in the standard LeRobot dataset factory. New
commands may call ``lerobot-train`` directly; this path remains for existing
runbooks and shell launchers.
"""

import lerobot.scripts.lerobot_train as train_module
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@parser.wrap()
def train_pi05_lora(cfg: TrainPipelineConfig) -> None:
    train_module.train(cfg)


if __name__ == "__main__":
    train_pi05_lora()
