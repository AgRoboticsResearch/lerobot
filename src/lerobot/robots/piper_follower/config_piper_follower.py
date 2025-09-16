#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PIPERFollowerConfig(RobotConfig):
    # Piper uses CAN; no serial port is required.
    # Keep an optional field for compatibility; it will be ignored if provided.
    port: str | None = None

    # CAN interface name to open (e.g. "can0" or custom name)
    can_name: str = "can_follower"

    # Piper SDK constructor options
    can_auto_init: bool = True
    start_sdk_gripper_limit: bool = True

    # ConnectPort behavior
    connect_can_init: bool = False
    connect_piper_init: bool = True
    connect_start_thread: bool = True

    # Mode control defaults (see Piper SDK ModeCtrl)
    ctrl_mode: int = 0x01  # CAN command control mode
    move_mode: int = 0x01  # MOVE J (Joint)
    move_spd_rate_ctrl: int = 50  # 0-100
    is_mit_mode: int = 0x00  # Position-velocity mode

    # Joint configuration defaults
    acc_param_is_effective: bool = True
    max_joint_acc: int = 300  # 0..500 (0.01 rad/s^2)

    # Crash protection per joint (0..8). 0 disables detection.
    crash_protection_levels: tuple[int, int, int, int, int, int] = (3, 3, 3, 3, 3, 3)

    # Enable/clear flags on connect
    enable_all_on_connect: bool = True
    clear_errors_on_connect: bool = True

    disable_torque_on_disconnect: bool = True

    # Optional limit on relative joint movement for safety
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


