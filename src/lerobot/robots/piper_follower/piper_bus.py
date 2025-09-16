#!/usr/bin/env python

# Minimal Piper CAN bus wrapper using Piper SDK

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass


@dataclass
class PIPERMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]
    # SDK options
    can_auto_init: bool = True
    start_sdk_gripper_limit: bool = True
    # ConnectPort flags
    connect_can_init: bool = False
    connect_piper_init: bool = True
    connect_start_thread: bool = True
    # Mode control
    ctrl_mode: int = 0x01
    move_mode: int = 0x01
    move_spd_rate_ctrl: int = 50
    is_mit_mode: int = 0x00
    # Joint config
    acc_param_is_effective: bool = True
    max_joint_acc: int = 300
    # Crash protection
    crash_protection_levels: tuple[int, int, int, int, int, int] = (3, 3, 3, 3, 3, 3)
    # Connect actions
    enable_all_on_connect: bool = True
    clear_errors_on_connect: bool = True


class PIPERMotorsBus:
    """
    Lightweight wrapper around the Piper SDK to match the subset of API used by LeRobot robots.

    The implementation intentionally mirrors only what `PIPERFollower`/`PIPERLeader` require:
    - connect/disable/enable
    - read() -> dict[motor_name, float]
    - write(target_joints: list[float])
    - is_connected flag
    - torque_disabled() context (no-op for Piper)
    """

    def __init__(self, config: PIPERMotorsBusConfig):
        # Lazy import so environments without the SDK can still import the package.
        from piper_sdk import C_PiperInterface_V2  # type: ignore

        self.piper = C_PiperInterface_V2(
            config.can_name,
            can_auto_init=config.can_auto_init,
            start_sdk_gripper_limit=config.start_sdk_gripper_limit,
        )
        # Start reading thread and optionally perform CAN/Piper init
        self.piper.ConnectPort(
            can_init=config.connect_can_init,
            piper_init=config.connect_piper_init,
            start_thread=config.connect_start_thread,
        )
        self.motors: dict[str, tuple[int, str]] = config.motors
        self.config = config
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        # 1000 * 180 / pi, converts radians to 0.001 degrees units used by Piper
        self.factor = 57295.7795
        self._is_enable = False

    # Compatibility attributes used by LeRobot
    @property
    def is_connected(self) -> bool:
        return self._is_enable

    @property
    def is_calibrated(self) -> bool:
        # Piper does not require LeRobot calibration; treat as calibrated by default
        return True

    def connect(self) -> None:
        # Ensure mode and enable sequence is applied robustly
        # Check reader thread health
        if not self.piper.isOk():
            # Reconnect port with CAN init if needed (e.g., after USB power loss)
            self.piper.ConnectPort(
                can_init=self.config.connect_can_init,
                piper_init=self.config.connect_piper_init,
                start_thread=self.config.connect_start_thread,
            )
        # Mode control
        self.piper.ModeCtrl(
            ctrl_mode=self.config.ctrl_mode,
            move_mode=self.config.move_mode,
            move_spd_rate_ctrl=self.config.move_spd_rate_ctrl,
            is_mit_mode=self.config.is_mit_mode,
        )
        # Joint config
        self.piper.JointConfig(
            joint_num=7,
            set_zero=0x00,
            acc_param_is_effective=0xAE if self.config.acc_param_is_effective else 0x00,
            max_joint_acc=self.config.max_joint_acc,
            clear_err=0xAE if self.config.clear_errors_on_connect else 0x00,
        )
        # Crash protection
        l = self.config.crash_protection_levels
        self.piper.CrashProtectionConfig(l[0], l[1], l[2], l[3], l[4], l[5])

        # Enable arm
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        if self.config.enable_all_on_connect:
            self.piper.EnableArm(7, 0x02)
        self._is_enable = True

    def disconnect(self, disable_torque: bool | None = None) -> None:
        # Optionally move to a safe position before disabling
        if disable_torque:
            try:
                self.write(self.safe_disable_position)
            except Exception:
                pass
        try:
            self.piper.DisablePiper()
        finally:
            self._is_enable = False

    @contextlib.contextmanager
    def torque_disabled(self):
        # Piper SDK handles enable/disable; expose a no-op context for compatibility
        yield

    # Read present joints in radians and gripper in meters
    def read(self) -> dict[str, float]:
        # Auto-recover on reader thread failure
        if not self.piper.isOk():
            self.connect()
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        # Convert from 0.001 degrees to radians for joints
        inv_factor = 1.0 / self.factor

        return {
            "joint_1": joint_state.joint_1 * inv_factor,
            "joint_2": joint_state.joint_2 * inv_factor,
            "joint_3": joint_state.joint_3 * inv_factor,
            "joint_4": joint_state.joint_4 * inv_factor,
            "joint_5": joint_state.joint_5 * inv_factor,
            "joint_6": joint_state.joint_6 * inv_factor,
            "gripper": gripper_state.grippers_angle / 1_000_000.0,
        }

    # Write goal joints: expect order aligned to self.motors keys
    def write(self, target_joint: list[float]) -> None:
        if not self._is_enable:
            self.connect()
        # Convert radians to 0.001 degrees
        j0 = round(target_joint[0] * self.factor)
        j1 = round(target_joint[1] * self.factor)
        j2 = round(target_joint[2] * self.factor)
        j3 = round(target_joint[3] * self.factor)
        j4 = round(target_joint[4] * self.factor)
        j5 = round(target_joint[5] * self.factor)
        gr = round(abs(target_joint[6]) * 1_000_000)

        # Enable motion and send commands
        # Ensure motion mode parameters
        self.piper.ModeCtrl(
            ctrl_mode=self.config.ctrl_mode,
            move_mode=self.config.move_mode,
            move_spd_rate_ctrl=self.config.move_spd_rate_ctrl,
            is_mit_mode=self.config.is_mit_mode,
        )
        self.piper.JointCtrl(j0, j1, j2, j3, j4, j5)
        # Enable and clear error on gripper if needed
        self.piper.GripperCtrl(gr, 1000, 0x03, 0)


