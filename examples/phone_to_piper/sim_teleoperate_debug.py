#!/usr/bin/env python

"""
Debug version of sim_teleoperate.py that uses the REAL pipeline steps from robot_kinematic_processor.
Allows disabling individual steps via --disable-step to isolate the source of "wild movements".

Usage:
    # All steps enabled (same as sim_teleoperate.py)
    python sim_teleoperate_debug.py

    # Disable specific steps:
    python sim_teleoperate_debug.py --disable-step remap
    python sim_teleoperate_debug.py --disable-step reference
    python sim_teleoperate_debug.py --disable-step bounds
    python sim_teleoperate_debug.py --disable-step gripper
    python sim_teleoperate_debug.py --disable-step ik
"""

import time
import argparse
import numpy as np
import copy
from dataclasses import dataclass, field

from lerobot.model.kinematics_bac import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    RobotActionProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
# Piper imports
from lerobot_robot_piper.config_piper import PiperConfig
from lerobot_robot_piper.robot_kinematic_processor import (
    PiperEEBoundsAndSafety,
    PiperEEReferenceAndDelta,
    PiperGripperVelocityToJoint,
    PiperInverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation
from lerobot.processor import TransitionKey

# Placo visualization
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz

FPS = 30


# ============================================================================
# Custom steps that can be toggled for debugging
# ============================================================================

@ProcessorStepRegistry.register("debug_remap_axis_step")
@dataclass
class DebugRemapAxisStep(RobotActionProcessorStep):
    """
    Remaps phone axes to Piper axes (second inversion layer).
    """
    enabled: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        if not self.enabled:
            print("  [SKIP] RemapAxisStep disabled")
            return action
        
        if "target_x" in action:
            action["target_x"] = -action["target_x"]
        if "target_y" in action:
            action["target_y"] = -action["target_y"]
        if "target_wy" in action:
            action["target_wy"] = -action["target_wy"]
        if "target_wz" in action:
            action["target_wz"] = -action["target_wz"]
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("debug_piper_ee_reference_and_delta")
@dataclass
class DebugPiperEEReferenceAndDelta(PiperEEReferenceAndDelta):
    """
    Corrected version with debug output. RobotKinematics expects DEGREES.
    """
    debug_enabled: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        if not self.debug_enabled:
            print("  [SKIP] EEReferenceAndDelta disabled - passing through")
            # Pass through with minimal transformation
            action["ee.x"] = action.pop("target_x", 0.0)
            action["ee.y"] = action.pop("target_y", 0.0)
            action["ee.z"] = action.pop("target_z", 0.0)
            action["ee.wx"] = action.pop("target_wx", 0.0)
            action["ee.wy"] = action.pop("target_wy", 0.0)
            action["ee.wz"] = action.pop("target_wz", 0.0)
            action["ee.gripper_vel"] = action.pop("gripper_vel", 0.0)
            action.pop("enabled", None)
            return action

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            q_raw = np.array(
                [
                    float(v)
                    for k, v in observation.items()
                    if isinstance(k, str)
                    and k.endswith(".pos")
                    and k.removesuffix(".pos") in self.motor_names
                ],
                dtype=float,
            )

        if q_raw is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        # RobotKinematics.forward_kinematics expects DEGREES
        t_curr = self.kinematics.forward_kinematics(q_raw)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))

        # Debug print
        print(f"  [REF] enabled={enabled}, delta=({tx:.4f},{ty:.4f},{tz:.4f}), rot=({wx:.4f},{wy:.4f},{wz:.4f})")

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = t_curr.copy()
                    print(f"  [REF] Latched reference pose")
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            r_abs = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = r_abs @ ref[:3, :3]
            desired[:3, 3] = ref[:3, 3] + delta_p

            self._command_when_disabled = desired.copy()
        else:
            if self._command_when_disabled is None:
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        self._prev_enabled = enabled

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_vel"] = gripper_vel

        print(f"  [REF] target_pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f})")
        return action


@ProcessorStepRegistry.register("debug_piper_ee_bounds_and_safety")
@dataclass
class DebugPiperEEBoundsAndSafety(PiperEEBoundsAndSafety):
    """
    EEBoundsAndSafety with debug output.
    """
    debug_enabled: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        if not self.debug_enabled:
            print("  [SKIP] EEBoundsAndSafety disabled")
            return action
        
        # Call parent and add debug
        before_x = action.get("ee.x", 0)
        before_y = action.get("ee.y", 0)
        before_z = action.get("ee.z", 0)
        
        result = super().action(action)
        
        after_x = result.get("ee.x", 0)
        after_y = result.get("ee.y", 0)
        after_z = result.get("ee.z", 0)
        
        if (before_x != after_x) or (before_y != after_y) or (before_z != after_z):
            print(f"  [BOUNDS] Clamped: ({before_x:.4f},{before_y:.4f},{before_z:.4f}) -> ({after_x:.4f},{after_y:.4f},{after_z:.4f})")
        
        return result


@ProcessorStepRegistry.register("debug_piper_gripper_velocity_to_joint")
@dataclass
class DebugPiperGripperVelocityToJoint(PiperGripperVelocityToJoint):
    """
    GripperVelocityToJoint with debug output.
    """
    debug_enabled: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        if not self.debug_enabled:
            print("  [SKIP] GripperVelocityToJoint disabled")
            # Just pass through with a default gripper pos
            action["ee.gripper_pos"] = action.pop("ee.gripper_vel", 0.0)
            return action
        
        return super().action(action)


@ProcessorStepRegistry.register("debug_piper_inverse_kinematics_ee_to_joints")
@dataclass
class DebugPiperInverseKinematicsEEToJoints(PiperInverseKinematicsEEToJoints):
    """
    IK step with debug output. RobotKinematics expects DEGREES.
    """
    debug_enabled: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        if not self.debug_enabled:
            print("  [SKIP] InverseKinematics disabled - keeping previous joints")
            # Remove ee.* keys without computing IK
            action.pop("ee.x", None)
            action.pop("ee.y", None)
            action.pop("ee.z", None)
            action.pop("ee.wx", None)
            action.pop("ee.wy", None)
            action.pop("ee.wz", None)
            action.pop("ee.gripper_pos", None)
            return action

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError("Missing required end-effector pose components")

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        if self.initial_guess_current_joints:
            self.q_curr = q_raw
        else:
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute IK (input/output in DEGREES)
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        print(f"  [IK] target_joints={np.round(q_target, 2)}")

        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-os", type=str, default="ios", choices=["ios", "android"])
    parser.add_argument(
        "--disable-step",
        type=str,
        default=None,
        choices=["remap", "reference", "bounds", "gripper", "ik"],
        help="Disable a specific processing step for debugging"
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug info every frame")
    args = parser.parse_args()

    # Robot Config
    robot_config = PiperConfig(
        can_interface="can0",
        include_gripper=True,
        use_degrees=True,
        cameras={},
        joint_names=[f"joint{i+1}" for i in range(6)]
    )
    
    phone_os_enum = PhoneOS.IOS if args.phone_os == "ios" else PhoneOS.ANDROID
    teleop_config = PhoneConfig(phone_os=phone_os_enum)

    # Initialize Teleoperator
    teleop_device = Phone(teleop_config)
    teleop_device.connect()
    
    # Initialize Robot Kinematics
    urdf_path = "piper_description/urdf/piper_description.urdf"
    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_base",
        joint_names=robot_config.joint_names,
    )

    # Initialize Placo Viewer
    viz = robot_viz(kinematics_solver.robot)

    # Set initial state
    initial_joints = {
        'joint1': 0.0,
        'joint2': 1.0,
        'joint3': -0.5,
        'joint4': 0.0,
        'joint5': -0.7,
        'joint6': 0.0
    }
    
    for name, val in initial_joints.items():
        kinematics_solver.robot.set_joint(name, val)
    kinematics_solver.robot.update_kinematics()

    # Build pipeline with debug steps
    # Determine which steps are enabled
    remap_enabled = args.disable_step != "remap"
    reference_enabled = args.disable_step != "reference"
    bounds_enabled = args.disable_step != "bounds"
    gripper_enabled = args.disable_step != "gripper"
    ik_enabled = args.disable_step != "ik"

    print("=" * 60)
    print("DEBUG SIMULATION - Pipeline Steps:")
    print(f"  MapPhoneActionToRobotAction: ENABLED (always)")
    print(f"  RemapAxisStep:               {'ENABLED' if remap_enabled else 'DISABLED'}")
    print(f"  EEReferenceAndDelta:         {'ENABLED' if reference_enabled else 'DISABLED'}")
    print(f"  EEBoundsAndSafety:           {'ENABLED' if bounds_enabled else 'DISABLED'}")
    print(f"  GripperVelocityToJoint:      {'ENABLED' if gripper_enabled else 'DISABLED'}")
    print(f"  InverseKinematicsEEToJoints: {'ENABLED' if ik_enabled else 'DISABLED'}")
    print("=" * 60)

    pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=phone_os_enum),
            DebugRemapAxisStep(enabled=remap_enabled),
            DebugPiperEEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
                motor_names=robot_config.joint_names,
                use_latched_reference=True,
                debug_enabled=reference_enabled,
            ),
            DebugPiperEEBoundsAndSafety(
                end_effector_bounds={"min": [-0.6, -0.6, 0.0], "max": [0.6, 0.6, 0.8]},
                max_ee_step_m=0.10,
                debug_enabled=bounds_enabled,
            ),
            DebugPiperGripperVelocityToJoint(
                speed_factor=20.0,
                debug_enabled=gripper_enabled,
            ),
            DebugPiperInverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=robot_config.joint_names,
                initial_guess_current_joints=True,
                debug_enabled=ik_enabled,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    print("Press B1 on phone to Enable and Latch origin.")
    frame_count = 0

    try:
        while True:
            t0 = time.perf_counter()
            frame_count += 1

            # Get Teleop Action
            phone_obs = teleop_device.get_action()
            if not phone_obs:
                time.sleep(0.01)
                continue

            # Construct Robot Observation (joint positions in DEGREES)
            robot_obs = {}
            for name in robot_config.joint_names:
                rad_val = kinematics_solver.robot.get_joint(name)
                deg_val = np.rad2deg(rad_val)
                robot_obs[f"{name}.pos"] = deg_val
            robot_obs["gripper.pos"] = 0.0

            # Run Pipeline
            try:
                phone_obs_copy = copy.deepcopy(phone_obs)
                
                if args.verbose or frame_count % 30 == 0:
                    print(f"\n--- Frame {frame_count} ---")
                
                joint_action = pipeline((phone_obs_copy, robot_obs))
                
                # Update Simulation State
                if ik_enabled:
                    for name in robot_config.joint_names:
                        key = f"{name}.pos"
                        if key in joint_action:
                            deg_val = joint_action[key]
                            rad_val = np.deg2rad(deg_val)
                            kinematics_solver.robot.set_joint(name, rad_val)
                    kinematics_solver.robot.update_kinematics()
                
            except Exception as e:
                print(f"Pipeline Error: {e}")
                import traceback
                traceback.print_exc()

            # Visualize
            viz.display(kinematics_solver.robot.state.q)
            robot_frame_viz(kinematics_solver.robot, "gripper_base")
            
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        print("\nStopping sim...")
    finally:
        teleop_device.disconnect()


if __name__ == "__main__":
    main()
