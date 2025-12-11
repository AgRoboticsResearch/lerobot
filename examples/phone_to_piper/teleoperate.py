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

"""
Phone teleoperation for REAL Piper robot arm.

This script connects to a phone teleoperator and drives a real Piper robot
using the kinematic processing pipeline with multiple safety mechanisms:
- End-effector workspace bounds
- End-effector position/rotation step limits
- Joint position limits
- Joint step limits per frame

Usage:
    # Basic usage
    python teleoperate.py --phone-os ios --can-port can0

    # With custom safety parameters
    python teleoperate.py --max-ee-step 0.03 --max-joint-step 5.0

    # Dry run (print commands without sending to robot)
    python teleoperate.py --dry-run

Safety Notes:
    - Ensure CAN interface is active: bash piper_sdk/piper_sdk/can_activate.sh can0 1000000
    - Keep hand on emergency stop during operation
    - Robot uses current position as initial reference (no sudden movement at start)
    - Press Ctrl+C to safely stop
"""

import time
import argparse
import logging
import copy

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
# Piper imports
from lerobot_robot_piper.config_piper import PiperConfig
from lerobot_robot_piper.piper import Piper
from lerobot_robot_piper.robot_kinematic_processor import (
    PiperEEBoundsAndSafety,
    PiperEEReferenceAndDelta,
    PiperGripperVelocityToJoint,
    PiperInverseKinematicsEEToJoints,
    PiperJointSafetyClamp,
)
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep

# Optional: rerun visualization
try:
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

# Logger will be configured in main() based on --debug flag
logger = logging.getLogger(__name__)

FPS = 30


def main():
    parser = argparse.ArgumentParser(description="Phone teleoperation for real Piper robot arm")
    parser.add_argument("--phone-os", type=str, default="ios", choices=["ios", "android"], 
                        help="Phone OS (ios or android)")
    parser.add_argument("--can-port", type=str, default="can0", 
                        help="CAN interface port for Piper")
    
    # Safety parameters
    parser.add_argument("--max-ee-step", type=float, default=0.05, 
                        help="Maximum EE position step per frame in meters (default: 0.05)")
    parser.add_argument("--max-ee-rot-step", type=float, default=0.3, 
                        help="Maximum EE rotation step per frame in radians (default: 0.3, ~17°)")
    parser.add_argument("--max-joint-step", type=float, default=10.0, 
                        help="Maximum joint step per frame in degrees (default: 10.0)")
    parser.add_argument("--ee-step-scale", type=float, default=1.0, 
                        help="Scale factor for phone-to-EE movement (default: 1.0)")
    
    # Debug options
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print commands without sending to robot")
    parser.add_argument("--no-rerun", action="store_true", 
                        help="Disable rerun visualization")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print debug info every frame")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG level logging for gripper/pipeline debugging")
    
    args = parser.parse_args()

    # Configure logging based on --debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Also set debug level for lerobot_robot_piper modules
    if args.debug:
        logging.getLogger("lerobot_robot_piper").setLevel(logging.DEBUG)

    # Initialize the robot configuration
    # NOTE: joint_names must match SDK format (joint_1, joint_2, ...) for real robot
    # But URDF uses joint1, joint2, ... so we need separate names for kinematics
    sdk_joint_names = [f"joint_{i+1}" for i in range(6)]  # SDK format: joint_1, joint_2, ...
    urdf_joint_names = [f"joint{i+1}" for i in range(6)]  # URDF format: joint1, joint2, ...
    
    robot_config = PiperConfig(
        can_interface=args.can_port,
        include_gripper=True,
        use_degrees=True,
        cameras={},  # Disable default camera for teleoperation
        joint_names=sdk_joint_names,  # Use SDK format for real robot communication
    )
    
    phone_os_enum = PhoneOS.IOS if args.phone_os == "ios" else PhoneOS.ANDROID
    teleop_config = PhoneConfig(phone_os=phone_os_enum)

    # Initialize the robot and teleoperator
    # NOTE: Always create robot even in dry-run mode so we can READ observations
    # Dry-run only skips SENDING actions
    robot = Piper(robot_config)
    teleop_device = Phone(teleop_config)

    # Robot Kinematics for Piper
    # NOTE: URDF uses joint1, joint2, ... (without underscore)
    kinematics_solver = RobotKinematics(
        urdf_path="piper_description/urdf/piper_description.urdf",
        target_frame_name="gripper_base",
        joint_names=urdf_joint_names,  # URDF format for kinematics
    )

    # Build pipeline with safety steps
    # Pipeline order:
    # 1. MapPhoneActionToRobotAction: Phone coords -> Robot coords
    # 2. PiperEEReferenceAndDelta: Delta -> Absolute EE target
    # 3. PiperEEBoundsAndSafety: Workspace bounds + step limits
    # 4. PiperGripperVelocityToJoint: Gripper velocity -> position
    # 5. PiperInverseKinematicsEEToJoints: EE pose -> Joint angles
    # 6. PiperJointSafetyClamp: Joint limits + step limits (final safety layer)
    #
    # IMPORTANT: motor_names in pipeline must match SDK format (joint_1, joint_2, ...)
    # because observation/action dicts use SDK format keys
    
    pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            PiperEEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={
                    "x": args.ee_step_scale, 
                    "y": args.ee_step_scale, 
                    "z": args.ee_step_scale
                },
                motor_names=sdk_joint_names,  # SDK format for observation keys
                use_latched_reference=True,
            ),
            PiperEEBoundsAndSafety(
                end_effector_bounds={"min": [-0.6, -0.6, 0.0], "max": [0.6, 0.6, 0.8]},
                max_ee_step_m=args.max_ee_step,
                max_ee_rot_step_rad=args.max_ee_rot_step,
            ),
            PiperGripperVelocityToJoint(
                speed_factor=20.0,
            ),
            PiperInverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=sdk_joint_names,  # SDK format for action keys
                initial_guess_current_joints=True,
            ),
            PiperJointSafetyClamp(
                motor_names=sdk_joint_names,  # SDK format
                max_joint_step_deg=args.max_joint_step,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to devices
    # NOTE: Connect to robot even in dry-run mode to read observations
    logger.info("Connecting to Piper robot...")
    robot.connect()
    if args.dry_run:
        logger.info("  (Dry-run mode: will read observations but NOT send actions)")
    teleop_device.connect()

    # Init rerun viewer
    if HAS_RERUN and not args.no_rerun:
        init_rerun(session_name="phone_piper_teleop")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
    if not teleop_device.is_connected:
        raise ValueError("Phone teleoperator is not connected!")

    # Print safety parameters
    logger.info("=" * 60)
    logger.info("Phone Teleoperation for Piper Robot")
    logger.info("=" * 60)
    logger.info(f"  CAN Port: {args.can_port}")
    logger.info(f"  Phone OS: {args.phone_os}")
    logger.info(f"  Dry Run: {args.dry_run}")
    logger.info("Safety Parameters:")
    logger.info(f"  Max EE Step: {args.max_ee_step:.3f} m/frame")
    logger.info(f"  Max EE Rot Step: {args.max_ee_rot_step:.2f} rad/frame ({args.max_ee_rot_step * 180 / 3.14159:.1f}°)")
    logger.info(f"  Max Joint Step: {args.max_joint_step:.1f} °/frame")
    logger.info(f"  EE Step Scale: {args.ee_step_scale}")
    logger.info("=" * 60)
    logger.info("Controls:")
    logger.info("  B1 button: Enable teleoperation & latch origin")
    logger.info("  A3 slider: Gripper control (up=open, down=close)")
    logger.info("  Move phone: Control arm position/orientation")
    logger.info("Press Ctrl+C to safely stop.")
    logger.info("=" * 60)

    frame_count = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10

    try:
        while True:
            t0 = time.perf_counter()
            frame_count += 1

            # Get robot observation (always read from real robot, even in dry-run)
            try:
                robot_obs = robot.get_observation()
            except Exception as e:
                logger.error(f"Failed to get robot observation: {e}")
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error("Too many consecutive errors, stopping...")
                    break
                continue

            # Get teleop action
            phone_obs = teleop_device.get_action()
            if not phone_obs:
                time.sleep(0.01)
                continue

            # Reset error counter on successful read
            consecutive_errors = 0

            # Process phone action through pipeline
            try:
                phone_obs_copy = copy.deepcopy(phone_obs)
                
                # Debug: Print phone input before pipeline
                # NOTE: phone_obs from get_action() has "phone.enabled", not "enabled"
                # The "enabled" key is created by MapPhoneActionToRobotAction in the pipeline
                if args.verbose or args.dry_run:
                    enabled = phone_obs.get("phone.enabled", False)
                    # Get position/rotation from raw phone output (before pipeline transform)
                    pos = phone_obs.get("phone.pos", [0, 0, 0])
                    rot = phone_obs.get("phone.rot", None)
                    rotvec = rot.as_rotvec() if rot is not None else [0, 0, 0]
                    raw_inputs = phone_obs.get("phone.raw_inputs", {})
                    gripper = float(raw_inputs.get("a3", 0.0))  # iOS gripper
                    
                    # Always print gripper if it's non-zero (slider moved)
                    if abs(gripper) > 0.001:
                        print(f"[GRIPPER] a3 slider value: {gripper:.4f}")
                    
                    if enabled or frame_count % 30 == 0:  # Print when enabled or every 30 frames
                        print(f"\n{'='*70}")
                        print(f"Frame {frame_count} | Enabled: {enabled}")
                        print(f"  (raw_inputs b1={raw_inputs.get('b1', 'N/A')}, a3={raw_inputs.get('a3', 'N/A')})")
                        # Print all raw_inputs to see what we're getting
                        print(f"  All raw_inputs: {list(raw_inputs.keys())}")
                        print(f"{'='*70}")
                        print(f"[PHONE INPUT] (raw, before pipeline transform)")
                        print(f"  Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
                        print(f"  Rotation: rx={rotvec[0]:.4f}, ry={rotvec[1]:.4f}, rz={rotvec[2]:.4f}")
                        print(f"  Gripper vel (a3): {gripper:.3f}")
                        print(f"[ROBOT OBS] (degrees)")
                        obs_str = ", ".join([f"j{i+1}={robot_obs.get(f'joint_{i+1}.pos', 0):.1f}" for i in range(6)])
                        print(f"  Joints: {obs_str}")
                        print(f"  Gripper pos: {robot_obs.get('gripper.pos', 0):.2f} mm")
                
                joint_action = pipeline((phone_obs_copy, robot_obs))
                
                # Debug: Print pipeline output
                if args.verbose or args.dry_run:
                    # Use phone.enabled since enabled key only exists after pipeline
                    enabled = phone_obs.get("phone.enabled", False)
                    if enabled or frame_count % 30 == 0:
                        print(f"[PIPELINE OUTPUT] (degrees)")
                        action_str = ", ".join([f"j{i+1}={joint_action.get(f'joint_{i+1}.pos', 0):.1f}" for i in range(6)])
                        print(f"  Target joints: {action_str}")
                        gripper_target = joint_action.get('gripper.pos', 0)
                        gripper_current = robot_obs.get('gripper.pos', 0)
                        print(f"  Gripper target: {gripper_target:.1f} mm (delta: {gripper_target - gripper_current:+.1f})")
                        
                        # Calculate and show delta from current
                        print(f"[DELTA] (target - current)")
                        delta_str = ", ".join([
                            f"j{i+1}={joint_action.get(f'joint_{i+1}.pos', 0) - robot_obs.get(f'joint_{i+1}.pos', 0):.1f}" 
                            for i in range(6)
                        ])
                        print(f"  Joint deltas: {delta_str}")

                # Send action to robot
                if not args.dry_run:
                    robot.send_action(joint_action)

                # Visualize
                if HAS_RERUN and not args.no_rerun:
                    log_rerun_data(observation=phone_obs, action=joint_action)

            except Exception as e:
                logger.error(f"Pipeline error at frame {frame_count}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error("Too many consecutive errors, stopping...")
                    break
                # Don't send action on error - robot stays at last position
                continue

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        logger.info("\nCtrl+C received, stopping safely...")
    finally:
        logger.info("Disconnecting...")
        try:
            robot.disconnect()
            logger.info("Robot disconnected safely.")
        except Exception as e:
            logger.error(f"Error during robot disconnect: {e}")
        
        try:
            teleop_device.disconnect()
        except Exception:
            pass
        
        logger.info("Teleoperation ended.")


if __name__ == "__main__":
    main()
