#!/usr/bin/env python3
"""
Open-loop trajectory playback on real SO101 robot.

This script loads pre-recorded trajectories from txt files and sends commands
to a real SO101 robot in open loop (no policy inference, no visual feedback).

Usage:
    python so101_deploy_real.py \
        --trajectory /path/to/sample_10217_gt.txt \
        --robot_port /dev/ttyACM0 \
        --warm_start
"""

import logging
import time
from pathlib import Path

import numpy as np
import yaml

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, TransitionKey
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so101_follower.relative_ee_processor import pose10d_to_mat
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.relative_ee_processor import (
    Relative10DAccumulatedToAbsoluteEE,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.robot_utils import precise_sleep

# Motor names for SO101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default RESET pose (starting position) - from deploy script
RESET_POSE_DEG = np.array([
    -8.00,    # shoulder_pan
    -62.73,   # shoulder_lift
    65.05,    # elbow_flex
    0.86,     # wrist_flex
    -2.55,    # wrist_roll
    88.91,    # gripper
])

# Fixed safe joint positions for error recovery
INITIAL_SAFE_JOINTS = np.array([
    -7.91,    # shoulder_pan
    -106.51,  # shoulder_lift
    87.91,    # elbow_flex
    70.74,    # wrist_flex
    -0.53,    # wrist_roll
    1.18,     # gripper
], dtype=np.float64)

FPS = 10  # Control loop frequency


def load_trajectory(file_path: str) -> np.ndarray:
    """Load trajectory from txt file. Returns (T, 10) array."""
    data = np.loadtxt(file_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def parse_cameras_config(cameras_str: str | None) -> dict:
    """Parse cameras configuration from YAML string."""
    if not cameras_str or cameras_str.strip() == "":
        return {}

    cameras_dict = yaml.safe_load(cameras_str)
    if not isinstance(cameras_dict, dict):
        raise ValueError(f"Cameras config should be a dict, got {type(cameras_dict)}")

    cameras: dict = {}
    for name, config in cameras_dict.items():
        if not isinstance(config, dict):
            raise ValueError(f"Camera '{name}' config should be a dict, got {type(config)}")

        camera_type = config.pop("type", None)
        if camera_type is None:
            raise ValueError(f"Camera '{name}' missing 'type' field")

        camera_config_class = CameraConfig.get_choice_class(camera_type)
        cameras[name] = camera_config_class(**config)

    return cameras


def main():
    init_logging()
    logger = logging.getLogger("so101_deploy_real")

    import argparse
    parser = argparse.ArgumentParser(
        description="Open-loop trajectory playback on real SO101 robot"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="Path to trajectory txt file (e.g., /path/to/sample_10217_gt.txt)",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="../urdf/Simulation/SO101/so101_new_calib.urdf",
        help="Path to SO101 URDF file for IK",
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO101 robot connection",
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default="so101",
        help="Robot ID for loading/saving calibration files",
    )
    parser.add_argument(
        "--reset_pose",
        type=float,
        nargs=6,
        default=list(RESET_POSE_DEG),
        help="Reset joint positions in degrees (6 values)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Control loop frequency (Hz)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the trajectory continuously",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="Move to reset pose before starting",
    )
    parser.add_argument(
        "--ee_bounds_min",
        type=float,
        nargs=3,
        default=[-0.5, -0.5, 0.0],
        help="EE position bounds minimum (x, y, z) in meters",
    )
    parser.add_argument(
        "--ee_bounds_max",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.4],
        help="EE position bounds maximum (x, y, z) in meters",
    )
    parser.add_argument(
        "--max_ee_step_m",
        type=float,
        default=0.05,
        help="Maximum EE step size in meters (safety)",
    )
    parser.add_argument(
        "--gripper_lower",
        type=float,
        default=0.0,
        help="Gripper lower bound in degrees",
    )
    parser.add_argument(
        "--gripper_upper",
        type=float,
        default=100.0,
        help="Gripper upper bound in degrees",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='Camera configuration in YAML format (optional)',
    )

    args = parser.parse_args()

    # ========================================================================
    # Load Trajectory
    # ========================================================================
    logger.info(f"Loading trajectory from: {args.trajectory}")
    trajectory = load_trajectory(args.trajectory)
    logger.info(f"  Trajectory shape: {trajectory.shape}")
    logger.info(f"  Number of steps: {trajectory.shape[0]}")

    # ========================================================================
    # Initialize Kinematics
    # ========================================================================
    logger.info("Initializing kinematics solver...")

    urdf_path = Path(args.urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found at {urdf_path}")

    # For RobotKinematics, we need the actual URDF file path
    if urdf_path.is_dir():
        urdf_file = urdf_path / "robot.urdf"
    else:
        urdf_file = urdf_path

    kinematics = RobotKinematics(
        urdf_path=str(urdf_file),
        target_frame_name="gripper_frame_link",
        joint_names=MOTOR_NAMES,
    )
    logger.info(f"URDF loaded: {urdf_file}")

    # ========================================================================
    # Compute Base EE Pose from Reset Pose
    # ========================================================================
    logger.info("Computing base EE pose from reset pose...")
    logger.info(f"  Reset pose (deg): {RESET_POSE_DEG}")

    base_ee_T = kinematics.forward_kinematics(RESET_POSE_DEG)
    logger.info(f"  Base EE position: {base_ee_T[:3, 3]}")

    # ========================================================================
    # Build Processor Pipeline
    # ========================================================================
    logger.info("Building processor pipeline...")

    ee_to_joints_pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            Relative10DAccumulatedToAbsoluteEE(
                gripper_lower_deg=args.gripper_lower,
                gripper_upper_deg=args.gripper_upper,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": args.ee_bounds_min, "max": args.ee_bounds_max},
                max_ee_step_m=args.max_ee_step_m,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=MOTOR_NAMES,
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    logger.info("Pipeline built")

    # ========================================================================
    # Connect to Real Robot
    # ========================================================================
    logger.info("Connecting to SO101 robot...")

    cameras = parse_cameras_config(args.cameras)
    if cameras:
        logger.info(f"Configured cameras: {list(cameras.keys())}")

    robot_config = SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        use_degrees=True,
        cameras=cameras,
    )

    robot = SO101Follower(robot_config)
    robot.connect(calibrate=True)
    logger.info("Robot connected")

    # Move to reset pose if requested
    if args.warm_start:
        logger.info(f"Moving to reset pose: {args.reset_pose}")
        reset_pose = np.array(args.reset_pose, dtype=np.float64)
        reset_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, reset_pose)}
        robot.send_action(reset_action)
        time.sleep(2.0)

    # Start from reset pose for trajectory playback
    current_joints = RESET_POSE_DEG.copy()
    logger.info(f"Using reset pose as starting position: {current_joints}")

    # Reset processors
    ee_to_joints_pipeline.reset()

    # ========================================================================
    # Main Control Loop
    # ========================================================================
    logger.info("Starting control loop...")
    logger.info(f"Control frequency: {args.fps} Hz")
    logger.info(f"Total steps in trajectory: {trajectory.shape[0]}")
    logger.info(f"Loop: {args.loop}")
    logger.info("\nPress Ctrl+C to stop\n")

    traj_idx = 0
    step_count = 0

    try:
        while True:
            t0 = time.perf_counter()

            # Check if we've reached the end
            if traj_idx >= trajectory.shape[0]:
                if args.loop:
                    traj_idx = 0
                    # Reset robot to reset pose
                    reset_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, RESET_POSE_DEG)}
                    robot.send_action(reset_action)
                    ee_to_joints_pipeline.reset()
                    logger.info("Restarting trajectory loop...")
                    time.sleep(0.5)
                    continue
                else:
                    logger.info(f"Completed {trajectory.shape[0]} steps. Stopping.")
                    break

            # -------------------------------------------------------------------
            # Get relative action from trajectory
            # -------------------------------------------------------------------
            rel_action_10d = trajectory[traj_idx]

            # -------------------------------------------------------------------
            # Run through the pipeline
            # -------------------------------------------------------------------
            action_dict: RobotAction = {"rel_pose": rel_action_10d.copy()}
            robot_obs: RobotObservation = {
                f"{name}.pos": current_joints[i] for i, name in enumerate(MOTOR_NAMES)
            }

            transition = robot_action_observation_to_transition((action_dict, robot_obs))
            transition[TransitionKey.COMPLEMENTARY_DATA] = {
                "chunk_base_pose": base_ee_T.copy()
            }

            # Run pipeline: relative -> absolute -> bounds -> IK -> joints
            processed_transition = ee_to_joints_pipeline._forward(transition)
            joints_action = transition_to_robot_action(processed_transition)

            # -------------------------------------------------------------------
            # Send action to robot
            # -------------------------------------------------------------------
            robot.send_action(joints_action)

            # Update current joints (open loop - no feedback)
            for i, name in enumerate(MOTOR_NAMES):
                current_joints[i] = float(joints_action[f"{name}.pos"])

            # -------------------------------------------------------------------
            # Log progress
            # -------------------------------------------------------------------
            if step_count % 10 == 0:
                rel_T = pose10d_to_mat(rel_action_10d[:9])
                target_ee_pose = base_ee_T @ rel_T
                pos = target_ee_pose[:3, 3]
                logger.info(f"Step {step_count}: EE pos {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")

            # Advance to next step
            traj_idx += 1
            step_count += 1

            # -------------------------------------------------------------------
            # Maintain timing
            # -------------------------------------------------------------------
            elapsed = time.perf_counter() - t0
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                precise_sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Error during control loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Return to safe pose
        logger.info("Returning to safe initial pose...")
        safe_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, INITIAL_SAFE_JOINTS)}
        robot.send_action(safe_action)
        time.sleep(1.0)

        # Disconnect robot
        logger.info("Disconnecting robot...")
        robot.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    main()
