#!/usr/bin/env python3
"""
Simulate SO101 deployment using placo visualization with pre-recorded trajectories.

This script loads ground truth or predicted trajectories from txt files generated
by debug_relative_ee_inference.py and simulates the robot execution.

It uses the same processing pipeline as deploy_relative_ee_so101.py:
- Relative10DAccumulatedToAbsoluteEE
- EEBoundsAndSafety
- InverseKinematicsEEToJoints

Usage:
    python so101_deploy_sim.py --trajectory /path/to/sample_10217_gt.txt
    python so101_deploy_sim.py --trajectory /path/to/sample_10217_pred.txt
"""

import logging
import time
from pathlib import Path

import numpy as np
import placo
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, points_viz, robot_frame_viz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, TransitionKey
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower.relative_ee_processor import pose10d_to_mat
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.relative_ee_processor import (
    Relative10DAccumulatedToAbsoluteEE,
)

from lerobot.utils.utils import init_logging

# Motor names for SO101 (same as deploy script)
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default RESET pose (same as deploy script) - robot starts here
RESET_POSE_DEG = np.array([
    -5.54,    # shoulder_pan
    -114.59,  # shoulder_lift
    80.44,    # elbow_flex
    15.84,    # wrist_flex
    -5.19,    # wrist_roll
    35.13,    # gripper
])

# Fixed safe joint positions for error recovery (same as deploy script)
INITIAL_SAFE_JOINTS = np.array([
    -7.91,    # shoulder_pan
    -106.51,  # shoulder_lift
    87.91,    # elbow_flex
    70.74,    # wrist_flex
    -0.53,    # wrist_roll
    1.18,     # gripper
], dtype=np.float64)

FPS = 30  # Control loop frequency (Hz) - must match training fps

# Path to SO101 model directory
URDF_PATH = "../urdf/Simulation/SO101/so101_new_calib.urdf"


def load_trajectory(file_path: str) -> np.ndarray:
    """Load trajectory from txt file. Returns (T, 10) array."""
    data = np.loadtxt(file_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


class SimulatedSO101Robot:
    """Simulated SO101 robot using placo for visualization."""

    def __init__(self, urdf_path: str, motor_names: list[str]):
        self.urdf_path = urdf_path
        self.motor_names = motor_names

        # Setup placo robot
        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)

        self.joints_task = self.solver.add_joints_task()
        self.viz = robot_viz(self.robot)

        # Current joint state (in degrees, matching real robot convention)
        self.current_joints = INITIAL_SAFE_JOINTS.copy()

        # Initialize visualization
        self._update_robot_from_joints()

    def connect(self, calibrate: bool = False):
        """Simulated connection - just initialize."""
        print(f"Simulated robot connected")
        print(f"  URDF: {self.urdf_path}")
        print(f"  Motors: {self.motor_names}")

    def get_observation(self) -> dict:
        """
        Get current observation (simulated).

        Returns dict with keys like "shoulder_pan.pos", etc.
        Values are in degrees.
        """
        return {f"{name}.pos": self.current_joints[i] for i, name in enumerate(self.motor_names)}

    def send_action(self, action: dict):
        """
        Send action to robot (simulated).

        Args:
            action: Dict with keys like "shoulder_pan.pos", etc.
                    Values are in degrees.
        """
        # Update current joint state
        for i, name in enumerate(self.motor_names):
            if f"{name}.pos" in action:
                self.current_joints[i] = float(action[f"{name}.pos"])

        # Update visualization
        self._update_robot_from_joints()

    def _update_robot_from_joints(self):
        """Update placo robot from current joint state."""
        # Convert degrees to radians for placo
        joints_rad = np.deg2rad(self.current_joints)

        for i, name in enumerate(self.motor_names):
            self.robot.set_joint(name, joints_rad[i])

        self.robot.update_kinematics()
        self.viz.display(self.robot.state.q)

    def disconnect(self):
        """Simulated disconnection."""
        print("Simulated robot disconnected")


def main():
    init_logging()
    logger = logging.getLogger("so101_deploy_sim")

    import argparse
    parser = argparse.ArgumentParser(
        description="Simulate SO101 deployment with pre-recorded trajectories in placo"
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
        default=URDF_PATH,
        help="Path to SO101 URDF directory for IK and visualization",
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
    # Compute Base EE Pose from Reset Pose (same as deploy script)
    # ========================================================================
    logger.info("Computing base EE pose from reset pose...")
    logger.info(f"  Reset pose (deg): {RESET_POSE_DEG}")

    base_ee_T = kinematics.forward_kinematics(RESET_POSE_DEG)
    logger.info(f"  Base EE position: {base_ee_T[:3, 3]}")

    # ========================================================================
    # Build Processor Pipeline (same as deploy script)
    # ========================================================================
    logger.info("Building processor pipeline...")

    # Pipeline: relative EE action -> absolute EE -> bounds check -> IK -> joints
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
    # Connect to Simulated Robot
    # ========================================================================
    logger.info("Connecting to simulated SO101 robot...")

    robot = SimulatedSO101Robot(str(urdf_path), MOTOR_NAMES)
    robot.connect(calibrate=False)

    # ========================================================================
    # Compute Target Trajectory & Find Starting Pose
    # ========================================================================
    logger.info("Computing target trajectory for visualization...")

    # Compute absolute EE positions for all trajectory steps
    # Start with base EE pose (t=0), then apply each relative action
    traj_positions = [base_ee_T[:3, 3]]  # Include starting position
    for i in range(trajectory.shape[0]):
        rel_action_10d = trajectory[i]
        rel_T = pose10d_to_mat(rel_action_10d[:9])
        abs_T = base_ee_T @ rel_T
        traj_positions.append(abs_T[:3, 3])
    traj_positions = np.array(traj_positions)

    # Determine color based on trajectory type
    if "_gt.txt" in args.trajectory:
        traj_type = "Ground Truth"
        traj_color = 0x0000ff  # Blue
    elif "_pred.txt" in args.trajectory:
        traj_type = "Predicted"
        traj_color = 0xff0000  # Red
    else:
        traj_type = "Trajectory"
        traj_color = 0x00ff00  # Green

    # Display the trajectory as point cloud
    points_viz("target_trajectory", traj_positions, color=traj_color)
    logger.info(f"  Target trajectory: {len(traj_positions)} points (including start)")
    logger.info(f"  Type: {traj_type}")
    logger.info(f"  Start position (from reset pose): {traj_positions[0]}")
    logger.info(f"  Trajectory range: x[{traj_positions[:,0].min():.3f}, {traj_positions[:,0].max():.3f}] "
                f"y[{traj_positions[:,1].min():.3f}, {traj_positions[:,1].max():.3f}] "
                f"z[{traj_positions[:,2].min():.3f}, {traj_positions[:,2].max():.3f}]")

    # ========================================================================
    # Initialize robot at reset pose
    # ========================================================================
    # Robot starts at reset pose, trajectory visualization aligns from initial EE
    logger.info(f"Setting robot to reset pose")
    logger.info(f"  Reset pose (deg): {RESET_POSE_DEG}")
    logger.info(f"  Initial EE position: {base_ee_T[:3, 3]}")
    robot.current_joints = RESET_POSE_DEG.copy()
    robot.send_action({f"{name}.pos": val for name, val in zip(MOTOR_NAMES, RESET_POSE_DEG)})

    logger.info("Waiting...")
    time.sleep(5.0)  # Pause to visualize the starting pose

    # Reset processors
    ee_to_joints_pipeline.reset()

    # ========================================================================
    # Main Control Loop
    # ========================================================================
    logger.info("Starting control loop...")
    logger.info(f"Control frequency: {args.fps} Hz")
    logger.info(f"Total steps in trajectory: {trajectory.shape[0]}")
    logger.info(f"Loop: {args.loop}")
    logger.info("\nOpen http://127.0.0.1:7000/static/ in your browser to see the visualization!")
    logger.info("Press Ctrl+C to stop\n")

    # Global state for ischedule
    global sim_running, sim_dt
    sim_running = True
    sim_dt = 1.0 / args.fps

    # Trajectory playback state
    traj_idx = [0]  # Use list to allow modification in nested function

    @schedule(interval=sim_dt)
    def loop():
        global sim_running

        if not sim_running:
            return

        idx = traj_idx[0]

        # Check if we've reached the end
        if idx >= trajectory.shape[0]:
            if args.loop:
                traj_idx[0] = 0
                # Reset robot to reset pose
                robot.current_joints = RESET_POSE_DEG.copy()
                robot.send_action({f"{name}.pos": val for name, val in zip(MOTOR_NAMES, RESET_POSE_DEG)})
                ee_to_joints_pipeline.reset()
                logger.info("Restarting trajectory loop...")
                return
            else:
                logger.info(f"Completed {trajectory.shape[0]} steps. Stopping.")
                sim_running = False
                return

        # -------------------------------------------------------------------
        # Get current observation from robot
        # -------------------------------------------------------------------
        obs_dict = robot.get_observation()
        current_joints = np.array([obs_dict[f"{name}.pos"] for name in MOTOR_NAMES])

        # -------------------------------------------------------------------
        # Get relative action from trajectory
        # -------------------------------------------------------------------
        rel_action_10d = trajectory[idx]  # (10,) array

        # -------------------------------------------------------------------
        # Run through the full pipeline (same as deploy script)
        # -------------------------------------------------------------------
        action_dict: RobotAction = {"rel_pose": rel_action_10d.copy()}
        robot_obs: RobotObservation = {
            f"{name}.pos": obs_dict[f"{name}.pos"] for name in MOTOR_NAMES
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

        # Display EE frame
        robot_frame_viz(robot.robot, "gripper_frame_link")

        # -------------------------------------------------------------------
        # Log progress
        # -------------------------------------------------------------------
        if idx % 10 == 0:
            # Compute target EE pose for logging
            rel_T = pose10d_to_mat(rel_action_10d[:9])
            target_ee_pose = base_ee_T @ rel_T
            pos = target_ee_pose[:3, 3]
            logger.info(f"Step {idx}/{trajectory.shape[0]}: EE pos {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")

        # Advance to next step
        traj_idx[0] = idx + 1

    # Run the control loop
    try:
        run_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Return to safe pose
        logger.info("Returning to safe initial pose...")
        safe_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, INITIAL_SAFE_JOINTS)}
        robot.send_action(safe_action)
        time.sleep(0.5)

        # Disconnect robot
        robot.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    main()
