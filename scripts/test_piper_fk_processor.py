import time
import logging
import numpy as np
import sys
import os

# Add the root directory to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lerobot_robot_piper.config_piper import PiperConfig
from lerobot_robot_piper.piper import Piper
from lerobot_robot_piper.robot_kinematic_processor import PiperForwardKinematicsJointsToEEObservation
from lerobot.model.kinematics import RobotKinematics
from placo_utils.visualization import robot_viz, frame_viz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Setup Robot
    # SDK uses joint_1, joint_2, etc.
    sdk_joint_names = [f"joint_{i+1}" for i in range(6)]
    # URDF uses joint1, joint2, etc.
    urdf_joint_names = [f"joint{i+1}" for i in range(6)]
    
    robot_config = PiperConfig(
        can_interface="can0",
        include_gripper=True,
        use_degrees=True,
        cameras={},
        joint_names=sdk_joint_names,
    )
    
    robot = Piper(robot_config)
    
    logger.info("Connecting to robot...")
    try:
        robot.connect()
    except Exception as e:
        logger.error(f"Failed to connect to robot: {e}")
        logger.info("Ensure the robot is connected and CAN interface is up.")
        return

    # 2. Setup Kinematics
    # Note: We use "gripper_base" as target frame to match the processor logic
    kinematics_solver = RobotKinematics(
        urdf_path="piper_description/urdf/piper_description.urdf",
        target_frame_name="gripper_base",
        joint_names=urdf_joint_names,
    )

    # 3. Setup Processor
    processor = PiperForwardKinematicsJointsToEEObservation(
        kinematics=kinematics_solver,
        motor_names=sdk_joint_names,
    )

    # Setup Visualization
    viz = robot_viz(processor.kinematics.robot)

    logger.info("Reading robot state and calculating EE pose...")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # Read observation
            obs = robot.get_observation()
            
            # Process observation
            # The processor adds ee.x, ee.y, etc. to the observation dict
            obs_with_ee = processor.observation(obs)
            
            # Update visualization
            viz.display(processor.kinematics.robot.state.q)
            
            # Visualize EE frame
            T_ee = processor.kinematics.robot.get_T_world_frame("gripper_base")
            frame_viz("ee_frame", T_ee)

            # Print result
            ee_pos = [obs_with_ee["ee.x"], obs_with_ee["ee.y"], obs_with_ee["ee.z"]]
            ee_rot = [obs_with_ee["ee.wx"], obs_with_ee["ee.wy"], obs_with_ee["ee.wz"]]
            gripper = obs_with_ee.get("ee.gripper_pos", 0.0)
            
            # Print formatted output on the same line
            print(f"\rEE Pos: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}] | "
                  f"EE Rot: [{ee_rot[0]:.4f}, {ee_rot[1]:.4f}, {ee_rot[2]:.4f}] | "
                  f"Gripper: {gripper:.1f}", end="")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()