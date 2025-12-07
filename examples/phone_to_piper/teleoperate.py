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

import time
import argparse

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
)
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-os", type=str, default="ios", choices=["ios", "android"], help="Phone OS (ios or android)")
    parser.add_argument("--can-port", type=str, default="can0", help="CAN interface port for Piper")
    args = parser.parse_args()

    # Initialize the robot and teleoperator
    # Enable gripper since we want to control it
    robot_config = PiperConfig(
        can_interface=args.can_port,
        include_gripper=True,
        use_degrees=True,
        cameras={},  # Disable default camera for teleoperation unless specified
        joint_names=[f"joint{i+1}" for i in range(6)] # URDF uses joint1, joint2... not joint_1
    )
    
    phone_os_enum = PhoneOS.IOS if args.phone_os == "ios" else PhoneOS.ANDROID
    teleop_config = PhoneConfig(phone_os=phone_os_enum)

    # Initialize the robot and teleoperator
    robot = Piper(robot_config)
    teleop_device = Phone(teleop_config)

    # Robot Kinematics for Piper
    # Assuming the script is run from the root of the repo: python examples/phone_to_piper/teleoperate.py
    # and the piper_description is at piper_description/urdf/piper_description.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="piper_description/urdf/piper_description.urdf",
        target_frame_name="gripper_base", # Verified from compare_ik_solvers.py
        joint_names=robot.config.joint_names,
    )

    # Build pipeline to convert phone action to ee pose action to joint action
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            PiperEEReferenceAndDelta(
                kinematics=kinematics_solver,
                # Step sizes: how much the robot EE moves relative to phone movement
                # Reducing step size for safety and precision
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
                motor_names=robot_config.joint_names,
                use_latched_reference=True,
            ),
            PiperEEBoundsAndSafety(
                # Safety bounds in meters
                end_effector_bounds={"min": [-0.6, -0.6, 0.0], "max": [0.6, 0.6, 0.8]},
                max_ee_step_m=0.10,
            ),
            PiperGripperVelocityToJoint(
                # Piper gripper typically 0-100 or 0-70mm
                # We need to map -1..1 velocity to appropriate changes
                speed_factor=20.0,
            ),
            PiperInverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=robot_config.joint_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer
    init_rerun(session_name="phone_piper_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop. Move your phone to teleoperate the robot...")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = robot.get_observation()

            # Get teleop action
            phone_obs = teleop_device.get_action()

            # Phone -> EE pose -> Joints transition
            joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

            # Send action to robot
            _ = robot.send_action(joint_action)

            # Visualize
            log_rerun_data(observation=phone_obs, action=joint_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        print("Stopping teleop...")
    finally:
        robot.disconnect()
        # Teleop device might not need explicit disconnect but good practice
        # teleop_device.disconnect() 

if __name__ == "__main__":
    main()
