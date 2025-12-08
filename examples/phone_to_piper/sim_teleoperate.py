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
Piper simulation teleoperation using phone input.

This script connects to a phone teleoperator and drives a simulated Piper robot
using the standard processing pipeline (no custom Corrected* classes needed after
fixing the unit conversion bug in robot_kinematic_processor.py).
"""

import time
import argparse
import numpy as np
import copy

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
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
    PiperJointSafetyClamp,
)
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep

# Placo visualization
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz

FPS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-os", type=str, default="ios", choices=["ios", "android"], help="Phone OS (ios or android)")
    args = parser.parse_args()

    # Robot Config (Simulated)
    robot_config = PiperConfig(
        can_interface="can0",  # Placeholder
        include_gripper=True,
        use_degrees=True,
        cameras={},
        joint_names=[f"joint{i+1}" for i in range(6)]
    )
    
    phone_os_enum = PhoneOS.IOS if args.phone_os == "ios" else PhoneOS.ANDROID
    teleop_config = PhoneConfig(phone_os=phone_os_enum)

    # Initialize Teleoperator (Real Phone)
    teleop_device = Phone(teleop_config)
    teleop_device.connect()
    
    # Initialize Robot Kinematics (Simulated Robot)
    urdf_path = "piper_description/urdf/piper_description.urdf"
    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_base",
        joint_names=robot_config.joint_names,
    )

    # Initialize Placo Viewer
    viz = robot_viz(kinematics_solver.robot)

    # Set initial state (radians for Placo)
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

    # Initialize Pipeline
    # Pipeline Sequence:
    # 1. MapPhoneActionToRobotAction: Phone coordinates -> Robot coordinates
    # 2. PiperEEReferenceAndDelta: Relative delta -> Absolute EE target pose
    # 3. PiperEEBoundsAndSafety: Workspace bounds and step limiting
    # 4. PiperGripperVelocityToJoint: Gripper velocity -> Gripper position
    # 5. PiperInverseKinematicsEEToJoints: EE pose -> Joint angles
    # 6. PiperJointSafetyClamp: Joint limits and step limiting (final safety)
    
    pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=phone_os_enum),
            PiperEEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
                motor_names=robot_config.joint_names,
                use_latched_reference=True,
            ),
            PiperEEBoundsAndSafety(
                end_effector_bounds={"min": [-0.6, -0.6, 0.0], "max": [0.6, 0.6, 0.8]},
                max_ee_step_m=0.10,
                max_ee_rot_step_rad=0.3,
            ),
            PiperGripperVelocityToJoint(
                speed_factor=20.0,
            ),
            PiperInverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=robot_config.joint_names,
                initial_guess_current_joints=True,
            ),
            PiperJointSafetyClamp(
                motor_names=robot_config.joint_names,
                max_joint_step_deg=10.0,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    print("Starting SIMULATION teleop loop.")
    print("Press B1 on phone to Enable and Latch origin.")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Get Teleop Action
            phone_obs = teleop_device.get_action()
            if not phone_obs:
                time.sleep(0.01)
                continue

            # 2. Construct Robot Observation from Simulation State
            # Pipeline expects joint positions in DEGREES
            robot_obs = {}
            for name in robot_config.joint_names:
                rad_val = kinematics_solver.robot.get_joint(name)
                deg_val = np.rad2deg(rad_val)
                robot_obs[f"{name}.pos"] = deg_val
            
            # Add gripper position (pipeline needs it)
            robot_obs["gripper.pos"] = 0.0
            
            # 3. Run Pipeline
            try:
                phone_obs_copy = copy.deepcopy(phone_obs)
                joint_action = pipeline((phone_obs_copy, robot_obs))
                
                # 4. Update Simulation State
                # joint_action contains {joint_name.pos: val_deg, ...}
                for name in robot_config.joint_names:
                    key = f"{name}.pos"
                    if key in joint_action:
                        deg_val = joint_action[key]
                        rad_val = np.deg2rad(deg_val)
                        kinematics_solver.robot.set_joint(name, rad_val)
                
                kinematics_solver.robot.update_kinematics()
                
            except Exception as e:
                print(f"Pipeline Error: {e}")

            # 5. Visualize
            viz.display(kinematics_solver.robot.state.q)
            robot_frame_viz(kinematics_solver.robot, "gripper_base")
            
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        print("Stopping sim...")
    finally:
        teleop_device.disconnect()


if __name__ == "__main__":
    main()