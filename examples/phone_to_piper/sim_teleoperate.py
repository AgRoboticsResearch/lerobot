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
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
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
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Placo visualization
import placo
from placo_utils.visualization import robot_viz

FPS = 30

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-os", type=str, default="ios", choices=["ios", "android"], help="Phone OS (ios or android)")
    args = parser.parse_args()

    # Robot Config (Simulated)
    # We still use PiperConfig for joint names and structure, but we won't connect to hardware
    robot_config = PiperConfig(
        can_interface="can0", # Placeholder
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
    # Using local path assumption similar to other scripts
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
    
    # Set robot to initial state
    for name, val in initial_joints.items():
        kinematics_solver.robot.set_joint(name, val)
    kinematics_solver.robot.update_kinematics()

    # Latched state
    T_origin = kinematics_solver.robot.get_T_world_frame("gripper_base")
    is_latched = False

    print("Starting SIMULATION teleop loop.")
    print("Press B1 on phone to Enable and Latch origin.")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Get Teleop Action
            phone_obs = teleop_device.get_action()
            
            # Check enabled state from raw inputs (since we parse it manually now)
            # Note: teleop_device.get_action() returns a dict with 'phone.pos', etc.
            # but 'enabled' is derived. We can respect 'phone.enabled'.
            
            enabled = phone_obs.get("phone.enabled", False)
            
            if enabled:
                if not is_latched:
                    # Rising edge: Latch current robot pose as origin
                    T_origin = kinematics_solver.robot.get_T_world_frame("gripper_base")
                    is_latched = True
                    print("Latched origin.")

                # 2. Extract Deltas
                pos = phone_obs.get("phone.pos") # [x, y, z]
                rot = phone_obs.get("phone.rot") # Rotation object

                if pos is not None and rot is not None:
                    # 3. Construct Delta Transform
                    # Apply mapping: Phone -> Robot
                    # Phone: X, Y, Z. Robot: X (fwd), Y (left), Z (up).
                    # Mapping from phone_processor.py:
                    # tx = -pos[1] (Phone -Y -> Robot X)
                    # ty = pos[0]  (Phone X -> Robot Y)
                    # tz = pos[2]  (Phone Z -> Robot Z)
                    
                    dx = -pos[1]
                    dy = pos[0]
                    dz = pos[2]
                    
                    # Rotation Mapping
                    rvec = rot.as_rotvec()
                    twx = rvec[1]
                    twy = rvec[0]
                    twz = -rvec[2]
                    
                    # Construct T_delta
                    # Position delta
                    T_delta = np.eye(4)
                    T_delta[:3, 3] = [dx, dy, dz]
                    
                    # Rotation delta
                    from scipy.spatial.transform import Rotation
                    # Create rotation matrix from mapped rotvec
                    R_delta = Rotation.from_rotvec([twx, twy, twz]).as_matrix()
                    T_delta[:3, :3] = R_delta
                    
                    # 4. Apply Delta to Origin
                    # We treat T_delta as a displacement in the world frame relative to the origin.
                    # R_target = R_delta * R_origin (Apply rotation change)
                    # P_target = P_origin + P_delta (Apply position change)
                    
                    T_target = np.eye(4)
                    # Apply rotation: R_target = R_delta @ R_origin (rotate the original frame by delta)
                    # Or R_target = R_origin @ R_delta?
                    # Phone rotation is "delta from calibration". 
                    # If I tilt phone right, I want robot to tilt right.
                    # Assuming R_delta represents the absolute rotation of the phone relative to calibration (Identity).
                    # So we apply this rotation to the robot's origin rotation.
                    
                    # If R_origin is "neutral" (Identity-ish), R_target = R_delta * R_origin.
                    # Let's try: T_target.R = R_delta @ T_origin.R
                    
                    T_target[:3, :3] = R_delta @ T_origin[:3, :3]
                    
                    # Apply position: T_target.P = T_origin.P + [dx, dy, dz]
                    T_target[:3, 3] = T_origin[:3, 3] + [dx, dy, dz]
                    
                    # 5. Solve IK
                    # Get current joints in degrees for seed
                    current_joints_rad = []
                    for name in robot_config.joint_names:
                        current_joints_rad.append(kinematics_solver.robot.get_joint(name))
                    current_joints_deg = np.rad2deg(current_joints_rad)

                    computed_joints_deg = kinematics_solver.inverse_kinematics(
                        current_joint_pos=current_joints_deg,
                        desired_ee_pose=T_target,
                        position_weight=1.0,
                        orientation_weight=0.1 # Relax orientation for position control test
                    )

                    # 6. Update Robot
                    computed_joints_rad = np.deg2rad(computed_joints_deg)
                    for i, name in enumerate(robot_config.joint_names):
                        kinematics_solver.robot.set_joint(name, computed_joints_rad[i])
                    kinematics_solver.robot.update_kinematics()

            else:
                is_latched = False

            # 7. Visualize
            viz.display(kinematics_solver.robot.state.q)
            
            # Optional: Log to Rerun if desired, but user focused on Placo
            # log_rerun_data(...) 

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        print("Stopping sim...")
    finally:
        teleop_device.disconnect()

if __name__ == "__main__":
    main()
