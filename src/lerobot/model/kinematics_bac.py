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

import numpy as np

from lerobot.utils.rotation import Rotation


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
        """
        try:
            import placo  # type: ignore[import-not-found] # C++ library with Python bindings, no type stubs available. TODO: Create stub file or request upstream typing support.
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base
        
        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver with multi-start optimization
        to find solutions close to the initial guess.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])
        current_joint_deg = current_joint_pos[: len(self.joint_names)].copy()

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # Multi-start approach: try multiple initial guesses and select the best
        # This helps avoid converging to distant solution branches
        best_solution = None
        best_cost = float('inf')
        
        # Generate initial guesses: original + small perturbations
        initial_guesses = [
            current_joint_rad,  # Original guess
        ]
        
        # Add small perturbations (in radians) to explore nearby solutions
        perturbation_magnitudes = [0.05, 0.1, 0.2]  # ~3°, 6°, 11° in degrees
        for pert_mag in perturbation_magnitudes:
            for _ in range(3):  # 3 random perturbations per magnitude
                perturbation = np.random.uniform(-pert_mag, pert_mag, size=len(current_joint_rad))
                initial_guesses.append(current_joint_rad + perturbation)

        # Try each initial guess
        for initial_guess_rad in initial_guesses:
            # Set initial guess
            for i, joint_name in enumerate(self.joint_names):
                self.robot.set_joint(joint_name, initial_guess_rad[i])

            # Solve IK with multiple iterations
            max_solve_iterations = 10
            for _ in range(max_solve_iterations):
                self.solver.solve(True)
                self.robot.update_kinematics()
                
                # Check convergence
                current_pose = self.robot.get_T_world_frame(self.target_frame_name)
                pos_error = np.linalg.norm(desired_ee_pose[:3, 3] - current_pose[:3, 3])
                
                if pos_error < 1e-4:  # 0.1mm tolerance
                    break

            # Extract solution
            joint_pos_rad = []
            for joint_name in self.joint_names:
                joint = self.robot.get_joint(joint_name)
                joint_pos_rad.append(joint)
            
            joint_pos_deg = np.rad2deg(joint_pos_rad)
            
            # Compute achieved pose
            achieved_pose = self.robot.get_T_world_frame(self.target_frame_name)
            
            # Compute errors
            pos_error = np.linalg.norm(desired_ee_pose[:3, 3] - achieved_pose[:3, 3])
            
            # Rotation error using rotation vectors
            rot_desired = Rotation.from_matrix(desired_ee_pose[:3, :3])
            rot_achieved = Rotation.from_matrix(achieved_pose[:3, :3])
            rot_error = np.linalg.norm((rot_desired.inv() * rot_achieved).as_rotvec())
            
            # Joint deviation from initial guess (penalize large changes)
            joint_deviation = np.linalg.norm(joint_pos_deg - current_joint_deg)
            
            # Combined cost: prioritize position and orientation accuracy, 
            # but also penalize large joint deviations
            # Weights: position (mm), rotation (degrees), joint deviation (degrees)
            cost = (
                pos_error * 1000.0 +  # Position error in mm
                np.degrees(rot_error) * 10.0 +  # Rotation error in degrees (weighted more)
                joint_deviation * 0.1  # Joint deviation penalty (small weight)
            )
            
            # Update best solution if this is better
            if cost < best_cost:
                best_cost = cost
                best_solution = joint_pos_deg.copy()

        # If no solution found (shouldn't happen), use the last one
        if best_solution is None:
            best_solution = joint_pos_deg.copy()

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = best_solution
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return best_solution
