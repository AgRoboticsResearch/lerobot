# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import placo
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from placo_utils.visualization import robot_frame_viz, points_viz, robot_viz
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so100_leader,
    so101_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# Motor names for SO101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class SimulatedSO101Robot:
    """Simulated SO101 robot using placo for visualization."""

    def __init__(self, urdf_path: str, motor_names: list[str], initial_joints: np.ndarray | None = None):
        self.urdf_path = urdf_path
        self.motor_names = motor_names

        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.joints_task = self.solver.add_joints_task()
        self.viz = robot_viz(self.robot)

        # Initialize joint state
        self.current_joints = initial_joints if initial_joints is not None else np.zeros(len(motor_names))
        self._update_robot_from_joints()

    def set_joints(self, joints: np.ndarray):
        """Set joint positions (in degrees)."""
        self.current_joints = joints
        self._update_robot_from_joints()

    def _update_robot_from_joints(self):
        """Update placo robot from current joint state."""
        joints_rad = np.deg2rad(self.current_joints)
        for i, name in enumerate(self.motor_names):
            self.robot.set_joint(name, joints_rad[i])
        self.robot.update_kinematics()
        self.viz.display(self.robot.state.q)


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Enable placo visualization
    placo_vis: bool = False
    # Path to URDF file for placo visualization
    urdf_path: str = "src/lerobot/robots/so101/urdf/so101.urdf"


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    sim_robot: SimulatedSO101Robot | None = None,
    kinematics: RobotKinematics | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
        sim_robot: Optional SimulatedSO101Robot for placo visualization.
        kinematics: Optional RobotKinematics for FK computation.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    ee_history = []  # Track EE position history for GREEN visualization
    max_history = 500  # Maximum number of history points to keep

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()

        # Get current robot joints
        current_joints = np.array([obs[f"{name}.pos"] for name in MOTOR_NAMES])

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        print("robot_action_to_send:", robot_action_to_send)
        _ = robot.send_action(robot_action_to_send)

        # ========================================================================
        # Placo visualization
        # ========================================================================
        if sim_robot is not None and kinematics is not None:
            # Update sim_robot to reflect actual robot position
            sim_robot.set_joints(current_joints)

            # Compute EE pose from current joints (FK)
            current_ee_T = kinematics.forward_kinematics(current_joints)
            ee_pos = current_ee_T[:3, 3].copy()
            ee_history.append(ee_pos)
            if len(ee_history) > max_history:
                ee_history.pop(0)

            # Show EE history trajectory (GREEN)
            if len(ee_history) > 1:
                points_viz("ee_history", np.array(ee_history), color=0x00ff00)

            # Compute target EE pose from commanded action
            # Extract joint positions from robot_action_to_send
            target_joints = np.array([robot_action_to_send.get(f"{name}.pos", current_joints[i]) for i, name in enumerate(MOTOR_NAMES)])
            target_ee_T = kinematics.forward_kinematics(target_joints)
            target_pos = target_ee_T[:3, 3]

            # Show target EE pose point (RED)
            points_viz("target_ee", np.array([target_pos]), color=0xff0000)

            # Show robot frame at current position
            robot_frame_viz(sim_robot.robot, "gripper_frame_link")

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Initialize placo visualization if enabled
    sim_robot = None
    kinematics = None
    if cfg.placo_vis:
        logging.info("Initializing placo visualization...")
        placo_urdf_path = str(Path(cfg.urdf_path).resolve())
        if not Path(placo_urdf_path).exists():
            raise FileNotFoundError(f"URDF not found at {placo_urdf_path}")

        # Get initial robot joints for sim_robot
        robot.connect()
        obs = robot.get_observation()
        initial_joints = np.array([obs[f"{name}.pos"] for name in MOTOR_NAMES])

        sim_robot = SimulatedSO101Robot(placo_urdf_path, MOTOR_NAMES, initial_joints)
        kinematics = RobotKinematics(
            urdf_path=placo_urdf_path,
            target_frame_name="gripper_frame_link",
        )
        logging.info("Placo visualization initialized")
        logging.info("Open http://127.0.0.1:7000/static/ to see visualization")
        logging.info("  GREEN: EE position history")
        logging.info("  RED: Target EE pose (commanded position)")
    else:
        robot.connect()

    # Always connect teleoperator
    teleop.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            sim_robot=sim_robot,
            kinematics=kinematics,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        # Shutdown placo visualization
        if sim_robot is not None:
            try:
                logging.info("Shutting down placo visualization...")
                if hasattr(placo, 'kill_web_server'):
                    placo.kill_web_server()
                del sim_robot.viz
                del sim_robot.solver
                del sim_robot.robot
                sim_robot = None
                logging.info("Placo visualization shut down")
            except Exception as e:
                logging.warning(f"Error shutting down placo: {e}")
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
