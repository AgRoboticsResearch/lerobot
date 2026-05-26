#!/usr/bin/env python

r"""
Deploy ACT policy trained with UMI-style processor pipeline + rot6d on SO101.

The preprocessor/postprocessor handle all conversions automatically:
- Preprocessor: caches 7D aa state from robot FK, converts to 20D rot6d relative state
- Model: predicts 10D rot6d relative actions
- Postprocessor: converts 10D rot6d relative → 7D aa absolute

The deployment script only needs to:
1. Get 7D aa EE pose from robot FK as observation.state
2. Run preprocessor → model → postprocessor
3. Send 7D aa absolute action through IK to get joint positions

Usage:
    python deploy_relative_ee_processor_so101.py \
        --pretrained_path outputs/.../pretrained_model \
        --robot_port /dev/ttyACM0 \
        --cameras "{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG}}"
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.cameras import CameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import OBS_STATE

logger = logging.getLogger(__name__)

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def ee_pose_aa_from_fk(kinematics, joints_deg):
    """Get 7D EE pose [x,y,z,wx,wy,wz,gripper] from FK."""
    ee_T = kinematics.forward_kinematics(joints_deg)
    pos = ee_T[:3, 3]
    # Extract axis-angle from rotation matrix
    from lerobot.utils.rotation import Rotation
    aa = Rotation.from_matrix(ee_T[:3, :3]).as_rotvec()
    # Gripper from joint positions (last motor)
    gripper = joints_deg[-1] / 100.0  # Convert deg to [0,1]
    return np.concatenate([pos, aa, [gripper]]).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy UMI-style relative EE policy on SO101")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--robot_id", type=str, default="oscar_so101_follower")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--cameras", type=str, default='{}')
    parser.add_argument("--urdf_path", type=str, default="urdf/Simulation/SO101/so101_sroi.urdf")
    parser.add_argument("--deploy_frame", type=str, default="camera_link")
    parser.add_argument("--n_action_steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--display_data", action="store_true")
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--ee_bounds_min", type=float, nargs=3, default=[-0.5, -0.5, -0.1])
    parser.add_argument("--ee_bounds_max", type=float, nargs=3, default=[0.5, 0.5, 0.8])
    parser.add_argument("--max_ee_step_m", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load policy
    model_path = args.pretrained_path
    logger.info(f"Loading policy from: {model_path}")

    policy = ACTPolicy.from_pretrained(model_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    # Create processors (saved in checkpoint — handles relative rot6d pipeline)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    logger.info("Policy and processors loaded")

    # Initialize kinematics
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.deploy_frame,
        joint_names=MOTOR_NAMES,
    )

    # Build IK pipeline: 7D aa absolute → joints
    ik_pipeline = RobotProcessorPipeline(
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": args.ee_bounds_min, "max": args.ee_bounds_max},
                max_ee_step_m=args.max_ee_step_m,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=MOTOR_NAMES,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect robot
    cameras = eval(args.cameras) if args.cameras else {}
    robot_config = SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        use_degrees=True,
        cameras=cameras,
    )
    robot = SO101Follower(robot_config)
    robot.connect(calibrate=True)

    if args.warm_start:
        reset_pose = np.array([-3.43, -94.77, 82.92, 70.74, -0.53, 1.18])
        robot.send_action({f"{name}.pos": val for name, val in zip(MOTOR_NAMES, reset_pose)})
        time.sleep(2.0)

    # Reset processors for clean state
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Main control loop
    step_count = 0
    action_queue = []

    logger.info(f"Starting control loop at {args.fps} Hz")

    try:
        while args.num_steps == 0 or step_count < args.num_steps:
            t0 = time.perf_counter()

            # Read robot state
            obs_dict = robot.get_observation()
            current_joints = np.array([obs_dict[f"{name}.pos"] for name in MOTOR_NAMES])

            # Get 7D aa EE pose via FK (this is observation.state)
            ee_aa = ee_pose_aa_from_fk(kinematics, current_joints)
            state_tensor = torch.from_numpy(ee_aa).unsqueeze(0).to(device)  # (1, 7)

            batch = {OBS_STATE: state_tensor}

            # Add camera images
            for cam_name in cameras.keys():
                img = obs_dict.get(cam_name)
                if img is not None:
                    if args.display_data:
                        cv2.imshow(f"Camera: {cam_name}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    img = img.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
                    batch[f"observation.images.{cam_name}"] = torch.from_numpy(img).unsqueeze(0).to(device)

            # Predict new chunk when queue is empty
            if len(action_queue) == 0:
                with torch.no_grad():
                    processed = preprocessor(batch)
                    pred = policy.predict_action_chunk(processed)
                    pred = postprocessor(pred)

                if isinstance(pred, dict) and "action" in pred:
                    pred = pred["action"]

                # Postprocessor converts 10D rot6d → 7D aa absolute
                actions_aa = pred[0].cpu().numpy()  # (chunk_size, 7)
                action_queue = [actions_aa[i] for i in range(min(args.n_action_steps, len(actions_aa)))]

                logger.info(f"Predicted chunk of {len(action_queue)} actions")

            # Execute next action
            action_aa = action_queue.pop(0)

            # Convert 7D aa absolute → joints via IK
            action_dict = {
                "action": {
                    "ee.x": float(action_aa[0]),
                    "ee.y": float(action_aa[1]),
                    "ee.z": float(action_aa[2]),
                    "ee.wx": float(action_aa[3]),
                    "ee.wy": float(action_aa[4]),
                    "ee.wz": float(action_aa[5]),
                    "ee.gripper_pos": float(action_aa[6]),
                },
                "observation": {
                    f"{name}.pos": float(current_joints[i]) for i, name in enumerate(MOTOR_NAMES)
                },
            }

            result = ik_pipeline(action_dict)
            if "action" in result:
                robot.send_action(result["action"])

            step_count += 1

            # Timing
            elapsed = time.perf_counter() - t0
            if elapsed < 1.0 / args.fps:
                time.sleep(1.0 / args.fps - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.disconnect()
        logger.info(f"Disconnected after {step_count} steps")


if __name__ == "__main__":
    main()
