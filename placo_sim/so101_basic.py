#!/usr/bin/env python3
"""
Basic SO101 motion simulation using placo.
The robot's end effector (gripper_frame_link) follows a sinusoidal trajectory.
"""

import pinocchio
import placo
import numpy as np
import io
from PIL import Image, ImageDraw, ImageFont
import meshcat.geometry as g
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, get_viewer
from placo_utils.tf import tf

# Path to SO101 URDF file
URDF_PATH = "../urdf/Simulation/SO101/so101_new_calib.urdf"

# Loading the robot
robot = placo.RobotWrapper(URDF_PATH, placo.Flags.ignore_collisions)

# Creating the solver
solver = placo.KinematicsSolver(robot)

# The floating base is fixed (robot is not moving)
solver.mask_fbase(True)

# Creating a task for the gripper frame (end effector)
effector_task = solver.add_frame_task("gripper_frame_link", np.eye(4))
effector_task.configure("gripper_frame", "soft", 1.0, 1.0)

# Enable joints velocity limits
solver.enable_velocity_limits(True)

viz = robot_viz(robot)

t = 0
dt = 0.01
solver.dt = dt

# Visualize coordinate frames for link/body frames only.
FRAME_TYPES_TO_SHOW = {pinocchio.FrameType.BODY}
FRAME_NAMES = [
    frame.name
    for frame in robot.model.frames
    if frame.type in FRAME_TYPES_TO_SHOW and frame.name != "universe"
]

LABEL_NODES = {}
LABEL_SIZES = {}
LABEL_OFFSET_M = np.array([0.0, 0.0, 0.03])
LABEL_PIXEL_TO_M = 0.001


def _render_label_png(text: str, font_size: int = 18) -> bytes:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + 6
    height = bbox[3] - bbox[1] + 4

    img = Image.new("RGBA", (width, height), (255, 255, 255, 160))
    draw = ImageDraw.Draw(img)
    draw.text((3, 2), text, font=font, fill=(0, 0, 0, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ensure_label(frame_name: str) -> None:
    if frame_name in LABEL_NODES:
        return

    vis = get_viewer()
    png_data = _render_label_png(frame_name)
    img = g.PngImage(png_data)
    texture = g.ImageTexture(img)
    material = g.MeshBasicMaterial(map=texture, transparent=True)

    width_m = max(0.02, len(frame_name) * 0.008)
    height_m = 0.02
    depth_m = 0.001
    geometry = g.Box([width_m, height_m, depth_m])

    node = vis["labels"][frame_name]
    node.set_object(geometry, material)
    LABEL_NODES[frame_name] = node
    LABEL_SIZES[frame_name] = (width_m, height_m, depth_m)


@schedule(interval=dt)
def loop():
    global t
    t += dt

    # Define target position for the gripper
    # SO101 reaches forward with sinusoidal motion
    target_x = 0.15 + 0.05 * np.cos(t * 0.5)
    target_y = 0.0 + 0.08 * np.sin(t)
    target_z = 0.10 + 0.05 * np.sin(t * 0.7)

    # Create target transform (identity rotation, translation above)
    effector_task.T_world_frame = tf.translation_matrix([target_x, target_y, target_z])

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "gripper_frame_link")
    for frame_name in FRAME_NAMES:
        robot_frame_viz(robot, frame_name)
        _ensure_label(frame_name)
        T_label = np.eye(4)
        T_frame = robot.get_T_world_frame(frame_name)
        T_label[:3, 3] = T_frame[:3, 3] + LABEL_OFFSET_M
        LABEL_NODES[frame_name].set_transform(T_label)
    frame_viz("target", effector_task.T_world_frame)


if __name__ == "__main__":
    print("Starting SO101 basic motion simulation...")
    print(f"URDF: {URDF_PATH}")
    print("Press Ctrl+C to stop")
    run_loop()
