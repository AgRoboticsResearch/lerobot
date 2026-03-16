#!/usr/bin/env python

# This script visualizes future ORB-SLAM3 trajectories projected onto the current camera frame.
# It assumes CameraTrajectory.txt is in a 12-column flat 3x4 matrix format (T_w_c)
# and the camera frame uses standard OpenCV convention (+Z forward, +X right, +Y down).

import argparse
import json
import logging
from pathlib import Path
import io

import cv2
import imageio.v3 as iio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Lerobot imports to support dynamic URDF kinematic offsets
from lerobot.model.kinematics import RobotKinematics

logger = logging.getLogger(__name__)

def load_camera_matrix(dataset_root: Path) -> np.ndarray:
    camera_info_path = dataset_root / "camera_info_color.json"
    if not camera_info_path.exists():
        raise FileNotFoundError(f"Missing {camera_info_path}")
    with open(camera_info_path) as f:
        camera_info = json.load(f)
    
    K_flat = camera_info["K"]
    camera_matrix = np.array(K_flat, dtype=np.float64).reshape(3, 3)
    return camera_matrix

def load_trajectory(dataset_root: Path) -> np.ndarray:
    traj_path = dataset_root / "CameraTrajectory.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing {traj_path}")
    
    data = np.loadtxt(traj_path)
    # data should be N x 12
    N = len(data)
    poses = []
    for i in range(N):
        T = np.eye(4)
        T[:3, :] = data[i].reshape(3, 4)
        poses.append(T)
    return np.array(poses)

def get_camera_to_gripper_transform(urdf_path: Path) -> np.ndarray:
    """Calculate T_c_f (transform from OpenCV camera frame to gripper_frame_link) using placo/RobotKinematics."""
    # We use camera_optical_link as the root because the ORB-SLAM3 trajectory 
    # uses the OpenCV standard (Z-forward, X-right, Y-down).
    kin_cam = RobotKinematics(str(urdf_path), target_frame_name="camera_optical_link")
    kin_grip = RobotKinematics(str(urdf_path), target_frame_name="gripper_frame_link")

    # Since both camera and gripper frame are fixed to the physical gripper end-effector link,
    # we can use any joint configuration to get their relative transform.
    joints = np.zeros(len(kin_cam.joint_names))
    T_base_cam = kin_cam.forward_kinematics(joints)
    T_base_grip = kin_grip.forward_kinematics(joints)

    # Calculate offset natively in the OpenCV optical frame
    T_c_f = np.linalg.inv(T_base_cam) @ T_base_grip
    return T_c_f

def project_points(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points in OpenCV camera frame to 2D pixel coordinates."""
    # points_3d: (N, 3)
    z = points_3d[:, 2:3]
    # Avoid division by zero
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    
    points_2d_homogeneous = (K @ points_3d.T).T  # (N, 3)
    points_2d = points_2d_homogeneous[:, :2] / z  # (N, 2)
    return points_2d

def draw_trajectory(img: np.ndarray, points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    img_draw = img.copy()
    h, w = img.shape[:2]
    
    n = len(points_2d)
    if n == 0:
        return img_draw

    # Gradient from blue to red
    colors = [
        (int(255 * i / n), int(255 * (1 - i / n)), 0)  # BGR
        for i in range(max(1, n - 1))
    ]
    
    for i in range(n - 1):
        # Only draw if both points are strictly in front of camera
        if points_3d[i, 2] <= 0 or points_3d[i+1, 2] <= 0:
            continue
            
        pt1 = tuple(points_2d[i].astype(int))
        pt2 = tuple(points_2d[i + 1].astype(int))
        
        # Check if points are within image bounds
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(img_draw, pt1, pt2, colors[i % len(colors)], 2)
            
    # Draw start point (green) and end point (red)
    if points_3d[0, 2] > 0:
        start_pt = tuple(points_2d[0].astype(int))
        if 0 <= start_pt[0] < w and 0 <= start_pt[1] < h:
            cv2.circle(img_draw, start_pt, 5, (0, 255, 0), -1)  # Green

    if n > 0 and points_3d[-1, 2] > 0:
        end_pt = tuple(points_2d[-1].astype(int))
        if 0 <= end_pt[0] < w and 0 <= end_pt[1] < h:
            cv2.circle(img_draw, end_pt, 5, (0, 0, 255), -1)  # Red
        
    return img_draw

def get_3d_plot_img(fig, ax, P_ci, max_bound=None):
    ax.clear()
    
    # P_ci: (M, 3) in OpenCV frame (X right, Y down, Z forward)
    # We plot: X=X (right), Y=Z (forward/depth), Z=-Y (up)
    if len(P_ci) > 0:
        xs = P_ci[:, 0]
        ys = P_ci[:, 2]
        zs = -P_ci[:, 1]
        
        ax.plot(xs, ys, zs, marker='.', markersize=2, label="Future Traj")
        # Origin (Camera Center in current frame)
        ax.scatter([0], [0], [0], color='r', s=50, label='Origin')
        
        if max_bound is not None:
            ax.set_xlim([-max_bound, max_bound])
            ax.set_ylim([-max_bound, max_bound])
            ax.set_zlim([-max_bound, max_bound])
            
            try:
                ax.set_box_aspect((1, 1, 1))
            except AttributeError:
                pass
        
        ax.set_xlabel('X (Right)')
        ax.set_ylabel('Z (Forward)')
        ax.set_zlabel('-Y (Up)')
        ax.legend()
        
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:,:,:3] # shape (H, W, 3)
    return img.copy()

def process_dataset(dataset_root: Path, T_c_f: np.ndarray, output_dir: Path, fps: int, render_3d: bool = False):
    if not (dataset_root / "camera_info_color.json").exists() or not (dataset_root / "CameraTrajectory.txt").exists():
        logger.warning(f"Missing 'camera_info_color.json' or 'CameraTrajectory.txt' in {dataset_root}. Skipping.")
        return

    K = load_camera_matrix(dataset_root)
    T_w_c = load_trajectory(dataset_root)
    N = len(T_w_c)
    logger.info(f"Processing {dataset_root.name}: Loaded {N} poses.")
    
    frames = []
    frames_3d = []
    
    if render_3d:
        # Pre-calculate max bound so the 3D axes stay fixed instead of dynamically moving
        max_bound = 0
        for i in range(N):
            T_ci_w = np.linalg.inv(T_w_c[i])
            future_poses = T_w_c[i:] # (M, 4, 4)
            P_w_h = (future_poses @ T_c_f)[:, :4, 3] # (M, 4)
            
            P_ci = (T_ci_w @ P_w_h.T).T[:, :3]
            if len(P_ci) > 0:
                current_max = np.max(np.abs(P_ci))
                if current_max > max_bound:
                    max_bound = current_max
        max_bound = max_bound * 1.05  # slightly pad the bounds
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    for i in range(N):
        img_path = dataset_root / f"color_{i:06d}.png"
        if not img_path.exists():
            logger.warning(f"Image {img_path} not found, stopping at frame {i}.")
            break
            
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to read {img_path}.")
            break
            
        T_ci_w = np.linalg.inv(T_w_c[i])
        future_poses = T_w_c[i:]
        P_w_h = (future_poses @ T_c_f)[:, :4, 3]
        P_ci = (T_ci_w @ P_w_h.T).T[:, :3]
        
        points_2d = project_points(P_ci, K)
        img_drawn = draw_trajectory(img, P_ci, points_2d)
        img_drawn_rgb = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)
        frames.append(img_drawn_rgb)
        
        if render_3d:
            img_3d = get_3d_plot_img(fig, ax, P_ci, max_bound)
            frames_3d.append(img_3d)
        
        if (i+1) % 100 == 0:
            logger.info(f"Processed {i+1}/{N} frames.")

    if render_3d:
        plt.close(fig)
    
    if len(frames) == 0:
        logger.warning(f"No frames processed for {dataset_root.name}. Skipping text/video generation.")
        return

    scenario_name = dataset_root.name
    scenario_out_dir = output_dir / scenario_name
    scenario_out_dir.mkdir(parents=True, exist_ok=True)
    
    out_2d = scenario_out_dir / "projection.mp4"
    logger.info(f"Writing {len(frames)} frames to {out_2d}...")
    iio.imwrite(str(out_2d), np.stack(frames), fps=fps, codec='libx264')
    logger.info(f"Video saved to {out_2d}")
    
    if render_3d:
        out_3d = scenario_out_dir / "3d_projection.mp4"
        logger.info(f"Writing {len(frames_3d)} 3D frames to {out_3d}...")
        iio.imwrite(str(out_3d), np.stack(frames_3d), fps=fps, codec='libx264')
        logger.info(f"Video 3D saved to {out_3d}")


def main():
    parser = argparse.ArgumentParser(description="Visualize future ORB-SLAM3 trajectory projection inside current camera frame")
    parser.add_argument("dataset_root", type=str, help="Path to the ORB-SLAM3 scenario folder (or parent folder if -r is used)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process all subdirectories recursively")
    parser.add_argument("--urdf", type=str, default="urdf/Simulation/SO101/so101_sroi.urdf", help="Path to URDF to calculate gripper offsets")
    parser.add_argument("--output-dir", type=str, default="outputs/debug", help="Output parent directory for the videos")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--3d-projection", dest="render_3d", action="store_true", help="Generate 3D trajectory plot video")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    urdf_path = Path(args.urdf)
    
    if not urdf_path.exists():
        logger.error(f"URDF not found at {urdf_path}. Cannot compute gripper offsets.")
        return
        
    import warnings
    import sys
    import os
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            T_c_f = get_camera_to_gripper_transform(urdf_path)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
    logger.info(f"Using dynamically computed camera to gripper offset from URDF:\n{T_c_f}")
    
    root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.recursive:
        logger.info(f"Searching recursively in {root}...")
        # Find all folders containing 'CameraTrajectory.txt'
        traj_files = list(root.rglob("CameraTrajectory.txt"))
        if not traj_files:
            logger.warning(f"No CameraTrajectory.txt found in any subdirectories of {root}.")
            return
        for t_file in traj_files:
            process_dataset(t_file.parent, T_c_f, output_dir, args.fps, args.render_3d)
    else:
        process_dataset(root, T_c_f, output_dir, args.fps, args.render_3d)

if __name__ == "__main__":
    main()
