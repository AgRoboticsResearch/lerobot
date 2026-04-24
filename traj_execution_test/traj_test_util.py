"""Shared utilities for trajectory test scripts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


# ============================================================
# Helpers
# ============================================================

def make_transform(pos, rotvec):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = pos
    return T


def rad2deg(rad):
    return np.array(rad) * 180.0 / np.pi


def deg2rad(deg):
    return np.array(deg) * np.pi / 180.0


# ============================================================
# CSV loading
# ============================================================

def load_trajectory(csv_path, T_base, max_steps=None):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} steps from {csv_path}")

    traj = []
    for _, row in df.iterrows():
        if "action.ee.x" in df.columns:
            rel_pos = [row["action.ee.x"], row["action.ee.y"], row["action.ee.z"]]
            rel_rv = [row.get("action.ee.wx", 0), row.get("action.ee.wy", 0), row.get("action.ee.wz", 0)]
            gripper = row["action.ee.gripper_pos"]
        else:
            rel_pos = [row["state.ee.x"], row["state.ee.y"], row["state.ee.z"]]
            rel_rv = [row.get("state.ee.wx", 0), row.get("state.ee.wy", 0), row.get("state.ee.wz", 0)]
            gripper = row["state.ee.gripper_pos"]

        T_t = T_base @ make_transform(rel_pos, rel_rv)
        traj.append({"T_target": T_t.copy(), "gripper": gripper})

    if max_steps:
        traj = traj[:max_steps]
    print(f"Composed {len(traj)} absolute EE targets")
    return traj


# ============================================================
# Video
# ============================================================

def save_video(frames, out_path, fps=20):
    import imageio
    imageio.mimsave(str(out_path), frames, fps=fps)
    print(f"Saved {len(frames)} frames -> {out_path}")


# ============================================================
# Plot
# ============================================================

def plot_tcp_trajectory(positions: np.ndarray, output_path: Path, title: str = "TCP Trajectory"):
    """Plot TCP trajectory (XY/XZ/YZ + 3D + xyz vs steps)."""
    fig = plt.figure(figsize=(16, 12))

    projections = [
        (0, 1, 'X', 'Y'),
        (0, 2, 'X', 'Z'),
        (1, 2, 'Y', 'Z'),
    ]
    for idx, (x_idx, y_idx, xlabel, ylabel) in enumerate(projections):
        ax = fig.add_subplot(3, 3, idx + 1)
        ax.plot(positions[:, x_idx], positions[:, y_idx], 'b-', linewidth=1.5, alpha=0.7)
        ax.scatter([positions[0, x_idx]], [positions[0, y_idx]], c='green', s=80, marker='o', label='Start', zorder=10)
        ax.scatter([positions[-1, x_idx]], [positions[-1, y_idx]], c='red', s=80, marker='x', label='End', zorder=10)
        ax.set_xlabel(f'{xlabel} (m)'); ax.set_ylabel(f'{ylabel} (m)')
        ax.set_title(f'{xlabel}-{ylabel} Projection'); ax.legend(); ax.grid(True, alpha=0.3); ax.axis('equal')

    ax3d = fig.add_subplot(3, 3, (4, 6), projection='3d')
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
    ax3d.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], c='green', s=80, marker='o', label='Start', zorder=10)
    ax3d.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], c='red', s=80, marker='x', label='End', zorder=10)
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory'); ax3d.legend(); ax3d.grid(True, alpha=0.3)

    max_range = max(
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min(),
        0.1,
    )
    xc = (positions[:, 0].max() + positions[:, 0].min()) / 2
    yc = (positions[:, 1].max() + positions[:, 1].min()) / 2
    zc = (positions[:, 2].max() + positions[:, 2].min()) / 2
    ax3d.set_xlim(xc - max_range/2, xc + max_range/2)
    ax3d.set_ylim(yc - max_range/2, yc + max_range/2)
    ax3d.set_zlim(zc - max_range/2, zc + max_range/2)
    ax3d.view_init(elev=20, azim=45)

    ax_steps = fig.add_subplot(3, 3, (7, 9))
    steps = np.arange(len(positions))
    ax_steps.plot(steps, positions[:, 0], 'r-', linewidth=1.5, label='X')
    ax_steps.plot(steps, positions[:, 1], 'g-', linewidth=1.5, label='Y')
    ax_steps.plot(steps, positions[:, 2], 'b-', linewidth=1.5, label='Z')
    ax_steps.set_xlabel('Step'); ax_steps.set_ylabel('Position (m)')
    ax_steps.set_title('X, Y, Z vs Step'); ax_steps.legend(); ax_steps.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Saved TCP trajectory plot to {output_path}")
    plt.close()


def plot(result, out_dir, tcp_frame, title_suffix, file_prefix):
    """Plot 3D trajectory + 2D states (XYZ, RPY, Grip)."""
    sp, ap = result["sent_pos"], result["act_pos"]
    sr, ar = result["sent_rv"], result["act_rv"]
    gs, ga = result["g_sent"], result["g_act"]
    steps = np.arange(len(sp))

    se = R.from_rotvec(sr.reshape(-1, 3)).as_euler("xyz")
    ae = R.from_rotvec(ar.reshape(-1, 3)).as_euler("xyz")

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sp[:, 0], sp[:, 1], sp[:, 2], "b-o", label="Sent", ms=3)
    ax.plot(ap[:, 0], ap[:, 1], ap[:, 2], "r-s", label="Actual", ms=3)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    all_pts = np.vstack([sp, ap])
    mid = all_pts.mean(axis=0)
    half = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.05)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_title(f"3D {tcp_frame} Trajectory [{title_suffix}]"); ax.legend()
    plt.tight_layout(); fig.savefig(str(out_dir / f"{file_prefix}_traj_3d.png")); plt.close()

    # 2D plots
    labels = ["X(m)", "Y(m)", "Z(m)", "Roll", "Pitch", "Yaw", "Grip"]
    sd = np.column_stack([sp, se, gs])
    ad = np.column_stack([ap, ae, ga])
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    for a, lb, s, v in zip(axes, labels, sd.T, ad.T):
        a.plot(steps, s, "b-o", label="Sent", ms=3)
        a.plot(steps, v, "r-s", label="Actual", ms=3)
        a.set_ylabel(lb); a.legend()
        y_lo, y_hi = a.get_ylim()
        if y_hi - y_lo < 0.1:
            y_mid = (y_hi + y_lo) / 2
            a.set_ylim(y_mid - 0.05, y_mid + 0.05)
    axes[-1].set_xlabel("Step"); axes[0].set_title(f"{tcp_frame} State [{title_suffix}]")
    plt.tight_layout(); fig.savefig(str(out_dir / f"{file_prefix}_traj_2d_states.png")); plt.close()
    print(f"Saved plots to {out_dir}")


def plot_joints(result, out_dir, filename, joint_labels):
    """Plot joint commands vs observations. joint_labels: list of strings like ['J1', ...] or ['shoulder_pan', ...]."""
    cmd = result["cmd_joints"]
    obs = result["obs_joints"]
    num_joints = len(joint_labels)
    steps = np.arange(len(cmd))
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2 * num_joints), sharex=True)
    for j in range(num_joints):
        ax = axes[j]
        ax.plot(steps, cmd[:, j], "b-o", label="Command", ms=3)
        ax.plot(steps, obs[:, j], "r-s", label="Observed", ms=3)
        ax.set_ylabel(f"{joint_labels[j]} (deg)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        y_min = min(cmd[:, j].min(), obs[:, j].min())
        y_max = max(cmd[:, j].max(), obs[:, j].max())
        if y_max - y_min < 5.0:
            y_center = (y_max + y_min) / 2
            ax.set_ylim(y_center - 2.5, y_center + 2.5)

    axes[-1].set_xlabel("Step")
    axes[0].set_title("Joint Commands vs Observations")
    plt.tight_layout()
    fig.savefig(str(out_dir / filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved joints plot to {out_dir / filename}")


def save_result_to_csv(result, out_dir, filename):
    """Save trajectory execution result to CSV for analysis."""
    sp, ap = result["sent_pos"], result["act_pos"]
    sr, ar = result["sent_rv"], result["act_rv"]
    gs, ga = result["g_sent"], result["g_act"]
    cmd = result["cmd_joints"]
    obs = result["obs_joints"]
    num_joints = cmd.shape[1]

    sent_rpy = R.from_rotvec(sr.reshape(-1, 3)).as_euler("xyz", degrees=True)
    act_rpy = R.from_rotvec(ar.reshape(-1, 3)).as_euler("xyz", degrees=True)

    rows = []
    for i in range(len(sp)):
        row = {"step": i}
        for j in range(num_joints):
            row[f"cmd_j{j+1}"] = cmd[i, j]
            row[f"obs_j{j+1}"] = obs[i, j]
        row.update({
            "sent_x": sp[i, 0], "sent_y": sp[i, 1], "sent_z": sp[i, 2],
            "act_x": ap[i, 0], "act_y": ap[i, 1], "act_z": ap[i, 2],
            "sent_roll": sent_rpy[i, 0], "sent_pitch": sent_rpy[i, 1], "sent_yaw": sent_rpy[i, 2],
            "act_roll": act_rpy[i, 0], "act_pitch": act_rpy[i, 1], "act_yaw": act_rpy[i, 2],
            "cmd_gripper": gs[i], "act_gripper": ga[i],
            "pos_err_mm": np.linalg.norm(sp[i] - ap[i]) * 1000,
        })
        rows.append(row)

    csv_path = out_dir / filename
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved execution data to {csv_path}")
    return csv_path
