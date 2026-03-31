#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on a LIBERO environment.

The CSV trajectory is in the relative EE frame (e.g. from RelativeEEDataset),
with step 0 at the identity pose [0,0,0,0,0,0]. This script composes each relative
pose with the LIBERO env's initial EE pose using SE(3): T_target = T_base @ T_rel.

Supports two control modes via --mode:
  - delta:   Converts absolute targets to per-step scaled deltas. Tracks well.
  - absolute: Sends absolute axis-angle poses directly. Limited by OSC nullspace drift.

Usage:
    python libero_traj_test.py --traj-csv test_x_axis.csv --mode delta
    python libero_traj_test.py --traj-csv test_x_axis.csv --mode absolute --steps 50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from lerobot.envs.libero import LiberoEnv, _get_suite


# ============================================================
# Helpers
# ============================================================

def make_transform(pos, rotvec):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = pos
    return T


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-csv", required=True)
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--mode", default="absolute", choices=["absolute", "delta"])
    return p.parse_args()


# ============================================================
# CSV -> absolute poses
# ============================================================

def load_trajectory(csv_path, T_base, max_steps=None):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} steps from {csv_path}")

    traj = []
    for _, row in df.iterrows():
        # Use action columns for the command trajectory; fall back to state columns
        if "action.ee.x" in df.columns:
            rel_pos = [row["action.ee.x"], row["action.ee.y"], row["action.ee.z"]]
            rel_rv = [row.get("action.ee.wx", 0), row.get("action.ee.wy", 0), row.get("action.ee.wz", 0)]
            gripper = row["action.ee.gripper_pos"]
        else:
            rel_pos = [row["state.ee.x"], row["state.ee.y"], row["state.ee.z"]]
            rel_rv = [row.get("state.ee.wx", 0), row.get("state.ee.wy", 0), row.get("state.ee.wz", 0)]
            gripper = row["state.ee.gripper_pos"]

        T_t = T_base @ make_transform(rel_pos, rel_rv)
        abs_pos = T_t[:3, 3]
        abs_rv = R.from_matrix(T_t[:3, :3]).as_rotvec()
        # CSV gripper ~1=open ~0=closed -> LIBERO -1=open 1=closed
        g_lib = -(gripper * 2 - 1)

        traj.append(np.array([*abs_pos, *abs_rv, g_lib], dtype=np.float32))

    if max_steps:
        traj = traj[:max_steps]
    print(f"Composed {len(traj)} absolute poses")
    return traj


# ============================================================
# Absolute -> Delta conversion
# ============================================================

POS_RANGE = 0.05
ORI_RANGE = 0.5


def abs_to_delta(abs_traj):
    delta = []
    for i in range(len(abs_traj)):
        T_cur = make_transform(abs_traj[i][:3], abs_traj[i][3:6])
        T_prev = make_transform(abs_traj[i - 1][:3], abs_traj[i - 1][3:6]) if i > 0 else T_cur

        T_d = np.linalg.inv(T_prev) @ T_cur
        dp = np.clip(T_d[:3, 3] / POS_RANGE, -1, 1)
        dr = np.clip(R.from_matrix(T_d[:3, :3]).as_rotvec() / ORI_RANGE, -1, 1)
        delta.append(np.array([*dp, *dr, abs_traj[i][6]], dtype=np.float32))
    return delta


# ============================================================
# Environment
# ============================================================

def create_env(suite_name, task_id, mode):
    suite = _get_suite(suite_name)
    control = "absolute" if mode == "absolute" else "relative"
    env = LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name=suite_name,
        obs_type="pixels_agent_pos", control_mode=control,
    )
    if mode == "delta":
        for robot in env._env.robots:
            robot.controller.use_delta = True

    obs, info = env.reset()
    bp = obs["robot_state"]["eef"]["pos"]
    # Use the controller's ee_ori_mat (not the raw quat) to get the correct
    # orientation frame that the OSC controller actually tracks.
    eef_mat = obs["robot_state"]["eef"]["mat"]
    T_base = make_transform(bp, R.from_matrix(eef_mat).as_rotvec())
    print(f"Task: {env.task_description}")
    print(f"Mode: {mode} | Action space: {env.action_space}")
    print(f"Base pos: {bp}")
    return env, T_base


# ============================================================
# Execution
# ============================================================

def run_trajectory(env, traj):
    reward_sum = 0.0
    frames, sent_p, sent_r, act_p, act_r = [], [], [], [], []
    g_sent, g_act = [], []

    for i, action in enumerate(traj):
        sent_p.append(action[:3].copy())
        sent_r.append(action[3:6].copy())
        g_sent.append(action[6])

        obs, r, done, trunc, info = env.step(action)
        reward_sum += r
        frames.append(env.render())

        act_p.append(obs["robot_state"]["eef"]["pos"].copy())
        act_mat = obs["robot_state"]["eef"]["mat"]
        act_r.append(R.from_matrix(act_mat).as_rotvec())
        # Normalize gripper qpos [0, 0.04] -> [-1, 1] (matching action space)
        g_qpos = obs["robot_state"]["gripper"]["qpos"][0]
        g_act.append(-(g_qpos / 0.04 * 2 - 1))

        print(f"  Step {i:3d}: r={r:.3f}")
        print(f"    Sent: pos={action[:3].tolist()} rotvec={action[3:6].tolist()} grip={action[6]:.3f}")
        print(f"    True: pos={obs['robot_state']['eef']['pos'].tolist()} rotvec={R.from_matrix(act_mat).as_rotvec().tolist()} grip={g_qpos:.5f}")
        if done:
            print(f"  Done at step {i}")
            break

    print(f"\nTotal reward: {reward_sum:.3f}")
    result = dict(
        sent_pos=np.array(sent_p), sent_rv=np.array(sent_r), g_sent=np.array(g_sent),
        act_pos=np.array(act_p), act_rv=np.array(act_r), g_act=np.array(g_act),
    )
    return result, frames


# ============================================================
# Save & Plot
# ============================================================

def save_video(frames, out_dir, fps=20):
    import imageio
    p = out_dir / "libero_traj_test_output.mp4"
    imageio.mimsave(str(p), frames, fps=fps)
    print(f"Saved {len(frames)} frames -> {p}")


def plot(result, out_dir, mode):
    sp, ap = result["sent_pos"], result["act_pos"]
    sr, ar = result["sent_rv"], result["act_rv"]
    gs, ga = result["g_sent"], result["g_act"]
    steps = np.arange(len(sp))

    se = R.from_rotvec(sr.reshape(-1, 3)).as_euler("xyz")
    ae = R.from_rotvec(ar.reshape(-1, 3)).as_euler("xyz")

    # 3D — use equal axis ranges to avoid distortion
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sp[:, 0], sp[:, 1], sp[:, 2], "b-o", label="Sent", ms=3)
    ax.plot(ap[:, 0], ap[:, 1], ap[:, 2], "r-s", label="Actual", ms=3)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    # Equalize axis ranges so visual distances are proportional
    all_pts = np.vstack([sp, ap])
    mid = all_pts.mean(axis=0)
    half = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.05)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_title(f"3D EE Trajectory [{mode}]"); ax.legend()
    plt.tight_layout(); fig.savefig(str(out_dir / "traj_3d.png")); plt.close()

    # 2D
    if mode == "delta":
        labels = ["X(m)", "Y(m)", "Z(m)", "dOriX", "dOriY", "dOriZ", "Grip"]
    else:
        labels = ["X(m)", "Y(m)", "Z(m)", "Roll", "Pitch", "Yaw", "Grip"]
    sd = np.column_stack([sp, se, gs])
    ad = np.column_stack([ap, ae, ga])
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    for a, lb, s, v in zip(axes, labels, sd.T, ad.T):
        a.plot(steps, s, "b-o", label="Sent", ms=3)
        a.plot(steps, v, "r-s", label="Actual", ms=3)
        a.set_ylabel(lb); a.legend()
        # Enforce minimum axis range of 0.1 for position axes
        if a.get_ylim()[1] - a.get_ylim()[0] < 0.1:
            mid = (a.get_ylim()[0] + a.get_ylim()[1]) / 2
            a.set_ylim(mid - 0.05, mid + 0.05)
    axes[-1].set_xlabel("Step"); axes[0].set_title(f"EE State [{mode}]")
    plt.tight_layout(); fig.savefig(str(out_dir / "traj_2d_states.png")); plt.close()
    print(f"Saved plots to {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    env, T_base = create_env(args.suite, args.task_id, args.mode)
    abs_traj = load_trajectory(args.traj_csv, T_base, args.steps)
    traj = abs_to_delta(abs_traj) if args.mode == "delta" else abs_traj

    result, frames = run_trajectory(env, traj)
    if frames:
        save_video(frames, out)
    plot(result, out, args.mode)
    env.close()


if __name__ == "__main__":
    main()
