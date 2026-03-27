#!/usr/bin/env python
"""Minimal script to execute a trajectory on a LIBERO environment (absolute mode)."""

from pathlib import Path

import numpy as np
from lerobot.envs.libero import LiberoEnv, _get_suite

# --- Config ---
SUITE = "libero_object"
TASK_ID = 0

# --- Create environment ---
suite = _get_suite(SUITE)
env = LiberoEnv(
    task_suite=suite, task_id=TASK_ID, task_suite_name=SUITE,
    obs_type="pixels_agent_pos", control_mode="absolute",
)
print(f"Task: {env.task_description}")
print(f"Action space: {env.action_space}")

# --- Reset and read initial EE pose ---
obs, info = env.reset()
eef = obs["robot_state"]["eef"]["pos"]  # initial [x, y, z]
print(f"Initial EE pos: {eef}")

# --- Build trajectory: move down incrementally, then close gripper ---
traj = (
    [np.array([eef[0], eef[1], eef[2] - 0.02 * i, 0, 0, 0, -1], dtype=np.float32) for i in range(20)]
    + [np.array([eef[0], eef[1], eef[2] - 0.4, 0, 0, 0, 1], dtype=np.float32) for _ in range(10)]
)

# --- Execute ---
total_reward = 0.0
frames = []
for i, action in enumerate(traj):
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    frames.append(env.render())
    success = info.get("is_success", False)
    print(f"Step {i}: reward={reward:.3f} success={success}")
    if terminated or success:
        print(f"Episode done at step {i}")
        break

print(f"\nTotal reward: {total_reward:.3f}")
print(f"Success: {info.get('is_success', False)}")

# --- Save video ---
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
video_path = output_dir / "libero_traj_test_output.mp4"
if frames:
    import imageio
    imageio.mimsave(str(video_path), frames, fps=20)
    print(f"Saved {len(frames)} frames to {video_path}")

env.close()
