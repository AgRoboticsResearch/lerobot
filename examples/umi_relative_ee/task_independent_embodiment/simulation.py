"""Synchronous, simulation-only control. No hardware interfaces are imported."""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from .core import Config, interpolate, pose_features, rotation_errors


class CommandDelay:
    def __init__(self, ticks: int, initial: np.ndarray):
        if ticks < 0:
            raise ValueError("Delay must be nonnegative")
        self.queue = deque([initial.copy() for _ in range(ticks)])

    def push(self, command: np.ndarray) -> np.ndarray:
        self.queue.append(command.copy())
        return self.queue.popleft()


class IK:
    def __init__(self, cfg: Config):
        from piper_sim.model import ARM_JOINT_NAMES, packaged_urdf_path

        from lerobot.model.kinematics import RobotKinematics
        from lerobot.processor import RobotProcessorPipeline
        from lerobot.processor.converters import (
            robot_action_observation_to_transition,
            transition_to_robot_action,
        )
        from lerobot.robots.so_follower.robot_kinematic_processor import (
            EEBoundsAndSafety,
            InverseKinematicsEEToJoints,
        )

        self.names = ARM_JOINT_NAMES
        self.kinematics = RobotKinematics(str(packaged_urdf_path()), "camera_link", self.names)
        self.safety = EEBoundsAndSafety(
            end_effector_bounds={"min": cfg.lower_xyz, "max": cfg.upper_xyz},
            max_ee_step_m=cfg.max_ee_step,
        )
        self.pipeline = RobotProcessorPipeline(
            steps=[
                self.safety,
                InverseKinematicsEEToJoints(
                    kinematics=self.kinematics,
                    motor_names=self.names,
                    initial_guess_current_joints=True,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

    def reset(self):
        self.pipeline.reset()

    def fk(self, q):
        return self.kinematics.forward_kinematics(np.rad2deg(q)).copy()

    def command(self, pose, q):
        from scipy.spatial.transform import Rotation

        values = [*pose[:3, 3], *Rotation.from_matrix(pose[:3, :3]).as_rotvec(), 0.0]
        action = dict(
            zip(("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"), values, strict=False)
        )
        obs = {f"{name}.pos": float(v) for name, v in zip(self.names, np.rad2deg(q), strict=False)}
        output = self.pipeline((action, obs))
        adjusted = pose.copy()
        adjusted[:3, 3] = self.safety.last_position
        return np.deg2rad([output[f"{name}.pos"] for name in self.names]), adjusted


class Simulation:
    def __init__(self, cfg: Config, condition: str):
        import mujoco
        from piper_sim.config import ServerConfig
        from piper_sim.physics import SimCore

        if condition not in ("native", "delay40"):
            raise ValueError(condition)
        if cfg.physics_hz % cfg.control_hz or cfg.control_hz != 50 or cfg.ee_hz != 30:
            raise ValueError("This protocol requires 500 Hz physics, 50 Hz commands, and 30 Hz EE")
        self.mj, self.cfg, self.condition = mujoco, cfg, condition
        self.core = SimCore(ServerConfig(headless=True, max_joint_vel_deg_s=cfg.max_joint_vel_deg_s))
        if not np.isclose(self.core.model.opt.timestep, 1 / cfg.physics_hz):
            raise ValueError("Simulator timestep differs from protocol")
        self.substeps = cfg.physics_hz // cfg.control_hz
        self.site = mujoco.mj_name2id(self.core.model, mujoco.mjtObj.mjOBJ_SITE, "camera_link")
        if self.site < 0:
            raise ValueError("Packaged simulator has no camera_link site")
        self.low, self.high = self.core.info.arm_jnt_range_rad.T.copy()
        self.ik = IK(cfg)

    def state(self):
        core = self.core
        # mj_step advances qpos after computing site transforms. Refresh before
        # recording so the achieved pose and encoders refer to the SAME time.
        self.mj.mj_forward(core.model, core.data)
        pose = np.eye(4)
        pose[:3, 3] = core.data.site_xpos[self.site]
        pose[:3, :3] = core.data.site_xmat[self.site].reshape(3, 3)
        return (
            float(core.data.time),
            core.data.qpos[core.info.arm_qposadr].copy(),
            core.data.qvel[core.info.arm_dofadr].copy(),
            pose,
        )

    def reset(self, q):
        core = self.core
        self.mj.mj_resetData(core.model, core.data)
        core.data.qpos[core.info.arm_qposadr] = q
        core.goal_deg = np.rad2deg(q).copy()
        core.target_deg = core.goal_deg.copy()
        core.gripper_enabled = False
        core.gripper_zero = 0.0
        core.builtin_goal_mm = core.builtin_current_mm = 0.0
        self.mj.mj_forward(core.model, core.data)
        core.connect("task-independent-embodiment")
        # Identical settling and queue initialization for every method and seed.
        for _ in range(self.cfg.physics_hz // 2):
            core.step()
        core.data.time = 0.0
        self.delay = CommandDelay(2 if self.condition == "delay40" else 0, q)
        self.ik.reset()

    def advance(self, command, gripper=0.0):
        invalid = not np.isfinite(command).all()
        raw = self.state()[1] if invalid else np.asarray(command)
        issued = np.clip(raw, self.low, self.high)
        clamped = bool(np.any(np.abs(raw - issued) > 1e-10))
        delivered = self.delay.push(issued)
        self.core.set_joint_targets(np.rad2deg(delivered))
        self.core.gripper_mit_command(5.0, 0.5, -0.91 * float(np.clip(gripper, 0, 1)), 0.0, 0.0)
        for _ in range(self.substeps):
            self.core.step()
        return issued, delivered, clamped, invalid

    def sample_start(self, rng):
        margin = np.deg2rad(self.cfg.joint_margin_deg)
        for _ in range(5000):
            q = rng.uniform(self.low + margin, self.high - margin)
            xyz = self.ik.fk(q)[:3, 3]
            if np.all(xyz >= self.cfg.lower_xyz) and np.all(xyz <= self.cfg.upper_xyz):
                return q
        raise RuntimeError("Could not sample a start within configured workspace")

    def feasibility(self, times, poses, start):
        """Method-independent nominal sequential IK classification on the 50 Hz grid."""
        ticks = np.arange(int(np.floor(times[-1] * self.cfg.control_hz)) + 1) / self.cfg.control_hz
        desired, _ = interpolate(times, poses, ticks)
        q, max_p, max_r, max_v = start.copy(), 0.0, 0.0, 0.0
        bounds = bool(
            np.all(desired[:, :3, 3] >= self.cfg.lower_xyz)
            and np.all(desired[:, :3, 3] <= self.cfg.upper_xyz)
        )
        limits, error = True, None
        self.ik.reset()
        try:
            for pose in desired:
                target, _ = self.ik.command(pose, q)
                actual = self.ik.fk(target)
                max_p = max(max_p, float(np.linalg.norm(actual[:3, 3] - pose[:3, 3])))
                max_r = max(max_r, float(rotation_errors(actual[None], pose[None])[0]))
                max_v = max(max_v, float(np.rad2deg(np.max(np.abs(target - q))) * self.cfg.control_hz))
                limits &= bool(np.all(target >= self.low) and np.all(target <= self.high))
                q = target
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        self.ik.reset()
        return {
            "ik_feasible": bool(
                bounds
                and limits
                and error is None
                and max_p <= 0.005
                and max_r <= 3
                and max_v <= self.cfg.max_joint_vel_deg_s
            ),
            "workspace_valid": bounds,
            "joint_limits_valid": limits,
            "ik_error": error,
            "max_position_residual_m": max_p,
            "max_rotation_residual_deg": max_r,
            "max_nominal_velocity_deg_s": max_v,
        }

    def rollout(self, motion, start, controller=None, record_nominal=False):
        cfg = self.cfg
        self.reset(start)
        records = [self.state()]
        issued, delivered, clamp, invalid, adjusted, nominal, errors, latencies, grips = (
            [] for _ in range(9)
        )

        def tick(command, grip, adjusted_pose, nominal_command, error, latency):
            u, d, c, bad = self.advance(command, grip)
            issued.append(u)
            delivered.append(d)
            clamp.append(c)
            invalid.append(bad)
            adjusted.append(adjusted_pose)
            nominal.append(nominal_command)
            errors.append(error)
            latencies.append(latency)
            grips.append(self.core.gripper_pos_rad())
            records.append(self.state())

        for _ in range(cfg.history - 1):
            tick(start, motion["gripper"][0], records[-1][3], start, "", 0.0)
        anchor = records[-1][3].copy()
        query = anchor @ motion["relative"]
        query_time = records[-1][0] + motion["times"]
        feasibility = self.feasibility(motion["times"], query, records[-1][1])
        # Final tick never exceeds the original chunk's last timestamp.
        count = int(np.floor(motion["times"][-1] * cfg.control_hz))
        for _ in range(count):
            t, q, _, _ = records[-1]
            future, valid = interpolate(query_time, query, t + np.arange(cfg.horizon) / cfg.ee_hz)
            adjusted_pose = future[0].copy()
            nominal_command = np.zeros(6)
            error = ""
            began = time.perf_counter()
            try:
                if controller is None:
                    command, adjusted_pose = self.ik.command(future[0], q)
                else:
                    state = np.stack([np.concatenate((r[1], r[2])) for r in records[-cfg.history :]])
                    past = np.asarray(issued[-(cfg.history - 1) :])
                    command = controller(pose_features(future), valid, state, past, future[0], self.ik)
                    if controller.method == "residual":
                        adjusted_pose[:3, 3] = self.ik.safety.last_position
            except (ValueError, RuntimeError) as exc:
                command, error = q.copy(), str(exc)
            elapsed = time.perf_counter() - began
            grip = np.interp(t, query_time, motion["gripper"])
            tick(command, grip, adjusted_pose, nominal_command, error, elapsed)

        result = {
            "time": np.asarray([r[0] for r in records]),
            "q": np.asarray([r[1] for r in records]),
            "qd": np.asarray([r[2] for r in records]),
            "actual": np.asarray([r[3] for r in records]),
            "issued": np.asarray(issued),
            "delivered": np.asarray(delivered),
            "clamped": np.asarray(clamp),
            "invalid": np.asarray(invalid),
            "adjusted": np.asarray(adjusted),
            "errors": np.asarray(errors),
            "latency_s": np.asarray(latencies),
            "gripper_actual_rad": np.asarray(grips),
            "query": query,
            "query_time": query_time,
            "anchor": anchor,
            "start": start,
            "encoder_fk": np.asarray([self.ik.fk(q) for q in [r[1] for r in records]]),
            "gripper_query": motion["gripper"],
        }
        if record_nominal:
            # Counterfactual IK evaluated on ACHIEVED poses, seeded only with q at that tick.
            self.ik.reset()
            values, nominal_valid = [], []
            for q, pose in zip(result["q"][:-1], result["actual"][:-1], strict=False):
                try:
                    command, _ = self.ik.command(pose, q)
                    ok = bool(np.isfinite(command).all())
                except (ValueError, RuntimeError):
                    command, ok = q.copy(), False
                values.append(command)
                nominal_valid.append(ok)
            result["nominal_actual"] = np.asarray(values)
            result["nominal_valid"] = np.asarray(nominal_valid)
        return result, feasibility


def rollout_metrics(rollout: dict, cfg: Config) -> dict:
    begin = cfg.history - 1
    actual = rollout["actual"][begin:]
    desired, _ = interpolate(rollout["query_time"], rollout["query"], rollout["time"][begin:])
    position = np.linalg.norm(actual[:, :3, 3] - desired[:, :3, 3], axis=-1)
    rotation = rotation_errors(actual, desired)
    commands = rollout["issued"][begin:]
    failed = bool(rollout["invalid"][begin:].any() or np.any(rollout["errors"][begin:] != ""))
    latency = rollout["latency_s"][begin:]
    gripper_target = np.interp(
        rollout["time"][begin + 1 :],
        rollout["query_time"],
        rollout.get("gripper_query", np.zeros(len(rollout["query_time"]))),
    )
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position**2))),
        "position_end_m": float(position[-1]),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_end_deg": float(rotation[-1]),
        "command_acceleration_rad_s2": float(
            np.mean(np.abs(np.diff(commands, n=2, axis=0))) * cfg.control_hz**2
        ),
        "joint_limit_rate": float(rollout["clamped"][begin:].mean()),
        "invalid_output_rate": float(rollout["invalid"][begin:].mean()),
        "tracking_failed": float(failed or position.max() > 0.05 or rotation.max() > 15),
        "execution_error": float(failed),
        "latency_p95_ms": float(np.percentile(latency, 95) * 1000),
        "deadline_miss_rate": float(np.mean(latency > 1 / cfg.control_hz)),
        "gripper_rmse_rad": float(
            np.sqrt(np.mean((rollout["gripper_actual_rad"][begin:] + 0.91 * gripper_target) ** 2))
        ),
    }
