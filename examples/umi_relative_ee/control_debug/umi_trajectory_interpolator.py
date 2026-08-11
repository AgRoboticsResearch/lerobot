"""UMI-style Cartesian trajectory interpolator, ported for 7D relative-EE poses.

This is a faithful port of ``real-stanford/universal_manipulation_interface``'s
``PoseTrajectoryInterpolator`` (diffusion_policy/common/pose_trajectory_interpolator.py),
extended from 6D ``[x,y,z,rotvec]`` to our 7D ``[x,y,z,rotvec,gripper]``:

  * translation ``[x,y,z]``  -> ``scipy.interpolate.interp1d`` (linear, like UMI)
  * rotation   ``rotvec``    -> ``scipy.spatial.transform.Slerp``  (like UMI)
  * gripper    ``[6]``       -> ``scipy.interpolate.interp1d`` (linear; UMI treats the
                               gripper as a separate 1-D width interpolator)

The key behaviour copied verbatim is ``schedule_waypoint``: when a new chunk arrives,
the interpolator trims its *currently commanded* future and splices the new waypoint
onto the value of the **commanded trajectory** at the splice time (``trimmed_interp(end_time)``),
NOT the measured robot pose. That is UMI's anti-jump mechanism -- the comment in the
original Franka controller reads: *"since curr_pose always lag behind curr_target_pose,
if we start the next interpolation with curr_pose the command robot receive will have
discontinuity and cause jittery robot behavior."*

Like the original, ``__call__`` clips the requested time to ``[start, end]`` so a
stalled inference holds the last pose instead of extrapolating (C0, not C1/C2 -- there
is no cubic/quintic spline, so velocity can be discontinuous at waypoints).

Reference:
https://github.com/real-stanford/universal_manipulation_interface/blob/main/diffusion_policy/common/pose_trajectory_interpolator.py
"""

from __future__ import annotations

import numbers

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()


def pose_distance(start_pose, end_pose):
    """Return (xyz distance [m], rotation distance [rad]) between two 6D poses."""
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    pos_dist = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
    rot_dist = rotation_distance(
        st.Rotation.from_rotvec(start_pose[3:6]),
        st.Rotation.from_rotvec(end_pose[3:6]),
    )
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator7D:
    """Timestamped 7D EE-pose interpolator (linear XYZ + SLERP rot + linear gripper).

    ``poses`` rows are ``[x, y, z, rx, ry, rz, gripper]`` with ``rx,ry,rz`` an
    axis-angle rotation vector (the same convention our relative-EE postprocessor
    emits and UMI consumes). ``times`` must be non-decreasing.
    """

    def __init__(self, times, poses):
        if not isinstance(times, np.ndarray):
            times = np.array(times, dtype=np.float64)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses, dtype=np.float64)
        if times.ndim != 1 or len(times) < 1:
            raise ValueError(f"times must be a non-empty 1D array, got shape {times.shape}")
        if poses.shape != (len(times), 7):
            raise ValueError(f"poses must have shape ({len(times)}, 7), got {poses.shape}")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(poses)):
            raise ValueError("times and poses must be finite")
        if np.any(times[1:] < times[:-1]):
            raise ValueError("times must be non-decreasing")
        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            # Slerp requires STRICTLY increasing times. Keep the last pose at each
            # repeated timestamp: a newly scheduled target supersedes the stale one.
            if len(np.unique(times)) != len(times):
                keep = np.concatenate([np.diff(times) > 0, [True]])
                times = times[keep]
                poses = poses[keep]
                if len(times) == 1:
                    self.single_step = True
                    self._times = times
                    self._poses = poses
                    return
            self.single_step = False
            self._times = times
            self._poses = poses
            pos = poses[:, :3]
            grip = poses[:, 6:7]
            rot = st.Rotation.from_rotvec(poses[:, 3:6])
            self.pos_interp = si.interp1d(times, pos, axis=0, assume_sorted=True)
            self.grip_interp = si.interp1d(times, grip, axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)

    @property
    def times(self) -> np.ndarray:
        return self._times if self.single_step else self.pos_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        t = self.times
        poses = np.zeros((len(t), 7))
        poses[:, :3] = self.pos_interp.y
        poses[:, 3:6] = self.rot_interp(t).as_rotvec()
        poses[:, 6:7] = self.grip_interp.y
        return poses

    def trim(self, start_t: float, end_t: float) -> PoseTrajectoryInterpolator7D:
        assert start_t <= end_t
        times = self.times
        keep = (start_t < times) & (times < end_t)
        all_times = np.unique(np.concatenate([[start_t], times[keep], [end_t]]))
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator7D(all_times, all_poses)

    def drive_to_waypoint(self, pose, time, curr_time, max_pos_speed=np.inf, max_rot_speed=np.inf):
        assert max_pos_speed > 0 and max_rot_speed > 0
        time = max(time, curr_time)
        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        duration = max(time - curr_time, pos_dist / max_pos_speed, rot_dist / max_rot_speed)
        last_wp_time = curr_time + duration
        trimmed = self.trim(curr_time, curr_time)
        times = np.append(trimmed.times, [last_wp_time], axis=0)
        poses = np.append(trimmed.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator7D(times, poses)

    def schedule_waypoint(
        self, pose, time, curr_time=None, last_waypoint_time=None, max_pos_speed=np.inf, max_rot_speed=np.inf
    ):
        """Splice a new waypoint onto the *commanded* trajectory (UMI semantics).

        Mirrors the original algorithm exactly (incl. the zhenjia min-operations that
        guarantee ``start_time <= end_time <= time``). Returns a NEW interpolator.
        """
        assert max_pos_speed > 0 and max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None
        start_time = self.times[0]
        end_time = self.times[-1]
        if curr_time is not None:
            if time <= curr_time:
                return self  # inserting in the past: no-op
            start_time = max(curr_time, start_time)
            if last_waypoint_time is not None:
                end_time = curr_time if time <= last_waypoint_time else max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time
        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        trimmed = self.trim(start_time, end_time)
        duration = time - end_time
        end_pose = trimmed(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        duration = max(duration, pos_dist / max_pos_speed, rot_dist / max_rot_speed)
        last_wp_time = end_time + duration
        times = np.append(trimmed.times, [last_wp_time], axis=0)
        poses = np.append(trimmed.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator7D(times, poses)

    def __call__(self, t):
        is_single = isinstance(t, numbers.Number)
        t = np.array([t], dtype=np.float64) if is_single else np.asarray(t, dtype=np.float64)
        n = len(t)
        pose = np.zeros((n, 7))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            t = np.clip(t, self.times[0], self.times[-1])
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:6] = self.rot_interp(t).as_rotvec()
            pose[:, 6:7] = self.grip_interp(t)
        return pose[0] if is_single else pose
