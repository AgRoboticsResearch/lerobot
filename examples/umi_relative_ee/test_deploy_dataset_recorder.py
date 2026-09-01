#!/usr/bin/env python
"""Smoke test for ``deploy_dataset_recorder`` (the ``--save_dataset`` feature).

Hardware-free: FakeCameras + synthetic ``diag`` dicts drive the SHARED recorder
through the same lifecycle both deploy loops use (``record_tick`` on executed
ticks, ``note_loop_state`` every tick, ``close`` at shutdown), then the dataset
is reloaded with :class:`LeRobotDataset` and round-tripped. Run from the repo
root:

  uv run python examples/umi_relative_ee/test_deploy_dataset_recorder.py
"""

from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

try:
    from examples.umi_relative_ee.deploy_dataset_recorder import make_deploy_dataset_recorder
except ModuleNotFoundError:
    # Supports direct execution from inside examples/umi_relative_ee.
    from deploy_dataset_recorder import make_deploy_dataset_recorder  # type: ignore[no-redef]

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"This smoke test needs the dataset stack (pyarrow/av/PIL): {exc}\n"
        "Try: uv sync --locked --extra dataset"
    ) from exc

H, W = 48, 64
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


class FakeCam:
    """Minimal camera stand-in: factory probes one read() for (H, W)."""

    def __init__(self, height: int, width: int) -> None:
        self.frame = np.zeros((height, width, 3), np.uint8)

    def read(self) -> np.ndarray:
        return self.frame


def _diag(i: int, *, ik_ok: bool = True) -> dict:
    """Synthetic per-tick diag dict in the shared control-logger notation."""
    action = np.array([0.30 + 0.001 * i, -0.10, 0.20, 0.01, -0.02, 0.03, 0.5], np.float32)
    return {
        "popped": True,
        "ik_ok": ik_ok,
        "skip_reason": None if ik_ok else "ik_failed",
        "action_timestep": i,
        "action_ee": action,
        "chunk_id": 1 + i // 30,
        "chunk_ref_ee": np.full(7, 0.25, np.float32),
        "action_abs": action,  # pre-ensemble == executed in sync mode
        "action_agg": action,
        "action_rel": np.linspace(-0.5, 0.5, 10).astype(np.float32) if ik_ok else None,
        "ik_joints_rad": np.full(6, 5.0 + i) if ik_ok else None,  # degrees (misnomer)
    }


def _fresh_images(i: int) -> dict[str, np.ndarray]:
    # New arrays each tick (like camera.read()), never in-place mutation: the
    # async PNG writer may still hold the previous reference.
    return {name: np.full((H, W, 3), 40 + 8 * i, np.uint8) for name in ("cam", "cam2")}


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="deploy_ds_smoke_"))
    print(f"tmp root: {tmp}")

    # ── 1. Flag off → factory returns None (deploy loop untouched) ──────────
    assert (
        make_deploy_dataset_recorder(
            Namespace(save_dataset=None),
            prefix="sync",
            cameras={},
            fps=30,
            arm_joint_names=ARM_JOINTS,
        )
        is None
    )

    # ── 2. Factory + lazy create ────────────────────────────────────────────
    root = tmp / "ds"
    cameras = {"cam": FakeCam(H, W), "cam2": FakeCam(H, W)}
    rec = make_deploy_dataset_recorder(
        Namespace(save_dataset=str(root)),
        prefix="sync",
        cameras=cameras,
        fps=30,
        arm_joint_names=ARM_JOINTS,
    )
    assert rec is not None
    rec.note_loop_state("PAUSED")
    rec.note_loop_state("INFERENCE")
    assert not root.exists(), "dataset dir must not exist before the first executed tick"

    # ── 3. Episode 1: five executed ticks ───────────────────────────────────
    joints_deg = np.full(6, 10.0)
    ee_pose = np.array([0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.75], np.float32)
    for i in range(5):
        images = _fresh_images(i)
        for name, frame in images.items():
            cameras[name].frame = frame  # keep the probe camera in sync (unused later)
        rec.record_tick(
            images=images,
            current_joints=joints_deg,
            current_ee=ee_pose,
            diag=_diag(i),
            gripper_raw=-0.45,
        )
    rec.note_loop_state("PAUSED")  # disengage → save_episode (mp4 encode)
    assert root.exists(), "dataset dir must exist after the first saved episode"
    assert len(list(root.glob("videos/**/*.mp4"))) == 2, "one mp4 per camera"

    # ── 4. Episode 2: two ok ticks + one IK-failed tick; close is idempotent ─
    for i in range(5, 7):
        rec.record_tick(
            images=_fresh_images(i),
            current_joints=joints_deg,
            current_ee=ee_pose,
            diag=_diag(i),
            gripper_raw=-0.46,
            latencies={"e2e_ms": 12.5, "wire_ms": 40.0, "server_ms": 30.0},
        )
    rec.record_tick(  # IK-failed tick (async invalid-action equivalent)
        images=_fresh_images(7),
        current_joints=joints_deg,
        current_ee=ee_pose,
        diag=_diag(7, ik_ok=False),
        gripper_raw=-0.46,
    )
    rec.note_loop_state("PAUSED")
    rec.close()
    rec.close()  # idempotent

    # ── 5. Reload + round-trip ──────────────────────────────────────────────
    ds = LeRobotDataset(repo_id=root.name, root=root)
    assert len(ds) == 8, f"expected 8 frames, got {len(ds)}"
    assert ds.meta.total_episodes == 2

    f0 = ds[0]
    assert np.allclose(np.asarray(f0["observation.state"]), np.deg2rad(joints_deg), atol=1e-6), (
        "observation.state must be radians (converted from the loop's degrees)"
    )
    assert np.allclose(np.asarray(f0["observation.ee_pose"]), ee_pose, atol=1e-6)
    assert np.allclose(np.asarray(f0["action"]), _diag(0)["action_agg"], atol=1e-5)
    assert np.allclose(np.asarray(f0["action.joints"]), np.deg2rad(_diag(0)["ik_joints_rad"]), atol=1e-5)
    assert abs(float(np.asarray(f0["observation.gripper_position"])) - (-0.45)) < 1e-6
    assert float(np.asarray(f0["chunk_id"])) == 1.0
    assert bool(np.asarray(f0["action.executed_ok"]).reshape(-1)[0])
    assert np.isnan(np.asarray(f0["e2e_ms"])).all(), "no latencies passed → NaN (sync mode)"
    img = np.asarray(f0["observation.images.cam"])
    # Video frames decode as float32 CHW in [0, 1]; tick 0 wrote uniform gray 40.
    assert img.shape == (3, H, W)
    assert abs(float(img.mean()) - 40 / 255) < 0.02, (
        "frame content should survive the (lossy) h264 round-trip"
    )

    f5 = ds[5]
    assert np.allclose(
        [float(np.asarray(f5[k])) for k in ("e2e_ms", "wire_ms", "server_ms")], [12.5, 40.0, 30.0], atol=1e-4
    ), "latency columns must round-trip"

    f7 = ds[7]
    assert np.isnan(np.asarray(f7["action.joints"])).all(), "IK-failed tick → NaN joints"
    assert np.isnan(np.asarray(f7["action.relative"])).all(), "IK-failed tick → NaN relative"
    assert not bool(np.asarray(f7["action.executed_ok"]).reshape(-1)[0])
    assert np.allclose(np.asarray(f7["action"]), _diag(7)["action_agg"], atol=1e-5)

    assert tuple(ds.meta.features["action"]["shape"]) == (7,)
    assert "timestamp" in ds.meta.features, "auto-added default features present"

    # ── 6. Factory error cases ──────────────────────────────────────────────
    for bad_name in ("a/b", "bad name", ""):
        try:
            make_deploy_dataset_recorder(
                Namespace(save_dataset=str(tmp / "never")),
                prefix="sync",
                cameras={bad_name: FakeCam(H, W)},
                fps=30,
                arm_joint_names=ARM_JOINTS,
            )
            raise AssertionError(f"camera name {bad_name!r} must be rejected")
        except ValueError:
            pass
    try:
        make_deploy_dataset_recorder(
            Namespace(save_dataset=str(root)),
            prefix="sync",
            cameras={"cam": FakeCam(H, W)},
            fps=30,
            arm_joint_names=ARM_JOINTS,
        )
        raise AssertionError("existing root must be rejected")
    except ValueError:
        pass

    # ── 7. Zero engagements → nothing on disk ───────────────────────────────
    root2 = tmp / "never_created"
    rec2 = make_deploy_dataset_recorder(
        Namespace(save_dataset=str(root2)),
        prefix="async",
        cameras={"cam": FakeCam(H, W)},
        fps=30,
        arm_joint_names=ARM_JOINTS,
    )
    assert rec2 is not None
    rec2.note_loop_state("INFERENCE")
    rec2.note_loop_state("PAUSED")
    rec2.close()
    assert not root2.exists(), "zero executed ticks → no dataset dir"

    print("ALL SMOKE CHECKS PASSED")
    print(f"(dataset kept for inspection: {root})")


if __name__ == "__main__":
    main()
