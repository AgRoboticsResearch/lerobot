import numpy as np
import pytest
import torch

from examples.umi_relative_ee.embodiment_v2.train import (
    Controller,
    DirectControllerModel,
    PilotConfig,
    transform,
)


def fixture(joints=6):
    poses = np.zeros((30, 9), dtype=np.float32)
    poses[:, 3:] = [1, 0, 0, 0, 1, 0]
    poses[:, :3] = [0.3, -0.2, 0.1]
    poses[:, 0] += np.arange(30) * 0.001
    state = np.zeros((11, joints * 2), dtype=np.float32)
    state[:, :joints] = np.arange(joints) * 0.1
    past = state[:, :joints][:-1].copy() + 0.03
    current = np.eye(4)
    current[:3, 3] = [0.3, -0.2, 0.1]
    return poses, state, past, current


@pytest.mark.parametrize("joints", [5, 6])
@pytest.mark.parametrize("representation", ["q_delta", "command_delta"])
def test_initial_command_has_exact_physical_skip_connection(joints, representation):
    cfg = PilotConfig(representation=representation, width=32, layers=1)
    model = DirectControllerModel(cfg, joints).eval()
    stats = {
        k: {"mean": [0.02] * n, "std": [0.2] * n}
        for k, n in {
            "trajectory": 9,
            "state": 2 * joints,
            "past_commands": joints,
            "q_reference": joints,
            "target": joints,
        }.items()
    }
    model.set_statistics(stats)
    data = transform(*fixture(joints), representation)
    data["trajectory_valid"] = np.ones(30, dtype=bool)
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in data.items()}
    with torch.inference_mode():
        prediction = model.predict(batch)
    expected = data["base"][None, None, :].repeat(5, axis=1)
    np.testing.assert_allclose(prediction, expected, atol=1e-6)


def test_relative_targets_are_invariant_to_world_frame():
    poses, state, past, current = fixture()
    local = transform(poses, state, past, current, "command_delta")
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    shifted = poses.copy()
    shifted[:, :3] = poses[:, :3] @ rotation.T + [1, 2, 3]
    shifted[:, 3:] = rotation[:2].reshape(6)
    other = current.copy()
    other[:3, 3] = rotation @ current[:3, 3] + [1, 2, 3]
    other[:3, :3] = rotation
    transformed = transform(shifted, state, past, other, "command_delta")
    np.testing.assert_allclose(local["trajectory"], transformed["trajectory"], atol=2e-7)
    np.testing.assert_allclose(local["state"][:, :6], 0)
    np.testing.assert_allclose(local["past_commands"], 0.03, atol=1e-7)


def test_current_time_goal_cannot_affect_prediction():
    torch.manual_seed(1000)
    model = DirectControllerModel(PilotConfig(width=32, layers=1), 6).eval()
    data = transform(*fixture(), "command_delta")
    data["trajectory_valid"] = np.ones(30, dtype=bool)
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in data.items()}
    with torch.inference_mode():
        before = model.predict(batch)
        batch["trajectory"][:, 0] += 999
        after = model.predict(batch)
    torch.testing.assert_close(before, after)


def test_live_controller_has_no_ik_fallback_and_holds_at_exhausted_horizon():
    model = DirectControllerModel(PilotConfig(width=32, layers=1), 6).eval()
    controller = Controller(model, "command_delta", "cpu")
    poses, state, past, current = fixture()

    class Kinematics:
        def fk(self, q):
            return current

        def command(self, *args):
            raise AssertionError("Direct model must never solve IK")

    assert controller(poses, np.ones(30, dtype=bool), state, past, np.eye(4), Kinematics()).shape == (6,)
    only_now = np.zeros(30, dtype=bool)
    only_now[0] = True
    np.testing.assert_array_equal(controller(poses, only_now, state, past, np.eye(4), Kinematics()), past[-1])


def test_prepared_offsets_reconstruct_issued_commands(tmp_path, monkeypatch):
    from examples.umi_relative_ee.embodiment_v2 import train
    from examples.umi_relative_ee.task_independent_embodiment.core import Config, learning_window

    source_root = tmp_path / "original"
    source = source_root / "prepared/native/C0/hindsight"
    collection = source_root / "collections/native/D0/train"
    source.mkdir(parents=True)
    (collection / "rollouts").mkdir(parents=True)
    for path in (source / "manifest.json", source / "complete.json", collection / "manifest.json"):
        path.write_text("{}")
    windows = []
    for episode in range(2):
        size = 16
        poses = np.repeat(np.eye(4)[None], size, axis=0)
        poses[:, 0, 3] = episode + np.arange(size) * 0.01
        q = np.arange(size * 6).reshape(size, 6) * 0.001 + episode
        rollout = {
            "time": np.arange(size) / 50,
            "actual": poses,
            "q": q,
            "qd": q * 0,
            "issued": q[:-1] + 0.02,
        }
        np.savez(collection / "rollouts" / f"{episode:03d}.npz", **rollout)
        windows.extend(learning_window(rollout, i, Config(), True) for i in range(10, size - 1))
    for key in windows[0]:
        np.save(source / f"{key}.npy", np.stack([w[key] for w in windows]))
    monkeypatch.setitem(train.ROOTS, "piper", source_root)
    directory = train.prepare(tmp_path / "new", PilotConfig(representation="command_delta"))
    eligible = np.load(directory / "eligible.npy")
    # Last window has less than one EE sample of future support and is excluded.
    np.testing.assert_array_equal(eligible, [0, 1, 2, 3, 5, 6, 7, 8])
    target = np.load(directory / "target.npy")
    base = np.load(directory / "base.npy")
    valid = np.load(directory / "target_valid.npy")
    original = np.load(source / "target.npy")
    np.testing.assert_allclose((target + base[:, None])[valid], original[valid], atol=1e-6)
    assert not np.load(directory / "trajectory_valid.npy")[:, 0].any()


def test_watchdog_does_not_resume_while_orphaned_worker_runs(tmp_path, monkeypatch):
    from examples.umi_relative_ee.embodiment_v2 import watch

    lease = {"state": "active", "worker": {"pid": 22, "created": 2}, "paused": [{"pid": 11, "created": 1}]}
    watch.atomic_json(tmp_path / "gpu-lease.json", lease)
    monkeypatch.setattr(watch, "alive", lambda pid, created: True)
    signals = []
    monkeypatch.setattr(watch.os, "kill", lambda *args: signals.append(args))
    watch.restore_lease(tmp_path)
    assert signals == []
    monkeypatch.setattr(watch, "alive", lambda pid, created: pid == 11 and created == 1)
    watch.restore_lease(tmp_path)
    assert signals == [(11, watch.signal.SIGCONT)]


def test_watchdog_does_not_signal_reused_pid(tmp_path, monkeypatch):
    from examples.umi_relative_ee.embodiment_v2 import watch

    watch.atomic_json(tmp_path / "gpu-lease.json", {"state": "active", "paused": [{"pid": 11, "created": 1}]})
    monkeypatch.setattr(watch, "alive", lambda pid, created: False)
    signals = []
    monkeypatch.setattr(watch.os, "kill", lambda *args: signals.append(args))
    watch.restore_lease(tmp_path)
    assert signals == []


@pytest.mark.parametrize("delay", [0, 2])
def test_bounded_goals_preserve_every_rate_limited_physics_target(delay):
    from examples.umi_relative_ee.embodiment_v2.canonical import bounded_commands

    rng = np.random.default_rng(123)
    initial = rng.uniform(-1, 1, 6)
    raw = rng.uniform(-2, 2, (50, 6))
    max_step = np.deg2rad(90) / 50
    canonical = bounded_commands(raw, initial, max_step)
    assert np.max(np.abs(np.diff(np.concatenate([initial[None], canonical]), axis=0))) <= max_step + 1e-12

    def actuator_targets(commands):
        delayed = np.concatenate([np.repeat(initial[None], delay, axis=0), commands])[: len(commands)]
        target = initial.copy()
        values = []
        for command in delayed:
            for _ in range(10):
                target += np.clip(command - target, -max_step / 10, max_step / 10)
                values.append(target.copy())
        return np.asarray(values)

    np.testing.assert_allclose(actuator_targets(raw), actuator_targets(canonical), atol=1e-12)


def test_gpu_lease_gates_worker_and_releases_after_exit(tmp_path, monkeypatch):
    import json
    import sys

    from examples.umi_relative_ee.embodiment_v2 import watch

    # This test must never inspect or signal real experiment processes.
    monkeypatch.setattr(watch.psutil, "process_iter", lambda fields: [])
    output = tmp_path / "worker.txt"
    status = watch.run_exclusive(
        tmp_path,
        [
            sys.executable,
            "-c",
            'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("done")',
            str(output),
        ],
    )
    assert status == 0
    assert output.read_text() == "done"
    lease = json.loads((tmp_path / "gpu-lease.json").read_text())
    assert lease["state"] == "released"
    assert lease["paused"] == []
    assert not watch.alive(lease["worker"]["pid"], lease["worker"]["created"])


def test_paired_interval_preserves_constant_difference_with_repeated_episodes():
    from examples.umi_relative_ee.embodiment_v2.compare import paired_interval

    baseline = np.array([0.10, 0.12, 0.08, 0.15])
    result = paired_interval(baseline - 0.01, baseline, [1, 1, 2, 3])
    assert result["source_episode_clusters"] == 3
    np.testing.assert_allclose(result["mean_difference_m"], -0.01)
    np.testing.assert_allclose(result["cluster_bootstrap_95_m"], [-0.01, -0.01])


def test_completed_piper_reader_preserves_provenance_and_rejects_changed_inputs(tmp_path):
    import json

    from examples.umi_relative_ee.embodiment_v2.canonical import compatible_completed_piper

    old = {
        "robot": "piper",
        "sources": {"a.npz": "old-data"},
        "code": "78fa59984bf7acfb23f058547b57740ffebd59b132f01c07b3a028cb22365ffe",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(old))
    new = {**old, "code": "new-reader"}
    assert not compatible_completed_piper(tmp_path, new)  # Partial collections cannot mix generators.
    (tmp_path / "complete.json").write_text("{}")
    assert compatible_completed_piper(tmp_path, new)
    assert json.loads(path.read_text()) == old
    assert not compatible_completed_piper(tmp_path, {**new, "sources": {"a.npz": "changed-data"}})
    assert not compatible_completed_piper(tmp_path, {**new, "robot": "so101"})


def test_feasibility_partitions_use_separate_denominators():
    from examples.umi_relative_ee.embodiment_v2.strata import partition

    rows = [
        {
            "ik_feasible": feasible,
            "position_rmse_m": error,
            "rotation_mean_deg": 1.0,
            "tracking_failed": failure,
            "joint_limit_rate": 0.0,
            "execution_error": 0.0,
        }
        for feasible, error, failure in [(True, 0.01, 0), (True, 0.03, 0), (False, 0.10, 1), (False, 0.20, 0)]
    ]
    result = partition(rows)
    assert result["feasible"]["trials"] == result["infeasible"]["trials"] == 2
    assert result["feasible"]["position_rmse_m"] == pytest.approx(0.02)
    assert result["infeasible"]["position_rmse_m"] == pytest.approx(0.15)
    assert result["infeasible"]["tracking_failed"] == 0.5
    assert partition(rows[:2])["infeasible"]["position_rmse_m"] is None


def test_infeasibility_reasons_overlap_and_check_recorded_classification():
    from examples.umi_relative_ee.embodiment_v2.strata import reasons

    row = {
        "ik_feasible": False,
        "workspace_valid": True,
        "joint_limits_valid": True,
        "ik_error": None,
        "max_position_residual_m": 0.001,
        "max_rotation_residual_deg": 4.0,
        "max_nominal_velocity_deg_s": 100.0,
    }
    assert reasons(row, 90) == ["orientation_residual", "nominal_velocity"]
    with pytest.raises(ValueError, match="Stored feasibility"):
        reasons({**row, "ik_feasible": True}, 90)
