"""Scientific contracts: causality, time alignment, frames, masks, and paired evaluation."""

import json

import numpy as np
import pytest
import torch

from examples.umi_relative_ee.so101_task_independent_embodiment.core import (
    Config,
    interpolate,
    learning_window,
    manifest,
    pose_features,
    relative_poses,
)
from examples.umi_relative_ee.so101_task_independent_embodiment.evaluation import (
    episode_bootstrap,
    paired_comparison,
)
from examples.umi_relative_ee.so101_task_independent_embodiment.learning import EmbodimentTransformer
from examples.umi_relative_ee.so101_task_independent_embodiment.simulation import CommandDelay

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


def test_timestamped_attenuated_delayed_inverse_labels():
    """Issued command causes a later attenuated displacement; neither is a query label."""
    cfg = Config.smoke_config()
    n = 70
    commands = np.zeros((n, cfg.arm_joints))
    commands[12:, 0] = 1
    delay = CommandDelay(2, np.zeros(cfg.arm_joints))
    delivered = np.stack([delay.push(u) for u in commands])
    q = np.zeros((n + 1, cfg.arm_joints))
    q[1:] = np.cumsum(0.9 * delivered / cfg.control_hz, axis=0)
    actual = np.broadcast_to(np.eye(4), (n + 1, 4, 4)).copy()
    actual[:, 0, 3] = q[:, 0]
    query = actual.copy()
    query[:, 0, 3] += 100  # distinguish query versus hindsight unambiguously
    rollout = {
        "time": np.arange(n + 1) / 50,
        "q": q,
        "qd": np.zeros_like(q),
        "actual": actual,
        "issued": commands,
        "delivered": delivered,
        "query": query,
        "query_time": np.arange(n + 1) / 50,
    }
    window = learning_window(rollout, 12, cfg, True)
    np.testing.assert_array_equal(window["past_commands"], commands[2:12])
    np.testing.assert_array_equal(window["target"], commands[12:17])
    assert window["trajectory"][0, 0] == 0
    # At t+0.1 s only three of the five issued commands have arrived: .9 * 3 / 50.
    assert window["trajectory"][3, 0] == pytest.approx(0.9 * 3 / 50)
    assert learning_window(rollout, 12, cfg, False)["trajectory"][0, 0] == 100
    assert delivered[12, 0] == 0 and delivered[14, 0] == 1
    assert CommandDelay(2, np.zeros(5)).push(np.ones(5))[0] == 0  # reset clears queue
    # Corrupt all future state: the model's history must remain unchanged.
    rollout["q"][13:] = 999
    second = learning_window(rollout, 12, cfg, True)
    np.testing.assert_array_equal(second["state"], window["state"])
    tail = learning_window(rollout, n - 1, cfg, True)
    assert tail["target_valid"].tolist() == [True, False, False, False, False]
    assert tail["trajectory_valid"].sum() == 1


def test_rigid_frame_transplant_and_slerp_timing():
    pose = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    pose[1, 0, 3] = 1
    pose[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    times = np.array([0.0, 1.0])
    result, mask = interpolate(times, pose, np.array([0.0, 0.02, 0.5, 1.0, 1.1]))
    np.testing.assert_allclose(result[:, 0, 3], [0.0, 0.02, 0.5, 1.0, 1.0])
    assert mask.tolist() == [True, True, True, True, False]
    assert Rotation.from_matrix(result[2, :3, :3]).as_euler("xyz", degrees=True)[2] == pytest.approx(45)
    anchor = np.eye(4)
    anchor[:3, :3] = Rotation.from_euler("xyz", [40, 20, -30], degrees=True).as_matrix()
    anchor[:3, 3] = [0.2, -0.1, 0.3]
    np.testing.assert_allclose(relative_poses(anchor @ pose, anchor), pose, atol=1e-12)
    # Validate the repository's row-rot6d convention independently.
    from lerobot.processor.umi_relative_ee_processor import rot6d_to_matrix

    features = pose_features(anchor @ pose)
    decoded = rot6d_to_matrix(torch.tensor(features[:, 3:]))
    np.testing.assert_allclose(decoded, (anchor @ pose)[:, :3, :3], atol=1e-6)


def test_masked_future_cannot_affect_controller_output():
    torch.manual_seed(1000)
    cfg = Config.smoke_config()
    model = EmbodimentTransformer(cfg).eval()
    batch = {
        "trajectory": torch.randn(2, 30, 9),
        "trajectory_valid": torch.zeros(2, 30, dtype=torch.bool),
        "state": torch.randn(2, 11, 2 * cfg.arm_joints),
        "past_commands": torch.randn(2, 10, cfg.arm_joints),
    }
    batch["trajectory_valid"][:, :3] = True
    with torch.no_grad():
        before = model.predict(batch)
        batch["trajectory"][:, 3:] += 1000
        after = model.predict(batch)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    assert before.shape == (2, 5, cfg.arm_joints)


def test_provenance_refuses_overwrite_and_changed_resume(tmp_path):
    path = tmp_path / "manifest.json"
    manifest(path, {"source_episodes": [1, 2], "seed": 1000}, False)
    with pytest.raises(FileExistsError):
        manifest(path, json.loads(path.read_text()), False)
    manifest(path, json.loads(path.read_text()), True)
    with pytest.raises(ValueError, match="Provenance"):
        manifest(path, {"source_episodes": [3], "seed": 1000}, True)


def test_paired_bootstrap_clusters_episodes_and_keeps_failures():
    a = [{"trial": str(i), "episode": i // 2, "ik_feasible": i < 2, "loss": float(i)} for i in range(4)]
    b = [{**row, "loss": row["loss"] + 2} for row in a]
    differences = paired_comparison(a, b, "loss")
    assert differences == {0: -2.0, 1: -2.0}
    assert episode_bootstrap(differences, 100)["ci95"] == [-2.0, -2.0]
    assert paired_comparison(a, b, "loss", True) == {0: -2.0}
    with pytest.raises(ValueError, match="identical trial"):
        paired_comparison(a, b[:-1], "loss")
    b[0]["ik_feasible"] = False
    with pytest.raises(ValueError, match="classification"):
        paired_comparison(a, b, "loss")


def test_simulation_resets_units_and_exact_control_clock():
    pytest.importorskip("mujoco")
    pytest.importorskip("placo")
    pytest.importorskip("so101_sim")
    from examples.umi_relative_ee.so101_task_independent_embodiment.simulation import Simulation

    cfg = Config.smoke_config()
    sim = Simulation(cfg, "delay40")
    q = np.deg2rad([0, -30, 60, 30, 0])
    sim.reset(q)
    before = sim.state()
    command = q + 0.01
    _, delivered, _, _ = sim.advance(command)
    np.testing.assert_allclose(delivered, q)
    assert sim.state()[0] - before[0] == pytest.approx(0.02, abs=1e-12)
    sim.advance(command)
    _, delivered, _, _ = sim.advance(command)
    np.testing.assert_allclose(delivered, command)
    sim.reset(q)
    after = sim.state()
    for first, second in zip(before, after, strict=True):
        np.testing.assert_allclose(first, second, atol=1e-10)
    fk = sim.ik.fk(after[1])
    np.testing.assert_allclose(fk, after[3], atol=1e-6)
    issued, _, clamped, invalid = sim.advance(np.full(cfg.arm_joints, 100.0))
    assert clamped and not invalid
    np.testing.assert_allclose(issued, sim.high)
    q = sim.state()[1]
    issued, _, _, invalid = sim.advance(np.full(cfg.arm_joints, np.nan))
    assert invalid
    np.testing.assert_allclose(issued, np.clip(q, sim.low, sim.high))


def test_source_splits_are_disjoint_before_augmentation(tmp_path, monkeypatch):
    from examples.umi_relative_ee.so101_task_independent_embodiment import core

    episodes = {i: {"times": np.arange(60) / 30, "actions": np.zeros((60, 7))} for i in range(20)}
    monkeypatch.setattr(core, "read_episodes", lambda _: episodes)
    monkeypatch.setattr(core, "source_fingerprint", lambda _: {"fixture": "hash"})
    core.extract(tmp_path, Config.smoke_config(), False)
    splits = json.loads((tmp_path / "motions/manifest.json").read_text())["splits"]
    assert set(splits["train"]).isdisjoint(splits["dev"])
    assert set(splits["train"]) | set(splits["dev"]) == set(episodes)
    assert len(splits["dev"]) == 2
    core.extract(tmp_path, Config.smoke_config(), True)
    assert json.loads((tmp_path / "motions/manifest.json").read_text())["splits"] == splits


def test_training_resume_matches_uninterrupted_despite_eval_rng(tmp_path):
    from examples.umi_relative_ee.so101_task_independent_embodiment.learning import train

    torch.set_num_threads(2)
    cfg = Config.smoke_config()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}")
    rng = np.random.default_rng(3)
    for key, shape in {
        "trajectory": (30, 9),
        "state": (11, 2 * cfg.arm_joints),
        "past_commands": (10, cfg.arm_joints),
        "target": (5, cfg.arm_joints),
    }.items():
        np.save(prepared / f"{key}.npy", rng.normal(size=(16, *shape)).astype(np.float32))
    np.save(prepared / "trajectory_valid.npy", np.ones((16, 30), dtype=bool))
    np.save(prepared / "target_valid.npy", np.ones((16, 5), dtype=bool))
    stats = {
        key: {"mean": [0.0] * size, "std": [1.0] * size}
        for key, size in (
            ("trajectory", 9),
            ("state", 2 * cfg.arm_joints),
            ("past_commands", cfg.arm_joints),
            ("target", cfg.arm_joints),
        )
    }
    (prepared / "statistics.json").write_text(json.dumps(stats))

    def evaluate(_):
        torch.randn(100)  # mimic construction of the separate evaluation model
        return {"position_rmse_m": 1.0, "rotation_mean_deg": 1.0}

    full = tmp_path / "full"
    interrupted = tmp_path / "interrupted"
    train(full, prepared, "hindsight", cfg, 1000, "cpu", evaluate, False, {})
    calls = 0

    def stop_at_second_evaluation(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interruption")
        return evaluate(path)

    with pytest.raises(RuntimeError, match="interruption"):
        train(interrupted, prepared, "hindsight", cfg, 1000, "cpu", stop_at_second_evaluation, False, {})
    train(interrupted, prepared, "hindsight", cfg, 1000, "cpu", evaluate, True, {})
    expected = torch.load(full / "last.pt", weights_only=False)
    actual = torch.load(interrupted / "last.pt", weights_only=False)
    for name, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][name], atol=0, rtol=0)
    assert expected["numpy_rng"] == actual["numpy_rng"]
