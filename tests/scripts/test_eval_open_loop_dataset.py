import pytest
import torch

from examples.umi_relative_ee.eval_open_loop_dataset import (
    bootstrap_episode_confidence_intervals,
    choose_query_indices,
    rotation_error_deg,
    summarize,
)


def test_choose_query_indices_covers_every_episode_with_multiple_valid_frames():
    records = [
        {
            "episode_index": episode,
            "length": 80,
            "dataset_from_index": episode * 80,
        }
        for episode in range(100)
    ]

    queries = choose_query_indices(
        records,
        episode_indices=list(range(100)),
        chunk_size=30,
        samples_per_episode=10,
    )

    assert len(queries) == 1000
    assert {episode for _, episode, _ in queries} == set(range(100))
    assert all(1 <= frame <= 50 for _, _, frame in queries)


def test_choose_query_indices_rejects_nonpositive_sample_count():
    with pytest.raises(ValueError, match="samples_per_episode must be positive"):
        choose_query_indices([], [], chunk_size=30, samples_per_episode=0)


def test_bootstrap_episode_confidence_intervals_is_deterministic_and_episode_level():
    episode_means = {
        0: {"metric": 1.0},
        1: {"metric": 3.0},
        2: {"metric": 5.0},
    }

    first = bootstrap_episode_confidence_intervals(
        episode_means, ("metric",), num_resamples=1000, seed=7
    )
    second = bootstrap_episode_confidence_intervals(
        episode_means, ("metric",), num_resamples=1000, seed=7
    )

    assert first == second
    assert first["metric"]["low"] <= 3.0 <= first["metric"]["high"]


def test_rotation_error_deg_uses_absolute_pose_geodesic_error():
    predicted = torch.zeros(2, 7)
    target = torch.zeros(2, 7)
    target[1, 5] = torch.pi / 2

    error = rotation_error_deg(predicted, target)

    torch.testing.assert_close(error, torch.tensor([0.0, 90.0]), atol=1e-4, rtol=0)


def test_summarize_reports_episode_balanced_primary_metric():
    samples = [
        {
            "episode_index": 0,
            "frame_index": 1,
            "rotation_chunk_mean_deg": 1.0,
            "rotation_end_deg": 2.0,
            "xyz_chunk_mean_m": 0.01,
            "xyz_end_m": 0.02,
            "gripper_chunk_mean": 0.1,
            "gripper_end": 0.2,
        },
        {
            "episode_index": 0,
            "frame_index": 2,
            "rotation_chunk_mean_deg": 3.0,
            "rotation_end_deg": 4.0,
            "xyz_chunk_mean_m": 0.03,
            "xyz_end_m": 0.04,
            "gripper_chunk_mean": 0.3,
            "gripper_end": 0.4,
        },
        {
            "episode_index": 1,
            "frame_index": 1,
            "rotation_chunk_mean_deg": 9.0,
            "rotation_end_deg": 10.0,
            "xyz_chunk_mean_m": 0.09,
            "xyz_end_m": 0.10,
            "gripper_chunk_mean": 0.9,
            "gripper_end": 1.0,
        },
    ]
    added_metrics = (
        "rotation_chunk_rmse_deg",
        "rotation_chunk_mse_deg2",
        "xyz_chunk_rmse_m",
        "xyz_chunk_mse_m2",
        "gripper_chunk_rmse",
        "gripper_chunk_mse",
        "rot_jerk_deg",
        "xyz_jerk_m",
        "gt_rot_jerk_deg",
        "gt_xyz_jerk_m",
    )
    for index, sample in enumerate(samples):
        sample.update({name: float(index + 1) for name in added_metrics})

    summary = summarize(samples)

    assert summary["num_episodes"] == 2
    assert summary["num_samples"] == 3
    assert summary["primary_metric"] == "episode_balanced.rot_jerk_deg"
    assert summary["episode_balanced"]["rotation_end_deg"] == pytest.approx(6.5)
    assert summary["sample_weighted"]["rotation_end_deg"] == pytest.approx(16 / 3)
    assert summary["episode_balanced_95ci"]["rotation_end_deg"]["low"] <= 6.5
    assert summary["episode_balanced_95ci"]["rotation_end_deg"]["high"] >= 6.5
