import pytest
import torch

from examples.umi_relative_ee.eval_open_loop_dataset import (
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

    summary = summarize(samples)

    assert summary["num_episodes"] == 2
    assert summary["num_samples"] == 3
    assert summary["primary_metric"] == "episode_balanced.rotation_end_deg"
    assert summary["episode_balanced"]["rotation_end_deg"] == pytest.approx(6.5)
    assert summary["sample_weighted"]["rotation_end_deg"] == pytest.approx(16 / 3)
