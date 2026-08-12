from pathlib import Path

import numpy as np
import pytest

from examples.umi_relative_ee.act_flow_ablation.collect_results import (
    aggregate_episode_metrics,
    bootstrap_mean_interval,
    bootstrap_paired_improvement_interval,
    hierarchical_bootstrap_mean_interval,
    hierarchical_bootstrap_paired_improvement_interval,
    parse_log,
    parse_run_name,
    validate_evaluation_report,
)

METRICS = (
    "rotation_chunk_mean_deg",
    "rotation_end_deg",
    "xyz_chunk_mean_m",
    "xyz_end_m",
    "gripper_chunk_mean",
    "gripper_end",
    "rot_jerk_deg",
    "xyz_jerk_m",
)


def test_parse_run_name_preserves_variant_and_numeric_fields():
    assert parse_run_name("act_r18_flow_u_lr1e4_seed1000_30000steps") == {
        "run_name": "act_r18_flow_u_lr1e4_seed1000_30000steps",
        "variant": "act_r18_flow_u_lr1e4",
        "training_seed": 1000,
        "steps": 30000,
    }


def test_parse_log_collects_parameters_wall_time_and_validation(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "[2026-08-11 10:00:00] starting run on host GPU\n"
        "INFO num_total_params=123456 (123K)\n"
        "INFO step:200 loss:1.0 updt_s:0.040 data_s:0.003\n"
        "INFO step:400 loss:0.5 updt_s:0.060 data_s:0.003\n"
        "INFO Validation at step 10000: loss=0.123000, flow_loss=0.123000\n"
        "[2026-08-11 10:02:03] completed run\n"
    )

    parsed = parse_log(log)

    assert parsed["status"] == "complete"
    assert parsed["parameters"] == 123456
    assert parsed["wall_seconds"] == 123
    assert parsed["median_update_seconds"] == 0.05
    assert parsed["median_updates_per_second"] == 20
    assert parsed["validation"] == [{"step": 10000, "loss": 0.123, "flow_loss": 0.123}]


def test_parse_log_separates_online_parameters_from_ema_state(tmp_path: Path):
    log = tmp_path / "ema.log"
    log.write_text("INFO num_learnable_params=160 (160)\nINFO num_total_params=320 (320)\n")

    parsed = parse_log(log)

    assert parsed["parameters"] == 160
    assert parsed["learnable_parameters"] == 160
    assert parsed["total_parameters"] == 320


def test_aggregate_episode_metrics_averages_inference_seeds_within_episode():
    reports = []
    for seed_offset in (0.0, 2.0):
        per_episode = {}
        for episode_id, episode_offset in (("0", 1.0), ("1", 3.0)):
            per_episode[episode_id] = dict.fromkeys(
                METRICS,
                seed_offset + episode_offset,
            )
        reports.append({"summary": {"per_episode": per_episode}})

    episode_ids, metrics = aggregate_episode_metrics(reports)

    assert episode_ids == ("0", "1")
    assert metrics["xyz_end_m"].tolist() == pytest.approx([2.0, 4.0])


def test_aggregate_episode_metrics_rejects_mismatched_episode_ids():
    def report(*episode_ids: str):
        return {
            "summary": {
                "per_episode": {
                    episode_id: dict.fromkeys(METRICS, 1.0) for episode_id in episode_ids
                }
            }
        }

    with pytest.raises(ValueError, match="mismatched episode IDs"):
        aggregate_episode_metrics([report("0", "1"), report("0", "2")])


def test_validate_evaluation_report_checks_seed_and_fixed_query_cardinality():
    path = Path("/tmp/evaluation/seed1000/result_open_loop_metrics.json")
    report = {
        "seed": 1000,
        "summary": {
            "num_episodes": 2,
            "num_samples": 10,
            "per_episode": {"0": {}, "1": {}},
        },
    }
    assert (
        validate_evaluation_report(
            path, report, expected_num_episodes=2, expected_samples_per_episode=5
        )
        == 1000
    )

    with pytest.raises(ValueError, match="Inference seed mismatch"):
        validate_evaluation_report(
            path, {**report, "seed": 2000}, expected_num_episodes=2, expected_samples_per_episode=5
        )
    with pytest.raises(ValueError, match="Unexpected query count"):
        validate_evaluation_report(
            path,
            {**report, "summary": {**report["summary"], "num_samples": 9}},
            expected_num_episodes=2,
            expected_samples_per_episode=5,
        )


def test_paired_improvement_bootstrap_uses_ratio_of_aggregate_means():
    baseline = np.asarray([1.0, 100.0])
    candidate = np.asarray([2.0, 50.0])
    low, high = bootstrap_paired_improvement_interval(
        baseline, candidate, rng=np.random.default_rng(0), num_resamples=1000
    )

    aggregate_improvement = (baseline.mean() - candidate.mean()) / baseline.mean() * 100
    assert low <= aggregate_improvement <= high


def test_hierarchical_bootstrap_single_seed_exactly_reduces_to_episode_bootstrap():
    values = np.arange(1.0, 101.0)
    baseline = values + 10.0
    candidate = baseline * 0.8

    assert hierarchical_bootstrap_mean_interval(
        [values], rng=np.random.default_rng(0), num_resamples=1000
    ) == bootstrap_mean_interval(values, rng=np.random.default_rng(0), num_resamples=1000)
    assert hierarchical_bootstrap_paired_improvement_interval(
        [baseline], [candidate], rng=np.random.default_rng(0), num_resamples=1000
    ) == bootstrap_paired_improvement_interval(
        baseline, candidate, rng=np.random.default_rng(0), num_resamples=1000
    )


def test_hierarchical_bootstrap_resamples_training_seed_clusters():
    groups = [np.full(20, value) for value in (1.0, 10.0, 100.0)]

    low, high = hierarchical_bootstrap_mean_interval(
        groups, rng=np.random.default_rng(0), num_resamples=10_000
    )

    assert low == pytest.approx(1.0)
    assert high == pytest.approx(100.0)


def test_hierarchical_paired_bootstrap_preserves_seed_and_episode_pairing():
    baseline = [np.full(20, value) for value in (10.0, 20.0, 30.0)]
    candidate = [group * scale for group, scale in zip(baseline, (0.8, 0.9, 0.7), strict=True)]

    low, high = hierarchical_bootstrap_paired_improvement_interval(
        baseline, candidate, rng=np.random.default_rng(0), num_resamples=10_000
    )

    assert low == pytest.approx(10.0)
    assert high == pytest.approx(30.0)
