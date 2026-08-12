from __future__ import annotations

import numpy as np

from examples.umi_relative_ee.act_flow_ablation import collect_results


def test_summarize_variants_skips_unevaluated_live_candidate(monkeypatch) -> None:
    baseline = "act_r18_vae_seed1000_30000steps"
    candidate = "act_r18_diffusion_lr1e5_seed1000_30000steps"
    reports = {
        baseline: [
            {
                "inference_latency_seconds": {"median_seconds": 0.1, "p95_seconds": 0.2},
                "cuda_peak_memory_bytes": 1,
                "summary": {"episode_balanced": {"position_l2_mean": 1.0}},
            }
        ]
    }
    runs = {
        baseline: {
            "run_name": baseline,
            "variant": "act_r18_vae",
            "training_seed": 1000,
            "steps": 30000,
        },
        candidate: {
            "run_name": candidate,
            "variant": "act_r18_diffusion_lr1e5",
            "training_seed": 1000,
            "steps": 30000,
        },
    }
    monkeypatch.setattr(
        collect_results,
        "aggregate_episode_metrics",
        lambda _: (("episode_0",), {"position_l2_mean": np.array([1.0])}),
    )

    summary, comparisons, episode_data = collect_results.summarize_variants(reports, runs)

    assert [row["run_name"] for row in summary] == [baseline]
    assert comparisons == []
    assert set(episode_data) == {baseline}
