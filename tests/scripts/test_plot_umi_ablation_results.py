from examples.umi_relative_ee.act_flow_ablation.plot_results import (
    ACT_FLOW_VARIANTS,
    ACT_L1_VARIANTS,
    DIFFUSION_VARIANTS,
    ORDER,
    plot_paired_improvements,
)


def test_learning_curve_objective_groups_are_complete_and_disjoint():
    groups = (set(ACT_L1_VARIANTS), set(ACT_FLOW_VARIANTS), set(DIFFUSION_VARIANTS))

    assert set.union(*groups) == set(ORDER)
    assert all(not left & right for index, left in enumerate(groups) for right in groups[index + 1 :])
    assert "act_r18_diffusion_lr1e5" in DIFFUSION_VARIANTS


def test_paired_plot_is_independent_of_input_row_order(tmp_path):
    pairs = (("diffusion_r18", "act_r18_l1"), ("diffusion_r18", "act_r18_diffusion_lr1e5"))
    seed_rows = []
    run_rows = []
    for pair_index, (candidate, baseline) in enumerate(pairs):
        for metric_index, metric in enumerate(("xyz_end_m", "rotation_end_deg")):
            value = float(pair_index * 10 + metric_index + 1)
            common = {
                "variant": candidate,
                "baseline_variant": baseline,
                "metric": metric,
                "steps": "30000",
                "num_training_seeds": "1",
                "paired_improvement_percent_mean": str(value),
            }
            seed_rows.append(common)
            run_rows.append(
                {
                    **common,
                    "paired_improvement_percent_ci_low": str(value - 0.5),
                    "paired_improvement_percent_ci_high": str(value + 0.5),
                }
            )

    forward = tmp_path / "forward"
    reverse = tmp_path / "reverse"
    plot_paired_improvements(run_rows, seed_rows, forward)
    plot_paired_improvements(list(reversed(run_rows)), list(reversed(seed_rows)), reverse)

    assert (forward / "paired_endpoint_improvements.svg").read_bytes() == (
        reverse / "paired_endpoint_improvements.svg"
    ).read_bytes()
