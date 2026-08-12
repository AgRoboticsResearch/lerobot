from examples.umi_relative_ee.act_flow_ablation.plot_results import (
    ACT_FLOW_VARIANTS,
    ACT_L1_VARIANTS,
    DIFFUSION_VARIANTS,
    ORDER,
)


def test_learning_curve_objective_groups_are_complete_and_disjoint():
    groups = (set(ACT_L1_VARIANTS), set(ACT_FLOW_VARIANTS), set(DIFFUSION_VARIANTS))

    assert set.union(*groups) == set(ORDER)
    assert all(not left & right for index, left in enumerate(groups) for right in groups[index + 1 :])
    assert "act_r18_diffusion_lr1e5" in DIFFUSION_VARIANTS
