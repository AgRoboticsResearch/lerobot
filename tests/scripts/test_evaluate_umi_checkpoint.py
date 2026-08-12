from pathlib import Path


SCRIPT = Path("examples/umi_relative_ee/act_flow_ablation/evaluate_checkpoint.sh")
SUPERVISOR = Path(
    "examples/umi_relative_ee/act_flow_ablation/supervise_act_checkpoint_selection.sh"
)


def test_checkpoint_evaluator_is_explicit_and_noncanonical() -> None:
    text = SCRIPT.read_text()
    assert 'printf -v PADDED_STEP \'%06d\' "$CHECKPOINT_STEP"' in text
    assert 'NAMESPACE="eval_checkpoint_h32"' in text
    assert 'eval_common_h32' not in text
    assert '--pretrained_path="$CHECKPOINT"' in text
    assert '--query_min_action_offset=-1' in text
    assert '--query_max_action_offset=31' in text
    assert '--video_backend=pyav' in text
    assert 'Refusing to overwrite existing checkpoint evaluation' in text


def test_checkpoint_selection_waits_for_primary_evaluation_and_gpu_headroom() -> None:
    text = SUPERVISOR.read_text()
    assert 'umi_act_l1_100k_early_eval_20260812' in text
    assert 'UMI_ACT_SELECTION_STEP:-60000' in text
    assert 'eval_checkpoint_h32' in text
    assert 'memory.free' in text
    assert '"$SCRIPT_DIR/evaluate_checkpoint.sh"' in text
