from pathlib import Path

from examples.umi_relative_ee.act_flow_ablation.collect_results import parse_log, parse_run_name


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
