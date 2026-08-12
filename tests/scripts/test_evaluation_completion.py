import subprocess
from pathlib import Path


HELPER = Path("examples/umi_relative_ee/act_flow_ablation/evaluation_completion.sh").resolve()


def _check(output: Path, log: Path, steps: int) -> int:
    command = (
        f'. "{HELPER}"; canonical_evaluation_complete '
        f'"{output}" "{log}" run_seed1000_100000steps 1000 {steps}'
    )
    return subprocess.run(["bash", "-c", command], check=False).returncode


def test_completion_requires_exact_checkpoint_suffix(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    log = tmp_path / "evaluation.log"
    log.write_text(
        "[2026-08-12 00:00:00] completed evaluation run_seed1000_100000steps seed 1000\n"
    )
    (output / "run_030000_open_loop_metrics.json").write_text("{}")

    assert _check(output, log, 100000) != 0
    assert _check(output, log, 30000) == 0


def test_completion_rejects_duplicates_and_missing_marker(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    log = tmp_path / "evaluation.log"
    log.write_text("no completion marker\n")
    (output / "first_100000_open_loop_metrics.json").write_text("{}")

    assert _check(output, log, 100000) != 0
    log.write_text(
        "[2026-08-12 00:00:00] completed evaluation run_seed1000_100000steps seed 1000\n"
    )
    (output / "second_100000_open_loop_metrics.json").write_text("{}")
    assert _check(output, log, 100000) != 0
