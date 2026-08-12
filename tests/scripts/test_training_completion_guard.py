from __future__ import annotations

import json
import subprocess
from pathlib import Path


HELPER = Path("examples/umi_relative_ee/act_flow_ablation/training_completion.sh").resolve()


def _make_checkpoint(
    root: Path, run_name: str, steps: int, *, terminal_log: bool = True, zero_pad: bool = False
) -> Path:
    checkpoint_name = f"{steps:06d}" if zero_pad else str(steps)
    checkpoint = root / "train" / run_name / "checkpoints" / checkpoint_name
    pretrained = checkpoint / "pretrained_model"
    state = checkpoint / "training_state"
    pretrained.mkdir(parents=True)
    state.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "config.json",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (pretrained / name).write_text("state")
    for name in ("optimizer_state.safetensors", "optimizer_param_groups.json", "rng_state.safetensors"):
        (state / name).write_text("state")
    (state / "training_step.json").write_text(json.dumps({"step": steps}))
    log = root / "logs" / f"{run_name}.log"
    log.parent.mkdir(parents=True)
    log.write_text("End of training\n" if terminal_log else "Training: 100%\n")
    return checkpoint


def _run_helper(root: Path, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'. "{HELPER}"; {command}'],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_requires_exact_resumable_final_checkpoint_and_terminal_log(tmp_path: Path) -> None:
    run_name = "candidate_seed1000_30000steps"
    checkpoint = _make_checkpoint(tmp_path, run_name, 30_000)

    assert _run_helper(tmp_path, f'training_is_complete "{tmp_path}" {run_name} 30000').returncode == 0

    (checkpoint / "training_state" / "training_step.json").write_text(json.dumps({"step": 29_999}))
    assert _run_helper(tmp_path, f'training_is_complete "{tmp_path}" {run_name} 30000').returncode != 0

    (checkpoint / "training_state" / "training_step.json").write_text(json.dumps({"step": 30_000}))
    (checkpoint / "training_state" / "optimizer_state.safetensors").unlink()
    assert _run_helper(tmp_path, f'training_is_complete "{tmp_path}" {run_name} 30000').returncode != 0


def test_recovery_marker_is_explicit_and_idempotent(tmp_path: Path) -> None:
    run_name = "candidate_seed1000_30000steps"
    _make_checkpoint(tmp_path, run_name, 30_000)

    command = f'recover_training_completion "{tmp_path}" {run_name} 30000'
    assert _run_helper(tmp_path, command).returncode == 0
    assert _run_helper(tmp_path, command).returncode == 0

    log = (tmp_path / "logs" / f"{run_name}.log").read_text()
    assert log.count("recovered-complete:") == 1
    assert log.count(f"completed {run_name}") == 1


def test_accepts_canonical_zero_padded_checkpoint_directory(tmp_path: Path) -> None:
    run_name = "candidate_seed1000_30000steps"
    checkpoint = _make_checkpoint(tmp_path, run_name, 30_000, zero_pad=True)
    (checkpoint / "training_state" / "training_step.json").write_text(
        json.dumps({"step": 30_000}, indent=4) + "\n"
    )

    assert _run_helper(tmp_path, f'training_is_complete "{tmp_path}" {run_name} 30000').returncode == 0
