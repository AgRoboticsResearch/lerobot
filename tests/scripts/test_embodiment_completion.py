import json

import numpy as np
import pytest

from examples.umi_relative_ee.embodiment_v2 import completion
from examples.umi_relative_ee.embodiment_v2.research_report import cluster_interval


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(parent, done=True):
    train, evaluation = completion.expected_runs()
    for robot in ("piper", "so101"):
        root = parent / robot
        write(root / "full-sweep-status.json", {"state": "complete" if done else "running", "exit_code": 0})
        if done:
            for name in train:
                write(root / "train" / name / "complete.json", {})
            for name in evaluation:
                write(root / "eval" / name / "complete.json", {"trials": 3000})
                write(root / "eval" / name / "results.json", [0] * 3000)
            write(root / "report/summary.json", {"matrix_complete": True})
    for name in ("pilot-status.json", "bounded-status.json"):
        write(parent / "pilots" / name, {"state": "complete"})


def test_readiness_requires_both_complete_matrices_and_no_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "active_experiments", lambda: [])
    fixture(tmp_path)
    assert completion.readiness(tmp_path) == []
    monkeypatch.setattr(completion, "active_experiments", lambda: [123])
    assert any("workers" in text for text in completion.readiness(tmp_path))
    monkeypatch.setattr(completion, "active_experiments", lambda: [])
    marker = tmp_path / "so101/eval/delay40/C1/hindsight_seed3000/complete.json"
    marker.unlink()
    assert any("missing eval" in text for text in completion.readiness(tmp_path))


def test_running_sweeps_never_finalize_or_clean(tmp_path, monkeypatch):
    fixture(tmp_path, done=False)
    monkeypatch.setattr(completion, "active_experiments", lambda: [])
    monkeypatch.setattr(completion, "cleanup", lambda parent: pytest.fail("Cleanup ran early"))
    assert completion.finalize(tmp_path)["state"] == "waiting"


def test_cleanup_requires_final_report(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "readiness", lambda parent: [])
    with pytest.raises(RuntimeError, match="Final report"):
        completion.cleanup(tmp_path)


def test_cleanup_only_removes_expected_symlinks_and_preserves_data(tmp_path, monkeypatch):
    parent = tmp_path / "new"
    old = tmp_path / "old"
    (parent / "piper").mkdir(parents=True)
    data = parent / "piper/model.pt"
    data.write_bytes(b"checkpoint")
    old.symlink_to(parent / "piper")
    view = parent / "view"
    view.symlink_to(str(old / "model.pt"))
    for name in ("complete.json", "REPORT.md", "figure_atlas.pdf"):
        path = parent / "report/final" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    monkeypatch.setattr(completion, "ALIASES", {str(old): "piper"})
    monkeypatch.setattr(completion, "readiness", lambda parent: [])
    monkeypatch.setattr(completion.psutil, "process_iter", lambda fields: [])
    result = completion.cleanup(parent)
    assert not old.is_symlink()
    assert data.read_bytes() == b"checkpoint"
    assert view.resolve() == data
    assert result["removed_links"] == [str(old)]
    # An ordinary directory must never be deleted, even at a configured old path.
    old.mkdir()
    with pytest.raises(RuntimeError, match="Unexpected old path"):
        completion.cleanup(parent)
    assert old.is_dir()


def test_cluster_interval_does_not_treat_duplicate_seed_trials_as_new_episodes():
    rows = [{"episode": e, "seed": seed, "error": 0.01} for e in (1, 1, 2) for seed in (1000, 2000)]
    result = cluster_interval(rows, "error")
    assert result["episodes"] == 2
    assert result["trials"] == 6
    np.testing.assert_allclose(result["ci95"], [0.01, 0.01])
    assert cluster_interval([], "error") is None
