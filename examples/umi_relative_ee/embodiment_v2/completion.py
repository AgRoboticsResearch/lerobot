"""Wait for the full experiment matrix, build the final report, then remove only compatibility links."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import psutil

from .watch import atomic_json, utc

PARENT = Path("/mnt/data1/projects/lerobot-embodiment")
ALIASES = {
    "/mnt/data1/projects/lerobot-embodiment-exp-so101": "so101",
    "/mnt/data1/projects/lerobot-embodiment-exp": "piper",
    "/mnt/data1/projects/lerobot-embodiment-env": "env",
    "/mnt/data1/projects/lerobot-embodiment-smoke": "smoke",
    "/mnt/data1/projects/lerobot-embodiment-v2": "pilots",
}
METHODS = [
    ("C0", "query"),
    ("C0", "residual"),
    ("C0", "hindsight"),
    ("C1", "hindsight"),
    ("bootstrap20k", "hindsight"),
]


def expected_runs():
    train = {
        f"{condition}/{generation}/{method}_seed{seed}"
        for condition in ("native", "delay40")
        for generation, method in METHODS
        for seed in (1000, 2000, 3000)
    }
    evaluation = train | {f"{condition}/C0/ik_seed1000" for condition in ("native", "delay40")}
    return train, evaluation


def active_experiments():
    found = []
    modules = {
        f"examples.umi_relative_ee.{name}.experiment"
        for name in ("task_independent_embodiment", "so101_task_independent_embodiment")
    }
    modules.update(
        f"examples.umi_relative_ee.embodiment_v2.{name}"
        for name in ("train", "pilots", "canonical", "canonical_pipeline", "compare")
    )
    for process in psutil.process_iter(["pid", "cmdline", "status"]):
        args = process.info["cmdline"] or []
        if process.info["status"] != psutil.STATUS_ZOMBIE and any(arg in modules for arg in args):
            found.append(process.pid)
    return found


def readiness(parent):
    pending = []
    train, evaluation = expected_runs()
    for robot in ("piper", "so101"):
        root = parent / robot
        status = json.loads((root / "full-sweep-status.json").read_text())
        if status["state"] != "complete" or status.get("exit_code") != 0:
            pending.append(f"{robot}: sweep status {status['state']}")
            continue
        for kind, names in [("train", train), ("eval", evaluation)]:
            for name in sorted(names):
                marker = root / kind / name / "complete.json"
                if not marker.exists():
                    pending.append(f"{robot}: missing {kind}/{name}")
                elif kind == "eval":
                    results = marker.with_name("results.json")
                    if json.loads(marker.read_text()).get("trials") != 3000 or not results.exists():
                        pending.append(f"{robot}: incomplete trial matrix {name}")
                    elif len(json.loads(results.read_text())) != 3000:
                        pending.append(f"{robot}: wrong result count {name}")
        report = root / "report/summary.json"
        if not report.exists() or not json.loads(report.read_text()).get("matrix_complete"):
            pending.append(f"{robot}: final matrix report not complete")
    for name in ("pilot-status.json", "bounded-status.json"):
        status = json.loads((parent / "pilots" / name).read_text())
        if status["state"] != "complete":
            pending.append(f"{name}: {status['state']}")
    lease = parent / "pilots/gpu-lease.json"
    if lease.exists() and json.loads(lease.read_text()).get("state") == "active":
        pending.append("GPU lease still active")
    workers = active_experiments()
    if workers:
        pending.append(f"Experiment workers still running: {workers}")
    return pending


def replace_paths(value, parent):
    if isinstance(value, str):
        for old, child in ALIASES.items():
            value = value.replace(old, str(parent / child))
        return value
    if isinstance(value, list):
        return [replace_paths(v, parent) for v in value]
    if isinstance(value, dict):
        return {replace_paths(k, parent): replace_paths(v, parent) for k, v in value.items()}
    return value


def cleanup(parent):
    # Recheck after report generation: a newly started run must block deletion.
    pending = readiness(parent)
    if pending:
        raise RuntimeError("Cleanup blocked: " + "; ".join(pending))
    final_report = parent / "report/final"
    if not (final_report / "complete.json").exists() or any(
        not (final_report / name).is_file() for name in ("REPORT.md", "figure_atlas.pdf")
    ):
        raise RuntimeError("Final report must be successfully generated before cleanup")
    for process in psutil.process_iter(["pid", "cmdline", "status"]):
        if process.pid == os.getpid() or process.info["status"] == psutil.STATUS_ZOMBIE:
            continue
        if any(any(arg.startswith(old) for old in ALIASES) for arg in (process.info["cmdline"] or [])):
            raise RuntimeError(f"Process {process.pid} still uses an old executable/script path")
    # All aliases must be the exact expected links; never remove a real directory.
    for old, child in ALIASES.items():
        path = Path(old)
        if os.path.lexists(path) and (not path.is_symlink() or path.resolve() != (parent / child).resolve()):
            raise RuntimeError(f"Unexpected old path; refusing removal: {path}")
    archive = parent / "cleanup/original_operational_files"
    archive.mkdir(parents=True, exist_ok=True)
    changed = []
    # Repair dataset-view and environment symlinks before removing root aliases.
    for directory, dirs, files in os.walk(parent, followlinks=False):
        if Path(directory).is_relative_to(parent / "cleanup"):
            dirs[:] = []
            continue
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                old_target = os.readlink(path)
                new_target = replace_paths(old_target, parent)
                if old_target != new_target:
                    tmp = path.with_name(path.name + ".relocation-tmp")
                    tmp.symlink_to(new_target)
                    os.replace(tmp, path)
                    changed.append(str(path.relative_to(parent)))
    paths = list((parent / "env/bin").glob("*")) + [parent / "env/pyvenv.cfg"]
    paths += list((parent / "env/lib").glob("python*/site-packages/*.pth"))
    for robot in ("piper", "so101"):
        paths += [parent / robot / "supervise_full_run.py", parent / robot / "full-sweep-status.json"]
    paths += [
        p
        for pattern in ("*-status.json", "*-launch.json", "run_pilots.py")
        for p in (parent / "pilots").glob(pattern)
    ]
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
            continue
        try:
            original = path.read_text()
        except UnicodeDecodeError:
            continue
        if "\0" in original:
            continue
        updated = replace_paths(original, parent)
        if updated == original:
            continue
        backup = archive / path.relative_to(parent)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_text(original)
        mode = path.stat().st_mode
        tmp = path.with_name(path.name + ".relocation-tmp")
        tmp.write_text(updated)
        tmp.chmod(mode)
        os.replace(tmp, path)
        changed.append(str(path.relative_to(parent)))
    removed = []
    for old in ALIASES:
        path = Path(old)
        if path.is_symlink():
            path.unlink()
            removed.append(old)
    record = {
        "completed_utc": utc(),
        "removed_links": removed,
        "updated_operational_files": changed,
        "preserved": "All datasets, rollouts, checkpoints, immutable manifests and original logs",
    }
    atomic_json(parent / "cleanup/complete.json", record)
    return record


def finalize(parent):
    pending = readiness(parent)
    if pending:
        return {"state": "waiting", "pending": pending, "checked_utc": utc()}
    from .research_report import build

    with ExitStack() as locks:
        for path in [parent / robot / "full-sweep.lock" for robot in ("piper", "so101")] + [
            parent / "pilots" / name for name in ("pilots.lock", "bounded-pipeline.lock", "gpu-lease.lock")
        ]:
            handle = locks.enter_context(path.open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"state": "waiting", "pending": [f"Run lock is busy: {path}"], "checked_utc": utc()}
        pending = readiness(parent)
        if pending:
            return {"state": "waiting", "pending": pending, "checked_utc": utc()}
        result = build(parent, final=True)
        record = cleanup(parent)
    (parent / "README.md").write_text("""# Embodiment experiments

Both full sweeps and the controller pilots are complete.

- [Final illustrated Markdown report](report/final/REPORT.md): plots and trajectory figures.
- [PDF figure atlas](report/final/figure_atlas.pdf).
- [Feasible/infeasible tables](pilots/stratified/REPORT.md).

Folders: `piper/`, `so101/`, `pilots/`, `smoke/`, `env/`.

The old compatibility links were removed after verified completion. All data and
checkpoints are preserved. Immutable manifests retain their original paths for
provenance; `relocation.json` maps those historical paths to this parent folder.
Operational-file backups and cleanup records are in `cleanup/`.
""")
    return {"state": "complete", "finished_utc": utc(), "report": result, "cleanup": record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "watch", "once"))
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.mode == "preview":
        from .research_report import build

        print(json.dumps(build(args.parent), indent=2))
        return
    directory = args.parent / "completion"
    directory.mkdir(exist_ok=True)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(
            directory / "process.json",
            {"pid": os.getpid(), "created": psutil.Process().create_time(), "started_utc": utc()},
        )
        while True:
            try:
                if (directory / "status.json").exists() and json.loads(
                    (directory / "status.json").read_text()
                )["state"] == "complete":
                    return
                status = finalize(args.parent)
                atomic_json(directory / "status.json", status)
                print(
                    json.dumps(
                        {"state": status["state"], "time": utc(), "pending": status.get("pending", [])}
                    ),
                    flush=True,
                )
                if status["state"] == "complete" or args.mode == "once":
                    return
            except Exception as exc:
                atomic_json(directory / "status.json", {"state": "error", "time": utc(), "error": repr(exc)})
                print(repr(exc), flush=True)
                # A failed report/cleanup is inspectable; never delete files after an exception.
                if args.mode == "once":
                    raise
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
