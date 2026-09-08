"""Persistent experiment watchdog and recoverable exclusive-GPU leases for short diagnostics."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[3]
OUTPUT = Path("/mnt/data1/projects/lerobot-embodiment/pilots")
SWEEPS = {
    "piper": Path("/mnt/data1/projects/lerobot-embodiment/piper"),
    "so101": Path("/mnt/data1/projects/lerobot-embodiment/so101"),
}


def utc():
    return datetime.datetime.now(datetime.UTC).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def alive(pid, created=None):
    try:
        process = psutil.Process(pid)
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and (created is None or abs(process.create_time() - created) < 0.01)
        )
    except psutil.Error:
        return False


def restore_lease(root):
    path = root / "gpu-lease.json"
    if not path.exists():
        return
    lease = json.loads(path.read_text())
    if lease.get("state") != "active":
        return
    worker = lease.get("worker", {})
    if worker and alive(worker["pid"], worker["created"]):
        return  # An orphaned pilot still owns the GPU until it exits.
    for item in lease["paused"]:
        if alive(item["pid"], item["created"]):
            os.kill(item["pid"], signal.SIGCONT)
    lease.update(state="released", released_utc=utc())
    atomic_json(path, lease)


def run_exclusive(root, command):
    """Freeze only the experiment Python processes; leave their state/checkpoints intact."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "gpu-lease.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paused = []
        owner = psutil.Process()
        lease = {
            "state": "active",
            "owner_pid": owner.pid,
            "owner_created": owner.create_time(),
            "started_utc": utc(),
            "paused": paused,
            "command": command,
        }
        worker = None
        previous_handler = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise KeyboardInterrupt("GPU lease interrupted")

        signal.signal(signal.SIGTERM, terminate)
        try:
            for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
                args = process.info["cmdline"] or []
                if any(
                    arg
                    in (
                        "examples.umi_relative_ee.task_independent_embodiment.experiment",
                        "examples.umi_relative_ee.so101_task_independent_embodiment.experiment",
                    )
                    for arg in args
                ) and any(Path(arg).name.startswith("python") for arg in args[:1]):
                    if process.status() == psutil.STATUS_STOPPED:
                        raise RuntimeError(f"Experiment {process.pid} already stopped by another owner")
                    paused.append({"pid": process.pid, "created": process.info["create_time"]})
                    # Commit the recovery list BEFORE stopping each process.
                    atomic_json(root / "gpu-lease.json", lease)
                    process.send_signal(signal.SIGSTOP)
            atomic_json(root / "gpu-lease.json", lease)
            print(
                f"Exclusive diagnostic started; temporarily paused {[p['pid'] for p in paused]}", flush=True
            )
            # Gate exec until the worker PID is durably recorded. If the owner
            # dies in this window, EOF makes the child exit without taking the GPU.
            gate = (
                "import os, sys\n"
                "if sys.stdin.buffer.read(1) != b'1': sys.exit(125)\n"
                "os.execvp(sys.argv[1], sys.argv[1:])\n"
            )
            worker = subprocess.Popen([sys.executable, "-c", gate, *command], cwd=REPO, stdin=subprocess.PIPE)
            lease["worker"] = {"pid": worker.pid, "created": psutil.Process(worker.pid).create_time()}
            atomic_json(root / "gpu-lease.json", lease)
            worker.stdin.write(b"1")
            worker.stdin.close()
            return worker.wait()
        finally:
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            restore_lease(root)
            signal.signal(signal.SIGTERM, previous_handler)
            print("Original sweeps resumed", flush=True)


def snapshot(root):
    now = time.time()
    lease_file = root / "gpu-lease.json"
    lease = json.loads(lease_file.read_text()) if lease_file.exists() else {}
    if lease.get("state") == "active" and not alive(lease["owner_pid"], lease["owner_created"]):
        restore_lease(root)
        lease = json.loads(lease_file.read_text())
    summary = {"updated_utc": utc(), "sweeps": {}, "gpu_lease": lease, "alerts": []}
    summary["free_disk_gb"] = round(shutil.disk_usage(root).free / 1e9, 1)
    pilot_path = root / "pilot-status.json"
    summary["pilot"] = json.loads(pilot_path.read_text()) if pilot_path.exists() else {}
    if summary["pilot"].get("state") == "failed":
        summary["alerts"].append("Development pilot failed: " + summary["pilot"].get("error", "inspect log"))
    if summary["free_disk_gb"] < 30:
        summary["alerts"].append("Less than 30 GB free on the artifact volume")
    summary["pipelines"] = {}
    for name in ("pilot", "bounded"):
        state_path, launch_path = root / f"{name}-status.json", root / f"{name}-launch.json"
        if not state_path.exists() or not launch_path.exists():
            continue
        pipeline = json.loads(state_path.read_text())
        launch = json.loads(launch_path.read_text())
        pipeline["process_alive"] = alive(launch["pid"], launch.get("created"))
        logfile = Path(launch["log"])
        pipeline["log_idle_seconds"] = round(now - logfile.stat().st_mtime)
        with logfile.open("rb") as handle:
            handle.seek(max(0, logfile.stat().st_size - 2048))
            pipeline["recent_log"] = handle.read().decode(errors="replace").splitlines()[-2:]
        summary["pipelines"][name] = pipeline
        if pipeline["state"] == "failed" or (
            pipeline["state"] == "running" and not pipeline["process_alive"]
        ):
            summary["alerts"].append(f"{name}: pipeline stopped; inspect {logfile}")
        if pipeline["state"] == "running" and pipeline["log_idle_seconds"] > 3600:
            summary["alerts"].append(f"{name}: no pipeline log progress for over an hour")
    for name, source in SWEEPS.items():
        status_file = source / "full-sweep-status.json"
        status = json.loads(status_file.read_text())
        logfile = Path(status["log"])
        with logfile.open("rb") as handle:
            handle.seek(max(0, logfile.stat().st_size - 8192))
            recent = handle.read().decode(errors="replace").splitlines()
        progress = [
            line
            for line in recent
            if line.startswith(
                ("collect ", "eval ", "development:", "query seed=", "hindsight seed=", "residual seed=")
            )
        ]
        completed = sorted(
            str(p.parent.relative_to(source / "train")) for p in source.glob("train/*/*/*/complete.json")
        )
        running = alive(status["supervisor_pid"])
        idle = now - logfile.stat().st_mtime
        data = {
            "status": status["state"],
            "supervisor_alive": running,
            "log_idle_seconds": round(idle),
            "completed_training": len(completed),
            "expected_training": 30,
            "completed_runs": completed,
            "completed_evaluations": len(list(source.glob("eval/*/*/*/complete.json"))),
            "last_progress": progress[-1] if progress else recent[-1:],
            "log": str(logfile),
        }
        namespace = "task_independent_embodiment" if name == "piper" else "so101_task_independent_embodiment"
        code_root = REPO / "examples/umi_relative_ee" / namespace
        expected = json.loads((source / "preflight.json").read_text())["code"]
        data["source_hashes_match"] = all(
            (code_root / file).exists()
            and hashlib.sha256((code_root / file).read_bytes()).hexdigest() == digest
            for file, digest in expected.items()
        )
        if not data["source_hashes_match"]:
            summary["alerts"].append(
                f"{name}: source changed since preflight; do not resume into changed code"
            )
        summary["sweeps"][name] = data
        if status["state"] == "failed" or (status["state"] == "running" and not running):
            summary["alerts"].append(f"{name}: sweep stopped; inspect last error before resuming")
        if idle > 3600 and status["state"] == "running" and lease.get("state") != "active":
            summary["alerts"].append(
                f"{name}: no logged progress for over an hour (long evaluation possible)"
            )
    # Refresh the separate feasible/infeasible analysis as full evaluations finish.
    completed_sources = [
        path for source in SWEEPS.values() for path in source.glob("eval/*/*/*/complete.json")
    ] + list(root.glob("comparison/*/complete.json"))
    fingerprint = {str(p): p.stat().st_mtime_ns for p in sorted(completed_sources)}
    fingerprint["analysis_code"] = (Path(__file__).with_name("strata.py")).stat().st_mtime_ns
    marker = root / "stratified/source-state.json"
    try:
        if not marker.exists() or json.loads(marker.read_text()) != fingerprint:
            from .strata import report

            report(root)
            atomic_json(marker, fingerprint)
    except Exception as exc:
        summary["alerts"].append(f"Stratified analysis failed: {exc}")
    completion_status = root.parent / "completion/status.json"
    if completion_status.exists():
        summary["completion"] = json.loads(completion_status.read_text())
        if summary["completion"].get("state") == "error":
            summary["alerts"].append("Final report/cleanup error; inspect ../completion/status.json")
    atomic_json(root / "watch-status.json", summary)
    lines = ["# Experiment watchdog", "", summary["updated_utc"], ""]
    for name, data in summary["sweeps"].items():
        lines += [
            f"- {name}: {data['completed_training']}/30 trainings; alive={data['supervisor_alive']}",
            f"  Latest: {data['last_progress']}",
        ]
    lines += [
        "",
        "Pipelines: " + json.dumps(summary["pipelines"]),
        "GPU lease: " + lease.get("state", "none"),
        f"Free disk: {summary['free_disk_gb']} GB",
        "",
        "Alerts: " + ("; ".join(summary["alerts"]) or "none"),
        "",
        "[Feasible / infeasible analysis](stratified/REPORT.md)",
        "[Illustrated Markdown report](../report/README.md)",
        "Final report / cleanup: " + summary.get("completion", {}).get("state", "not scheduled"),
        "",
        "Existing v1 sweeps share a GPU outside exclusive diagnostics: their latency metrics are not isolated.",
    ]
    (root / "WATCH.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("watch", "once", "exclusive"))
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--interval", type=int, default=60)
    args, command = parser.parse_known_args()
    if args.mode == "exclusive":
        if command[:1] == ["--"]:
            command = command[1:]
        if not command:
            parser.error("exclusive requires a command after --")
        sys.exit(run_exclusive(args.root, command))
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == "once":
        print(json.dumps(snapshot(args.root), indent=2))
        return
    with (args.root / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(
            args.root / "watch-process.json",
            {"pid": os.getpid(), "created": psutil.Process().create_time(), "started_utc": utc()},
        )
        while True:
            try:
                status = snapshot(args.root)
                print(json.dumps({"updated": status["updated_utc"], "alerts": status["alerts"]}), flush=True)
            except Exception as exc:
                atomic_json(args.root / "watch-error.json", {"time": utc(), "error": repr(exc)})
                print(repr(exc), flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
