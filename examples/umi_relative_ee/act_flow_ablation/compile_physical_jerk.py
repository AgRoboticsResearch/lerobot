#!/usr/bin/env python3
"""Compile the §9.2.13 physical-jerk re-evaluation from the kiwi jerk tree.

Inputs:
  --jerk_root     tree of extended-protocol re-evals (seed 1000 only):
                  <RUN>/seed1000/*_open_loop_metrics.json, produced by
                  jerk_sweep_kiwi.sh over the archived report checkpoints
                  (2026-08-23). Each JSON carries BOTH the legacy second-
                  difference metrics (keys unchanged) and the new physical-
                  unit velocity/acceleration/jerk at dt = 1/fps.
  --unified_root  the §9.2.9 archived tree — used (a) to cross-validate that
                  the re-eval reproduces the archived endpoint/2nd-diff
                  numbers, and (b) to carry the four JAX openpi rows (whose
                  physical-jerk re-eval runs on the host and is marked
                  pending here).

Outputs (in --out_dir):
  physical_jerk_h10.csv          per-run episode-balanced metrics + 95% CI
  physical_jerk_h10.md           markdown table for the report
  validation.txt                 max |Δ| vs the archived tree (repro proof)
And per-run compact per-episode files under --per_episode_dir (repo-tracked
repro evidence): <run>.json.gz with episode_balanced, CI, and per_episode.

Run from the repo root of this script's directory with `uv run`.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

STEP_RE = re.compile(r"_(?P<step>\d{6,7})_open_loop_metrics\.json$")

# Legacy cross-check metrics (must reproduce the archived tree).
CROSSCHECK = ["xyz_end_m", "rotation_end_deg", "rot_jerk_deg", "xyz_jerk_m"]
# New physical-unit metrics (mean over chunk; rotation uses |signed| diffs).
PHYSICAL = [
    "rot_vel_deg_s",
    "rot_accel_deg_s2",
    "rot_jerk_deg_s3",
    "xyz_vel_m_s",
    "xyz_accel_m_s2",
    "xyz_jerk_m_s3",
    "gt_rot_vel_deg_s",
    "gt_rot_accel_deg_s2",
    "gt_rot_jerk_deg_s3",
    "gt_xyz_vel_m_s",
    "gt_xyz_accel_m_s2",
    "gt_xyz_jerk_m_s3",
]
ALL_METRICS = CROSSCHECK + PHYSICAL

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0

# openpi JAX rows re-scored on the host (physical metrics pending there).
OPENPI_RUNS = {
    "pi05_lora_sroi_rot6d_seed1000_0020000steps",
    "pi05_lora_sroi_rotvec_seed1000_0020000steps",
    "pi05_lora_sroi_rot6d_h30_seed1000_0020000steps",
    "pi05_openpi1m_seed1000_0100001steps",
}


def step_of(path: str) -> str:
    m = STEP_RE.search(os.path.basename(path))
    if m is not None:
        return m.group("step")
    # Shadow-ckpt runs (scheduler-flag patch) name the JSON from the run
    # instead of the checkpoint dir: <prefix>_<run>_020000steps_open_...
    m = re.search(r"_(\d{6,7})steps_open_loop_metrics", os.path.basename(path))
    if m is not None:
        return m.group(1)
    raise ValueError(f"no step in {path}")


def bootstrap_ci(per_episode: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, per_episode.shape[0], size=(N_BOOTSTRAP, per_episode.shape[0]))
    means = per_episode[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def episode_means(report: dict) -> dict[int, dict[str, float]]:
    by_ep: dict[int, list[dict]] = defaultdict(list)
    for s in report["samples"]:
        by_ep[int(s["episode_index"])].append(s)
    out = {}
    for ep, samples in by_ep.items():
        out[ep] = {m: float(np.mean([s[m] for s in samples])) for m in ALL_METRICS if m in samples[0]}
    return out


def check_protocol(run: str, report: dict) -> float:
    if report.get("query_action_offset_bounds") != {"min": -1, "max": 31}:
        raise ValueError(f"{run}: non-canonical query bounds")
    if report.get("eval_horizon") != 10:
        raise ValueError(f"{run}: eval_horizon != 10")
    if len(report["samples"]) != 500:
        raise ValueError(f"{run}: {len(report['samples'])} queries, expected 500")
    fps = report.get("control_fps")
    if fps != 30.0:
        raise ValueError(f"{run}: control_fps {fps}, expected 30.0")
    return float(fps)  # type: ignore[arg-type]  # narrowed by the != 30.0 check above


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jerk_root",
                    default="/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/eval_unified_h10_jerk")
    ap.add_argument("--unified_root",
                    default="/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/eval_unified_h10")
    ap.add_argument("--out_dir",
                    default="/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results_physical_jerk")
    ap.add_argument("--per_episode_dir",
                    default=os.path.join(os.path.dirname(__file__), "repro", "per_episode"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.per_episode_dir, exist_ok=True)

    # --- load the re-eval tree -------------------------------------------------
    rows = {}
    for path in sorted(glob.glob(os.path.join(args.jerk_root, "*", "seed1000", "*_open_loop_metrics.json"))):
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        report = json.load(open(path))
        fps = check_protocol(run, report)
        per_ep = episode_means(report)
        eps = sorted(per_ep)
        balanced = {m: float(np.mean([per_ep[e][m] for e in eps])) for m in ALL_METRICS if m in per_ep[eps[0]]}
        ci = {}
        for m in balanced:
            lo, hi = bootstrap_ci(np.array([per_ep[e][m] for e in eps]))
            ci[m] = (lo, hi)
        rows[run] = {
            "per_episode": per_ep,
            "balanced": balanced,
            "ci": ci,
            "fps": fps,
            "checkpoint": report["checkpoint"],
            "steps": int(step_of(path)),
        }

    # --- cross-check against the archived §9.2.9 tree -------------------------
    deltas = {m: 0.0 for m in CROSSCHECK}
    n_checked = 0
    for run, entry in sorted(rows.items()):
        arch = sorted(glob.glob(os.path.join(args.unified_root, run, "seed1000", "*_open_loop_metrics.json")))
        if not arch:
            print(f"WARN: no archived counterpart for {run}", file=sys.stderr)
            continue
        arch_report = json.load(open(arch[0]))
        if "samples" in arch_report:
            arch_ep = episode_means(arch_report)
            eps = sorted(arch_ep)
            for m in CROSSCHECK:
                arch_val = float(np.mean([arch_ep[e][m] for e in eps]))
                deltas[m] = max(deltas[m], abs(arch_val - entry["balanced"][m]))
        else:  # openpi schema: episode_balanced verbatim
            ab = arch_report["summary"]["episode_balanced"]
            for m in CROSSCHECK:
                deltas[m] = max(deltas[m], abs(ab[m] - entry["balanced"][m]))
        n_checked += 1

    # --- openpi rows carried from the archived tree (physical pending) --------
    carried = {}
    for run in sorted(OPENPI_RUNS):
        arch = sorted(glob.glob(os.path.join(args.unified_root, run, "seed1000", "*_open_loop_metrics.json")))
        if not arch:
            print(f"WARN: openpi row {run} missing from archived tree", file=sys.stderr)
            continue
        rep = json.load(open(arch[0]))
        carried[run] = {
            "balanced": {m: rep["summary"]["episode_balanced"][m] for m in CROSSCHECK},
            "per_episode": rep["summary"].get("per_episode", {}),
            "steps": int(step_of(arch[0])),
            "physical_pending": True,
        }

    # --- outputs ---------------------------------------------------------------
    def fmt(v, scale=1.0, nd=3):
        return f"{v * scale:.{nd}f}"

    md = ["| Run | step | XYZ end (mm) | rot 2nd-diff (deg) | rot vel (deg/s) | rot accel (deg/s²) | rot jerk (deg/s³) | xyz vel (mm/s) | xyz accel (mm/s²) | xyz jerk (mm/s³) | GT rot jerk (deg/s³) | GT xyz jerk (mm/s³) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    csv = ["run,steps,xyz_end_mm,rot_2nd_diff_deg,rot_vel_deg_s,rot_accel_deg_s2,rot_jerk_deg_s3,rot_jerk_deg_s3_lo,rot_jerk_deg_s3_hi,xyz_vel_mm_s,xyz_accel_mm_s2,xyz_jerk_mm_s3,xyz_jerk_mm_s3_lo,xyz_jerk_mm_s3_hi,gt_rot_vel_deg_s,gt_rot_accel_deg_s2,gt_rot_jerk_deg_s3,gt_xyz_vel_mm_s,gt_xyz_accel_mm_s2,gt_xyz_jerk_mm_s3"]
    for run in sorted(set(rows) | set(carried)):
        if run in rows:
            e = rows[run]
            b, c = e["balanced"], e["ci"]
            rj_lo, rj_hi = c["rot_jerk_deg_s3"]
            xj_lo, xj_hi = c["xyz_jerk_m_s3"]
            md.append(
                f"| {run} | {e['steps']} | {fmt(b['xyz_end_m'], 1000)} | {fmt(b['rot_jerk_deg'])} | "
                f"{fmt(b['rot_vel_deg_s'], 1, 2)} | {fmt(b['rot_accel_deg_s2'], 1, 1)} | "
                f"{fmt(b['rot_jerk_deg_s3'], 1, 0)} [{fmt(rj_lo, 1, 0)}, {fmt(rj_hi, 1, 0)}] | "
                f"{fmt(b['xyz_vel_m_s'], 1000, 1)} | {fmt(b['xyz_accel_m_s2'], 1000, 1)} | "
                f"{fmt(b['xyz_jerk_m_s3'], 1000, 1)} [{fmt(xj_lo, 1000, 1)}, {fmt(xj_hi, 1000, 1)}] | "
                f"{fmt(b['gt_rot_jerk_deg_s3'], 1, 0)} | {fmt(b['gt_xyz_jerk_m_s3'], 1000, 1)} |")
            csv.append(
                f"{run},{e['steps']},{b['xyz_end_m'] * 1000:.3f},{b['rot_jerk_deg']:.4f},"
                f"{b['rot_vel_deg_s']:.3f},{b['rot_accel_deg_s2']:.2f},{b['rot_jerk_deg_s3']:.1f},{rj_lo:.1f},{rj_hi:.1f},"
                f"{b['xyz_vel_m_s'] * 1000:.2f},{b['xyz_accel_m_s2'] * 1000:.2f},{b['xyz_jerk_m_s3'] * 1000:.2f},{xj_lo * 1000:.2f},{xj_hi * 1000:.2f},"
                f"{b['gt_rot_vel_deg_s']:.3f},{b['gt_rot_accel_deg_s2']:.2f},{b['gt_rot_jerk_deg_s3']:.1f},"
                f"{b['gt_xyz_vel_m_s'] * 1000:.2f},{b['gt_xyz_accel_m_s2'] * 1000:.2f},{b['gt_xyz_jerk_m_s3'] * 1000:.2f}")
        else:
            e = carried[run]
            b = e["balanced"]
            md.append(f"| {run} | {e['steps']} | {fmt(b['xyz_end_m'], 1000)} | {fmt(b['rot_jerk_deg'])} | "
                      f"— | — | host re-eval pending | — | — | host re-eval pending | — | — |")
            csv.append(f"{run},{e['steps']},{b['xyz_end_m'] * 1000:.3f},{b['rot_jerk_deg']:.4f},"
                      f"nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan")

    open(os.path.join(args.out_dir, "physical_jerk_h10.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(args.out_dir, "physical_jerk_h10.csv"), "w").write("\n".join(csv) + "\n")

    val = [f"cross-checked {n_checked} runs against the archived §9.2.9 tree (episode-balanced, seed 1000):"]
    val += [f"  max |Δ{m}| = {d:.3g}" for m, d in deltas.items()]
    val.append(f"re-eval rows: {len(rows)}, openpi rows carried (physical pending): {len(carried)}")
    open(os.path.join(args.out_dir, "validation.txt"), "w").write("\n".join(val) + "\n")
    print("\n".join(val))

    # --- per-episode repro files ----------------------------------------------
    for run, e in rows.items():
        payload = {
            "run": run,
            "checkpoint": e["checkpoint"],
            "steps": e["steps"],
            "protocol": {"bounds": [-1, 31], "eval_horizon": 10, "queries": 500, "seed": 1000, "fps": e["fps"]},
            "episode_balanced": e["balanced"],
            "episode_balanced_95ci": {m: list(v) for m, v in e["ci"].items()},
            "per_episode": {str(ep): e["per_episode"][ep] for ep in sorted(e["per_episode"])},
        }
        with gzip.open(os.path.join(args.per_episode_dir, f"{run}.json.gz"), "wt") as f:
            json.dump(payload, f)
    print(f"wrote {len(rows)} per-episode repro files to {args.per_episode_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
