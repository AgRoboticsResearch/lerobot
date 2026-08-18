#!/usr/bin/env python3
"""Cross-schema compiler for the unified NATIVE-h30 (full-chunk) evaluation
(§9.2.11) — every model whose action chunk supports 30 steps, scored over
the full chunk under the canonical 500-query window.

Mirrors compile_unified_h10.py (same aggregation + assertions), with the
horizon expectation inverted: LeRobot reports must record eval_horizon=None
(the flag was not passed → full 30-step scoring) or 30; openpi reports
action_horizon/eval_horizon == 30. Scans the eval_common_h32 tree, which
holds:

  - the historical R18-VAE 30-checkpoint budget curve (§9.2.7);
  - the fresh R50-V1 1M 10-checkpoint curve (§9.2.8);
  - seed-23k/2000/3000 companions (ACT-L1, ACT-flow, R50-VAE);
  - the π0.5-port h30 curve — run names pi05_port_<STEP>_h30_v2 (host
    front-run 50k–900k minus 650k/700k; kiwi K2 owns 650K/700K/1M);
  - SmolVLA rows as they land (notation ×3 seeds via K3, the 1M full-width
    curve, the kiwi masked-subspace 1M curve).

Protocol assertions enforced on EVERY report before it is admitted:
  - query bounds exactly {min: -1, max: 31} (canonical 500-query set);
  - full-chunk scoring horizon (LeRobot eval_horizon None/30; openpi 30);
  - 500 scored queries / 100 episodes;
  - accuracy@τ per-dim half-ranges identical across all admitted rows;
  - ground-truth jerk episode-balanced values identical across all rows.

Usage:
  uv run python compile_unified_h30.py \
    [--h30_root /mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/eval_common_h32] \
    [--out_dir /mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# Run-name grammars kept in lockstep with collect_results.py's v2 pass.
RUN_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)_(?P<steps>\d+)steps$")
HIST_RUN_RE = re.compile(r"^(?P<variant>act_umi_identity_rot6d_1459)_(?P<steps>\d{7})steps$")
# kiwi K2 / host front-run π0.5-port h30 naming (no _seed<..>_ suffix)
PORT_H30_RE = re.compile(r"^pi05_port_(?P<steps>\d{6,7})_h30_v2$")
# Authoritative evaluated checkpoint step, from the report filename (companions
# sit in 100000steps run dirs but were evaluated at their true 80k/50k ckpts).
STEP_RE = re.compile(r"_(?P<step>\d{6,7})_open_loop_metrics\.json$")

# Co-primary metrics (§9.2.9 definitions) + reference columns.
METRICS = [
    "xyz_end_m",
    "rotation_end_deg",
    "xyz_chunk_mean_m",
    "rotation_chunk_mean_deg",
    "xyz_l1_per_dim_m",
    "xyz_mse_per_dim_m2",
    "rotvec_l1_per_dim_deg",
    "rotvec_mse_per_dim_deg2",
    "action_acc_at_0p5",
    "action_acc_at_0p1",
    "xyz_acc_at_0p5",
    "xyz_acc_at_0p1",
    "rotvec_acc_at_0p5",
    "rotvec_acc_at_0p1",
    "rot_jerk_deg",
    "xyz_jerk_m",
    "gt_rot_jerk_deg",
    "gt_xyz_jerk_m",
    "gripper_end",
]
# Invariance checks: GT-only metrics must be identical across every row.
GT_METRICS = ["gt_rot_jerk_deg", "gt_xyz_jerk_m"]

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0


def bootstrap_ci(per_episode: np.ndarray) -> tuple[float, float]:
    """Deterministic 95% nonparametric bootstrap over episode means."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, per_episode.shape[0], size=(N_BOOTSTRAP, per_episode.shape[0]))
    means = per_episode[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def episode_means_lerobot(report: dict) -> dict[int, dict[str, float]]:
    """Per-episode metric means from a LeRobot-schema report's samples."""
    by_ep: dict[int, list[dict]] = defaultdict(list)
    for s in report["samples"]:
        by_ep[int(s["episode_index"])].append(s)
    out: dict[int, dict[str, float]] = {}
    for ep, samples in by_ep.items():
        out[ep] = {
            m: float(np.mean([s[m] for s in samples])) for m in METRICS if m in samples[0]
        }
    return out


def check_protocol(run: str, path: str, report: dict, tau_ref: list[float] | None) -> list[float]:
    bounds = report.get("query_action_offset_bounds")
    if bounds != {"min": -1, "max": 31}:
        raise ValueError(f"{run}: non-canonical query bounds {bounds} ({path})")
    eh, ah = report.get("eval_horizon"), report.get("action_horizon")
    if eh not in (None, 30) and ah != 30:
        raise ValueError(
            f"{run}: scoring horizon is not full-chunk/30 (eval_horizon={eh}, action_horizon={ah})"
        )
    if "samples" in report:
        n = len(report["samples"])
    else:
        n = int(report.get("num_samples", len(report.get("query_frames", []))))
    if n != 500:
        raise ValueError(f"{run}: {n} scored queries, expected 500 ({path})")
    tau = report["accuracy_at_tau_normalization"]["per_dim_half_ranges"]
    if len(tau) != 7:
        raise ValueError(f"{run}: expected 7 per-dim half-ranges, got {len(tau)}")
    if tau_ref is not None and not np.allclose(tau, tau_ref, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{run}: acc@τ normalization scales differ from reference ({path})")
    return [float(t) for t in tau]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--h30_root",
        default="/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/eval_common_h32",
    )
    ap.add_argument(
        "--out_dir",
        default="/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # run -> inference seed -> report path
    runs: dict[str, dict[int, str]] = defaultdict(dict)
    for path in sorted(
        glob.glob(os.path.join(args.h30_root, "*", "seed*", "*_open_loop_metrics.json"))
    ):
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m = RUN_RE.match(run) or HIST_RUN_RE.match(run) or PORT_H30_RE.match(run)
        if not m:
            print(f"WARN: unparseable run dir {run}, skipping", file=sys.stderr)
            continue
        seed_dir = os.path.basename(os.path.dirname(path))
        runs[run][int(seed_dir.removeprefix("seed"))] = path

    tau_ref: list[float] | None = None
    gt_ref: dict[str, float] | None = None
    summary_rows: list[dict] = []
    eval_rows: list[dict] = []

    for run in sorted(runs):
        m = RUN_RE.match(run) or HIST_RUN_RE.match(run) or PORT_H30_RE.match(run)
        assert m is not None, run  # filtered during collection
        gd = m.groupdict()
        variant = gd["variant"] if "variant" in gd else "pi05_port_h30"
        train_seed = int(gd.get("seed", 1000))
        # authoritative evaluated step: from the run name (port grammar) or
        # the (shared) report filename (companions' true 80k/50k ckpts)
        fname = os.path.basename(next(iter(runs[run].values())))
        sm = STEP_RE.search(fname)
        assert sm is not None, fname
        step = int(gd["steps"]) if "steps" in gd and PORT_H30_RE.match(run) else int(sm.group("step"))

        # Schema detection: every report of a run shares one schema.
        reports = {}
        for seed, path in sorted(runs[run].items()):
            with open(path) as f:
                reports[seed] = json.load(f)
            tau_ref = check_protocol(run, path, reports[seed], tau_ref)

        first = reports[next(iter(reports))]
        if "samples" in first:
            per_seed_eps = {seed: episode_means_lerobot(rep) for seed, rep in reports.items()}
            ep_sets = [set(e) for e in per_seed_eps.values()]
            if any(s != ep_sets[0] for s in ep_sets[1:]):
                raise ValueError(f"{run}: episode sets differ across inference seeds")
            episodes = sorted(ep_sets[0])
            if len(episodes) != 100:
                raise ValueError(f"{run}: {len(episodes)} episodes, expected 100")
            # average inference seeds within each episode
            per_episode = {
                met: np.array(
                    [
                        float(np.mean([per_seed_eps[s][ep][met] for s in sorted(per_seed_eps)]))
                        for ep in episodes
                    ]
                )
                for met in METRICS
                if met in next(iter(per_seed_eps.values()))[episodes[0]]
            }
            row = {"run": run, "variant": variant, "train_seed": train_seed, "step": step}
            for seed in sorted(per_seed_eps):
                ev = {"run": run, "inference_seed": seed, "step": step}
                for met, vals in per_episode.items():
                    if met in GT_METRICS:
                        continue
                    ev[met] = float(np.mean([per_seed_eps[seed][ep][met] for ep in episodes]))
                eval_rows.append(ev)
        else:
            eb = first["summary"]["episode_balanced"]
            per_episode = None
            row = {"run": run, "variant": variant, "train_seed": train_seed, "step": step}

        row["inference_seeds"] = ",".join(str(s) for s in sorted(reports))
        row["n_inference_seeds"] = len(reports)
        row["n_episodes"] = 100

        if per_episode is not None:  # LeRobot schema
            for met, vals in per_episode.items():
                mean = float(vals.mean())
                row[met] = mean
                if met in GT_METRICS:
                    continue
                lo, hi = bootstrap_ci(vals)
                row[f"{met}__ci_low"], row[f"{met}__ci_high"] = lo, hi
        else:  # openpi schema: take the evaluator's own episode-balanced + CI
            eb = first["summary"]["episode_balanced"]
            ci = first["summary"]["episode_balanced_95ci"]
            for met in METRICS:
                if met not in eb:
                    continue
                row[met] = float(eb[met])
                if met in GT_METRICS:
                    continue
                row[f"{met}__ci_low"] = float(ci[met]["low"])
                row[f"{met}__ci_high"] = float(ci[met]["high"])

        # GT invariance across all rows (protocol check, not a model metric).
        for met in GT_METRICS:
            if met in row:
                if gt_ref is None:
                    gt_ref = {g: row[g] for g in GT_METRICS if g in row}
                elif abs(row[met] - gt_ref[met]) > 1e-6 * max(abs(gt_ref[met]), 1e-9):
                    raise ValueError(
                        f"{run}: GT {met}={row[met]!r} differs from reference {gt_ref[met]!r}"
                    )
        summary_rows.append(row)

    # ---- outputs -----------------------------------------------------------
    fieldnames = (
        ["run", "variant", "train_seed", "step", "inference_seeds", "n_inference_seeds", "n_episodes"]
        + METRICS
        + [f"{m}__ci_{s}" for m in METRICS for s in ("low", "high") if f"{m}__ci_{s}" in summary_rows[0]]
    )
    # union of available columns (metric presence can differ while the sweep
    # is mid-flight; keep the CSV superset)
    seen: list[str] = []
    for r in summary_rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    fieldnames = [c for c in fieldnames if c in seen] + [c for c in seen if c not in fieldnames]

    out_csv = os.path.join(args.out_dir, "unified_h30_run_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    eval_csv = os.path.join(args.out_dir, "unified_h30_evaluations.csv")
    ev_fields = ["run", "inference_seed", "step"] + [
        m for m in METRICS if any(m in e for e in eval_rows)
    ]
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(eval_rows)

    # ---- markdown table (co-primary metrics; mm / deg / µm² / pp) ---------
    def cell(r: dict, met: str, scale: float, fmt: str = "{:.2f}", ci: bool = True) -> str:
        v = r[met] * scale
        if not ci or f"{met}__ci_low" not in r:
            return fmt.format(v)
        return f"{fmt.format(v)} [{fmt.format(r[f'{met}__ci_low'] * scale)}, {fmt.format(r[f'{met}__ci_high'] * scale)}]"

    print("| Run | step | XYZ end (mm) | Rot end (deg) | XYZ L1/dim (mm) | XYZ MSE/dim (µm²) | Rotvec L1/dim (deg) | Rotvec MSE/dim (deg²) | acc@0.5 | acc@0.1 | Rot jerk (deg) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        print(
            f"| {r['run']} | {r['step']} | {cell(r, 'xyz_end_m', 1000)} | {cell(r, 'rotation_end_deg', 1)} "
            f"| {cell(r, 'xyz_l1_per_dim_m', 1000, ci=False)} | {cell(r, 'xyz_mse_per_dim_m2', 1e6, '{:.1f}', False)} "
            f"| {cell(r, 'rotvec_l1_per_dim_deg', 1, '{:.3f}', False)} | {cell(r, 'rotvec_mse_per_dim_deg2', 1, '{:.2f}', False)} "
            f"| {cell(r, 'action_acc_at_0p5', 1, '{:.3f}', False)} "
            f"| {cell(r, 'action_acc_at_0p1', 1, '{:.3f}')} | {cell(r, 'rot_jerk_deg', 1, '{:.3f}', False)} |"
        )
    assert gt_ref is not None and tau_ref is not None  # set with the first admitted row
    print(
        f"\nGT reference (identical across all rows): rot jerk {gt_ref['gt_rot_jerk_deg']:.3f} deg, "
        f"xyz jerk {gt_ref['gt_xyz_jerk_m'] * 1000:.2f} mm; acc@τ half-ranges "
        f"{[round(t, 5) for t in tau_ref]}"
    )
    print(f"\nwrote {out_csv} ({len(summary_rows)} runs) and {eval_csv} ({len(eval_rows)} evals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
