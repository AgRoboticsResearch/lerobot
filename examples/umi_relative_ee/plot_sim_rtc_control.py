#!/usr/bin/env python

"""Comparison plots for sim_rtc_control_test.py JSON logs.

Reads the per-tick control log (commanded/achieved EE, underruns, chunk
boundaries) and renders RTC vs no-RTC execution-dynamics comparisons:

  - timeseries: tracking error + EE speed per episode, underrun shading
  - summary bars: track error, chunk-switch step vs in-chunk step, stall %, jerk
  - 3D absolute EE paths per episode
  - summary JSON with the aggregate numbers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Okabe-Ito colorblind-safe pair; fixed assignment (color follows the arm)
COLORS = {"no_rtc": "#0072B2", "rtc": "#D55E00"}
LABELS = {"no_rtc": "no RTC (clear & replan)", "rtc": "RTC (guided replan)"}
GRID = {"color": "0.85", "linewidth": 0.6}
FPS = 30  # control tick rate; converts per-tick deltas to per-second units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="outputs/debug/sim_rtc_control/sim_rtc_control.json")
    parser.add_argument("--out_dir", default=None, help="default: alongside the log")
    return parser.parse_args()


def run_arrays(run: dict) -> dict:
    ticks = run["ticks"]
    ee = np.array([t["ee"] for t in ticks], dtype=float)
    cmd = np.array([t["cmd_ee"] for t in ticks], dtype=float)
    t = np.array([t["t"] for t in ticks], dtype=float)
    underrun = np.array([t["underrun"] for t in ticks], dtype=bool)
    boundary = np.array([t["boundary"] for t in ticks], dtype=bool)
    executed = ~underrun

    track_err = np.linalg.norm(cmd[:, :3] - ee[:, :3], axis=1) * 1000.0  # mm
    speed = np.linalg.norm(np.diff(ee[:, :3], axis=0), axis=1) * 1000.0 * FPS  # mm/s
    jerk = np.abs(np.diff(speed)) * FPS  # mm/s^2
    # Jerk measured only while the arm is actually moving: stalled ticks (queue
    # underrun during inference) contribute zeros and deflate the mean.
    moving = speed > 0.2 * FPS  # mm/s
    jerk_moving = jerk[moving[1:]] if moving.sum() > 2 else jerk

    # Commanded step between consecutive EXECUTED targets, split at chunk switches
    exec_idx = np.where(executed)[0]
    steps = np.full(len(ticks), np.nan)
    for a, b in zip(exec_idx[:-1], exec_idx[1:], strict=False):
        steps[b] = np.linalg.norm(cmd[b, :3] - cmd[a, :3]) * 1000.0
    switch_steps = steps[boundary & ~np.isnan(steps)]
    inchunk_steps = steps[~boundary & ~np.isnan(steps)]

    return {
        "t": t, "ee": ee, "cmd": cmd, "track_err": track_err, "speed": speed,
        "jerk": jerk, "jerk_moving": jerk_moving, "underrun": underrun, "boundary": boundary,
        "switch_steps": switch_steps, "inchunk_steps": inchunk_steps,
        "infer_s": [c["infer_s"] for c in run["chunks"]],
        "n_ticks": len(ticks),
    }


def shade_underrun(ax, data) -> None:
    for i, flag in enumerate(data["underrun"]):
        if flag:
            ax.axvspan(data["t"][i] - 1e-9, data["t"][min(i + 1, len(data["t"]) - 1)],
                       color="0.0", alpha=0.06, linewidth=0)


def plot_timeseries(runs_by, episodes, out_path) -> None:
    fig, axes = plt.subplots(
        len(episodes), 2, figsize=(11.5, 2.6 * len(episodes)), squeeze=False,
        sharex="col",
    )
    for row, ep in enumerate(episodes):
        ax_err, ax_spd = axes[row][0], axes[row][1]
        for arm in ("no_rtc", "rtc"):
            data = runs_by[(arm, ep)]
            ax_err.plot(data["t"], data["track_err"], color=COLORS[arm],
                        label=LABELS[arm], linewidth=1.4)
            shade_underrun(ax_err, data)
            ax_spd.plot(data["t"][1:], data["speed"], color=COLORS[arm],
                        label=LABELS[arm], linewidth=1.4)
            for b in np.where(data["boundary"])[0]:
                ax_spd.axvline(data["t"][b], color=COLORS[arm], alpha=0.25,
                               linewidth=0.8, linestyle=":")
        ax_err.set_title(f"episode {ep}: EE tracking error (cmd vs achieved)", fontsize=10)
        ax_spd.set_title(f"episode {ep}: achieved EE speed (dotted = chunk switch)", fontsize=10)
        ax_err.set_ylabel("error [mm]")
        ax_spd.set_ylabel("speed [mm/s]")
        for ax in (ax_err, ax_spd):
            ax.grid(True, **GRID)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax_err.legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary(summary, out_path) -> None:
    metrics = [
        ("track_err_mean", "mean tracking error [mm]", "%.1f"),
        ("switch_step_mean", "chunk-switch step [mm]", "%.1f"),
        ("inchunk_step_mean", "in-chunk step [mm]", "%.2f"),
        ("stall_pct", "stalled ticks [%]", "%.0f"),
        ("speed_p95", "peak EE speed [mm/s]", "%.0f"),
        ("jerk_moving", "|Δspeed| while moving\n[mm/s²]", "%.0f"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.5 * len(metrics), 3.1))
    arms = ["no_rtc", "rtc"]
    for ax, (key, title, fmt) in zip(axes, metrics, strict=True):
        values = [summary[arm][key] for arm in arms]
        bars = ax.bar([LABELS[a] for a in arms], values, color=[COLORS[a] for a in arms],
                      width=0.62, edgecolor="white", linewidth=1)
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(fmt % value, (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=8.5, color="0.15")
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, axis="y", **GRID)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_traj3d(runs_by, episodes, out_path) -> None:
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    origin = runs_by[("no_rtc", episodes[0])]["ee"][0, :3] * 1000.0  # mm, common start
    for ep in episodes:
        for arm in ("no_rtc", "rtc"):
            data = runs_by[(arm, ep)]
            ee = data["ee"][:, :3] * 1000.0 - origin  # start-relative mm
            ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=COLORS[arm], linewidth=1.2,
                    label=f"{LABELS[arm]} (ep {ep})" if ep == episodes[0] else None,
                    alpha=0.5 + 0.5 * (ep == episodes[0]))
    ax.scatter(0, 0, 0, color="0.2", s=28, marker="o", label="start")
    ax.set_xlabel("Δx from start [mm]")
    ax.set_ylabel("Δy from start [mm]")
    ax.set_zlabel("Δz from start [mm]")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Achieved EE paths in piper-sim (start-relative)")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.view_init(elev=28, azim=-58)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_dir = Path(args.out_dir) if args.out_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path) as f:
        payload = json.load(f)
    runs = payload["runs"]
    episodes = sorted({r["episode"] for r in runs})
    runs_by = {(r["arm"], r["episode"]): run_arrays(r) for r in runs}

    summary: dict[str, dict] = {}
    for arm in ("no_rtc", "rtc"):
        errs = np.concatenate([runs_by[(arm, ep)]["track_err"] for ep in episodes])
        switches = np.concatenate([runs_by[(arm, ep)]["switch_steps"] for ep in episodes])
        inchunk = np.concatenate([runs_by[(arm, ep)]["inchunk_steps"] for ep in episodes])
        stalls = np.concatenate([runs_by[(arm, ep)]["underrun"] for ep in episodes])
        jerks = np.concatenate([runs_by[(arm, ep)]["jerk"] for ep in episodes])
        jerks_moving = np.concatenate([runs_by[(arm, ep)]["jerk_moving"] for ep in episodes])
        speeds = np.concatenate([runs_by[(arm, ep)]["speed"] for ep in episodes])
        infers = np.concatenate([runs_by[(arm, ep)]["infer_s"] for ep in episodes])
        summary[arm] = {
            "track_err_mean_mm": float(errs.mean()),
            "track_err_median_mm": float(np.median(errs)),
            "track_err_p95_mm": float(np.percentile(errs, 95)),
            "switch_step_mean_mm": float(switches.mean()) if len(switches) else float("nan"),
            "switch_step_max_mm": float(switches.max()) if len(switches) else float("nan"),
            "inchunk_step_mean_mm": float(inchunk.mean()) if len(inchunk) else float("nan"),
            "stall_pct": float(stalls.mean() * 100.0),
            "jerk_mean": float(jerks.mean()),
            "jerk_moving_mean": float(jerks_moving.mean()),
            "speed_mean_mm_s": float(speeds.mean()),
            "speed_p95_mm_s": float(np.percentile(speeds, 95)),
            "speed_max_mm_s": float(speeds.max()),
            "infer_s_mean": float(infers.mean()),
            "n_ticks": int(sum(runs_by[(arm, ep)]["n_ticks"] for ep in episodes)),
        }

    plot_timeseries(runs_by, episodes, out_dir / "sim_rtc_control_timeseries.png")
    plot_summary(
        {arm: {
            "track_err_mean": summary[arm]["track_err_mean_mm"],
            "switch_step_mean": summary[arm]["switch_step_mean_mm"],
            "inchunk_step_mean": summary[arm]["inchunk_step_mean_mm"],
            "stall_pct": summary[arm]["stall_pct"],
            "speed_p95": summary[arm]["speed_p95_mm_s"],
            "jerk_moving": summary[arm]["jerk_moving_mean"],
        } for arm in ("no_rtc", "rtc")},
        out_dir / "sim_rtc_control_summary.png",
    )
    plot_traj3d(runs_by, episodes, out_dir / "sim_rtc_control_traj3d.png")

    with open(out_dir / "sim_rtc_control_summary.json", "w") as f:
        json.dump({"summary": summary, "log": str(log_path)}, f, indent=2)

    for arm in ("no_rtc", "rtc"):
        s = summary[arm]
        print(
            f"{arm:7s}: track {s['track_err_mean_mm']:.1f}mm (p95 {s['track_err_p95_mm']:.1f}) | "
            f"switch {s['switch_step_mean_mm']:.1f}mm vs in-chunk {s['inchunk_step_mean_mm']:.2f}mm | "
            f"stall {s['stall_pct']:.0f}% | peak speed {s['speed_p95_mm_s']:.0f}mm/s | "
            f"jerk(moving) {s['jerk_moving_mean']:.0f}mm/s2 | infer {s['infer_s_mean']:.2f}s"
        )
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
