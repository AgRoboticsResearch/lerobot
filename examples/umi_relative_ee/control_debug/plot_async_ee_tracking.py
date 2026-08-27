"""Plot commanded-vs-measured EE tracking for ONE async ``--log`` run.

Reads a control-log triplet (.npz/.csv/.json) written by the async Piper client
(same schema as ``make_control_debug_figs.py``) and produces, under
``outputs/research_report/async_ee_tracking/<stem>/figures/``:

  ee_tracking.png     7 stacked panels — x = time, y = commanded absolute EE
                      target (``action_ee``) vs sensor FK (``current_ee``)
  joint_tracking.png  6 stacked panels — IK commanded joints vs measured joints
  diagnostics.png     queue size + chunk round-trip timings (one axis per panel)

PAUSED spans are shaded with the neutral surface tint; queue-empty (underrun)
spans while running with a faint amber tint, so gaps in the commanded trace can
be read at a glance.

Usage:
  python examples/umi_relative_ee/control_debug/plot_async_ee_tracking.py \
      [stem] [out_dir]
  stem    : path or stem of a control log under logs/ (default: latest async_*)
  out_dir : output dir (default: outputs/research_report/async_ee_tracking/<stem>)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Sibling module (script dir is on sys.path both run-from-repo-root and in-dir);
# importing it also applies the shared dataviz rcParams.
from make_control_debug_figs import AQUA, BLUE, EE_LABELS, INK2, MUTED, PAUSED_FILL, resolve_stem

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS = REPO / "logs"

JOINT_LABELS = [f"joint{i+1} [rad]" for i in range(6)]
UNDERRUN_FILL = "#f7ecd2"  # faint amber tint for queue-empty-while-running


def spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs where mask is True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1] + 1]
    return list(zip(starts, stops))


def shade_spans(ax, mask: np.ndarray, t: np.ndarray, color: str) -> None:
    for s, e in spans(mask):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=color, lw=0, zorder=0)


def main() -> None:
    stem_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if stem_arg is None:
        candidates = sorted(LOGS.glob("async_*.npz"))
        if not candidates:
            raise SystemExit("no async_* logs under logs/ — pass a stem")
        stem = candidates[-1].with_suffix("")
        print(f"using latest log: {stem.name}")
    else:
        stem = resolve_stem(stem_arg)

    out_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "outputs" / "research_report" / "async_ee_tracking" / stem.name
    )
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    d = dict(np.load(stem.with_suffix(".npz"), allow_pickle=True))
    meta = json.loads(stem.with_suffix(".json").read_text())
    t = d["t_s"]
    t = t - t[0]
    cmd_ee, meas_ee = d["action_ee"], d["current_ee"]
    cmd_j, meas_j = d["ik_joints_rad"], d["current_joints_rad"]
    queue = d["queue"]

    # String columns (state) live in the csv.
    import csv as _csv

    with open(stem.with_suffix(".csv")) as fh:
        rows = list(_csv.DictReader(fh))
    state = np.array([r.get("state", "") for r in rows])
    valid_ee = ~np.isnan(cmd_ee[:, 0])
    valid_j = ~np.isnan(cmd_j[:, 0]) & valid_ee

    paused = state == "PAUSED"
    underrun = (state == "INFERENCE") & (queue == 0)

    for fig_out, (cmd, meas, labels, v, title) in {
        "ee_tracking.png": (
            cmd_ee,
            meas_ee,
            EE_LABELS,
            valid_ee,
            f"EE commanded vs measured — {stem.name} ({meta['policy_type']}, "
            f"{meta['aggregate_fn_name']}, thr={meta['chunk_size_threshold']})",
        ),
        "joint_tracking.png": (
            cmd_j,
            meas_j,
            JOINT_LABELS,
            valid_j,
            f"Joint commanded (IK) vs measured — {stem.name}",
        ),
    }.items():
        fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(9, 1.55 * len(labels)))
        bmin = meta.get("ee_bounds_min")
        bmax = meta.get("ee_bounds_max")
        for i, ax in enumerate(np.atleast_1d(axes)):
            shade_spans(ax, paused, t, PAUSED_FILL)
            shade_spans(ax, underrun, t, UNDERRUN_FILL)
            ax.plot(t[v], cmd[v, i], color=BLUE, lw=1.4, label="commanded", zorder=3)
            ax.plot(t, meas[:, i], color=AQUA, lw=1.1, alpha=0.9, label="measured", zorder=2)
            if i < 3 and bmin is not None and bmax is not None:
                # Workspace box from EEBoundsAndSafety — commanded is logged
                # PRE-clamp, so it may cross these while measured cannot.
                ax.axhline(bmax[i], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
                ax.axhline(bmin[i], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax.set_ylabel(labels[i], rotation=0, ha="right", va="center", labelpad=28)
        np.atleast_1d(axes)[0].legend(loc="upper right", ncol=2)
        np.atleast_1d(axes)[-1].set_xlabel("time [s]")
        fig.suptitle(title, x=0.11, y=0.995, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(figs / fig_out, dpi=150)
        plt.close(fig)
        print(f"wrote {figs / fig_out}")

    # ── diagnostics: queue depth + per-chunk timings (one axis per panel) ──
    fig, (ax_q, ax_t) = plt.subplots(2, 1, sharex=True, figsize=(9, 4.2))
    shade_spans(ax_q, paused, t, PAUSED_FILL)
    ax_q.step(t, queue, where="post", color=BLUE, lw=1.2)
    ax_q.set_ylabel("queue [actions]")
    ax_q.set_title(f"Queue depth and chunk round-trip — {stem.name}")
    got_chunk = ~np.isnan(d["chunk_id"]) & np.r_[True, np.diff(np.nan_to_num(d["chunk_id"], -1)) != 0]
    ax_q.scatter(t[got_chunk], queue[got_chunk], s=14, color=INK2, zorder=3, label="chunk arrival")
    ax_q.legend(loc="upper right")
    for key, label, color in (
        ("e2e_ms", "e2e [ms]", BLUE),
        ("server_ms", "server [ms]", AQUA),
        ("wire_ms", "wire [ms]", MUTED),
    ):
        y = d[key]
        m = ~np.isnan(y)
        ax_t.plot(t[m], y[m], color=color, lw=1.0, label=label)
    ax_t.set_ylabel("latency [ms]")
    ax_t.set_xlabel("time [s]")
    ax_t.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(figs / "diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"wrote {figs / 'diagnostics.png'}")

    # ── headline tracking-error stats ──
    err = cmd_ee[valid_ee] - meas_ee[valid_ee]
    stats = {
        "stem": stem.name,
        "executed_ticks": int(valid_ee.sum()),
        "underrun_ticks": int(underrun.sum()),
        "paused_ticks": int(paused.sum()),
        "ee_abs_err_mean": dict(zip(EE_LABELS, np.round(np.abs(err).mean(axis=0), 5).tolist())),
        "ee_abs_err_max": dict(zip(EE_LABELS, np.round(np.abs(err).max(axis=0), 5).tolist())),
        "xyz_err_norm_mean_m": float(np.linalg.norm(err[:, :3], axis=1).mean()),
        "xyz_err_norm_max_m": float(np.linalg.norm(err[:, :3], axis=1).max()),
    }
    (out_dir / "computed_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
