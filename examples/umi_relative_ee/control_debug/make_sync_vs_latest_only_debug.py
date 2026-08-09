"""Deep debug comparison: SYNC ACT vs ASYNC `latest_only` (same ACT checkpoint).

Answers: why does async `latest_only` reverse direction 29% while SYNC is 6%, when
`latest_only` does no blending? Generates a detailed plot set keyed on the per-tick
chunk_id (now logged) so each re-plan / chunk arrival is visible on the timeline:

  ee_pose_with_arrivals.png      EE pose (7 dims) with chunk-arrival vlines
  step_reversal_queue_arrivals.png per-tick EE step + signed step (reversal) + queue, arrivals marked
  stitching_jumps.png            discontinuity injected at each re-plan (async merge.csv / sync switch)
  reversal_clustering.png        do reversals cluster at chunk arrivals?
  consecutive_chunks.png         consecutive chunks' EE-x predictions overlaid (agree vs disagree)

Reads logs/sync_20260807_150336 + logs/async_20260807_150005; writes into
outputs/research_report/low_level_control_debug/sync_vs_latest_only/figures/.
Joint columns are DEGREES (mislabeled "_rad").
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS = REPO / "logs"
OUTDIR = REPO / "outputs" / "research_report" / "low_level_control_debug" / "sync_vs_latest_only"

SYNC = "sync_20260807_150336"
ASYNC = "async_20260807_150005"

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, RED, VIOLET, AQUA = "#2a78d6", "#eb6834", "#008300", "#e34948", "#4a3aa7", "#1baf7a"
CAT = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", GREEN, VIOLET, RED]
ARR = "#bdb006"  # arrival-marker color
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 110, "legend.frameon": False,
})
EE_LABELS = ["x [m]", "y [m]", "z [m]", "rot_x [rad]", "rot_y [rad]", "rot_z [rad]", "gripper"]


def load(stem: str) -> dict:
    base = LOGS / stem
    d = {k: v for k, v in np.load(base.with_suffix(".npz")).items()}
    with base.with_suffix(".csv").open() as fh:
        rows = list(csv.DictReader(fh))
    d["state"] = np.array([r["state"] for r in rows])
    return d


def valid_idx(d: dict) -> np.ndarray:
    return np.where(d["ik_ok"].astype(bool))[0]


def arrivals(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (arrival_times, arrival_global_idx) — first executed tick of each chunk_id."""
    idx = valid_idx(d)
    cid = d["chunk_id"][idx]
    if not np.isfinite(cid).any():
        return np.array([]), np.array([])
    change = np.concatenate([[True], cid[1:] != cid[:-1]])
    pos = np.where(change)[0]
    return d["t_s"][idx[pos]], idx[pos]


def ee_steps(d: dict) -> dict:
    idx = valid_idx(d)
    ee = d["action_ee"][idx, :3]
    steps = np.diff(ee, axis=0)
    mag = np.linalg.norm(steps, axis=1) * 1e3
    mdir = steps.mean(0)
    n = np.linalg.norm(mdir)
    mdir = mdir / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    signed = steps @ mdir * 1e3
    reversal = float(np.mean(np.sum(steps[1:] * steps[:-1], axis=1) < 0)) if steps.shape[0] > 1 else float("nan")
    return {"t": d["t_s"][idx][1:], "mag": mag, "signed": signed, "reversal": reversal, "idx": idx}


def mark_arrivals(ax, t_arr: np.ndarray) -> None:
    for x in t_arr:
        ax.axvline(x, color=ARR, lw=0.5, alpha=0.45, zorder=1)


def xlim_valid(ax, d: dict) -> None:
    idx = valid_idx(d)
    t = d["t_s"][idx]
    if t.size:
        pad = max((t[-1] - t[0]) * 0.03, 1e-3)
        ax.set_xlim(t[0] - pad, t[-1] + pad)


SYNC_D = load(SYNC)
ASYNC_D = load(ASYNC)
RUNS = [("SYNC · ACT", SYNC_D), ("ASYNC · latest_only", ASYNC_D)]
FIG = OUTDIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ───────────── Figure A: EE pose with chunk-arrival markers ─────────────
def fig_ee_arrivals() -> None:
    fig, axes = plt.subplots(7, 2, figsize=(12, 12.5), sharex="col")
    for c, (label, d) in enumerate(RUNS):
        t_arr, _ = arrivals(d)
        idx = valid_idx(d)
        axes[0, c].set_title(f"{label}  —  {len(t_arr)} chunk arrivals (yellow)", color=INK)
        for r in range(7):
            ax = axes[r, c]
            mark_arrivals(ax, t_arr)
            ax.plot(d["t_s"][idx], d["action_ee"][idx, r], color=CAT[r], lw=1.0,
                    marker="o", markersize=2.5, markeredgecolor=SURF, markeredgewidth=0.3, zorder=3)
            ax.set_ylabel(EE_LABELS[r], fontsize=8)
            ax.tick_params(labelsize=7)
        xlim_valid(axes[0, c], d)
        axes[-1, c].set_xlabel("loop time [s]  (executed window)")
    fig.suptitle("EE pose vs time, chunk arrivals marked — SYNC replans ~every 32 ticks, ASYNC ~every 11",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG / "ee_pose_with_arrivals.png", bbox_inches="tight")
    plt.close(fig)


# ─────────── Figure B: step / reversal / queue + arrivals ──────────────
def fig_step_reversal_queue() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="col")
    for c, (label, d) in enumerate(RUNS):
        t_arr, _ = arrivals(d)
        es = ee_steps(d)
        ax = axes[0, c]
        mark_arrivals(ax, t_arr)
        ax.scatter(es["t"], es["mag"], s=10, color=BLUE, alpha=0.7, zorder=3)
        ax.set_title(label, color=INK)
        ax.set_ylabel("EE step [mm]")

        ax = axes[1, c]
        mark_arrivals(ax, t_arr)
        s = es["signed"]
        ax.scatter(es["t"][s >= 0], s[s >= 0], s=10, color=BLUE, alpha=0.7, zorder=3)
        ax.scatter(es["t"][s < 0], s[s < 0], s=12, color=RED, alpha=0.85, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel("signed EE step\nalong motion [mm]")
        ax.set_title(f"reversal {es['reversal']:.0%}", color=INK2)

        ax = axes[2, c]
        mark_arrivals(ax, t_arr)
        idx = valid_idx(d)
        ax.plot(d["t_s"][idx], d["queue"][idx], color=VIOLET, lw=1.0, zorder=3)
        ax.set_ylabel("queue depth")
        ax.set_xlabel("loop time [s]")
        xlim_valid(ax, d)
    fig.suptitle("Per-tick EE step, direction reversal, and queue — yellow = chunk arrival",
                 fontsize=11.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "step_reversal_queue_arrivals.png", bbox_inches="tight")
    plt.close(fig)


# ─────────── Figure C: discontinuity injected at each re-plan ───────────
def fig_stitching() -> None:
    fig, (axs, axa) = plt.subplots(1, 2, figsize=(12, 4.8))
    # SYNC: switch jump at each chunk boundary = |action[new chunk first] - action[old chunk last]|
    idx = valid_idx(SYNC_D)
    cid = SYNC_D["chunk_id"][idx]
    ch = np.where(np.concatenate([[True], cid[1:] != cid[:-1]]))[0]
    jumps_t, jumps_m = [], []
    for i in range(1, len(ch)):
        a0 = SYNC_D["action_ee"][idx[ch[i] - 1], :3]      # last executed of old chunk
        a1 = SYNC_D["action_ee"][idx[ch[i]], :3]          # first executed of new chunk
        jumps_t.append(SYNC_D["t_s"][idx[ch[i]]])
        jumps_m.append(np.linalg.norm(a1 - a0) * 1e3)
    axs.stem(jumps_t, jumps_m, basefmt=" ", linefmt=BLUE, markerfmt="o")
    axs.set_title(f"SYNC — chunk-switch jump (n={len(jumps_m)})", color=INK2)
    axs.set_ylabel("|ΔEE across boundary| [mm]")
    axs.set_xlabel("loop time [s]")
    xlim_valid(axs, SYNC_D)

    # ASYNC: |incoming_abs - existing_abs| per overlap, from merge.csv
    mp = LOGS / (ASYNC + "_merge.csv")
    jt, jm = [], []
    if mp.exists():
        for r in csv.DictReader(mp.open()):
            inc_s, ex_s = r.get("incoming_abs"), r.get("existing_abs")
            if not inc_s or not ex_s:
                continue
            inc = np.fromstring(inc_s, sep=";")[:3]
            ex = np.fromstring(ex_s, sep=";")[:3]
            jt.append(int(r["timestep"]))
            jm.append(float(np.linalg.norm(inc - ex) * 1e3))
    axa.stem(jt, jm, basefmt=" ", linefmt=ORANGE, markerfmt="o")
    axa.set_title(f"ASYNC latest_only — |incoming − existing| per overlap (n={len(jm)})", color=INK2)
    axa.set_ylabel("stitching jump [mm]")
    axa.set_xlabel("action timestep")
    fig.suptitle("Discontinuity injected at each re-plan — ASYNC stitches frequently, SYNC only at chunk ends",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "stitching_jumps.png", bbox_inches="tight")
    plt.close(fig)


# ─────────── Figure D: reversal clustering at arrivals ─────────────────
def fig_clustering() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (label, d) in zip(axes, RUNS):
        _, arr_idx = arrivals(d)
        idx = valid_idx(d)
        steps = np.diff(d["action_ee"][idx, :3], axis=0)           # arrives at idx[1:]
        oppose = np.sum(steps[1:] * steps[:-1], axis=1) < 0         # aligns with steps[1:] → idx[2:]
        step_arr = idx[2:]                                          # ticks where a reversal can be detected
        rev_arr = step_arr[oppose]

        def since(gidx):
            last = arr_idx[np.searchsorted(arr_idx, gidx, side="right") - 1]
            return gidx - last
        all_d = since(step_arr)
        rev_d = since(rev_arr)
        bins = np.arange(0, 16, 1)
        ax.hist(all_d, bins=bins, density=True, alpha=0.45, color=BLUE, label="all steps")
        ax.hist(rev_d, bins=bins, density=True, alpha=0.6, color=RED, label="reversal steps")
        ax.axvline(0.5, color=ARR, lw=1.0, ls="--", label="≤1 tick after arrival")
        within_all = float(np.mean(all_d <= 1))
        within_rev = float(np.mean(rev_d <= 1)) if rev_d.size else 0.0
        ax.set_title(f"{label}\nreversal within 1 tick of arrival: {within_rev:.0%} vs {within_all:.0%} overall",
                     color=INK2, fontsize=9)
        ax.set_xlabel("ticks since last chunk arrival")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
    fig.suptitle("Do direction reversals cluster at chunk arrivals? (ASYNC yes, SYNC n/a — few arrivals)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "reversal_clustering.png", bbox_inches="tight")
    plt.close(fig)


# ─────────── Figure E: consecutive chunks overlaid (EE-x) ──────────────
def fig_chunks() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, (label, d) in zip(axes, RUNS):
        idx = valid_idx(d)
        cid = d["chunk_id"][idx]
        ids = sorted(set(cid.tolist()))
        for k, cidv in enumerate(ids[:8]):
            m = cid == cidv
            pos = np.arange(m.sum())
            ax.plot(pos, d["action_abs"][idx[m], 0] * 1e3, color=CAT[k % len(CAT)], lw=1.0,
                    marker="o", markersize=2.5, markeredgecolor=SURF, markeredgewidth=0.2,
                    label=f"chunk {int(cidv)}" if k < 5 else None)
        ax.set_title(f"{label} — first ~8 chunks' EE-x target", color=INK2)
        ax.set_xlabel("position within chunk (executed ticks)")
        ax.set_ylabel("action_abs EE-x [mm]")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Consecutive chunks overlaid: SYNC continues the trajectory; "
                 "ASYNC latest_only re-predicts (overlapping, disagreeing)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "consecutive_chunks.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_ee_arrivals()
    fig_step_reversal_queue()
    fig_stitching()
    fig_clustering()
    fig_chunks()
    print(f"Debug figures written to {FIG}")
    for label, d in RUNS:
        es = ee_steps(d)
        t_arr, _ = arrivals(d)
        idx = valid_idx(d)
        print(f"  {label:22s} valid={len(idx)} chunks={len(t_arr)} "
              f"replan every ~{len(idx)/max(len(t_arr),1):.1f} ticks  reversal={es['reversal']:.0%}")


if __name__ == "__main__":
    main()
