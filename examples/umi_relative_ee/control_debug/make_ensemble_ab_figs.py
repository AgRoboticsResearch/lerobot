"""Multi-run analysis of the afternoon UMI relative-EE control logs.

Compares the seven 2026-08-07 afternoon `--log` runs — three controlled ASYNC A/B
pairs (aggregation fn × policy, and SmolVLA flow-matching padding mode) plus one
short SYNC run — on low-level control metrics, and audits the temporal ensemble on
the real `_merge.csv` data.

Outputs (into outputs/research_report/low_level_control_debug/ — the merged report's
home; figures numbered 07-09 to follow the deep-dive's 01-06):
  figures/07_ab_comparison.png, 08_ensemble_audit.png, 09_speed_vs_jitter.png
  computed_stats_ab.json    per-run metrics + audit numbers (separate from the
                            deep-dive's computed_stats.json)

Joint columns are DEGREES despite the "_rad" suffix (see the merged report).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS = REPO / "logs"
OUTDIR = REPO / "outputs" / "research_report" / "low_level_control_debug"

# (label, stem, policy, what_varies, A/B group)
# group: "A" = baseline variant, "B" = changed variant, "S" = sync
RUNS = [
    ("pi05 · latest_only",     "async_20260807_145035", "pi05 LoRA",   "aggregation", "A"),
    ("pi05 · weighted_average", "async_20260807_145157", "pi05 LoRA",   "aggregation", "B"),
    ("SmolVLA · fullwidth",    "async_20260807_145638", "SmolVLA",     "flow padding", "A"),
    ("SmolVLA · masked",       "async_20260807_145815", "SmolVLA",     "flow padding", "B"),
    ("ACT · latest_only",      "async_20260807_150005", "ACT",         "aggregation", "A"),
    ("ACT · weighted_average", "async_20260807_150157", "ACT",         "aggregation", "B"),
    ("SYNC · ACT",             "sync_20260807_150336",  "ACT",         "deploy mode", "S"),
]

# ── palette (dataviz skill, light) ──
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, RED, VIOLET, AQUA = "#2a78d6", "#eb6834", "#008300", "#e34948", "#4a3aa7", "#1baf7a"
YELLOW, MAGENTA = "#eda100", "#e87ba4"
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GREY = "#b9b7af"
EE_LABELS = ["x [m]", "y [m]", "z [m]", "rot_x [rad]", "rot_y [rad]", "rot_z [rad]", "gripper"]
JOINT_LABELS = [f"joint{i+1} [deg]" for i in range(6)]  # Piper reports degrees

# One representative run per policy family (the ensembled/smoothed variant).
REPR = [
    ("ACT · weighted_average", "async_20260807_150157"),
    ("SmolVLA · masked", "async_20260807_145815"),
    ("pi05 · weighted_average", "async_20260807_145157"),
]


def _xlim_valid(ax, t, valid, pad_frac=0.03):
    tv = t[valid]
    if tv.size == 0:
        return
    lo, hi = float(tv.min()), float(tv.max())
    ax.set_xlim(lo - max((hi - lo) * pad_frac, 1e-3), hi + max((hi - lo) * pad_frac, 1e-3))
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 110, "legend.frameon": False,
})
GCOLOR = {"A": BLUE, "B": ORANGE, "S": GREY}


def load(stem: str) -> dict:
    base = LOGS / stem
    d = {k: v for k, v in np.load(base.with_suffix(".npz")).items()}
    with base.with_suffix(".csv").open() as fh:
        rows = list(csv.DictReader(fh))
    d["state"] = np.array([r["state"] for r in rows])
    d["meta"] = json.loads(base.with_suffix(".json").read_text())
    return d


def metrics(d: dict) -> dict:
    v = d["ik_ok"].astype(bool)
    out = {"n_valid": int(v.sum()), "dur_s": float(d["t_s"][-1] - d["t_s"][0])}
    e2e = d["meta"].get("summary", {}).get("e2e_ms") or {}
    out["e2e_p50"] = e2e.get("p50", float("nan")) if e2e else float("nan")
    out["n_underrun"] = d["meta"].get("summary", {}).get("n_underrun", 0)
    if v.sum() < 2:
        return out
    qc = d["ik_joints_rad"][v].astype(float)      # degrees (mislabeled "_rad")
    qm = d["current_joints_rad"][v].astype(float)
    out["j_track"] = float(np.max(np.abs(qc - qm), axis=1).mean())     # °
    out["j_meas"] = float(np.max(np.abs(np.diff(qm, axis=0)), axis=1).mean())
    ee = d["action_ee"][v, :3]
    out["ee_step_mm"] = float(np.linalg.norm(np.diff(ee, axis=0), axis=1).mean() * 1e3)
    st = np.diff(ee, axis=0)
    out["reversal"] = float(np.mean(np.sum(st[1:] * st[:-1], axis=1) < 0))
    return out


def audit(stem: str) -> dict | None:
    p = LOGS / (stem + "_merge.csv")
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return None

    def vec(s):
        return np.array([float(x) for x in s.split(";")]) if s else None

    ex = np.array([vec(r["existing_abs"]) for r in rows if r["existing_abs"]])
    inc = np.array([vec(r["incoming_abs"]) for r in rows if r["incoming_abs"]])
    agg = np.array([vec(r["aggregated_abs"]) for r in rows if r["aggregated_abs"]])
    w = np.array([float(r["weight"]) for r in rows if r["weight"] not in ("", "nan")])
    ref = np.array([vec(r["ref_ee"]) for r in rows if r["ref_ee"]])
    lo, hi = np.minimum(ex, inc), np.maximum(ex, inc)
    between = float(np.all((agg >= lo - 1e-6) & (agg <= hi + 1e-6), axis=1).mean())
    return {
        "n_blends": len(rows), "n_chunks": len({r["chunk_id"] for r in rows}),
        "convex_frac": between,
        "w_mean": float(np.mean(w)) if w.size else float("nan"),
        "anchor_std_x_mm": float(np.std(ref[:, 0])) if ref.size else float("nan"),
    }


# ───────────────────────── load + compute ──────────────────────────────
data = []
for label, stem, pol, varies, grp in RUNS:
    d = load(stem)
    m = metrics(d)
    m.update(label=label, stem=stem, policy=pol, varies=varies, group=grp,
             agg=d["meta"].get("args", {}).get("aggregate_fn_name"))
    m["audit"] = audit(stem)
    data.append(m)

labels = [r["label"] for r in data]
colors = [GCOLOR[r["group"]] for r in data]
figdir = OUTDIR / "figures"
figdir.mkdir(parents=True, exist_ok=True)


# ───────────────────────── Figure 1: A/B metrics ───────────────────────
def fig_ab() -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    x = np.arange(len(data))
    rev = [r.get("reversal", float("nan")) for r in data]
    trg = [r.get("j_track", float("nan")) for r in data]
    ax1.bar(x, rev, color=colors)
    for xi, v in zip(x, rev):
        ax1.text(xi, v, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax1.set_ylabel("tick-to-tick direction reversal")
    ax1.set_title("Direction-reversal (jitter FREQUENCY) — policy/speed-driven, not aggregation", color=INK2)
    ax1.set_ylim(0, max(r for r in rev if r == r) * 1.25)

    ax2.bar(x, trg, color=colors)
    for xi, v in zip(x, trg):
        ax2.text(xi, v, f"{v:.2f}°", ha="center", va="bottom", fontsize=8, color=INK)
    ax2.set_ylabel("joint tracking gap [deg]")
    ax2.set_title("Joint tracking gap (amplitude) — halved by weighted_average / masked padding", color=INK2)
    ax2.set_ylim(0, max(t for t in trg if t == t) * 1.25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)

    # legend for A/B/S
    for g, c, t in [("A (baseline)", BLUE, "latest_only / fullwidth"),
                    ("B (variant)", ORANGE, "weighted_average / masked"),
                    ("S (sync)", GREY, "open-loop replay")]:
        ax1.bar([], [], color=c, label=f"{g} — {t}")
    ax1.legend(loc="upper right", fontsize=7)
    fig.suptitle("Afternoon A/B: aggregation & flow-padding vs low-level control (7 runs)", fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figdir / "07_ab_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ───────────────────── Figure 2: ensemble audit ────────────────────────
def fig_audit() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: convexity + weight per audited run
    audited = [r for r in data if r["audit"]]
    xa = np.arange(len(audited))
    w = [r["audit"]["w_mean"] for r in audited]
    cc = [ORANGE if r["audit"]["w_mean"] > 0.1 else BLUE for r in audited]
    ax1.bar(xa, w, color=cc)
    for xi, v, r in zip(xa, w, audited):
        ax1.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax1.text(xi, -0.05, f"{r['audit']['convex_frac']:.0%}\nconvex", ha="center", va="top",
                 fontsize=7, color=INK2)
    ax1.set_xticks(xa)
    ax1.set_xticklabels([r["label"] for r in audited], rotation=20, ha="right", fontsize=7)
    ax1.set_ylabel("blend weight on EXISTING target  (agg = w·existing + (1−w)·incoming)")
    ax1.set_ylim(-0.12, 0.45)
    ax1.set_title("Ensemble weight (real _merge.csv) — fixed, averages ABSOLUTES", color=INK2)
    ax1.bar([], [], color=ORANGE, label="weighted_average (w≈0.30)")
    ax1.bar([], [], color=BLUE, label="latest_only (w≈0.00, replace)")
    ax1.legend(fontsize=7)

    # Right: anchor variation across chunks (frame-mismatch is a non-issue)
    sig = [r["audit"]["anchor_std_x_mm"] for r in audited]
    ax2.bar(xa, sig, color=cc)
    for xi, v in zip(xa, sig):
        ax2.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax2.set_xticks(xa)
    ax2.set_xticklabels([r["label"] for r in audited], rotation=20, ha="right", fontsize=7)
    ax2.set_ylabel("σ of per-chunk anchor EE-x [mm]")
    ax2.set_title("Anchor barely moves between chunks → no frame-mismatch hazard", color=INK2)

    fig.suptitle("Temporal-ensemble audit on real data — confirms ABSOLUTE blending", fontsize=11.5, y=1.0)
    fig.tight_layout()
    fig.savefig(figdir / "08_ensemble_audit.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────── Figure 3: inference speed → jitter ───────────────
def fig_speed() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pol_color = {"ACT": BLUE, "pi05 LoRA": ORANGE, "SmolVLA": VIOLET}
    for r in data:
        if r.get("e2e_p50") != r.get("e2e_p50") or r["group"] == "S":
            continue  # sync has no e2e
        ax.scatter(r["e2e_p50"], r.get("reversal", float("nan")) * 100,
                   s=90, color=pol_color.get(r["policy"], GREY), edgecolor=SURF, linewidth=0.8,
                   zorder=3)
        ax.annotate(r["label"].split(" · ")[-1], (r["e2e_p50"], r.get("reversal", 0) * 100),
                    fontsize=7, color=INK2, xytext=(5, 4), textcoords="offset points")
    for p, c in pol_color.items():
        ax.scatter([], [], color=c, label=p)
    ax.set_xlabel("inference e2e latency p50 [ms]  (30 Hz loop → 33 ms target)")
    ax.set_ylabel("tick-to-tick direction reversal [%]")
    ax.set_title("Slower inference → more reversal (starved ensemble)", color=INK2)
    ax.axvline(33.3, color=GREEN, lw=1.0, ls="--", label="33 ms (30 Hz)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "09_speed_vs_jitter.png", bbox_inches="tight")
    plt.close(fig)


# ────────────────── Figure 10: afternoon EE pose (executed only) ────────
def fig_afternoon_ee() -> None:
    runs = [(label, load(stem)) for label, stem in REPR]
    fig, axes = plt.subplots(7, 3, figsize=(12, 12.5), sharex="col")
    for c, (label, d) in enumerate(runs):
        t, valid = d["t_s"], d["ik_ok"].astype(bool)
        axes[0, c].set_title(label, color=INK)
        for r in range(7):
            ax = axes[r, c]
            ax.plot(t[valid], d["action_ee"][valid, r], color=CAT[r], lw=1.0,
                    marker="o", markersize=2.5, markeredgecolor=SURF, markeredgewidth=0.3)
            ax.set_ylabel(EE_LABELS[r], fontsize=8)
            ax.tick_params(labelsize=7)
        _xlim_valid(axes[0, c], t, valid)
        axes[-1, c].set_xlabel("loop time [s]  (executed window)")
    fig.suptitle("EE pose command sent to IK — representative afternoon runs (executed only)",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(figdir / "10_afternoon_ee_pose.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────── Figure 11: afternoon joint cmd vs read-back ───────────
def fig_afternoon_joints() -> None:
    runs = [(label, load(stem)) for label, stem in REPR]
    fig, axes = plt.subplots(6, 3, figsize=(12, 11.5), sharex="col")
    for c, (label, d) in enumerate(runs):
        t, valid = d["t_s"], d["ik_ok"].astype(bool)
        axes[0, c].set_title(label, color=INK)
        for r in range(6):
            ax = axes[r, c]
            ax.plot(t[valid], d["ik_joints_rad"][valid, r], color=BLUE, lw=0.7,
                    marker="o", markersize=2.5, markeredgecolor=SURF, markeredgewidth=0.3)
            ax.plot(t[valid], d["current_joints_rad"][valid, r], color=ORANGE, lw=0.6, alpha=0.9,
                    marker="o", markersize=2, markeredgecolor=SURF, markeredgewidth=0.25)
            ax.set_ylabel(JOINT_LABELS[r], fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(["IK joint cmd", "read-back"], loc="upper right", fontsize=7)
        _xlim_valid(axes[0, c], t, valid)
        axes[-1, c].set_xlabel("loop time [s]  (executed window)")
    fig.suptitle("Joint command vs read-back — representative afternoon runs (executed only)",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(figdir / "11_afternoon_joints.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_ab()
    fig_audit()
    fig_speed()
    fig_afternoon_ee()
    fig_afternoon_joints()
    out = {"runs": [{k: v for k, v in r.items() if k != "audit"} | {"audit": r["audit"]} for r in data]}
    (OUTDIR / "computed_stats_ab.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"Figures + stats written to {OUTDIR}")
    hdr = f"{'run':26s}{'valid':>6s}{'jTrk°':>8s}{'jMeas°':>8s}{'EEmm':>7s}{'revers':>8s}{'e2e_ms':>8s}{'w_avg':>7s}{'convex':>8s}"
    print(hdr)
    for r in data:
        a = r["audit"] or {}
        print(f"{r['label']:26s}{r['n_valid']:6d}{r.get('j_track',float('nan')):8.2f}{r.get('j_meas',float('nan')):8.2f}"
              f"{r.get('ee_step_mm',float('nan')):7.2f}{r.get('reversal',float('nan')):8.0%}"
              f"{r.get('e2e_p50',float('nan')):8.0f}{a.get('w_mean',float('nan')):7.2f}{a.get('convex_frac',float('nan')):8.0%}")


if __name__ == "__main__":
    main()
