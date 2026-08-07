#!/usr/bin/env python
"""Generate polished figures for the open-loop cross-model report.

Reads <model>_<step>_open_loop_metrics.json + val_loss_summary.json from a folder
(default outputs/research_report/open_loop_val_compare) and writes PNGs into <folder>/figures/.
Follows the dataviz skill: validated categorical palette, 2px lines, >=8px markers
with a surface ring, recessive grid, no dual-axis (small multiples instead),
legend + selective direct labels.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _default_folder() -> Path:
    """Latest run folder under outputs/research_report/ (newest mtime)."""
    base = Path("outputs/research_report")
    if base.is_dir():
        subs = [p for p in base.iterdir() if p.is_dir()]
        if subs:
            return max(subs, key=lambda p: p.stat().st_mtime)
    return base / "open_loop_val_compare"


FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_folder()
FIGDIR = FOLDER / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- palette / chrome (dataviz skill reference palette, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR = {  # fixed categorical order, curated to avoid low-contrast yellow
    "ACT (1302)":            "#2a78d6",  # blue
    "ACT (1459)":            "#eb6834",  # orange
    "π0.5 38M (1302)":       "#1baf7a",  # aqua
    "π0.5 220M (1302)":      "#e87ba4",  # magenta
    "SmolVLA-masked (1302)": "#4a3aa7",  # violet
    "SmolVLA-fullwidth (1302)": "#008300",  # green
}
MODEL_LABEL = {
    "act_umi_identity_rot6d_1302": "ACT (1302)",
    "act_umi_identity_rot6d_1459": "ACT (1459)",
    "pi05_openpi_split_lora_masked_1302_bs4": "π0.5 38M (1302)",
    "pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full": "π0.5 220M (1302)",
    "smolvla_masked_subspace_1302_1M": "SmolVLA-masked (1302)",
    "smolvla_openpi_fullwidth_1302_1M": "SmolVLA-fullwidth (1302)",
}
ORDER = ["ACT (1302)", "ACT (1459)", "π0.5 38M (1302)", "π0.5 220M (1302)",
         "SmolVLA-masked (1302)", "SmolVLA-fullwidth (1302)"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 1.0, "grid.color": GRID,
    "grid.linewidth": 0.8, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})


def load():
    runs = []
    for p in sorted(FOLDER.glob("*_open_loop_metrics.json")):
        m = re.match(r"^(.*)_(\d+)_open_loop_metrics\.json$", p.name)
        if not m:
            continue
        model, step = m.group(1), int(m.group(2))
        d = json.loads(p.read_text())
        eb = d.get("summary", {}).get("episode_balanced", {})
        runs.append({"model": model, "step": step, "label": MODEL_LABEL.get(model, model), "eb": eb})
    val = json.loads((FOLDER / "val_loss_summary.json").read_text())
    for r in runs:
        traj = val.get(r["model"], [])
        if traj:
            best = min(traj, key=lambda t: abs(t["step"] - r["step"]))
            r["val_loss"], r["val_step"] = best["loss"], best["step"]
        else:
            r["val_loss"] = r["val_step"] = None
    return runs


def style_ax(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def line(ax, xs, ys, label, marker=True):
    ax.plot(xs, ys, color=COLOR[label], linewidth=2.0,
            marker=("o" if marker else None), markersize=7,
            markerfacecolor=COLOR[label], markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=label, zorder=3)


# metrics: key -> (title, factor, fmt)
MXYZ_END = ("XYZ endpoint (mm)", 1000.0, "{:.1f}")
MROT_END = ("Rotation endpoint (°)", 1.0, "{:.2f}")
MROT_JRK = ("Rotation jitter (°)", 1.0, "{:.3f}")
MXYZ_MEAN = ("XYZ trajectory mean (mm)", 1000.0, "{:.1f}")


def fig1_compare_bars(runs):
    """2x2 horizontal-bar small multiples, each model's best ckpt by XYZ end."""
    best = {}
    for r in runs:
        cur = best.get(r["label"])
        if cur is None or r["eb"].get("xyz_end_m", 1e9) < cur["eb"]["xyz_end_m"]:
            best[r["label"]] = r
    panels = [("xyz_end_m", *MXYZ_END), ("rotation_end_deg", *MROT_END),
              ("rot_jerk_deg", *MROT_JRK), ("xyz_chunk_mean_m", *MXYZ_MEAN)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title, fac, _) in zip(axes.flat, panels):
        rows = sorted(best.values(), key=lambda r: r["eb"].get(key, 1e9))  # best (low) on top
        labels = [r["label"] for r in rows]
        vals = [r["eb"][key] * fac for r in rows]
        colors = [COLOR[l] for l in labels]
        y = range(len(rows))
        ax.barh(y, vals, color=colors, height=0.62, zorder=3)
        ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        vmax = max(vals)
        for yi, v in zip(y, vals):
            ax.text(v + vmax * 0.012, yi, f"{v:.1f}" if fac == 1000 else f"{v:.2f}",
                    va="center", ha="left", fontsize=8, color=INK2)
        ax.set_xlim(0, vmax * 1.18)
        style_ax(ax); ax.grid(axis="y", visible=False)
    fig.suptitle("Cross-model comparison — each model's best checkpoint (by XYZ endpoint)",
                 fontsize=13, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGDIR / "fig1_compare_bars.png", dpi=160)
    plt.close(fig)


def _step_lines(runs, family_labels, metrics, title, fname):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (key, mtitle, fac, _) in zip(axes.flat, metrics):
        for lab in family_labels:
            rs = sorted([r for r in runs if r["label"] == lab], key=lambda r: r["step"])
            if len(rs) < 2:
                continue
            xs = [r["step"] / 1000 for r in rs]
            ys = [r["eb"][key] * fac for r in rs]
            line(ax, xs, ys, lab)
        ax.set_ylabel(mtitle, fontsize=9)
        ax.set_title(mtitle, fontsize=10, color=INK)
        style_ax(ax)
    axes.flat[-2].set_xlabel("training step (k)"); axes.flat[-1].set_xlabel("training step (k)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGDIR / fname, dpi=160)
    plt.close(fig)


def fig2_steps(runs):
    metrics = [("xyz_end_m", *MXYZ_END), ("rotation_end_deg", *MROT_END),
               ("rot_jerk_deg", *MROT_JRK), ("xyz_chunk_mean_m", *MXYZ_MEAN)]
    _step_lines(runs, ["ACT (1302)", "ACT (1459)"], metrics,
                "ACT — decoded metrics vs training step", "fig2_act_steps.png")
    _step_lines(runs, ["π0.5 38M (1302)", "π0.5 220M (1302)"], metrics,
                "π0.5 — decoded metrics vs training step", "fig2_pi05_steps.png")


def fig3_val_vs_decoded(runs):
    """2 rows x 4 cols small multiples: top=XYZ end vs step, bottom=val loss vs step.
    No dual axis; min of each marked so val-best vs decoded-best divergence is visible."""
    models = ["ACT (1302)", "ACT (1459)", "π0.5 38M (1302)", "π0.5 220M (1302)"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex="col")
    for ci, lab in enumerate(models):
        rs = sorted([r for r in runs if r["label"] == lab and r["val_loss"] is not None],
                    key=lambda r: r["step"])
        xs = [r["step"] / 1000 for r in rs]
        # top: decoded XYZ end
        axt = axes[0, ci]
        ys = [r["eb"]["xyz_end_m"] * 1000 for r in rs]
        line(axt, xs, ys, lab)
        xi = min(range(len(ys)), key=lambda i: ys[i])
        axt.axvline(xs[xi], color=MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
        axt.scatter([xs[xi]], [ys[xi]], s=70, facecolor=COLOR[lab], edgecolor=SURFACE,
                    linewidth=1.5, zorder=5)
        axt.set_title(lab, fontsize=10, color=INK)
        if ci == 0:
            axt.set_ylabel("XYZ endpoint (mm)", fontsize=9)
        style_ax(axt)
        # bottom: val loss
        axb = axes[1, ci]
        yv = [r["val_loss"] for r in rs]
        axb.plot(xs, yv, color=COLOR[lab], linewidth=2.0, marker="o", markersize=6,
                 markerfacecolor=COLOR[lab], markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=3)
        xj = min(range(len(yv)), key=lambda i: yv[i])
        axb.axvline(xs[xj], color=MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
        axb.scatter([xs[xj]], [yv[xj]], s=70, facecolor=COLOR[lab], edgecolor=SURFACE,
                    linewidth=1.5, zorder=5)
        axb.set_xlabel("training step (k)", fontsize=9)
        if ci == 0:
            axb.set_ylabel("validation loss", fontsize=9)
        style_ax(axb)
    fig.suptitle("Validation loss vs decoded quality (within each model). "
                 "Dashed = each curve's minimum — they land on different steps.",
                 fontsize=11.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGDIR / "fig3_val_vs_decoded.png", dpi=160)
    plt.close(fig)


def fig4_pareto(runs):
    """Accuracy vs smoothness: x=rot jitter, y=XYZ endpoint; each model's best-by-XYZ-end ckpt."""
    best = {}
    for r in runs:
        cur = best.get(r["label"])
        if cur is None or r["eb"].get("xyz_end_m", 1e9) < cur["eb"]["xyz_end_m"]:
            best[r["label"]] = r
    fig, ax = plt.subplots(figsize=(9, 6.5))
    # GT jitter reference (rot) ~0.158 deg
    gt = runs[0]["eb"].get("gt_rot_jerk_deg")
    if gt:
        ax.axvline(gt, color=MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
        ax.text(gt, ax.get_ylim()[1] if False else 0.5, "  GT jitter", color=MUTED, fontsize=8,
                va="top", transform=ax.get_xaxis_transform())
    for lab in ORDER:
        r = best.get(lab)
        if not r:
            continue
        x = r["eb"]["rot_jerk_deg"]; y = r["eb"]["xyz_end_m"] * 1000
        ax.scatter([x], [y], s=130, facecolor=COLOR[lab], edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.annotate(f"{lab}\n({r['step']//1000}k)", (x, y), fontsize=8, color=INK2,
                    textcoords="offset points", xytext=(9, 6))
    ax.set_xlabel("Rotation jitter (°)  →  smoother", fontsize=10)
    ax.set_ylabel("XYZ endpoint error (mm)  →  more accurate", fontsize=10)
    ax.set_title("Accuracy vs smoothness — each model's best checkpoint", fontsize=12, color=INK)
    style_ax(ax)
    ax.invert_yaxis()  # lower endpoint error higher up = "better"
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_pareto.png", dpi=160)
    plt.close(fig)


def main():
    runs = load()
    if not runs:
        print("No eval JSONs in", FOLDER); return
    fig1_compare_bars(runs)
    fig2_steps(runs)
    fig3_val_vs_decoded(runs)
    fig4_pareto(runs)
    print(f"Wrote figures to {FIGDIR}/ : {sorted(p.name for p in FIGDIR.glob('*.png'))}")


if __name__ == "__main__":
    main()
