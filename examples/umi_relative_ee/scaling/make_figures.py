#!/usr/bin/env python
"""Build scaling-report figures (per FIGURE_SPEC.md). Static PNG, matplotlib."""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

SD = "/home/zfei/code/lerobots/lerobot/outputs/train/ee_vs_joints/scaling_analysis"
OUT = SD

# ---- style (dataviz: recessive grid/axes, text in ink not series color) ----
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
    "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
    "axes.edgecolor": "#b8b8b5", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.7,
    "axes.titleweight": "bold", "axes.titlecolor": "#0b0b0b",
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.labelsize": 9, "ytick.labelsize": 9,
})
# sequential blue ramp: more data -> darker (100/500/1012 ep)
C = {"100": "#3987e5", "500": "#1c5cab", "1012": "#0d366b"}
LBL = {"100": "100 ep", "500": "500 ep", "1012": "1012 ep"}

# ---- load ----
H = {t: pd.read_csv(f"{SD}/_{t}_history.csv").sort_values("_step") for t in C}
EP = {"100": 100, "500": 500, "1012": 1012}

def train_series(t):
    d = H[t].dropna(subset=["train/loss"])
    s = d["_step"].to_numpy(); y = d["train/loss"].to_numpy()
    y = pd.Series(y).rolling(25, min_periods=1, center=True).mean().to_numpy()  # light smooth
    return s, y

def val_series(t):
    d = H[t].dropna(subset=["val/loss"])
    d = d[d["_step"] >= 10000]
    return d["_step"].to_numpy(), d["val/loss"].to_numpy()

def train_interp(t, steps):
    d = H[t].dropna(subset=["train/loss"]).sort_values("_step")
    return np.interp(steps, d["_step"].to_numpy(), d["train/loss"].to_numpy())

best = {}
for t in C:
    s, y = val_series(t)
    i = int(np.argmin(y))
    best[t] = (int(s[i]), float(y[i]))
final = {}
for t in C:
    s, y = val_series(t)
    final[t] = (int(s[-1]), float(y[-1]))

# ============ FIGURE 1: 2x2 scaling ============
fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))

# --- (A) val loss vs steps ---
a = ax[0, 0]
for t in C:
    s, y = val_series(t)
    a.plot(s, y, color=C[t], lw=1.8, label=LBL[t])
    bs, by = best[t]
    a.scatter([bs], [by], color=C[t], s=55, zorder=5, edgecolor="white", linewidth=0.8)
a.set_xscale("log"); a.set_xlabel("training steps"); a.set_ylabel("validation loss")
a.set_title("(A) Validation loss vs steps")
a.set_xlim(8e3, 2.7e6)
# direct labels at curve ends
for t in C:
    s, y = val_series(t)
    a.annotate(LBL[t], (s[-1], y[-1]), color=C[t], fontsize=9, fontweight="bold",
               xytext=(6, 0), textcoords="offset points", va="center")
a.text(0.02, 0.06, "★ = best val", transform=a.transAxes, fontsize=8, color="#52514e")

# --- (B) train loss vs steps ---
b = ax[0, 1]
for t in C:
    s, y = train_series(t)
    b.plot(s, y, color=C[t], lw=1.6, label=LBL[t])
b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel("training steps"); b.set_ylabel("training loss")
b.set_title("(B) Training loss vs steps"); b.set_xlim(1e2, 2.7e6); b.set_ylim(4e-3, 30)
for t in C:
    s, y = train_series(t)
    b.annotate(LBL[t], (s[-1], y[-1]), color=C[t], fontsize=9, fontweight="bold",
               xytext=(6, 0), textcoords="offset points", va="center")

# --- (C) scaling law: val loss vs dataset size (log-log) + fit ---
c = ax[1, 0]
D = np.array([EP[t] for t in C], float)
Lbest = np.array([best[t][1] for t in C])
Lfinal = np.array([final[t][1] for t in C])
# power-law fit on best: log L = log a - b log D  (b positive)
b_, alog = np.polyfit(np.log(D), np.log(Lbest), 1)
a_ = np.exp(alog); expo = -b_
Df = np.geomspace(80, 1200, 200)
c.plot(Df, a_ * Df**b_, color="#0d366b", lw=1.4, ls="--",
       label=f"fit  $L\\propto D^{{-{expo:.2f}}}$")
for t in C:
    c.scatter([EP[t]], [best[t][1]], color=C[t], s=70, zorder=5, edgecolor="white", linewidth=0.9)
for t in C:
    c.scatter([EP[t]], [final[t][1]], color=C[t], s=42, marker="o", facecolor="none",
              linewidth=1.4, zorder=4)
c.set_xscale("log"); c.set_yscale("log"); c.set_xlabel("training episodes  D")
c.set_ylabel("validation loss"); c.set_title("(C) Scaling law — val loss vs dataset size")
c.set_xlim(80, 1300)
# direct labels next to best-val points
for t in C:
    c.annotate(f"{LBL[t]}  ({best[t][1]:.3f})", (EP[t], best[t][1]), color=C[t],
               fontsize=8.5, fontweight="bold", xytext=(8, 6), textcoords="offset points")
c.legend(loc="upper right", frameon=False, fontsize=9)
c.text(0.02, 0.06, "● best val   ○ final val\nfit on best val", transform=c.transAxes,
       fontsize=8, color="#52514e", va="bottom")

# --- (D) generalization gap (val - train) vs steps ---
d = ax[1, 1]
for t in C:
    s, y = val_series(t)
    gap = y - train_interp(t, s)
    d.plot(s, gap, color=C[t], lw=1.8, label=LBL[t])
d.set_xscale("log"); d.set_xlabel("training steps"); d.set_ylabel("val loss − train loss")
d.set_title("(D) Generalization gap vs steps"); d.set_xlim(8e3, 2.7e6)
for t in C:
    s, y = val_series(t)
    gap = y - train_interp(t, s)
    d.annotate(LBL[t], (s[-1], gap[-1]), color=C[t], fontsize=9, fontweight="bold",
               xytext=(6, 0), textcoords="offset points", va="center")

fig.suptitle("ACT scaling — strawberry picking (relative-EE, rot6d, chunk30) · "
             "100 / 500 / 1012 episodes, shared 100-ep validation",
             fontsize=12.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}/fig1_scaling.png", bbox_inches="tight")
print("saved fig1_scaling.png")
plt.close(fig)

# ============ FIGURE 2: loss components ============
fig2, bx = plt.subplots(1, 2, figsize=(12.5, 4.2))
for ax_, comp, title in [(bx[0], "val/l1_loss", "(A) Validation L1 loss"),
                         (bx[1], "val/kld_loss", "(B) Validation KLD loss")]:
    for t in C:
        d = H[t].dropna(subset=[comp]); d = d[d["_step"] >= 10000]
        ax_.plot(d["_step"], d[comp], color=C[t], lw=1.7, label=LBL[t])
        ax_.set_xscale("log"); ax_.set_xlabel("training steps"); ax_.set_title(title)
        if comp == "val/kld_loss":
            ax_.set_yscale("symlog", linthresh=1e-4)
    ax_.set_xlim(8e3, 2.7e6)
    for t in C:
        d = H[t].dropna(subset=[comp]); d = d[d["_step"] >= 10000]
        ax_.annotate(LBL[t], (d["_step"].iloc[-1], d[comp].iloc[-1]), color=C[t],
                     fontsize=9, fontweight="bold", xytext=(6, 0),
                     textcoords="offset points", va="center")
fig2.suptitle("Loss components vs steps (validation)", fontsize=12, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig(f"{OUT}/fig2_loss_components.png", bbox_inches="tight")
print("saved fig2_loss_components.png")
plt.close(fig2)

# ---- summary print ----
print("\n== SUMMARY ==")
for t in C:
    print(f"{LBL[t]:>8}: best val={best[t][1]:.5f} @step {best[t][0]:>7} | "
          f"final val={final[t][1]:.5f} @step {final[t][0]}")
print(f"power-law (best val, episodes): L = {a_:.4f} * D^-({expo:.3f})")
print(f"power-law check: pred={[f'{a_*d**b_:.4f}' for d in D]} actual={[f'{x:.4f}' for x in Lbest]}")
