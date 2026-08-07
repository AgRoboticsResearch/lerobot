#!/usr/bin/env python
"""Compile a cross-model open-loop comparison report from eval_open_loop_dataset
JSONs + val_loss_summary.json in one folder.

Reads every ``<model>_<step>_open_loop_metrics.json`` and writes:
  * REPORT.md   - human report (general compare, step progression, val-vs-decoded)
  * summary.csv - one row per checkpoint with all metrics + nearest val loss
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

def _default_folder() -> Path:
    """Latest run folder under outputs/research_report/ (newest mtime), so standalone
    invocations auto-target the most recent eval without needing the path."""
    base = Path("outputs/research_report")
    if base.is_dir():
        subs = [p for p in base.iterdir() if p.is_dir()]
        if subs:
            return max(subs, key=lambda p: p.stat().st_mtime)
    return base / "open_loop_val_compare"


FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_folder()

LABEL = {
    "act_umi_identity_rot6d_1302": "ACT (1302)",
    "act_umi_identity_rot6d_1459": "ACT (1459)",
    "pi05_openpi_split_lora_masked_1302_bs4": "π0.5 38M (1302)",
    "pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full": "π0.5 220M (1302)",
    "smolvla_masked_subspace_1302_1M": "SmolVLA-masked (1302)",
    "smolvla_openpi_fullwidth_1302_1M": "SmolVLA-fullwidth (1302)",
}
# metrics: key -> (header, factor to multiply for display, fmt)
METRICS = [
    ("xyz_end_m",            "XYZ end (mm)",   1000.0, "{:.1f}"),
    ("rotation_end_deg",     "Rot end (°)",    1.0,    "{:.2f}"),
    ("xyz_chunk_mean_m",     "XYZ mean (mm)",  1000.0, "{:.1f}"),
    ("rotation_chunk_mean_deg","Rot mean (°)", 1.0,    "{:.2f}"),
    ("xyz_chunk_rmse_m",     "XYZ RMSE (mm)",  1000.0, "{:.1f}"),
    ("rotation_chunk_rmse_deg","Rot RMSE (°)", 1.0,    "{:.2f}"),
    ("gripper_end",          "Grip end",       1.0,    "{:.3f}"),
    ("rot_jerk_deg",         "Rot jerk (°)",   1.0,    "{:.3f}"),
    ("xyz_jerk_m",           "XYZ jerk (mm)",  1000.0, "{:.2f}"),
]
ENDPOINT_KEYS = ["xyz_end_m", "rotation_end_deg"]


def fmt(key, eb):
    *_, fac, f = next((m for m in METRICS if m[0] == key))
    v = eb.get(key)
    return "n/a" if v is None else f.format(v * fac)


def load_runs():
    runs = []  # dicts
    for p in sorted(FOLDER.glob("*_open_loop_metrics.json")):
        stem = p.name[: -len("_open_loop_metrics.json")]
        m = re.match(r"^(.*)_(\d+)$", stem)
        if not m:
            continue
        model, step = m.group(1), int(m.group(2))
        d = json.loads(p.read_text())
        eb = d.get("summary", {}).get("episode_balanced", {})
        runs.append({
            "model": model, "step": step, "label": LABEL.get(model, model),
            "policy_type": d.get("policy_type", "?"),
            "eb": eb,
            "num_samples": d.get("summary", {}).get("num_samples"),
            "num_episodes": d.get("summary", {}).get("num_episodes"),
        })
    return runs


def nearest_val(val_traj, step):
    if not val_traj:
        return None, None
    best = min(val_traj, key=lambda r: abs(r["step"] - step))
    return best["loss"], best["step"]


def main():
    val = json.loads((FOLDER / "val_loss_summary.json").read_text())
    runs = load_runs()
    if not runs:
        print("No eval JSONs found in", FOLDER); return
    for r in runs:
        vl, vs = nearest_val(val.get(r["model"], []), r["step"])
        r["val_loss"], r["val_step"] = vl, vs

    models = sorted({r["model"] for r in runs}, key=lambda m: LABEL.get(m, m))

    lines = []
    lines.append("# Open-loop cross-model comparison report\n")
    lines.append(f"Folder: `{FOLDER}`\n")
    any0 = runs[0]
    lines.append(
        f"Dataset: **{Path('/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation').name}** "
        f"— all {any0['num_episodes']} episodes, 5 evenly-spaced non-padded query frames each "
        f"({any0['num_samples']} query frames per checkpoint), seed 1000.\n"
    )
    lines.append(f"Models: **{len(models)}**, checkpoints evaluated: **{len(runs)}**.\n")
    lines.append(
        "\n## How to read this\n"
        "- Every number is a **decoded physical metric** from open-loop inference "
        "(recorded observation → predicted chunk → checkpoint's saved postprocessor → absolute 7D pose vs GT), "
        "**not training loss.** Lower is better.\n"
        "- xyz in mm, rotation in degrees, gripper is absolute error in [0,1].\n"
        "- **Validation loss is NOT comparable across models** (ACT = L1+KL; SmolVLA/π0.5 = flow MSE; "
        "SmolVLA-fullwidth even logs a 32-D-averaged loss). Compare val loss only **within** a model.\n"
        "- **gt_*_jerk** (the GT trajectory's own within-chunk jitter) is a floor for the jitter metrics; "
        "a model near GT jitter is smooth, not necessarily accurate.\n"
    )

    # GT jitter reference (model-independent ~constant)
    gt = runs[0]["eb"]
    lines.append(
        "Reference (GT trajectory): rot jerk {:.3f}°, xyz jerk {:.2f} mm.\n".format(
            gt.get("gt_rot_jerk_deg", float("nan")), gt.get("gt_xyz_jerk_m", 0.0) * 1000
        )
    )

    # ---------- Section A: general cross-model comparison ----------
    lines.append("\n## A. General cross-model comparison\n")
    lines.append("Each model represented by its own best checkpoint by translation endpoint (XYZ end). "
                 "Bold = best in that column across models.\n")
    # pick best-by-xyz_end per model
    best_rows = []
    for m in models:
        mr = [r for r in runs if r["model"] == m]
        if not mr:
            continue
        best = min(mr, key=lambda r: r["eb"].get("xyz_end_m", 1e9))
        best_rows.append(best)
    # column minima
    colmin = {}
    for key, *_ in METRICS:
        vals = [r["eb"].get(key) for r in best_rows if r["eb"].get(key) is not None]
        colmin[key] = min(vals) if vals else None

    def cell(key, eb):
        *_, fac, f = next(x for x in METRICS if x[0] == key)
        v = eb.get(key)
        if v is None:
            return "n/a"
        s = f.format(v * fac)
        return f"**{s}**" if abs(v - colmin[key]) < 1e-12 else s

    hdr = "| Model | ckpt | " + " | ".join(h for _, h, _, _ in METRICS) + " |"
    sep = "|" + "---|" * (2 + len(METRICS))
    lines.append(hdr); lines.append(sep)
    for r in sorted(best_rows, key=lambda r: r["eb"].get("xyz_end_m", 1e9)):
        cells = [r["label"], f"{r['step']//1000}k"] + [cell(k, r["eb"]) for k, *_ in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(f"\nRanked by XYZ endpoint error: "
                 + ", ".join(f"{r['label']} ({r['step']//1000}k)" for r in sorted(best_rows, key=lambda r: r['eb'].get('xyz_end_m', 1e9)))
                 + ".\n")

    # ---------- Section B: step progression ----------
    lines.append("\n## B. Training-step progression (within each model)\n")
    lines.append("How decoded metrics evolve with training. val = nearest logged validation loss for that checkpoint "
                 "(not cross-model comparable).\n")
    for m in models:
        mr = sorted([r for r in runs if r["model"] == m], key=lambda r: r["step"])
        if len(mr) < 2:
            continue
        lines.append(f"\n**{LABEL.get(m,m)}** ({mr[0]['policy_type']})\n")
        h = "| step | val loss | XYZ end (mm) | Rot end (°) | XYZ mean (mm) | Rot jerk (°) | XYZ jerk (mm) |"
        lines.append(h); lines.append("|" + "---|" * 7)
        for r in mr:
            vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "n/a"
            lines.append("| {sk} | {vl} | {xe} | {re} | {xm} | {rj} | {xj} |".format(
                sk=r["step"]//1000, vl=vl,
                xe=fmt("xyz_end_m", r["eb"]), re=fmt("rotation_end_deg", r["eb"]),
                xm=fmt("xyz_chunk_mean_m", r["eb"]),
                rj=fmt("rot_jerk_deg", r["eb"]), xj=fmt("xyz_jerk_m", r["eb"])))
        # per-model best-by-metric among evaluated checkpoints
        be = min(mr, key=lambda r: r["eb"].get("xyz_end_m", 1e9))
        bj = min(mr, key=lambda r: r["eb"].get("rot_jerk_deg", 1e9))
        bv = min(mr, key=lambda r: r["val_loss"] if r["val_loss"] is not None else 1e9)
        lines.append(f"_best XYZ end: {be['step']//1000}k · best rot jerk: {bj['step']//1000}k · "
                     f"lowest val loss: {bv['step']//1000}k_\n")

    # ---------- Section C: does best val loss = best decoded? ----------
    lines.append("\n## C. Does the best validation loss mean the best (decoded) model?\n")
    lines.append("Within each model: is the lowest-val-loss checkpoint also the best-decoded checkpoint? "
                 "(Val-best and decoded-best are both taken over the **evaluated** checkpoints.)\n")
    lines.append("| Model | val-best ckpt (loss) | decoded-best by XYZ end | by Rot end | by Rot jerk | verdict |")
    lines.append("|" + "---|" * 6)
    findings_c = []
    for m in models:
        mr = [r for r in runs if r["model"] == m]
        if len(mr) < 2 or any(r["val_loss"] is None for r in mr):
            continue
        bv = min(mr, key=lambda r: r["val_loss"])
        bx = min(mr, key=lambda r: r["eb"].get("xyz_end_m", 1e9))
        br = min(mr, key=lambda r: r["eb"].get("rotation_end_deg", 1e9))
        bj = min(mr, key=lambda r: r["eb"].get("rot_jerk_deg", 1e9))
        matches = [bx["step"] == bv["step"], br["step"] == bv["step"], bj["step"] == bv["step"]]
        verdict = "val-loss-best IS decoded-best" if all(matches) else (
            "val-loss-best = best on " + ", ".join(n for n, ok in zip(["XYZend","Rotend","Rotjerk"], matches) if ok)
            if any(matches) else "val-loss-best is NOT the decoded-best")
        lines.append("| {} | {}k ({:.4f}) | {}k ({:.1f}mm) | {}k ({:.2f}°) | {}k ({:.3f}°) | {} |".format(
            LABEL.get(m, m), bv["step"]//1000, bv["val_loss"],
            bx["step"]//1000, bx["eb"]["xyz_end_m"]*1000,
            br["step"]//1000, br["eb"]["rotation_end_deg"],
            bj["step"]//1000, bj["eb"]["rot_jerk_deg"], verdict))
        findings_c.append((LABEL.get(m,m), all(matches), verdict))

    # ---------- Section D: findings ----------
    lines.append("\n## D. Findings\n")
    ranked = sorted(best_rows, key=lambda r: r["eb"].get("xyz_end_m", 1e9))
    lines.append(f"- **Best endpoint accuracy (XYZ end):** {ranked[0]['label']} @ {ranked[0]['step']//1000}k "
                 f"({ranked[0]['eb']['xyz_end_m']*1000:.1f} mm). Worst: {ranked[-1]['label']} ({ranked[-1]['eb']['xyz_end_m']*1000:.1f} mm).")
    rankedj = sorted(best_rows, key=lambda r: r["eb"].get("rot_jerk_deg", 1e9))
    lines.append(f"- **Smoothest (rot jitter, GT floor {gt.get('gt_rot_jerk_deg',0):.3f}°):** "
                 f"{rankedj[0]['label']} @ {ranked[0]['step']//1000}k ({rankedj[0]['eb']['rot_jerk_deg']:.3f}°).")
    n_agree = sum(1 for _, ok, _ in findings_c if ok)
    n_tested = len(findings_c)
    if n_tested:
        lines.append(f"- **Best-val-loss = best-decoded:** agrees in {n_agree}/{n_tested} models. " +
                     ("So in this set, val loss is an unreliable proxy for decoded quality — confirm on the per-model rows above."
                      if n_agree < n_tested else "Val loss tracked decoded quality in every tested model."))
    lines.append("\n> Caveats: open-loop (recorded observations, no environment); 5 frames/episode × 100 episodes; "
                 "flow-policy inference is stochastic under the seed. For deployment selection, pair this with the "
                 "section-22 style full audit and closed-loop robot trials.")

    # Figures (embed if the PNGs exist under <folder>/figures/)
    fig_specs = [
        ("fig1_compare_bars.png", "Figure 1 — cross-model comparison",
         "Each model's best checkpoint (by XYZ endpoint) across four decoded metrics. Lower is better."),
        ("fig2_act_steps.png", "Figure 2 — ACT step progression",
         "ACT decoded metrics vs training step."),
        ("fig2_pi05_steps.png", "Figure 3 — π0.5 step progression",
         "π0.5 decoded metrics vs training step."),
        ("fig3_val_vs_decoded.png", "Figure 4 — validation loss vs decoded quality",
         "Within each model: top row XYZ endpoint, bottom row validation loss, both vs step. "
         "Dashed line = each curve's minimum — they land on different steps, so val-loss-best ≠ decoded-best."),
        ("fig4_pareto.png", "Figure 5 — accuracy vs smoothness",
         "Each model's best checkpoint: x = rotation jitter (smoother left), y = XYZ endpoint error (more accurate up). "
         "Dashed = GT jitter floor."),
    ]
    fig_lines = ["\n## Figures\n"]
    if any((FOLDER / "figures" / fn).exists() for fn, *_ in fig_specs):
        for fn, title, cap in fig_specs:
            if (FOLDER / "figures" / fn).exists():
                fig_lines.append(f"### {title}\n\n![{title}](figures/{fn})\n\n_{cap}_\n")
        lines += fig_lines

    (FOLDER / "REPORT.md").write_text("\n".join(lines) + "\n")

    # CSV
    with (FOLDER / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "label", "policy_type", "step", "val_loss", "val_step"]
                   + [k for k, *_ in METRICS])
        for r in sorted(runs, key=lambda r: (r["model"], r["step"])):
            w.writerow([r["model"], r["label"], r["policy_type"], r["step"],
                        r["val_loss"], r["val_step"]]
                       + [r["eb"].get(k) for k, *_ in METRICS])
    print(f"Wrote {FOLDER/'REPORT.md'} and {FOLDER/'summary.csv'} ({len(runs)} checkpoints, {len(models)} models).")


if __name__ == "__main__":
    main()
