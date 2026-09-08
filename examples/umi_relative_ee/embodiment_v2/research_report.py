"""Portable illustrated experiment report built from completed, immutable evaluations."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .strata import infeasible_category
from .watch import utc

METHODS = [
    ("C0", "ik"),
    ("C0", "query"),
    ("C0", "residual"),
    ("C0", "hindsight"),
    ("C1", "hindsight"),
    ("bootstrap20k", "hindsight"),
]
LABELS = [
    "Runtime IK",
    "Query imitation",
    "IK + residual",
    "Direct hindsight C0",
    "Direct hindsight C1",
    "Extra bootstrap",
]
COLORS = ["#64748b", "#d97706", "#0891b2", "#7c3aed", "#059669", "#db2777"]


def cluster_interval(rows, metric):
    """Trial-weighted mean, resampling source episodes; seeds remain explicit in plots."""
    if not rows:
        return None
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["episode"]].append(row[metric])
    values = list(clusters.values())
    totals = np.array([sum(v) for v in values])
    counts = np.array([len(v) for v in values])
    ids = np.random.default_rng(9041).integers(len(values), size=(2000, len(values)))
    samples = totals[ids].sum(1) / counts[ids].sum(1)
    return {
        "mean": float(totals.sum() / counts.sum()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "trials": len(rows),
        "episodes": len(values),
        "seeds": sorted({r["seed"] for r in rows}),
    }


def load_evaluations(parent):
    groups = defaultdict(list)
    provenance = {}
    for robot in ("piper", "so101"):
        for file in sorted((parent / robot).glob("eval/*/*/*/results.json")):
            if not (file.parent / "complete.json").exists():
                continue
            content = file.read_bytes()
            provenance[str(file.relative_to(parent))] = hashlib.sha256(content).hexdigest()
            for row in json.loads(content):
                row["_directory"] = str(file.parent)
                groups[(robot, row["condition"], row["target"], row["generation"], row["method"])].append(row)
    return groups, provenance


def build(parent: Path, final=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out = parent / "report" / ("final" if final else "preview")
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 170,
            "axes.titleweight": "bold",
        }
    )
    groups, provenance = load_evaluations(parent)
    if not groups:
        raise ValueError("No completed evaluations to illustrate")
    gallery, statistics, trajectories = [], [], []
    pdf = PdfPages(out / "figure_atlas.pdf")

    def save(fig, name, caption):
        fig.savefig(figures / f"{name}.png", bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        gallery.append((name, caption))

    # Identical method order and separate denominators in every comparison panel.
    for robot in ("piper", "so101"):
        for target in ("predicted", "truth"):
            for metric, unit, scale in [
                ("position_rmse_m", "Position RMSE (mm)", 1000),
                ("rotation_mean_deg", "Rotation error (degrees)", 1),
                ("tracking_failed", "Tracking failure (%)", 100),
            ]:
                fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
                for row_index, condition in enumerate(("native", "delay40")):
                    for col, feasible in enumerate((True, False)):
                        ax = axes[row_index, col]
                        any_data = False
                        for index, method in enumerate(METHODS):
                            rows = [
                                r
                                for r in groups.get((robot, condition, target, *method), [])
                                if bool(r["ik_feasible"]) == feasible
                            ]
                            stat = cluster_interval(rows, metric)
                            if stat is None:
                                continue
                            any_data = True
                            mean = stat["mean"] * scale
                            low, high = np.array(stat["ci95"]) * scale
                            ax.plot([low, high], [index, index], color=COLORS[index], lw=3)
                            ax.plot(mean, index, "s", color=COLORS[index], ms=6)
                            for seed in stat["seeds"]:
                                seed_mean = np.mean([r[metric] for r in rows if r["seed"] == seed]) * scale
                                ax.plot(seed_mean, index + 0.13, ".", color=COLORS[index], ms=7)
                            statistics.append(
                                {
                                    "robot": robot,
                                    "condition": condition,
                                    "target": target,
                                    "generation": method[0],
                                    "method": method[1],
                                    "subset": "feasible" if feasible else "infeasible",
                                    "metric": metric,
                                    **stat,
                                }
                            )
                        ax.set_yticks(range(len(METHODS)), LABELS)
                        ax.invert_yaxis()
                        ax.set_xlim(left=0)
                        ax.set_title(f"{condition} · {'Feasible' if feasible else 'Infeasible'}")
                        ax.set_xlabel(unit)
                        ax.grid(axis="x", alpha=0.2)
                        if not any_data:
                            ax.text(0.5, 0.5, "Not completed yet", transform=ax.transAxes, ha="center")
                fig.suptitle(
                    f"{robot.upper()} · {'ACT-predicted' if target == 'predicted' else 'Recorded ground-truth'} targets"
                )
                save(
                    fig,
                    f"{robot}_{target}_{metric}",
                    f"{robot.upper()}, {target}: {unit}. Squares are trial-weighted means; lines are 95% "
                    "source-episode bootstrap intervals, conditional on available training seeds. Small dots "
                    "show individual seed means. Feasible and infeasible trials use separate denominators. "
                    "Panel scales adapt independently; missing runs are not zero-valued results.",
                )

    # Feasibility coverage and failure modes, deduplicated across methods/seeds.
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    categories = ["feasible", "velocity_only", "outside_workspace", "other_kinematic"]
    coverage, names = [], []
    for robot in ("piper", "so101"):
        speed = json.loads((parent / robot / "config.json").read_text())["max_joint_vel_deg_s"]
        for condition in ("native", "delay40"):
            for target in ("predicted", "truth"):
                unique = {}
                for key, rows in groups.items():
                    if key[:3] == (robot, condition, target):
                        for row in rows:
                            if (
                                row["trial"] in unique
                                and unique[row["trial"]]["ik_feasible"] != row["ik_feasible"]
                            ):
                                raise ValueError("Feasibility changed across controllers")
                            unique[row["trial"]] = row
                if not unique:
                    continue
                counts = dict.fromkeys(categories, 0)
                for row in unique.values():
                    counts[infeasible_category(row, speed) or "feasible"] += 1
                names.append(f"{robot} / {condition} / {target} (n={len(unique)})")
                coverage.append([counts[c] / len(unique) * 100 for c in categories])
    values = np.array(coverage)
    left = np.zeros(len(values))
    for i, (label, color) in enumerate(
        zip(categories, ["#059669", "#eab308", "#dc2626", "#7c3aed"], strict=True)
    ):
        ax.barh(range(len(values)), values[:, i], left=left, label=label.replace("_", " "), color=color)
        left += values[:, i]
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of unique trajectory queries (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("What the nominal IK feasibility screen rejects")
    save(
        fig,
        "feasibility_coverage",
        "Each query is counted once across methods and seeds. Categories are "
        "mutually exclusive: velocity-only failures, then any workspace violation, then other IK/kinematic "
        "failures. Failed screening does not prove that every possible controller would fail.",
    )

    # Paired iteration comparison, with matched seeds and trajectory IDs.
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), constrained_layout=True)
    for ax, reference in zip(axes, ["C0", "bootstrap20k"], strict=True):
        labels, idx = [], 0
        for robot in ("piper", "so101"):
            for condition in ("native", "delay40"):
                a = groups.get((robot, condition, "predicted", "C1", "hindsight"), [])
                b = groups.get((robot, condition, "predicted", reference, "hindsight"), [])
                # Use only seeds with completed evaluations on both sides.
                shared = {r["seed"] for r in a} & {r["seed"] for r in b}
                right = {(r["seed"], r["trial"]): r for r in b if r["seed"] in shared}
                left_rows = [r for r in a if r["seed"] in shared]
                if {(r["seed"], r["trial"]) for r in left_rows} != right.keys():
                    raise ValueError("Iteration comparison has unmatched trials")
                for feasible in (True, False):
                    differences = []
                    for r in left_rows:
                        other = right[(r["seed"], r["trial"])]
                        if r["ik_feasible"] != other["ik_feasible"]:
                            raise ValueError("Paired feasibility differs")
                        if bool(r["ik_feasible"]) == feasible:
                            differences.append(
                                {**r, "position_rmse_m": r["position_rmse_m"] - other["position_rmse_m"]}
                            )
                    stat = cluster_interval(differences, "position_rmse_m")
                    if stat:
                        ax.plot(np.array(stat["ci95"]) * 1000, [idx, idx], color="#7c3aed", lw=3)
                        ax.plot(stat["mean"] * 1000, idx, "o", color="#7c3aed")
                        labels.append(
                            f"{robot}/{condition}/{'feasible' if feasible else 'infeasible'} · {len(shared)} seeds"
                        )
                        idx += 1
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(range(idx), labels)
        ax.invert_yaxis()
        ax.set_title(f"C1 minus {reference}")
        ax.set_xlabel("Paired position RMSE difference (mm) ← better")
        ax.grid(axis="x", alpha=0.2)
        if not idx:
            ax.text(0.5, 0.5, "Awaiting matched runs", transform=ax.transAxes, ha="center")
    save(
        fig,
        "iteration",
        "ACT-target paired comparisons. Negative differences favor C1. C0 comparisons "
        "combine extra data and recollection; extra-bootstrap comparisons control for data quantity. "
        "Only jointly completed seeds are included; intervals resample source episodes.",
    )

    # Original learning curves show all saved development checkpoints, not only winners.
    for robot in ("piper", "so101"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        for row_index, condition in enumerate(("native", "delay40")):
            for index, method in enumerate(METHODS[1:], start=1):
                for seed in (1000, 2000, 3000):
                    directory = (
                        parent / robot / f"train/{condition}/{method[0]}/{method[1]}_seed{seed}/development"
                    )
                    points = [
                        (int(p.stem.removeprefix("step")), json.loads(p.read_text()))
                        for p in sorted(directory.glob("step*.json"))
                    ]
                    if not points:
                        continue
                    for col, metric in enumerate(("position_rmse_m", "loss")):
                        scale = 1000 if col == 0 else 1
                        axes[row_index, col].plot(
                            [x for x, _ in points],
                            [r[metric] * scale for _, r in points],
                            color=COLORS[index],
                            alpha=0.65,
                            label=LABELS[index] if seed == 1000 else None,
                        )
            for col, unit in enumerate(("Development position RMSE (mm)", "Training normalized L1")):
                axes[row_index, col].set(title=condition, xlabel="Training updates", ylabel=unit)
                axes[row_index, col].grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=3)
        fig.suptitle(f"{robot.upper()} learning curves · individual seeds")
        save(
            fig,
            f"{robot}_learning",
            "Each line is one seed at saved checkpoints. Training loss and "
            "closed-loop tracking have different units and need not improve together. Missing delayed runs remain blank.",
        )

    # Matched development ablation, deliberately separated from held-out task evaluation.
    for complete in sorted((parent / "pilots").glob("comparison/*/complete.json")):
        directory = complete.parent
        reports = {
            p.stem: json.loads(p.read_text())["trials"]
            for p in directory.glob("*.json")
            if p.name not in ("manifest.json", "complete.json")
        }
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        for i, feasible in enumerate((True, False)):
            for j, (metric, scale, unit) in enumerate(
                [
                    ("position_rmse_m", 1000, "Position RMSE (mm)"),
                    ("rotation_mean_deg", 1, "Rotation (degrees)"),
                ]
            ):
                ax = axes[i, j]
                for k, (_name, rows) in enumerate(sorted(reports.items())):
                    subset = [r for r in rows if bool(r["ik_feasible"]) == feasible]
                    if subset:
                        ax.barh(k, np.mean([r[metric] for r in subset]) * scale)
                ax.set_yticks(range(len(reports)), [x.replace("_", " ") for x in sorted(reports)])
                ax.invert_yaxis()
                ax.set(title="Feasible" if feasible else "Infeasible", xlabel=unit)
                ax.grid(axis="x", alpha=0.2)
        save(
            fig,
            f"ablation_{directory.name}",
            "Exploratory 64-rollout embodiment-development comparison. "
            "Raw-command and bounded-command direct models share architecture, seed and 10k update budget; "
            "v1 baselines have 30k updates. These are not task-validation or manipulation-success results.",
        )

    # Representative rollouts selected using the IK baseline only, never a preferred model.
    for robot in ("piper", "so101"):
        for condition in ("native", "delay40"):
            baseline = groups.get((robot, condition, "predicted", "C0", "ik"), [])
            for feasible in (True, False):
                subset = sorted(
                    [r for r in baseline if bool(r["ik_feasible"]) == feasible],
                    key=lambda r: (r["position_rmse_m"], r["trial"]),
                )
                if not subset:
                    continue
                chosen = subset[len(subset) // 2]
                with np.load(Path(chosen["_directory"]) / "trials" / (chosen["trial"] + ".npz")) as data:
                    query = data["query"][:, :3, 3]
                    qt = data["query_time"]
                    origin = query[0]
                fig = plt.figure(figsize=(13, 5), constrained_layout=True)
                ax = fig.add_subplot(121, projection="3d")
                error_ax = fig.add_subplot(122)
                desired = (query - origin) * 1000
                ax.plot(*desired.T, "k--", label="Desired EE")
                item = {
                    "name": f"{robot}/{condition}/{'feasible' if feasible else 'infeasible'}/{chosen['trial']}",
                    "query": desired.tolist(),
                    "methods": {},
                }
                for index in (0, 2, 3, 4):
                    rows = groups.get((robot, condition, "predicted", *METHODS[index]), [])
                    matched = next(
                        (r for r in rows if r["seed"] == 1000 and r["trial"] == chosen["trial"]), None
                    )
                    if matched is None:
                        continue
                    with np.load(Path(matched["_directory"]) / "trials" / (chosen["trial"] + ".npz")) as data:
                        times = data["time"][10:]
                        actual = data["actual"][10:, :3, 3]
                    xyz = (actual - origin) * 1000
                    desired_now = np.stack([np.interp(times, qt, query[:, j]) for j in range(3)], -1)
                    error = np.linalg.norm(actual - desired_now, axis=-1) * 1000
                    ax.plot(*xyz.T, label=LABELS[index], color=COLORS[index])
                    error_ax.plot(times - times[0], error, label=LABELS[index], color=COLORS[index])
                    item["methods"][LABELS[index]] = {
                        "xyz": xyz.tolist(),
                        "time": (times - times[0]).tolist(),
                        "color": COLORS[index],
                    }
                trajectories.append(item)
                ax.set(
                    xlabel="X (mm)",
                    ylabel="Y (mm)",
                    zlabel="Z (mm)",
                    title="EE path relative to initial target",
                )
                ax.legend(fontsize=7)
                error_ax.set(
                    xlabel="Elapsed time (s)", ylabel="Position error (mm)", title="Tracking error over time"
                )
                error_ax.grid(alpha=0.2)
                name = f"path_{robot}_{condition}_{'feasible' if feasible else 'infeasible'}"
                save(
                    fig,
                    name,
                    f"{item['name']}. Selected as the median IK-baseline error within its subset; "
                    "same query/start across controllers, seed 1000. This illustration is not a distribution summary.",
                )
    pdf.close()
    state = "FINAL" if final else "PREVIEW — experiments still incomplete"
    protocol = (
        "Feasibility is the existing nominal sequential-IK screen (workspace/joint limits, ≤5 mm "
        "position residual, ≤3° orientation residual, and configured speed limit). It is not proof "
        "of physical impossibility. Reported failures use the existing >50 mm / >15° or execution-error "
        "criterion. Neither metric measures task success. ACT-R18 is frozen; task-validation and "
        "embodiment-development results are kept separate. This single-task simulation does not establish "
        "cross-task transfer, real backlash compensation, or external-tracker benefits. GPU/CPU contention "
        "means recorded latency is not an isolated hardware benchmark."
    )
    lines = [
        "# Task-independent embodiment learning",
        "",
        state,
        "",
        f"Generated: {utc()}",
        "",
        protocol,
        "",
        "Read the method comparisons first, then feasibility coverage, iteration controls, learning curves "
        "and the development ablation. Finally inspect representative paths. Missing runs are excluded, "
        "never assigned zero error. Seed dots expose training variability; bootstrap intervals reflect "
        "episode sampling conditional on the observed seeds.",
        "",
    ]
    findings = [
        "## Results at a glance",
        "",
        "ACT-target position error below summarizes completed runs only. Best observed means are descriptive, "
        "not significance tests. Consult the seed dots and paired comparisons before claiming an advantage.",
        "",
        "| Robot / dynamics / subset | Best observed method | Position (mm) | Best direct method | Position (mm) |",
        "|---|---|---:|---|---:|",
    ]
    for robot in ("piper", "so101"):
        for condition in ("native", "delay40"):
            for subset in ("feasible", "infeasible"):
                choices = [
                    s
                    for s in statistics
                    if s["robot"] == robot
                    and s["condition"] == condition
                    and s["target"] == "predicted"
                    and s["subset"] == subset
                    and s["metric"] == "position_rmse_m"
                ]
                direct = [s for s in choices if s["method"] == "hindsight"]
                if not choices or not direct:
                    continue
                best, motor = min(choices, key=lambda s: s["mean"]), min(direct, key=lambda s: s["mean"])
                label = f"{robot} / {condition} / {subset}"
                values = [
                    label,
                    f"{best['generation']}/{best['method']}",
                    f"{best['mean'] * 1000:.2f}",
                    f"{motor['generation']}/{motor['method']}",
                    f"{motor['mean'] * 1000:.2f}",
                ]
                findings.append("| " + " | ".join(values) + " |")
    lines += findings + [
        "",
        "The reusable-embodiment hypothesis needs strong feasible-motion tracking, "
        "an advantage from hindsight over query imitation, and C1 gains beyond an "
        "equal-data bootstrap control. Poor results on infeasible queries require a "
        "separate feasibility or projection solution; they should not hide failures "
        "on feasible queries. Cross-task and physical-robot claims remain untested.",
        "",
    ]
    for name, caption in gallery:
        lines += [f"## {name.replace('_', ' ')}", "", f"![{name}](figures/{name}.png)", "", caption, ""]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    (out / "trajectory_examples.json").write_text(json.dumps(trajectories, indent=2) + "\n")
    result = {
        "state": state,
        "generated_utc": utc(),
        "figure_count": len(gallery),
        "input_sha256": provenance,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    destination = "final" if (parent / "report/final/complete.json").exists() else out.name
    (parent / "report/README.md").write_text(
        f"# Embodiment research report\n\n[Read the illustrated Markdown report]({destination}/REPORT.md).\n"
    )
    return result
