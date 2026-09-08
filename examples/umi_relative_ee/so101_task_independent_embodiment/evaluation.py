"""Fixed ACT queries, paired simulation evaluation, and episode-clustered reporting."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import (
    HERE,
    Config,
    digest,
    interpolate,
    manifest,
    poses_from_actions,
    read_episodes,
    relative_poses,
    rotation_errors,
    save_npz,
    source_fingerprint,
    write_json,
)


def cache_act(root: Path, cfg: Config, device: str, resume: bool):
    import torch

    from examples.umi_relative_ee.eval_open_loop_dataset import predict_chunk_at_frame
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from .experiment import code_identity

    queries_file = HERE.parent / "act_flow_ablation/repro/query_frames_h10_seed1000.json"
    queries = json.loads(queries_file.read_text())["queries"]
    if cfg.query_limit:
        indices = np.linspace(0, len(queries) - 1, min(cfg.query_limit, len(queries)), dtype=int)
        queries = [queries[i] for i in indices]
    checkpoint = Path(cfg.task_checkpoint)
    directory = root / "act_cache"
    expected = {
        "config": cfg.to_dict(),
        "query_manifest_sha256": digest(queries_file),
        "queries": queries,
        "checkpoint": {p.name: digest(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
        "validation": source_fingerprint(Path(cfg.val_root)),
        "validation_videos": {
            str(p.relative_to(cfg.val_root)): digest(p)
            for p in sorted(Path(cfg.val_root).glob("videos/**/*.mp4"))
        },
        "code": code_identity(),
    }
    manifest(directory / "manifest.json", expected, resume)
    if (directory / "complete.json").exists():
        return
    pcfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    pcfg.device = device
    # The full saved weights replace initialization; avoid downloading ImageNet weights.
    pcfg.pretrained_backbone_weights = None
    policy = get_policy_class(pcfg.type).from_pretrained(checkpoint, config=pcfg, local_files_only=True)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    repo_id = "sroi/" + Path(cfg.val_root).name
    metadata = LeRobotDatasetMetadata(repo_id, root=cfg.val_root)
    dataset = LeRobotDataset(
        repo_id,
        root=cfg.val_root,
        delta_timestamps=resolve_delta_timestamps(pcfg, metadata),
        return_uint8=True,
        video_backend="pyav",
    )
    episodes = read_episodes(Path(cfg.val_root))
    metrics = []
    for query in queries:
        eid, frame = query["episode_index"], query["frame_index"]
        output = directory / f"queries/{eid:06d}_{frame:06d}.npz"
        record = output.with_suffix(".json")
        if output.exists() and record.exists():
            cached = json.loads(record.read_text())
            if cached["sha256"] != digest(output):
                raise ValueError(f"Corrupt ACT cache: {output}")
            metrics.append(cached)
            continue
        index = int(metadata.episodes[eid]["dataset_from_index"]) + frame
        predicted, truth, seconds = predict_chunk_at_frame(
            dataset,
            index,
            frame,
            cfg.seed,
            policy,
            pre,
            post,
            torch.device(device),
            "pick the strawberry",
        )
        predicted, truth = predicted.numpy()[: cfg.horizon], truth.numpy()[: cfg.horizon]
        source = episodes[eid]
        if len(predicted) != cfg.horizon or not np.allclose(
            truth, source["actions"][frame : frame + cfg.horizon], atol=1e-6
        ):
            raise ValueError("ACT decoder/ground-truth anchor alignment differs from protocol")
        anchor = poses_from_actions(source["actions"][frame : frame + 1])[0]
        pred_pose, gt_pose = poses_from_actions(predicted), poses_from_actions(truth)
        p = np.linalg.norm(pred_pose[:, :3, 3] - gt_pose[:, :3, 3], axis=-1)
        r = rotation_errors(pred_pose, gt_pose)
        save_npz(
            output,
            predicted_relative=relative_poses(pred_pose, anchor),
            truth_relative=relative_poses(gt_pose, anchor),
            predicted_gripper=predicted[:, 6],
            truth_gripper=truth[:, 6],
            times=np.arange(cfg.horizon) / cfg.ee_hz,
        )
        value = {
            **query,
            "position_rmse_m": float(np.sqrt(np.mean(p**2))),
            "position_end_m": float(p[-1]),
            "rotation_mean_deg": float(r.mean()),
            "rotation_end_deg": float(r[-1]),
            "inference_ms": seconds * 1000,
            "sha256": digest(output),
        }
        write_json(record, value)
        metrics.append(value)
        print(f"cached ACT episode={eid} frame={frame}", flush=True)
    write_json(directory / "task_prediction_metrics.json", metrics)
    write_json(directory / "complete.json", {"queries": len(queries)})


def evaluate_stage(root, cfg, condition, generation, method, seed, device, resume):
    from .experiment import code_identity, simulator_identity, training_path
    from .learning import LearnedController
    from .simulation import Simulation, rollout_metrics

    cache = root / "act_cache"
    if not (cache / "complete.json").exists():
        raise ValueError("Complete cache-act before evaluation")
    if method == "ik" and (generation != "C0" or seed != 1000):
        raise ValueError("Deterministic IK is evaluated once as C0 seed1000")
    checkpoint = (
        None if method == "ik" else training_path(root, condition, generation, method, seed) / "best.pt"
    )
    controller = None if checkpoint is None else LearnedController(checkpoint, device)
    directory = root / f"eval/{condition}/{generation}/{method}_seed{seed}"
    files = sorted(cache.glob("queries/*.npz"))
    expected = {
        "config": cfg.to_dict(),
        "condition": condition,
        "generation": generation,
        "method": method,
        "seed": seed,
        "checkpoint": digest(checkpoint) if checkpoint else "ik",
        "cache": digest(cache / "manifest.json"),
        "queries": {p.name: digest(p) for p in files},
        "simulator": simulator_identity(),
        "code": code_identity(),
    }
    manifest(directory / "manifest.json", expected, resume)
    if (directory / "complete.json").exists():
        return
    sim = Simulation(cfg, condition)
    rows = []
    for file in files:
        eid, frame = map(int, file.stem.split("_"))
        with np.load(file) as source:
            source = dict(source)
        for start_id in range(cfg.starts_per_query):
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, eid, frame, start_id, 99]))
            start = sim.sample_start(rng)
            for target in ("truth", "predicted"):
                trial = f"{file.stem}_{start_id}_{target}"
                output = directory / f"trials/{trial}.json"
                rollout_path = output.with_suffix(".npz")
                if output.exists() and rollout_path.exists():
                    row = json.loads(output.read_text())
                    if row["rollout_sha256"] != digest(rollout_path):
                        raise ValueError(f"Corrupt evaluation: {rollout_path}")
                    rows.append(row)
                    continue
                motion = {
                    "relative": source[target + "_relative"],
                    "times": source["times"],
                    "gripper": source[target + "_gripper"],
                }
                result, feasible = sim.rollout(motion, start, controller)
                result["gripper_query"] = motion["gripper"]
                values = rollout_metrics(result, cfg)
                begin = cfg.history - 1
                ground_truth = result["anchor"] @ source["truth_relative"]
                gt, _ = interpolate(result["query_time"], ground_truth, result["time"][begin:])
                actual = result["actual"][begin:]
                p = np.linalg.norm(actual[:, :3, 3] - gt[:, :3, 3], axis=-1)
                values["execution_vs_demo_position_rmse_m"] = float(np.sqrt(np.mean(p**2)))
                values["execution_vs_demo_rotation_mean_deg"] = float(rotation_errors(actual, gt).mean())
                save_npz(rollout_path, **result)
                row = {
                    "condition": condition,
                    "generation": generation,
                    "method": method,
                    "seed": seed,
                    "episode": eid,
                    "frame": frame,
                    "start_id": start_id,
                    "target": target,
                    "trial": trial,
                    "rollout_sha256": digest(rollout_path),
                    **feasible,
                    **values,
                }
                write_json(output, row)
                rows.append(row)
        print(f"eval {condition}/{generation}/{method} seed={seed} episode={eid} frame={frame}", flush=True)
    write_json(directory / "results.json", rows)
    write_json(directory / "complete.json", {"trials": len(rows)})


METRICS = (
    "position_rmse_m",
    "position_end_m",
    "rotation_mean_deg",
    "rotation_end_deg",
    "execution_vs_demo_position_rmse_m",
    "execution_vs_demo_rotation_mean_deg",
    "command_acceleration_rad_s2",
    "joint_limit_rate",
    "invalid_output_rate",
    "tracking_failed",
    "execution_error",
    "latency_p95_ms",
    "deadline_miss_rate",
    "gripper_rmse_norm",
)


def episode_bootstrap(values: dict[int, float], resamples: int, seed=1000):
    if not values:
        return {"mean": None, "ci95": [None, None], "episodes": 0}
    array = np.asarray([values[k] for k in sorted(values)])
    rng = np.random.default_rng(seed)
    boot = array[rng.integers(len(array), size=(resamples, len(array)))].mean(1)
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "episodes": len(array),
    }


def clustered(rows, metric):
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["episode"]].append(row[metric])
    return {eid: float(np.mean(vals)) for eid, vals in episodes.items()}


def paired_comparison(a, b, metric, feasible_only=False):
    left = {row["trial"]: row for row in a}
    right = {row["trial"]: row for row in b}
    if left.keys() != right.keys():
        raise ValueError(
            "Paired comparison requires identical trial sets; partial evaluations are not comparable"
        )
    differences = []
    for key in sorted(left):
        x, y = left[key], right[key]
        if x["ik_feasible"] != y["ik_feasible"]:
            raise ValueError("Method-independent feasibility classification changed across methods")
        if not feasible_only or x["ik_feasible"]:
            differences.append({"episode": x["episode"], metric: x[metric] - y[metric]})
    return clustered(differences, metric)


def report(root, cfg):
    rows = []
    for file in sorted((root / "eval").glob("*/*/*/results.json")):
        if (file.parent / "complete.json").exists():
            rows.extend(json.loads(file.read_text()))
    if not rows:
        raise ValueError("No completed evaluations")
    directory = root / "report"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "trials.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["target"], row["generation"], row["method"], row["seed"])].append(row)
    seeds = (1000,) if cfg.smoke else (1000, 2000, 3000)
    expected_groups = set()
    for condition in ("native", "delay40"):
        for target in ("truth", "predicted"):
            expected_groups.add((condition, target, "C0", "ik", 1000))
            for seed in seeds:
                for generation, method in (
                    ("C0", "query"),
                    ("C0", "residual"),
                    ("C0", "hindsight"),
                    ("C1", "hindsight"),
                    ("bootstrap20k", "hindsight"),
                ):
                    expected_groups.add((condition, target, generation, method, seed))
    missing_groups = sorted(expected_groups - groups.keys())
    summaries, comparisons = [], []
    for key, values in sorted(groups.items()):
        for subset in ("all", "ik_feasible"):
            included = values if subset == "all" else [v for v in values if v["ik_feasible"]]
            summaries.append(
                {
                    "condition": key[0],
                    "target": key[1],
                    "generation": key[2],
                    "method": key[3],
                    "seed": key[4],
                    "subset": subset,
                    "trials": len(included),
                    "total_trials": len(values),
                    "feasibility_coverage": float(np.mean([v["ik_feasible"] for v in values])),
                    "metrics": {
                        m: episode_bootstrap(clustered(included, m), cfg.bootstrap_resamples) for m in METRICS
                    },
                }
            )
        condition, target, generation, method, seed = key
        reference_keys = [(condition, target, "C0", "ik", 1000)] if method != "ik" else []
        if generation == "C0" and method == "hindsight":
            reference_keys += [(condition, target, "C0", m, seed) for m in ("query", "residual")]
        if generation == "C1":
            reference_keys += [(condition, target, g, "hindsight", seed) for g in ("C0", "bootstrap20k")]
        for reference in reference_keys:
            if reference not in groups:
                continue
            for subset in ("all", "ik_feasible"):
                comparisons.append(
                    {
                        "candidate": list(key),
                        "reference": list(reference),
                        "subset": subset,
                        "difference_candidate_minus_reference": {
                            m: episode_bootstrap(
                                paired_comparison(values, groups[reference], m, subset == "ik_feasible"),
                                cfg.bootstrap_resamples,
                            )
                            for m in METRICS
                        },
                    }
                )
    task_metrics = json.loads((root / "act_cache/task_prediction_metrics.json").read_text())
    task_rows = [{"episode": v["episode_index"], **v} for v in task_metrics]
    task_summary = {
        m: episode_bootstrap(clustered(task_rows, m), cfg.bootstrap_resamples)
        for m in ("position_rmse_m", "position_end_m", "rotation_mean_deg", "rotation_end_deg")
    }
    write_json(
        directory / "summary.json",
        {
            "smoke": cfg.smoke,
            "matrix_complete": not missing_groups,
            "missing_groups": [list(key) for key in missing_groups],
            "task_prediction": task_summary,
            "groups": summaries,
            "paired_comparisons": comparisons,
        },
    )
    flat = []
    for group in summaries:
        flat.append(
            {
                **{k: v for k, v in group.items() if k != "metrics"},
                **{m: v["mean"] for m, v in group["metrics"].items()},
            }
        )
    with (directory / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Piper embodiment experiment",
        "",
        "SMOKE RUN: software validation only; insufficient data/training for research conclusions."
        if cfg.smoke
        else "Paired validation results; negative differences favor the candidate.",
        "",
        "ACT prediction error, controller tracking error, and execution-versus-demonstration error are separate metrics.",
        "Confidence intervals resample source episodes; training seeds are reported separately.",
        "Comparison matrix complete."
        if not missing_groups
        else f"PARTIAL MATRIX: {len(missing_groups)} groups remain.",
        "",
        "| Dynamics | Targets | Generation / method / seed | Subset | Coverage | Position RMSE (mm) | Rotation (deg) |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in flat:
        p, r = row["position_rmse_m"], row["rotation_mean_deg"]
        lines.append(
            f"| {row['condition']} | {row['target']} | {row['generation']}/{row['method']}/{row['seed']} | {row['subset']} | {row['feasibility_coverage']:.1%} | {'n/a' if p is None else f'{p * 1000:.2f}'} | {'n/a' if r is None else f'{r:.2f}'} |"
        )
    lines += [
        "",
        "Full episode-level confidence intervals and paired differences are in `summary.json`.",
        "This single-task rigid-body simulation does not measure grasp success, cross-task transfer, real backlash, or external-tracker benefits.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (condition, target) in zip(
        axes.flat, ((c, t) for c in ("native", "delay40") for t in ("truth", "predicted")), strict=False
    ):
        entries = [
            s
            for s in summaries
            if s["condition"] == condition and s["target"] == target and s["subset"] == "all"
        ]
        for i, entry in enumerate(entries):
            metric = entry["metrics"]["position_rmse_m"]
            lo, hi = metric["ci95"]
            ax.plot([lo * 1000, hi * 1000], [i, i], color="C0")
            ax.plot(metric["mean"] * 1000, i, "o", color="C0")
        ax.set_yticks(
            range(len(entries)),
            [f"{e['generation']}/{e['method']}/s{e['seed']}" for e in entries],
            fontsize=8,
        )
        ax.set(title=f"{condition}, {target}", xlabel="Tracking position RMSE (mm), episode bootstrap 95% CI")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("SMOKE — not research evidence" if cfg.smoke else "Task-independent embodiment learning")
    fig.savefig(directory / "tracking.png", dpi=160)
    fig.savefig(directory / "tracking.pdf")
    plt.close(fig)
    print(f"Report: {directory / 'REPORT.md'}", flush=True)
