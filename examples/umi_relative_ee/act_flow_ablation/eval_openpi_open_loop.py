#!/usr/bin/env python3
"""Open-loop eval of an official-openpi (JAX) SROI LoRA checkpoint on the strawberry
validation set, using the SAME decoded-metric protocol as eval_open_loop_dataset.py
so rot6d-vs-rotvec (and the SmolVLA notation) numbers are directly comparable.

Runs in the openpi venv (pinned lerobot v2.1 reads the v2.1 validation dataset).
For each query frame: builds a lerobot-format obs (image + state + prompt), converts
the state rotvec->rot6d for the rot6d variant, calls policy.infer, decodes the
predicted chunk to rotvec (rot6d variant -> rotvec; rotvec variant -> passthrough),
and compares to the GT rotvec action chunk. Metrics mirror eval_open_loop_dataset.py.

Usage:
  python eval_openpi_open_loop.py --config-name pi05_lora_sroi_rotvec \
      --checkpoint <ckpt_dir> --output <out.json> [--samples_per_episode 5]
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# --- metric helpers (mirror eval_open_loop_dataset.py; local axis_angle_to_matrix) ---
def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues: (..., 3) axis-angle -> (..., 3, 3)."""
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    c, s = angle.cos(), angle.sin()
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(*x.shape, 3, 3)
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(*x.shape, 3, 3)
    return I + s.unsqueeze(-1) * K + (1 - c).unsqueeze(-1) * (K @ K)


def rotation_error_deg(predicted, target):
    pr = axis_angle_to_matrix(predicted[..., 3:6])
    tr = axis_angle_to_matrix(target[..., 3:6])
    rel = pr.transpose(-2, -1) @ tr
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def so3_angle_deg(matrix):
    cos = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def within_chunk_jerk(poses):
    steps = poses.shape[0]
    if steps < 3:
        return 0.0, 0.0
    rot = axis_angle_to_matrix(poses[:, 3:6])
    step_rot = rot[:-1].transpose(-2, -1) @ rot[1:]
    rot_jerk = so3_angle_deg(step_rot[:-1].transpose(-2, -1) @ step_rot[1:]).mean()
    step_xyz = poses[1:, :3] - poses[:-1, :3]
    xyz_jerk = (step_xyz[1:] - step_xyz[:-1]).norm(dim=-1).mean()
    return float(rot_jerk), float(xyz_jerk)


def rot6d_to_rotvec_chunk(chunk):  # (T,10) -> (T,7): xyz + rotvec(rot6d) + gripper
    from scipy.spatial.transform import Rotation as R
    r6 = chunk[:, 3:9].reshape(-1, 2, 3)  # 2 rows
    a, b = r6[:, 0], r6[:, 1]
    r1 = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    r2 = b - (r1 * b).sum(-1, keepdims=True) * r1
    r2 = r2 / (np.linalg.norm(r2, axis=-1, keepdims=True) + 1e-8)
    r3 = np.cross(r1, r2)
    mat = np.stack([r1, r2, r3], axis=1)
    rv = R.from_matrix(mat).as_rotvec()
    return np.concatenate([chunk[:, :3], rv, chunk[:, 9:10]], axis=1)


def rotvec_to_rot6d_state(state):  # (7,) -> (10,) for the rot6d variant input state
    from scipy.spatial.transform import Rotation as R
    xyz, rot, grip = state[:3], state[3:6], state[6:7]
    M = R.from_rotvec(rot).as_matrix()[:2, :].reshape(6)
    return np.concatenate([xyz, M, grip]).astype(np.float32)


def gt_quantile_scales(ground_truths):
    # Mirror of eval_open_loop_dataset.gt_quantile_scales: per-dim (q99-q01)/2
    # half-ranges of the pooled GT chunks -> the protocol-fixed "normalized
    # action units" for accuracy@tau (q01 -> -1, q99 -> +1, pi0.5 convention).
    pooled = torch.cat(list(ground_truths), dim=0).to(torch.float32)
    q01 = torch.quantile(pooled, 0.01, dim=0)
    q99 = torch.quantile(pooled, 0.99, dim=0)
    return ((q99 - q01) / 2).clamp_min(1e-9)


def accuracy_at_tau(predicted, ground_truth, scales, taus=(0.5, 0.1)):
    # Mirror of eval_open_loop_dataset.accuracy_at_tau: pi0.5-style thresholded
    # accuracy -- fraction of action dims (steps x dims) within tau of GT in
    # normalized action units; inclusive <=; component views per the L1/MSE
    # split; raw axis-angle components, no geodesic correction.
    normalized_error = (predicted - ground_truth).abs().to(torch.float32) / scales.to(torch.float32)
    metrics = {}
    for group, columns in {"action": slice(None), "xyz": slice(0, 3), "rotvec": slice(3, 6)}.items():
        for tau in taus:
            key = f"{group}_acc_at_{str(tau).replace('.', 'p')}"
            metrics[key] = float((normalized_error[:, columns] <= tau).to(torch.float32).mean())
    return metrics


def bootstrap_ci(episode_means, metric_names, num_resamples=10000, seed=0, confidence=0.95):
    vals = np.asarray([[r[n] for n in metric_names] for r in episode_means.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(num_resamples, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2], axis=0)
    return {n: {"low": float(lo[i]), "high": float(hi[i])} for i, n in enumerate(metric_names)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", default="/mnt/data1/sroi/lerobot/sroiv2_strawberry_validation_rotvec")
    p.add_argument("--samples_per_episode", type=int, default=5)
    p.add_argument("--action_horizon", type=int, default=10,
                   help="policy action chunk length (h10 arms -> 10; h30 arm -> 30)")
    p.add_argument("--max_queries", type=int, default=None,
                   help="cap total queries (smoke tests); None = all episodes")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    is_rot6d = "rot6d" in args.config_name
    from lerobot.common.datasets import lerobot_dataset as ld
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    repo_id = "sroiv2_strawberry_validation_rotvec"
    meta = ld.LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    action_horizon = args.action_horizon
    ds = ld.LeRobotDataset(repo_id, root=args.dataset_root,
                           delta_timestamps={"action": [t / meta.fps for t in range(action_horizon)]})

    # query frames: spread within each episode, leaving room for the chunk
    query = []
    for ep in range(meta.total_episodes):
        length = meta.episodes[ep]["length"]
        lo, hi = 0, max(0, length - action_horizon - 1)
        if hi <= lo:
            continue
        for j in range(args.samples_per_episode):
            fi = int(round(lo + (hi - lo) * (j + 0.5) / args.samples_per_episode))
            # map (ep, fi) -> global dataset index via episode_data_index
            query.append((ep, fi))
    if args.max_queries is not None:
        query = query[: args.max_queries]
    ep_data_index_from = [0]
    for ep in range(meta.total_episodes):
        ep_data_index_from.append(ep_data_index_from[-1] + meta.episodes[ep]["length"])

    print(f"loading policy {args.config_name} from {args.checkpoint} (rot6d={is_rot6d})")
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint,
                                                  default_prompt="pick the strawberry")

    metric_names = ("rotation_chunk_mean_deg", "rotation_chunk_rmse_deg", "rotation_chunk_mse_deg2",
                    "rotation_end_deg",
                    "xyz_chunk_mean_m", "xyz_chunk_rmse_m", "xyz_chunk_mse_m2", "xyz_end_m",
                    "gripper_chunk_mean", "gripper_chunk_rmse", "gripper_chunk_mse", "gripper_end",
                    "xyz_l1_per_dim_m", "xyz_mse_per_dim_m2",
                    "rotvec_l1_per_dim_deg", "rotvec_mse_per_dim_deg2",
                    "action_acc_at_0p5", "action_acc_at_0p1",
                    "xyz_acc_at_0p5", "xyz_acc_at_0p1",
                    "rotvec_acc_at_0p5", "rotvec_acc_at_0p1",
                    "rot_jerk_deg", "xyz_jerk_m", "gt_rot_jerk_deg", "gt_xyz_jerk_m")
    samples = []
    scored_chunks = []
    infer_s = []
    for si, (ep, fi) in enumerate(query):
        didx = ep_data_index_from[ep] + fi
        item = ds[didx]
        img = item["observation.images.camera"]  # torch C,H,W float
        state = np.asarray(item["observation.state"], dtype=np.float32)  # rotvec 7D
        if is_rot6d:
            state = rotvec_to_rot6d_state(state)
        # openpi's serving contract: obs arrives PRE-REPACKED with the short keys
        # (create_trained_policy does not apply the data config's RepackTransform;
        # official serve_policy.py passes none either -- LiberoInputs/SroiInputs
        # read data["image"]/data["state"] directly).
        obs = {"image": img, "state": state}
        gt = torch.as_tensor(np.asarray(item["action"]), dtype=torch.float32)  # (T,7) rotvec
        t0 = time.perf_counter()
        out = policy.infer(obs)
        infer_s.append(time.perf_counter() - t0)
        pred = np.asarray(out["actions"])  # (T, 7 or 10)
        if is_rot6d:
            pred = rot6d_to_rotvec_chunk(pred)  # -> (T,7) rotvec
        pred = torch.as_tensor(pred, dtype=torch.float32)
        steps = min(len(pred), len(gt))
        pred, gt = pred[:steps], gt[:steps]
        rot_err = rotation_error_deg(pred, gt)
        xyz_err = torch.linalg.vector_norm(pred[:, :3] - gt[:, :3], dim=-1)
        grip_err = (pred[:, 6] - gt[:, 6]).abs()
        rot_mse = float((rot_err ** 2).mean()); xyz_mse = float((xyz_err ** 2).mean()); grip_mse = float((grip_err ** 2).mean())
        # per-component L1/MSE (must mirror eval_open_loop_dataset.py per_component_l1_mse):
        # component-wise over steps x dims -- the action-space sense the training
        # objectives optimize. xyz per-dim MSE = norm-based xyz chunk MSE / 3.
        xyz_delta = pred[:, :3] - gt[:, :3]
        rotvec_delta_deg = torch.rad2deg(pred[:, 3:6] - gt[:, 3:6])
        xyz_l1 = float(xyz_delta.abs().mean()); xyz_mse_pd = float((xyz_delta ** 2).mean())
        rv_l1 = float(rotvec_delta_deg.abs().mean()); rv_mse_pd = float((rotvec_delta_deg ** 2).mean())
        prj, pxj = within_chunk_jerk(pred); grj, gxj = within_chunk_jerk(gt)
        samples.append({
            "episode_index": ep, "frame_index": fi,
            "rotation_chunk_mean_deg": float(rot_err.mean()), "rotation_chunk_rmse_deg": rot_mse ** 0.5,
            "rotation_chunk_mse_deg2": rot_mse, "rotation_end_deg": float(rot_err[-1]),
            "xyz_chunk_mean_m": float(xyz_err.mean()), "xyz_chunk_rmse_m": xyz_mse ** 0.5,
            "xyz_chunk_mse_m2": xyz_mse, "xyz_end_m": float(xyz_err[-1]),
            "gripper_chunk_mean": float(grip_err.mean()), "gripper_chunk_rmse": grip_mse ** 0.5,
            "gripper_chunk_mse": grip_mse, "gripper_end": float(grip_err[-1]),
            "xyz_l1_per_dim_m": xyz_l1, "xyz_mse_per_dim_m2": xyz_mse_pd,
            "rotvec_l1_per_dim_deg": rv_l1, "rotvec_mse_per_dim_deg2": rv_mse_pd,
            "rot_jerk_deg": prj, "xyz_jerk_m": pxj, "gt_rot_jerk_deg": grj, "gt_xyz_jerk_m": gxj,
        })
        scored_chunks.append((pred, gt))
        if (si + 1) % 50 == 0 or si + 1 == len(query):
            print(f"  {si+1}/{len(query)} queries done")

    # accuracy@tau needs a global normalization (see gt_quantile_scales).
    tau_scales = gt_quantile_scales([gt for _, gt in scored_chunks])
    for s, (pred, gt) in zip(samples, scored_chunks, strict=True):
        s.update(accuracy_at_tau(pred, gt, tau_scales))

    ep_samples = defaultdict(list)
    for s in samples:
        ep_samples[s["episode_index"]].append(s)
    ep_means = {e: {n: float(np.mean([s[n] for s in rows])) for n in metric_names} for e, rows in ep_samples.items()}
    eb = {n: float(np.mean([r[n] for r in ep_means.values()])) for n in metric_names}
    ci = bootstrap_ci(ep_means, metric_names)
    report = {
        "policy_type": args.config_name, "checkpoint": args.checkpoint,
        "is_rot6d": is_rot6d, "action_horizon": args.action_horizon,
        "num_episodes": len(ep_samples), "num_samples": len(samples),
        "inference_latency_seconds": {"mean": float(np.mean(infer_s)), "median": float(np.median(infer_s))},
        "accuracy_at_tau_normalization": {
            "definition": "per-dim error / ((q99-q01)/2 of pooled GT chunks); inclusive <= tau",
            "per_dim_half_ranges": [float(scale) for scale in tau_scales],
        },
        "summary": {"episode_balanced": eb, "episode_balanced_95ci": ci},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(f"\n=== {args.config_name} ===")
    for n in ["xyz_end_m", "rotation_end_deg", "rot_jerk_deg", "gt_rot_jerk_deg"]:
        print(f"  {n}: {eb[n]* (1000 if 'xyz' in n else 1):.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
