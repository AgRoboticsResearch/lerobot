#!/usr/bin/env python
"""Eval SmolVLA UMI EE-pose on training data.

Measures how well the model reconstructs the training actions:
  - Load a checkpoint
  - Run inference on training batches (with action chunks via delta_timestamps)
  - Compare predicted actions vs ground truth actions (both in absolute space)
  - Report MSE, per-dim error

Verbose mode (--verbose) traces one batch through the full pipeline:
  raw → relative → normalize → model predict → unnormalize → absolute
"""

import torch
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/sroi_lab_picking_all"
DATASET_REPO = "sroi_lab_picking_all"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _print_tensor(label, tensor, indent=2):
    """Print shape, range, and per-dim stats for a tensor."""
    prefix = " " * indent
    if tensor.ndim == 3:
        print(f"{prefix}{label}: shape={tensor.shape}")
        print(f"{prefix}  range: [{tensor.min():.6f}, {tensor.max():.6f}]")
        print(f"{prefix}  mean per dim (first timestep):")
        for d in range(tensor.shape[-1]):
            print(f"{prefix}    dim {d}: mean={tensor[:, 0, d].mean():.6f}, "
                  f"std={tensor[:, 0, d].std():.6f}, "
                  f"range=[{tensor[:, 0, d].min():.6f}, {tensor[:, 0, d].max():.6f}]")
    elif tensor.ndim == 2:
        print(f"{prefix}{label}: shape={tensor.shape}")
        print(f"{prefix}  range: [{tensor.min():.6f}, {tensor.max():.6f}]")
        for d in range(tensor.shape[-1]):
            print(f"{prefix}    dim {d}: mean={tensor[:, d].mean():.6f}, "
                  f"std={tensor[:, d].std():.6f}")
    else:
        print(f"{prefix}{label}: shape={tensor.shape}, range=[{tensor.min():.6f}, {tensor.max():.6f}]")


def eval_checkpoint(checkpoint_dir: str, n_batches: int = 50, verbose: bool = False, max_steps: int = 50):
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor.relative_action_processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
    )
    from torch.utils.data import DataLoader

    checkpoint_dir = Path(checkpoint_dir)
    print(f"=== Evaluating checkpoint: {checkpoint_dir} ===")
    print(f"Device: {DEVICE}")

    # Build config
    policy_cfg = SmolVLAConfig(
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        device=DEVICE,
        resize_imgs_with_padding=(512, 512),
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        load_vlm_weights=False,
        push_to_hub=False,
        pretrained_path=str(checkpoint_dir),
    )

    # Resolve delta_timestamps
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    print(f"\nDelta timestamps:")
    for k, v in delta_timestamps.items():
        print(f"  {k}: {len(v)} frames ({v[0]:.3f}s .. {v[-1]:.3f}s)")

    # Load dataset
    ds = LeRobotDataset(
        DATASET_REPO,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
    )
    print(f"Dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")

    # Load policy
    print("\nLoading policy from checkpoint...")
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy = policy.to(DEVICE)
    policy.eval()

    # Build preprocessor and postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint_dir),
        dataset_stats=ds_meta.stats,
    )

    # Show pipeline structure
    print("\nPreprocessor pipeline:")
    for j, step in enumerate(preprocessor.steps):
        print(f"  [{j}] {type(step).__name__}")
        if isinstance(step, RelativeActionsProcessorStep):
            mask = step._build_mask(7)
            print(f"      enabled={step.enabled}, exclude_joints={step.exclude_joints}")
            print(f"      relative_mask={mask}")
    print("Postprocessor pipeline:")
    for j, step in enumerate(postprocessor.steps):
        print(f"  [{j}] {type(step).__name__}")
        if isinstance(step, AbsoluteActionsProcessorStep):
            print(f"      enabled={step.enabled}")

    # Eval metrics
    dim_names = ds_meta.features["action"]["names"]
    n_action_dims = ds_meta.features["action"]["shape"][0]
    total_mse = 0.0
    total_mae = 0.0
    dim_errors = torch.zeros(n_action_dims)
    n_samples = 0

    print(f"\nEvaluating on {n_batches} batches (bs=16)...")
    print(f"Action dims: {list(dim_names)}")

    dataloader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, drop_last=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=n_batches)):
            if i >= n_batches:
                break

            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            gt_action = batch["action"]          # (B, 50, 7) absolute
            gt_state = batch["observation.state"] # (B, 1, 7)

            # ---- VERBOSE: trace first batch in detail ----
            if i == 0 and verbose:
                print(f"\n{'='*70}")
                print(f"  VERBOSE TRACE — Batch 0 (first of {n_batches})")
                print(f"{'='*70}")

                print(f"\n--- STEP 0: Raw batch from DataLoader ---")
                _print_tensor("gt_action (absolute)", gt_action)
                _print_tensor("gt_state", gt_state)
                print(f"  task: {batch['task'][0]}")
                print(f"  Sample action[0, :3, :] (first 3 timesteps):")
                for t in range(3):
                    print(f"    t={t}: {gt_action[0, t].cpu().tolist()}")

            # ---- Preprocess: raw → relative → normalize ----
            # The preprocessor chains: DeviceStep → RelativeActions → Normalize
            # Relative: action[..., :6] -= state[..., :6]  (gripper excluded)
            # Normalize: (action - mean) / std
            processed = preprocessor(batch)

            if i == 0 and verbose:
                print(f"\n--- STEP 1: After preprocess (relative + normalize) ---")
                _print_tensor("processed['action']", processed["action"])

                # Verify relative conversion worked
                # Relative action = absolute_action - state (for dims 0-5)
                state_2d = gt_state[:, -1, :]  # (B, 7) squeeze obs_steps
                expected_rel = gt_action.clone()
                expected_rel[..., :6] -= state_2d[:, :6].unsqueeze(1)
                print(f"\n  Verify relative conversion (sample 0, t=0):")
                print(f"    abs action:      {gt_action[0, 0].cpu().tolist()}")
                print(f"    state:           {state_2d[0].cpu().tolist()}")
                print(f"    expected rel:    {expected_rel[0, 0].cpu().tolist()}")
                print(f"    (gripper should be UNCHANGED)")

            # ---- Model inference: predict action chunk ----
            # predict_action_chunk does flow matching denoising (10 steps)
            # Input: normalized relative actions + observations
            # Output: predicted normalized relative action chunk
            pred_action = policy.predict_action_chunk(processed)  # (B, 50, 7)

            if i == 0 and verbose:
                print(f"\n--- STEP 2: Model prediction (normalized relative space) ---")
                _print_tensor("pred_action (model output)", pred_action)
                print(f"  Sample pred[0, :3, :] (first 3 timesteps):")
                for t in range(3):
                    print(f"    t={t}: {pred_action[0, t].cpu().tolist()}")

            # ---- Postprocess: unnormalize → absolute ----
            # The postprocessor chains: Unnormalize → AbsoluteActions → Device(cpu)
            # Unnormalize: action = action * std + mean (using relative stats)
            # Absolute: action[..., :6] += cached_state[..., :6]
            pred_abs = postprocessor(pred_action)

            if i == 0 and verbose:
                print(f"\n--- STEP 3: After postprocess (absolute space) ---")
                _print_tensor("pred_abs (final absolute)", pred_abs)

                # Show roundtrip quality
                print(f"\n--- STEP 4: Comparison (sample 0, first 5 timesteps) ---")
                print(f"  {'t':>3}  {'gt_x':>8} {'pred_x':>8} {'err_x':>8}  "
                      f"{'gt_y':>8} {'pred_y':>8} {'err_y':>8}  "
                      f"{'gt_grip':>8} {'pred_grip':>8} {'err_grip':>8}")
                print(f"  {'':>3}  {'':>8} {'':>8} {'(mm)':>8}  "
                      f"{'':>8} {'':>8} {'(mm)':>8}  "
                      f"{'':>8} {'':>8}")
                for t in range(min(5, gt_action.shape[1])):
                    gt_x = gt_action[0, t, 0].item()
                    pred_x = pred_abs[0, t, 0].item()
                    gt_y = gt_action[0, t, 1].item()
                    pred_y = pred_abs[0, t, 1].item()
                    gt_g = gt_action[0, t, 6].item()
                    pred_g = pred_abs[0, t, 6].item()
                    print(f"  {t:>3}  {gt_x:>8.4f} {pred_x:>8.4f} {abs(gt_x-pred_x)*1000:>8.2f}  "
                          f"{gt_y:>8.4f} {pred_y:>8.4f} {abs(gt_y-pred_y)*1000:>8.2f}  "
                          f"{gt_g:>8.4f} {pred_g:>8.4f} {abs(gt_g-pred_g):>8.4f}")

                # Show error across full chunk
                min_len_v = min(gt_action.shape[1], pred_abs.shape[1])
                err = (pred_abs[0, :min_len_v].cpu() - gt_action[0, :min_len_v].cpu()).abs()
                print(f"\n  Error across full chunk (50 steps), sample 0:")
                print(f"    Position MAE: x={err[:, 0].mean()*1000:.2f}mm, "
                      f"y={err[:, 1].mean()*1000:.2f}mm, z={err[:, 2].mean()*1000:.2f}mm")
                print(f"    Rotation MAE: wx={err[:, 3].mean():.4f}rad, "
                      f"wy={err[:, 4].mean():.4f}rad, wz={err[:, 5].mean():.4f}rad")
                print(f"    Gripper MAE:  {err[:, 6].mean():.4f}")

                # Error accumulation along chunk
                print(f"\n  Error growth along chunk (sample 0, xyz position):")
                for t in [0, 5, 10, 20, 30, 49]:
                    if t < min_len_v:
                        pos_err = (err[t, :3] * 1000).tolist()
                        print(f"    t={t:>2}: xyz error = [{pos_err[0]:.1f}, {pos_err[1]:.1f}, {pos_err[2]:.1f}] mm")

                print(f"\n{'='*70}")
                print(f"  Verbose trace complete. Running remaining batches...")
                print(f"{'='*70}\n")

            # Compute errors (limit to max_steps for comparison)
            min_len = min(gt_action.shape[1], pred_abs.shape[1], max_steps)
            gt = gt_action[:, :min_len, :n_action_dims].cpu()
            pred = pred_abs[:, :min_len, :n_action_dims].cpu()

            error = (pred - gt).abs()
            bs = gt.shape[0]
            total_mse += error.pow(2).sum().item()
            total_mae += error.sum().item()
            dim_errors[:n_action_dims] += error.sum(dim=(0, 1)).cpu()[:n_action_dims]
            n_samples += bs * min_len

    # Report
    print(f"\n{'='*60}")
    print(f"  Evaluation Results ({n_batches} batches, {n_samples} action steps)")
    print(f"{'='*60}")
    print(f"  MSE (per-step avg):  {total_mse / n_samples:.6f}")
    print(f"  MAE (per-step avg):  {total_mae / n_samples:.6f}")
    print(f"\n  Per-dimension MAE:")
    for j, name in enumerate(dim_names):
        val = dim_errors[j] / n_samples
        unit = "mm" if j < 3 else ("rad" if j < 6 else "")
        scale = 1000 if j < 3 else 1
        print(f"    {name:20s}: {val:.6f} ({val*scale:.2f} {unit})")

    pos_error = (dim_errors[:3] / n_samples).sum().item()
    print(f"\n  Position error (sum xyz): {pos_error:.6f} (~{pos_error*1000:.2f} mm)")
    print(f"{'='*60}")

    return {
        "mse": total_mse / n_samples,
        "mae": total_mae / n_samples,
        "dim_errors": {name: (dim_errors[j] / n_samples).item() for j, name in enumerate(dim_names)},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval SmolVLA UMI EE-pose checkpoint")
    parser.add_argument("checkpoint_dir", help="Path to checkpoint pretrained_model dir")
    parser.add_argument("--n-batches", type=int, default=50, help="Number of batches to eval")
    parser.add_argument("--verbose", action="store_true", help="Trace first batch in detail")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps to compare per chunk")
    args = parser.parse_args()

    eval_checkpoint(args.checkpoint_dir, n_batches=args.n_batches, verbose=args.verbose, max_steps=args.max_steps)
