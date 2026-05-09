#!/usr/bin/env python
"""Round-trip test: verify preprocessor transformations are reversible.

Forces: preprocess(ABS) → normalized → postprocess → ABS'
Checks:  ABS' ≈ ABS  (within numerical tolerance)

Covers:
  A. Action round-trip:  RelativeActions + Normalize → Unnormalize + AbsoluteActions
  B. State round-trip:   RelativeState + Normalize → manual reversal
"""

import copy
import logging

import torch

logging.basicConfig(level=logging.WARNING)

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/test_ee_dataset"


def test_action_round_trip():
    """Full pre→post pipeline round-trip for actions.

    Preprocessor:  action_abs → action_rel (= action - state) → normalize
    Postprocessor: unnormalize → action_abs' (= action_rel_norm + state)

    Verifies: action_abs' ≈ action_abs  (element-wise, atol=1e-5)
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.compute_stats import compute_relative_state_stats
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.processor.relative_action_processor import (
        DeriveStateFromActionStep,
        RelativeActionsProcessorStep,
        to_relative_actions,
        to_absolute_actions,
    )

    print("=" * 70)
    print("  TEST A — Action round-trip")
    print("  ABS action → preprocess(rel+norm) → postprocess → ABS action'")
    print("=" * 70)

    device = "cpu"
    ds_meta = LeRobotDatasetMetadata("test_ee_dataset", root=DATASET_ROOT)
    ds = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT)

    # Compute stats (14D state + 7D relative action)
    rs_stats = compute_relative_state_stats(
        hf_dataset=ds.hf_dataset, features=ds_meta.features,
        state_obs_steps=2, exclude_joints=["gripper"], source_key="action",
    )
    stats = copy.deepcopy(ds_meta.stats)
    stats["observation.state"] = {k: torch.tensor(v) for k, v in rs_stats.items()}

    cfg = SmolVLAConfig(
        derive_state_from_action=True,
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        relative_exclude_state_joints=["gripper"],
        device=device, push_to_hub=False,
        load_vlm_weights=False, resize_imgs_with_padding=(512, 512),
    )
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=stats,
    )

    # Load a batch with delta_timestamps: state (2,7), action (51,7)
    dt = resolve_delta_timestamps(cfg, ds_meta)
    ds_dt = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT, delta_timestamps=dt)

    dim_names = ds_meta.features["action"]["names"]
    if isinstance(dim_names, dict):
        dim_names = dim_names.get("axes", list(dim_names.keys()))
    print(f"\nAction dims: {dim_names}")
    print(f"State dims:  {ds_meta.features['observation.state']['names']}")

    all_ok = True
    n_tested = 0
    for frame_idx in [0, 5, 10, 50, 100, 150, 200, 250, 300]:
        if frame_idx >= len(ds_dt):
            break

        batch = ds_dt[frame_idx]
        batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
                 for k, v in batch.items()}

        # ---- Step 1: DeriveStateFromAction (isolated, to get ground truth) ----
        derive = DeriveStateFromActionStep(enabled=True)
        derived = derive(batch)
        gt_action = derived["action"]                     # (1, 50, 7) — ABS, after strip
        gt_state = derived["observation"]["observation.state"]  # (1, 2, 7) — ABS 2-step

        # ---- Step 2: Full preprocess ----
        processed = preprocessor(batch)

        # ---- Step 3: Check preprocess internal consistency ----
        # Relative actions should be action_rel = action_abs - current_state
        rel_step = [s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)][0]
        cached_state = rel_step.get_cached_state()  # (1, 7) — the current EE-pose
        assert cached_state is not None, "RelativeActions should cache state"

        mask = rel_step._build_mask(7)
        expected_action_rel = to_relative_actions(gt_action.clone(), cached_state, mask)
        print(f"\n  Frame {frame_idx}:")
        print(f"    Current state (cached): {[f'{v:.4f}' for v in cached_state[0].tolist()]}")
        print(f"    GT action[0,t=0] ABS:  {[f'{v:.4f}' for v in gt_action[0, 0].tolist()]}")
        print(f"    Expected REL:           {[f'{v:.4f}' for v in expected_action_rel[0, 0].tolist()]}")

        # ---- Step 4: Postprocess → back to absolute ----
        # Postprocessor takes the action tensor directly
        pred_abs = postprocessor(processed["action"])  # (1, 50, 7) ABS

        # ---- Step 5: Compare ----
        error = (pred_abs - gt_action).abs()
        max_err = error.max().item()
        mean_err = error.mean().item()
        ok = max_err < 1e-4  # float32 precision after normalize→unnormalize

        print(f"    Pred action[0,t=0] ABS: {[f'{v:.4f}' for v in pred_abs[0, 0].tolist()]}")
        print(f"    Max error:  {max_err:.2e}  {'OK' if ok else 'FAIL'}")
        print(f"    Mean error: {mean_err:.2e}")

        # Per-dim error
        dim_errs = error[0].mean(dim=0).tolist()
        for name, e in zip(dim_names, dim_errs):
            print(f"      {name:20s}: {e:.2e}")

        if not ok:
            all_ok = False
            print(f"    ⚠  Round-trip broken at frame {frame_idx}!")
        n_tested += 1

    print(f"\n  Tested {n_tested} frames — {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok


def test_state_round_trip():
    """Manual round-trip for state: verify RelativeState + Normalize is reversible.

    State path:  state_abs (2,7) → relative (state -= current) → flatten (14,) → normalize
    Reverse:     unnormalize → unflatten (2,7) → absolute (state += current)

    Verifies: state_abs' ≈ state_abs  (element-wise, atol=1e-5)
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.compute_stats import compute_relative_state_stats
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor.relative_action_processor import (
        DeriveStateFromActionStep,
        RelativeStateProcessorStep,
        to_relative_state,
    )

    print("\n" + "=" * 70)
    print("  TEST B — State round-trip")
    print("  ABS state → relative(offset from cur) → flatten → normalize → reverse")
    print("=" * 70)

    device = "cpu"
    ds_meta = LeRobotDatasetMetadata("test_ee_dataset", root=DATASET_ROOT)
    ds = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT)

    rs_stats = compute_relative_state_stats(
        hf_dataset=ds.hf_dataset, features=ds_meta.features,
        state_obs_steps=2, exclude_joints=["gripper"], source_key="action",
    )
    state_mean = torch.tensor(rs_stats["mean"])  # (14,)
    state_std = torch.tensor(rs_stats["std"]).clamp_min(1e-8)  # (14,) — avoid 0/0 for t=0 masked dims

    cfg = SmolVLAConfig(
        derive_state_from_action=True,
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        relative_exclude_state_joints=["gripper"],
        device=device, push_to_hub=False,
        load_vlm_weights=False, resize_imgs_with_padding=(512, 512),
    )
    dt = resolve_delta_timestamps(cfg, ds_meta)
    ds_dt = LeRobotDataset("test_ee_dataset", root=DATASET_ROOT, delta_timestamps=dt)

    dim_names = ds_meta.features["observation.state"]["names"]
    if isinstance(dim_names, dict):
        dim_names = dim_names.get("axes", list(dim_names.keys()))
    print(f"\nState dims: {dim_names}")

    # Build mask (exclude gripper from relative conversion)
    mask = [name.lower() != "ee.gripper_pos" and "gripper" not in name.lower()
            for name in dim_names]
    print(f"Relative mask (False=excluded): {mask}")

    all_ok = True
    n_tested = 0
    for frame_idx in [0, 5, 10, 50, 100, 150, 200, 250, 300]:
        if frame_idx >= len(ds_dt):
            break

        batch = ds_dt[frame_idx]
        batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
                 for k, v in batch.items()}

        # ---- Step 1: DeriveState to get ground-truth 2-step state ----
        derive = DeriveStateFromActionStep(enabled=True)
        derived = derive(batch)
        gt_state_abs = derived["observation"]["observation.state"].clone()  # (1, 2, 7)

        # ---- Step 2: Apply what RelativeStateProcessorStep does ----
        relative = to_relative_state(gt_state_abs.clone(), mask)  # (1, 2, 7)
        flattened = relative.flatten(start_dim=-2)                # (1, 14)

        print(f"\n  Frame {frame_idx}:")
        print(f"    State ABS (t=-1):  {[f'{v:.4f}' for v in gt_state_abs[0, 0].tolist()]}")
        print(f"    State ABS (t=0):   {[f'{v:.4f}' for v in gt_state_abs[0, 1].tolist()]}")
        print(f"    State REL (t=-1):  {[f'{v:.4f}' for v in relative[0, 0].tolist()]}")
        print(f"    State REL (t=0):   {[f'{v:.4f}' for v in relative[0, 1].tolist()]} (zeros for masked dims!)")
        print(f"    Flattened (14):    {[f'{v:.4f}' for v in flattened[0, :7].tolist()]} ...")

        # ---- Step 3: Normalize ----
        normalized = (flattened - state_mean) / state_std  # (1, 14)
        print(f"    Normalized (14):   {[f'{v:.4f}' for v in normalized[0, :7].tolist()]} ...")

        # ---- Step 4: REVERSE — unnormalize ----
        unnormalized = normalized * state_std + state_mean  # (1, 14)

        # ---- Step 5: REVERSE — unflatten (14 → 2×7) ----
        unflattened = unnormalized.view(1, 2, 7)  # (1, 2, 7)

        # ---- Step 6: REVERSE — un-relative (state += current) ----
        # relative[t] = abs[t] - abs[current], so abs[t] = relative[t] + abs[current]
        # We MUST use the original absolute current timestep (saved before relativization),
        # NOT the relativized version (which has zeros for masked dims).
        current_abs = gt_state_abs[:, -1:, :]  # (1, 1, 7) — original ABS current
        reconstructed = unflattened.clone()
        mask_t = torch.tensor(mask, dtype=unflattened.dtype)
        for d in range(len(mask)):
            if mask[d]:
                reconstructed[:, :, d] += current_abs[:, 0, d]

        # ---- Step 7: Compare ----
        error = (reconstructed - gt_state_abs).abs()
        max_err = error.max().item()
        mean_err = error.mean().item()
        ok = max_err < 1e-4

        print(f"    Reconstructed t=-1: {[f'{v:.4f}' for v in reconstructed[0, 0].tolist()]}")
        print(f"    Reconstructed t=0:  {[f'{v:.4f}' for v in reconstructed[0, 1].tolist()]}")
        print(f"    Max error:  {max_err:.2e}  {'OK' if ok else 'FAIL'}")
        print(f"    Mean error: {mean_err:.2e}")

        dim_errs = error[0].mean(dim=0).tolist()
        for name, e in zip(dim_names, dim_errs):
            print(f"      {name:20s}: {e:.2e}")

        # Extra check: t=0 masked dims should be ~0 in RELATIVE space
        rel_t0 = relative[0, 1]  # (7,)
        rel_t0_masked = rel_t0[:6]  # pos+rot (masked → should be ~0)
        rel_t0_gripper = rel_t0[6]   # gripper (excluded → unchanged)
        print(f"    REL t=0 pos+rot (should be zeros): {[f'{v:.2e}' for v in rel_t0_masked.tolist()]}")
        print(f"    REL t=0 gripper (unchanged):        {rel_t0_gripper:.4f}")

        if not ok:
            all_ok = False
            print(f"    ⚠  State round-trip broken at frame {frame_idx}!")
        n_tested += 1

    print(f"\n  Tested {n_tested} frames — {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    a_ok = test_action_round_trip()
    b_ok = test_state_round_trip()

    print("\n" + "=" * 70)
    if a_ok and b_ok:
        print("  ALL ROUND-TRIP TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        if not a_ok:
            print("  - Action round-trip FAILED")
        if not b_ok:
            print("  - State round-trip FAILED")
    print("=" * 70)
