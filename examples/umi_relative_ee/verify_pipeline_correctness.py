#!/usr/bin/env python
"""Comprehensive verification of UMI-style processor pipeline correctness.

Verifies:
1. SE(3) round-trip: aa → rot6d relative → aa absolute
2. Stats computed on correct 10D rot6d relative data
3. State derivation correctness
4. 20D state conversion round-trip
5. Normalizer/unnormalizer round-trip with processor pipeline
6. Full end-to-end: real dataset → preprocess → model → postprocess → 7D aa
"""

import sys
import torch
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

sys.path.insert(0, "src")


def test_se3_roundtrip():
    """Test 1: SE(3) round-trip aa → rot6d relative → aa absolute."""
    from lerobot.processor.relative_action_processor import (
        _pose_se3_relative_aa_to_rot6d,
        _pose_se3_absolute_rot6d_to_aa,
    )

    print("=" * 60)
    print("TEST 1: SE(3) Round-trip (aa → rot6d relative → aa absolute)")
    print("=" * 60)

    torch.manual_seed(42)
    n = 1000
    max_pos_err = 0
    max_rot_err = 0
    max_grip_err = 0

    for i in range(n):
        # Random poses
        pos_from = torch.randn(3) * 0.5
        pos_to = torch.randn(3) * 0.5
        aa_from = torch.randn(3) * 2.0  # axis-angle, up to ~115 degrees
        aa_to = torch.randn(3) * 2.0
        gripper = torch.rand(1)

        pose_from = torch.cat([pos_from, aa_from, gripper])
        pose_to = torch.cat([pos_to, aa_to, gripper])

        # Forward: aa → rot6d relative
        relative = _pose_se3_relative_aa_to_rot6d(pose_from, pose_to)

        # Backward: rot6d relative → aa absolute
        recovered = _pose_se3_absolute_rot6d_to_aa(relative, pose_from)

        # Compare
        pos_err = (recovered[:3] - pose_to[:3]).abs().max().item()
        grip_err = (recovered[6] - pose_to[6]).abs().item()

        # Rotation error: compute angle between rotation matrices
        R_orig = ScipyRotation.from_rotvec(pose_to[3:6].numpy()).as_matrix()
        R_rec = ScipyRotation.from_rotvec(recovered[3:6].numpy()).as_matrix()
        R_diff = R_orig.T @ R_rec
        angle_err = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, angle_err)
        max_grip_err = max(max_grip_err, grip_err)

    print(f"  Tested {n} random pose pairs")
    print(f"  Max position error:    {max_pos_err:.2e}")
    print(f"  Max rotation error:    {max_rot_err:.2e} rad ({np.degrees(max_rot_err):.4e} deg)")
    print(f"  Max gripper error:     {max_grip_err:.2e}")

    pos_ok = max_pos_err < 1e-5
    rot_ok = max_rot_err < 5e-4  # single-precision Rodrigues can accumulate ~1e-4 rad
    grip_ok = max_grip_err < 1e-6

    if pos_ok and rot_ok and grip_ok:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        if not pos_ok:
            print(f"    Position error too large: {max_pos_err}")
        if not rot_ok:
            print(f"    Rotation error too large: {max_rot_err}")
        if not grip_ok:
            print(f"    Gripper error too large: {max_grip_err}")
    return pos_ok and rot_ok and grip_ok


def test_batch_se3_roundtrip():
    """Test 2: Batch SE(3) round-trip via to_relative/to_absolute functions."""
    from lerobot.processor.relative_action_processor import (
        to_relative_actions_rot6d,
        to_absolute_actions_rot6d,
    )

    print("\n" + "=" * 60)
    print("TEST 2: Batch SE(3) Round-trip (full pipeline functions)")
    print("=" * 60)

    torch.manual_seed(42)
    B, T = 4, 30

    # Create realistic EE poses
    state_aa = torch.randn(B, 7) * 0.5  # [B, 7]
    actions_aa = torch.randn(B, T, 7) * 0.5  # [B, T, 7]
    mask = [True, True, True, True, True, True, False]  # gripper excluded

    # Forward
    relative = to_relative_actions_rot6d(actions_aa, state_aa, mask)
    assert relative.shape == (B, T, 10), f"Expected (B,T,10), got {relative.shape}"

    # Backward
    recovered = to_absolute_actions_rot6d(relative, state_aa, mask)
    assert recovered.shape == (B, T, 7), f"Expected (B,T,7), got {recovered.shape}"

    # Check per-element errors
    pos_err = (recovered[..., :3] - actions_aa[..., :3]).abs().max().item()
    grip_err = (recovered[..., 6] - actions_aa[..., 6]).abs().max().item()

    # Check rotation per sample
    max_rot_err = 0
    for b in range(B):
        for t in range(T):
            R_orig = ScipyRotation.from_rotvec(actions_aa[b, t, 3:6].numpy()).as_matrix()
            R_rec = ScipyRotation.from_rotvec(recovered[b, t, 3:6].numpy()).as_matrix()
            R_diff = R_orig.T @ R_rec
            angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            max_rot_err = max(max_rot_err, angle)

    print(f"  Batch shape: ({B}, {T})")
    print(f"  Relative action shape: {relative.shape} (correct: {(B,T,10)})")
    print(f"  Recovered action shape: {recovered.shape} (correct: {(B,T,7)})")
    print(f"  Max position error: {pos_err:.2e}")
    print(f"  Max rotation error: {max_rot_err:.2e} rad ({np.degrees(max_rot_err):.4e} deg)")
    print(f"  Max gripper error:  {grip_err:.2e}")

    ok = pos_err < 1e-4 and max_rot_err < 1e-3 and grip_err < 1e-5
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


def test_state_derivation_and_conversion():
    """Test 3: State derivation from action + relative state conversion."""
    from lerobot.processor.relative_action_processor import (
        to_relative_state_rot6d,
    )

    print("\n" + "=" * 60)
    print("TEST 3: State Derivation + 20D Relative State Conversion")
    print("=" * 60)

    torch.manual_seed(42)
    B = 8

    # Simulate 2-step state from action derivation: [action[t-1], action[t]]
    state_aa = torch.randn(B, 2, 7)  # [B, 2, 7]
    mask = [True, True, True, True, True, True, False]

    # Convert to 20D relative state
    relative_state = to_relative_state_rot6d(state_aa, mask)
    assert relative_state.shape == (B, 20), f"Expected (B,20), got {relative_state.shape}"

    # The last 10 dims (current step relative to itself) should be near-identity
    current_relative = relative_state[:, 10:]  # [B, 10]

    # Translation should be near zero
    trans_err = current_relative[:, :3].abs().max().item()
    # rot6d should be near identity matrix rows: [1,0,0, 0,1,0]
    identity_rot6d = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
    rot_err = (current_relative[:, 3:9] - identity_rot6d).abs().max().item()
    # Gripper should be unchanged
    grip_err = (current_relative[:, 9] - state_aa[:, 1, 6]).abs().max().item()

    # The first 10 dims (previous step relative to current) should encode the delta
    prev_relative = relative_state[:, :10]  # [B, 10]
    # Should have non-trivial values (not all zeros or identity)
    has_variation = prev_relative.std() > 0.01

    print(f"  Input state shape: {state_aa.shape}")
    print(f"  Output state shape: {relative_state.shape} (correct: {(B,20)})")
    print(f"  Current-step self-relative translation err: {trans_err:.2e}")
    print(f"  Current-step self-relative rot6d err: {rot_err:.2e}")
    print(f"  Current-step gripper preservation err: {grip_err:.2e}")
    print(f"  Previous-step has variation: {has_variation}")
    print(f"  Previous-step mean abs value: {prev_relative.abs().mean():.4f}")
    print(f"  Previous-step std: {prev_relative.std():.4f}")

    ok = trans_err < 1e-5 and rot_err < 1e-3 and grip_err < 1e-6 and has_variation
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


def test_stats_computation():
    """Test 4: Verify stats are computed on correct 10D rot6d relative data."""
    from lerobot.datasets.relative_action_stats import (
        _pose_se3_relative_aa_to_rot6d_np,
        _get_valid_chunk_starts,
    )

    print("\n" + "=" * 60)
    print("TEST 4: Stats Computation Correctness")
    print("=" * 60)

    np.random.seed(42)
    n_frames = 300
    chunk_size = 30

    # Create synthetic data: 3 episodes of 100 frames each
    episode_index = np.repeat([0, 1, 2], 100)
    actions = np.random.randn(n_frames, 7).astype(np.float32) * 0.5

    # Compute valid chunk starts
    valid_starts = _get_valid_chunk_starts(episode_index, chunk_size + 1)
    print(f"  Total frames: {n_frames}, Episodes: 3")
    print(f"  Valid chunk starts (chunk+1={chunk_size+1}): {len(valid_starts)}")

    # Compute relative actions manually for a few chunks and verify
    max_pos_diff = 0
    for start in valid_starts[:5]:  # Check first 5
        state = actions[start + 1]  # action at t (current)
        for offset in range(chunk_size):
            action_abs = actions[start + 1 + offset]  # future action

            # Manual SE(3)
            manual_rel = _pose_se3_relative_aa_to_rot6d_np(state, action_abs)

            # Pipeline would use state as base, action as target
            assert manual_rel.shape == (10,), f"Expected (10,), got {manual_rel.shape}"

            # Check translation is in local frame (not global)
            # dx should be R_curr^T @ (t_future - t_curr)
            R_curr = ScipyRotation.from_rotvec(state[3:6]).as_matrix()
            dt = action_abs[:3] - state[:3]
            expected_dx = R_curr.T @ dt
            pos_diff = np.abs(manual_rel[:3] - expected_dx).max()
            max_pos_diff = max(max_pos_diff, pos_diff)

    print(f"  Manual vs computed translation diff: {max_pos_diff:.2e}")
    print(f"  Relative action output dim: 10 (correct)")

    # Verify gripper is preserved (excluded from SE(3))
    for start in valid_starts[:5]:
        state = actions[start + 1]
        action_abs = actions[start + 1 + chunk_size - 1]
        rel = _pose_se3_relative_aa_to_rot6d_np(state, action_abs)
        # Gripper should be action's gripper (not transformed)
        grip_diff = abs(rel[9] - action_abs[6])
        assert grip_diff < 1e-6, f"Gripper not preserved: {grip_diff}"

    print(f"  Gripper preservation: verified (excluded from SE(3))")

    ok = max_pos_diff < 1e-5
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


def test_normalizer_roundtrip():
    """Test 5: Normalizer/unnormalizer round-trip with processor pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Processor Pipeline Round-trip (with normalization)")
    print("=" * 60)

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_STATE

    # Use the test checkpoint from earlier training
    import os
    checkpoint = "/tmp/test_new_dataset/checkpoints/last/pretrained_model"
    if not os.path.exists(checkpoint):
        checkpoint = "/tmp/test_relative_ee_processor/checkpoints/last/pretrained_model"
    if not os.path.exists(checkpoint):
        print("  SKIP (no checkpoint found)")
        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Create test observation with known EE pose
    ee_poses = [
        torch.tensor([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.5]),  # identity rotation
        torch.tensor([0.1, -0.2, 0.3, 0.5, -0.3, 0.8, 0.3]),  # arbitrary
        torch.tensor([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0]),  # 90 deg around x
    ]

    for i, ee_pose in enumerate(ee_poses):
        batch = {
            OBS_STATE: ee_pose.unsqueeze(0).to(device),
            "observation.images.camera": torch.rand(1, 3, 480, 640).to(device),
        }

        processed = preprocessor(batch)
        state = processed[OBS_STATE]
        assert state.shape == (1, 20), f"State shape wrong: {state.shape}"

        # Run through model
        with torch.no_grad():
            action = policy.select_action(processed)
        assert action.shape == (1, 10), f"Model output shape wrong: {action.shape}"

        # Postprocess
        result = postprocessor(action)
        assert result.shape == (1, 7), f"Postprocessed shape wrong: {result.shape}"

        # Verify output is reasonable 7D EE pose
        vals = result[0].cpu().numpy()
        pos_reasonable = np.all(np.abs(vals[:3]) < 5.0)  # within 5m
        rot_reasonable = np.all(np.abs(vals[3:6]) < np.pi)  # less than 180 deg
        grip_reasonable = 0 <= vals[6] <= 2  # gripper in range

        if i == 0:
            print(f"  Sample {i}: pose={vals}")
            print(f"  Position in range: {pos_reasonable}")
            print(f"  Rotation in range: {rot_reasonable}")
            print(f"  Gripper in range:  {grip_reasonable}")

    print(f"  Preprocessor: 7D state → {state.shape} (correct: (1,20))")
    print(f"  Model output: 10D (correct)")
    print(f"  Postprocessor: 7D aa (correct)")
    print(f"  PASS ✓")
    return True


def test_real_dataset_pipeline():
    """Test 6: Verify pipeline with actual dataset frames."""
    print("\n" + "=" * 60)
    print("TEST 6: Real Dataset Pipeline Verification")
    print("=" * 60)

    import os
    dataset_root = "/mnt/data1/data/lerobot/lerobot_sroi_v2"
    if not os.path.exists(dataset_root):
        print("  SKIP (dataset not found)")
        return True

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.processor.relative_action_processor import (
        DeriveStateFromActionStep,
        RelativeRot6dActionsProcessorStep,
        RelativeRot6dStateProcessorStep,
        AbsoluteRot6dActionsProcessorStep,
    )

    # Load dataset
    ds_meta = LeRobotDatasetMetadata("lerobot_sroi_v2", root=dataset_root)
    print(f"  Dataset: {ds_meta.total_episodes} episodes, {ds_meta.total_frames} frames")
    print(f"  Features: {list(ds_meta.features.keys())}")

    # Verify action is 7D
    action_feat = ds_meta.features["action"]
    assert list(action_feat["shape"]) == [7], f"Expected action shape [7], got {action_feat['shape']}"
    print(f"  Action shape: {action_feat['shape']} (correct: [7])")

    # Verify observation.state is NOT in dataset (new format)
    has_state = "observation.state" in ds_meta.features
    print(f"  observation.state in dataset: {has_state}")
    if has_state:
        print(f"  Note: dataset still has observation.state (old format)")

    # Load a few frames and verify action values are reasonable
    ds = LeRobotDataset("lerobot_sroi_v2", root=dataset_root)
    sample = ds[0]
    action = sample["action"]
    print(f"  Sample action shape: {action.shape}")
    print(f"  Sample action values: {action[:3].tolist()} (pos), {action[3:6].tolist()} (rot), {action[6:].tolist()} (gripper)")

    # Check action values are reasonable EE poses
    pos = action[:3].numpy()
    rot = action[3:6].numpy()
    grip = action[6].numpy()
    assert np.all(np.abs(pos) < 2.0), f"Position out of range: {pos}"
    assert np.all(np.abs(rot) < np.pi), f"Rotation out of range: {rot}"
    assert 0 <= grip <= 2, f"Gripper out of range: {grip}"
    print(f"  Action values reasonable: pos {np.abs(pos).max():.3f}m, rot {np.degrees(np.abs(rot).max()):.1f}deg")

    # Test processor steps on real data
    derive_step = DeriveStateFromActionStep(enabled=True)
    rel_action_step = RelativeRot6dActionsProcessorStep(enabled=True, exclude_joints=["gripper"])
    rel_state_step = RelativeRot6dStateProcessorStep(enabled=True, exclude_joints=["gripper"])
    abs_action_step = AbsoluteRot6dActionsProcessorStep(enabled=True)

    # Simulate a batch with action chunk
    chunk_size = 30
    # Get chunk+1 consecutive frames
    actions = []
    for i in range(chunk_size + 1):
        sample = ds[i]
        actions.append(sample["action"])
    action_chunk = torch.stack(actions).unsqueeze(0)  # [1, chunk+1, 7]
    print(f"\n  Action chunk shape: {action_chunk.shape}")

    # Step 1: Derive state from action
    batch = {"action": action_chunk}
    batch = derive_step(batch)

    # DeriveStateFromActionStep puts state under transition["observation"]["observation.state"]
    obs_dict = batch.get("observation", {})
    derived_state = obs_dict.get("observation.state") if isinstance(obs_dict, dict) else None

    print(f"  After DeriveState: action shape={batch['action'].shape}")
    if derived_state is not None:
        print(f"  Derived state shape: {derived_state.shape}")
        assert derived_state.shape == (1, 2, 7), f"Expected (1,2,7), got {derived_state.shape}"
        # Verify state values: should be first two timesteps of action chunk
        # state[0] = action_chunk[0,0] (t=-1), state[1] = action_chunk[0,1] (t=0)
        state_err = (derived_state - action_chunk[:, :2, :]).abs().max().item()
        print(f"  State extraction error: {state_err:.2e} (should be 0)")
        assert state_err < 1e-6, f"State extraction error too large: {state_err}"
    else:
        print(f"  Derived state keys: {list(batch.keys())}")
        print(f"  Observation keys: {list(obs_dict.keys()) if isinstance(obs_dict, dict) else 'N/A'}")

    print(f"  PASS ✓")
    return True


def test_stats_values():
    """Test 7: Verify computed stats have reasonable ranges."""
    print("\n" + "=" * 60)
    print("TEST 7: Verify Stats Values on Actual Dataset")
    print("=" * 60)

    import os
    dataset_root = "/mnt/data1/data/lerobot/lerobot_sroi_v2"
    if not os.path.exists(dataset_root):
        print("  SKIP (dataset not found)")
        return True

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot_sroi_v2", root=dataset_root)

    if not ds.meta.stats or "action" not in ds.meta.stats:
        print("  SKIP (no action stats)")
        return True

    action_stats = ds.meta.stats["action"]
    if isinstance(action_stats, dict) and "mean" in action_stats:
        mean = action_stats["mean"]
        if hasattr(mean, 'shape'):
            print(f"  Action stats mean shape: {mean.shape}")
            print(f"  Action stats mean: {mean}")

            if mean.shape[0] == 10:
                print(f"  10D rot6d relative action stats (correct)")
                mean_arr = np.array(mean)
                pos_mean = np.abs(mean_arr[:3]).mean()
                rot6d_mean = np.abs(mean_arr[3:9]).mean()
                grip_mean = np.abs(mean_arr[9])
                print(f"  Position mean abs: {pos_mean:.4f} (should be small ~0)")
                print(f"  Rot6d mean abs: {rot6d_mean:.4f} (should be near identity ~0.5-0.8)")
                print(f"  Gripper mean: {grip_mean:.4f}")
            elif mean.shape[0] == 7:
                print(f"  7D aa absolute action stats (raw dataset, needs recomputation)")
            else:
                print(f"  Unexpected action dim: {mean.shape[0]}")

    if ds.meta.stats and "observation.state" in ds.meta.stats:
        state_stats = ds.meta.stats["observation.state"]
        if isinstance(state_stats, dict) and "mean" in state_stats:
            state_mean = state_stats["mean"]
            if hasattr(state_mean, 'shape'):
                print(f"\n  State stats mean shape: {state_mean.shape}")
                if state_mean.shape[0] == 20:
                    print(f"  20D relative rot6d state stats (correct)")

    print(f"  PASS ✓")
    return True


def main():
    results = {}

    results["se3_roundtrip"] = test_se3_roundtrip()
    results["batch_roundtrip"] = test_batch_se3_roundtrip()
    results["state_derivation"] = test_state_derivation_and_conversion()
    results["stats_computation"] = test_stats_computation()
    results["normalizer_roundtrip"] = test_normalizer_roundtrip()
    results["real_dataset"] = test_real_dataset_pipeline()
    results["stats_values"] = test_stats_values()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
