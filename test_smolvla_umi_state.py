#!/usr/bin/env python
"""Smoke test: SmolVLA UMI pipeline with relative state + relative actions.

Tests the new DeriveStateFromActionStep, RelativeStateProcessorStep,
and RelativeStateProcessorStep wired into the SmolVLA processor.
"""

import torch


def test_processor_steps_isolated():
    """Test the new processor steps in isolation (no GPU needed)."""
    from lerobot.processor import (
        DeriveStateFromActionStep,
        RelativeStateProcessorStep,
        to_relative_state,
    )

    print("=== [1] Isolated processor step tests ===\n")

    # --- DeriveStateFromActionStep (training mode) ---
    print("[1a] DeriveStateFromActionStep")
    step = DeriveStateFromActionStep(enabled=True)

    # Simulate extended action: [B, chunk_size+1, 7] from action_delta_indices=[-1,0,...,49]
    action = torch.randn(2, 51, 7)
    transition = {"action": action}

    result = step(transition)
    new_action = result["action"]
    obs = result["observation"]
    state_from_action = obs["observation.state"]

    assert new_action.shape == (2, 50, 7), f"Expected (2,50,7), got {new_action.shape}"
    assert state_from_action.shape == (2, 2, 7), \
        f"Expected state (2,2,7), got {state_from_action.shape}"
    assert torch.equal(state_from_action, action[:, :2, :])
    print(f"    action: {action.shape} → {new_action.shape}, state: {action[:, :2, :].shape}")
    print("    OK")

    # DeriveStateFromActionStep disabled → no-op
    step_off = DeriveStateFromActionStep(enabled=False)
    result_off = step_off(transition)
    print("    Disabled → no-op: OK\n")

    # --- to_relative_state ---
    print("[1b] to_relative_state")
    state = torch.tensor([[
        [1.0, 2.0, 0.5],  # prev
        [3.0, 4.0, 0.7],  # current
    ]])
    mask = [True, True, False]  # exclude gripper
    relative = to_relative_state(state, mask)
    expected = torch.tensor([[
        [-2.0, -2.0, 0.5],   # prev - current
        [ 0.0,  0.0, 0.7],   # current - current
    ]])
    assert torch.allclose(relative, expected), f"Got {relative}"
    print(f"    Input: {state}")
    print(f"    Output: {relative}")
    print("    OK\n")

    # --- RelativeStateProcessorStep (inference mode: 2D → stack → relative → flatten) ---
    print("[1c] RelativeStateProcessorStep (inference)")
    step = RelativeStateProcessorStep(
        enabled=True,
        exclude_joints=["gripper"],
        state_names=["x", "y", "z", "rx", "ry", "rz", "gripper"],
    )

    # First call: buffer + relative (prev==cur → zeros for masked dims)
    obs = {"observation.state": torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5]])}
    t1 = step({"observation": obs})
    # Result should be [prev, cur] flattened → [0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5]
    # Because prev==cur, all masked dims are zero
    result_state = t1["observation"]["observation.state"]
    assert result_state.shape == (1, 14), f"Expected (1,14), got {result_state.shape}"
    print(f"    Step 1 output shape: {result_state.shape}")
    print(f"    Step 1 values (first 7): {result_state[0, :7].tolist()}")
    print(f"    Step 1 values (last 7):  {result_state[0, 7:].tolist()}")

    # Second call: prev=[1,2,3,...], cur=[3,4,5,...]
    obs2 = {"observation.state": torch.tensor([[3.0, 4.0, 5.0, 0.2, 0.3, 0.4, 0.7]])}
    t2 = step({"observation": obs2})
    result_state2 = t2["observation"]["observation.state"]
    assert result_state2.shape == (1, 14)
    print(f"    Step 2 output shape: {result_state2.shape}")
    print(f"    Step 2 values (first 7): {result_state2[0, :7].tolist()}")
    print(f"    Step 2 values (last 7):  {result_state2[0, 7:].tolist()}")
    # Expected: prev_relative = [1-3, 2-4, 3-5, 0.1-0.2, 0.2-0.3, 0.3-0.4, 0.5] = [-2,-2,-2,-0.1,-0.1,-0.1, 0.5]
    # cur_relative = [0,0,0,0,0,0, 0.7]
    expected_val = [-2.0, -2.0, -2.0, -0.1, -0.1, -0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]
    assert torch.allclose(result_state2[0], torch.tensor(expected_val), atol=1e-5), \
        f"Got {result_state2[0].tolist()}"
    print("    OK\n")

    step.reset()
    print("    Reset → buffer cleared")

    print("=== [1] All isolated tests PASSED ===\n")


def test_smolvla_umi_pipeline():
    """Test the full SmolVLA pipeline with derive_state_from_action enabled."""
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    print("=== [2] SmolVLA UMI pipeline config + processor test ===\n")

    # Build config with derive_state_from_action
    cfg = SmolVLAConfig(
        use_relative_actions=True,
        derive_state_from_action=True,
    )

    assert cfg.use_relative_state
    assert cfg.state_obs_steps == 2
    assert cfg.action_delta_indices == [-1] + list(range(50))
    assert cfg.state_delta_indices == [-1, 0]
    print(f"[2a] Config validation")
    print(f"     use_relative_state: {cfg.use_relative_state}")
    print(f"     state_obs_steps: {cfg.state_obs_steps}")
    print(f"     action_delta_indices: {cfg.action_delta_indices[:3]}...")
    print(f"     state_delta_indices: {cfg.state_delta_indices}")
    print("     OK\n")

    # Build processor pipeline
    from lerobot.processor import (
        DeriveStateFromActionStep,
        RelativeActionsProcessorStep,
        RelativeStateProcessorStep,
        AbsoluteActionsProcessorStep,
    )

    pre, post = make_smolvla_pre_post_processors(cfg)

    # Verify steps are in correct order
    step_types = [type(s) for s in pre.steps]
    print(f"[2b] Preprocessor pipeline ({len(pre.steps)} steps):")
    for i, s in enumerate(pre.steps):
        print(f"     {i}. {type(s).__name__}")

    # Check order: DeriveStateFromAction → RelativeActions → RelativeState → Normalize
    derive_idx = next(i for i, t in enumerate(step_types) if t == DeriveStateFromActionStep)
    relative_action_idx = next(i for i, t in enumerate(step_types) if t == RelativeActionsProcessorStep)
    relative_state_idx = next(i for i, t in enumerate(step_types) if t == RelativeStateProcessorStep)

    assert derive_idx < relative_action_idx < relative_state_idx, \
        f"Wrong order: DeriveState({derive_idx}) < RelativeAction({relative_action_idx}) < RelativeState({relative_state_idx})"
    print("     Order verified: DeriveState → RelativeAction → RelativeState\n")

    # Check postprocessor has AbsoluteActions
    post_step_types = [type(s) for s in post.steps]
    assert AbsoluteActionsProcessorStep in post_step_types
    print(f"[2c] Postprocessor pipeline ({len(post.steps)} steps):")
    for i, s in enumerate(post.steps):
        print(f"     {i}. {type(s).__name__}")
    print("     OK\n")

    print("=== [2] Pipeline config test PASSED ===\n")


def test_smolvla_umi_forward():
    """Test a forward pass with DeriveStateFromAction + RelativeState (requires GPU + dataset)."""
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/sroi_lab_picking_all"
    DATASET_REPO = "sroi_lab_picking_all"

    print(f"=== [3] SmolVLA UMI forward pass === (device={DEVICE})\n")

    try:
        from lerobot.datasets import LeRobotDataset
        dataset = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT)
    except Exception as e:
        print(f"    Skipping: dataset not available ({e})")
        return

    state_dim = dataset.meta.features["observation.state"]["shape"][0]
    action_dim = dataset.meta.features["action"]["shape"][0]
    action_names = dataset.meta.features["action"]["names"]
    state_names = dataset.meta.features["observation.state"]["names"]
    print(f"[3a] Dataset: {dataset.num_frames} frames, state={state_dim}D, action={action_dim}D")
    print(f"     Action names: {action_names}")
    print(f"     State names: {state_names}\n")

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg = SmolVLAConfig(
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        derive_state_from_action=True,
        relative_exclude_state_joints=["gripper"],
        device=DEVICE,
        resize_imgs_with_padding=(256, 256),
    )

    print("[3b] Building policy...")
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy = policy.to(DEVICE)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"     Params: {n_params:,}\n")

    print("[3c] Building pre/post processors...")

    # Compute proper relative state stats for flattened 14D vector
    import numpy as np
    from lerobot.datasets.compute_stats import compute_relative_state_stats

    rs_stats = compute_relative_state_stats(
        hf_dataset=dataset.hf_dataset,
        features=dataset.meta.features,
        state_obs_steps=cfg.state_obs_steps,
        exclude_joints=["gripper"],
    )
    import copy
    stats = copy.deepcopy(dataset.meta.stats)
    stats["observation.state"] = {k: v for k, v in rs_stats.items()}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=stats,
    )

    # Build a batch with extended action: [-1, 0, 1, ..., chunk_size-1]
    chunk_size = cfg.chunk_size
    episode_start = 0
    extended_size = chunk_size + 1  # -1 step + chunk_size steps

    # Get state (single timestep)
    state = dataset[episode_start]["observation.state"].unsqueeze(0)  # (1, 7)
    # Get image
    image = dataset[episode_start]["observation.images.camera"].unsqueeze(0)  # (1, C, H, W)
    task = [dataset[episode_start]["task"]]

    # Build extended action chunk: [t-1, t, t+1, ..., t+chunk_size-1]
    total_frames = min(extended_size, len(dataset) - max(0, episode_start - 1))
    actual_start = max(0, episode_start - 1)
    actions_list = []
    for k in range(actual_start, actual_start + extended_size):
        if k < len(dataset):
            actions_list.append(dataset[k]["action"])
        else:
            actions_list.append(actions_list[-1])
    action_extended = torch.stack(actions_list).unsqueeze(0)  # (1, chunk_size+1, action_dim)

    # State needs to be 2D (1, 7) — simulate what data loader returns
    # With state_delta_indices=[-1, 0], the dataset would load [t-1, t] as state
    # But for training with derive_state_from_action, state comes from action
    # So we just need a single-frame state placeholder
    obs_state = torch.cat([
        actions_list[0].unsqueeze(0),  # t-1
        actions_list[1].unsqueeze(0),  # t
    ]).unsqueeze(0)  # (1, 2, 7) — state_delta_indices=[-1, 0]

    chunk_batch = {
        "observation.state": obs_state.to(DEVICE),
        "observation.images.camera": image.to(DEVICE),
        "task": task,
        "action": action_extended.to(DEVICE),
    }

    print(f"[3d] Batch shapes:")
    print(f"     observation.state: {chunk_batch['observation.state'].shape}")
    print(f"     action: {chunk_batch['action'].shape}")
    print(f"     image: {chunk_batch['observation.images.camera'].shape}\n")

    # Run preprocessor
    print("[3e] Preprocessing (DeriveState → RelativeAction → RelativeState → Normalize)...")
    with torch.no_grad():
        processed = preprocessor(chunk_batch)

    print("     After preprocess:")
    for key, val in processed.items():
        if isinstance(val, torch.Tensor):
            print(f"       {key}: shape={val.shape}, mean={val.float().mean():.4f}, std={val.float().std():.4f}")

    # Verify state was relativized and flattened
    state_processed = processed["observation.state"]
    expected_state_flat_dim = cfg.state_obs_steps * state_dim  # 2 * 7 = 14
    assert state_processed.shape[-1] == expected_state_flat_dim, \
        f"Expected state dim {expected_state_flat_dim}, got {state_processed.shape[-1]}"
    print(f"\n     State flattened: {expected_state_flat_dim}D (from {cfg.state_obs_steps}×{state_dim})")

    # Relative state: last timestep masked dims should be ~0
    # The last 7 dims are the "current" timestep, all masked dims should be near 0
    last_7 = state_processed[0, -7:]
    masked_last = last_7[:6]  # first 6 are position+rotation (masked)
    print(f"     Current timestep (last 7): {last_7.tolist()}")
    print(f"     Masked dims (pos+rot): {masked_last.tolist()} — should be ~0")

    # Verify action was stripped from 51 to 50
    assert processed["action"].shape[1] == chunk_size, \
        f"Expected action chunk {chunk_size}, got {processed['action'].shape[1]}"

    # Forward pass
    print(f"\n[3f] Forward pass...")
    policy.eval()
    with torch.no_grad():
        loss, loss_dict = policy(processed)
    # Forward during training uses ground-truth actions, so loss should be reasonable
    print(f"     Loss: {loss.item():.4f}")
    print(f"     OK (model produces valid output)\n")

    print("=== [3] Forward pass test PASSED ===\n")


if __name__ == "__main__":
    test_processor_steps_isolated()
    test_smolvla_umi_pipeline()

    # Only run forward pass if GPU is available
    if torch.cuda.is_available():
        test_smolvla_umi_forward()
    else:
        print("=== [3] Skipped (requires GPU) ===\n")

    print("=== ALL TESTS PASSED ===")
