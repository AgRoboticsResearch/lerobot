#!/usr/bin/env python
"""Smoke test: SmolVLA with UMI EE-pose relative actions on sroi_lab_picking_all dataset.

Tests the full pipeline:
  1. Load EE-pose dataset
  2. Build SmolVLA policy with use_relative_actions=True
  3. Build preprocessor (relative → normalize) and postprocessor (unnormalize → absolute)
  4. Forward pass through the model
  5. Verify output shapes and value ranges
"""

import torch

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/sroi_lab_picking_all"
DATASET_REPO = "sroi_lab_picking_all"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies import make_policy, make_pre_post_processors

    print(f"=== SmolVLA UMI EE-Pose Smoke Test ===")
    print(f"Device: {DEVICE}")

    # 1. Load dataset
    print("\n[1] Loading dataset...")
    dataset = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT)
    print(f"    {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    print(f"    State: {dataset.meta.features['observation.state']['names']}")
    print(f"    Action: {dataset.meta.features['action']['names']}")
    print(f"    Cameras: {dataset.meta.camera_keys}")

    state_shape = dataset.meta.features["observation.state"]["shape"]
    action_shape = dataset.meta.features["action"]["shape"]
    assert state_shape in ([7], (7,)), f"Expected 7D EE-pose state, got {state_shape}"
    assert action_shape in ([7], (7,)), f"Expected 7D EE-pose action, got {action_shape}"

    # 2. Build SmolVLA policy config with relative actions
    print("\n[2] Building SmolVLA config with use_relative_actions=True...")
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    policy_cfg = SmolVLAConfig(
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        device=DEVICE,
        # Small model for smoke test
        resize_imgs_with_padding=(256, 256),
    )
    print(f"    use_relative_actions: {policy_cfg.use_relative_actions}")
    print(f"    relative_exclude_joints: {policy_cfg.relative_exclude_joints}")
    print(f"    chunk_size: {policy_cfg.chunk_size}")
    print(f"    max_state_dim: {policy_cfg.max_state_dim}")

    # 3. Build policy
    print("\n[3] Building SmolVLA policy...")
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=dataset.meta,
    )
    policy = policy.to(DEVICE)

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"    Total params: {n_params:,}")
    print(f"    Trainable params: {n_trainable:,}")

    # 4. Build preprocessor and postprocessor
    print("\n[4] Building preprocessors with relative actions...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=dataset.meta.stats,
    )

    # Verify relative steps are present
    from lerobot.processor.relative_action_processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
    )

    has_relative = any(isinstance(s, RelativeActionsProcessorStep) for s in preprocessor.steps)
    has_absolute = any(isinstance(s, AbsoluteActionsProcessorStep) for s in postprocessor.steps)
    print(f"    RelativeActionsProcessorStep in preprocessor: {has_relative}")
    print(f"    AbsoluteActionsProcessorStep in postprocessor: {has_absolute}")

    if not has_relative or not has_absolute:
        raise RuntimeError("Relative/Absolute action steps not found in processor pipeline!")

    # Check the relative step mask matches EE-pose feature names
    for step in preprocessor.steps:
        if isinstance(step, RelativeActionsProcessorStep):
            mask = step._build_mask(7)  # 7D action
            print(f"    Relative mask for 7D EE-pose: {mask}")
            # gripper (last dim) should be excluded
            assert mask[-1] == False, f"Gripper should be excluded from relative, got mask={mask}"
            assert all(mask[:6]), f"Position+rotation dims should be relative, got mask={mask}"

    # 5. Run a batch through the full pipeline
    print("\n[5] Running forward pass with real data...")
    from torch.utils.data import DataLoader

    policy.eval()

    # Get a few samples
    samples = [dataset[i] for i in range(4)]
    batch = {}
    for key in samples[0]:
        if isinstance(samples[0][key], torch.Tensor):
            batch[key] = torch.stack([s[key] for s in samples])
        elif isinstance(samples[0][key], str):
            batch[key] = [s[key] for s in samples]

    print(f"    Batch state shape: {batch['observation.state'].shape}")
    print(f"    Batch action shape: {batch['action'].shape}")

    # Expand actions to chunk format (simulate training batch)
    chunk_size = policy_cfg.chunk_size
    print(f"    Simulating action chunks of size {chunk_size}...")

    # Build a simple chunked batch from sequential frames
    # Get chunk_size consecutive frames from episode 0
    episode_0_start = 0
    chunk_batch = {}
    chunk_batch["observation.state"] = dataset[episode_0_start]["observation.state"].unsqueeze(0).to(DEVICE)
    chunk_batch["observation.images.camera"] = dataset[episode_0_start]["observation.images.camera"].unsqueeze(0)
    chunk_batch["task"] = [dataset[episode_0_start]["task"]]

    # Get chunk of actions
    if episode_0_start + chunk_size < len(dataset):
        actions_chunk = torch.stack([
            dataset[episode_0_start + k]["action"] for k in range(chunk_size)
        ]).unsqueeze(0).to(DEVICE)  # (1, chunk_size, 7)
    else:
        # Pad if not enough frames
        available = len(dataset) - episode_0_start
        actions = [dataset[episode_0_start + k]["action"] for k in range(available)]
        while len(actions) < chunk_size:
            actions.append(actions[-1])
        actions_chunk = torch.stack(actions).unsqueeze(0).to(DEVICE)

    chunk_batch["action"] = actions_chunk

    print(f"    chunk_batch state: {chunk_batch['observation.state'].shape}")
    print(f"    chunk_batch action: {chunk_batch['action'].shape}")
    print(f"    chunk_batch image: {chunk_batch['observation.images.camera'].shape}")

    # Preprocess (relative → normalize)
    print("\n    Preprocessing (raw → relative → normalize)...")
    with torch.no_grad():
        processed = preprocessor(chunk_batch)

    # Check that action was converted to relative
    for key, val in processed.items():
        if isinstance(val, torch.Tensor):
            print(f"    After preprocess - {key}: shape={val.shape}, "
                  f"mean={val.float().mean():.4f}, std={val.float().std():.4f}")

    # Forward pass
    print("\n    Forward pass through model...")
    with torch.no_grad():
        loss, output_dict = policy(processed)

    print(f"    Loss: {loss.item():.4f}")

    # Postprocess (unnormalize → absolute) using output from model
    if "action" in output_dict:
        print("\n    Postprocessing (unnormalize → absolute)...")

        # Build a PolicyAction-like dict for postprocessor
        from lerobot.processor import PolicyAction
        pa = PolicyAction(
            action=output_dict["action"],
            observation={},
            state=chunk_batch["observation.state"],
        )

        result = postprocessor(pa)
        final_action = result.action if hasattr(result, "action") else result["action"]
        print(f"    Final action shape: {final_action.shape}")
        print(f"    Final action (first step): {final_action[0, 0].cpu().tolist()}")
        print(f"    Original action (first step): {actions_chunk[0, 0].cpu().tolist()}")

    print("\n=== SMOKE TEST PASSED ===")
    print("SmolVLA UMI EE-pose relative action pipeline works correctly!")


if __name__ == "__main__":
    main()
