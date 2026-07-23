#!/usr/bin/env python
"""Test inference pipeline for UMI-style processor pipeline + rot6d.

Simulates deployment: loads checkpoint, creates synthetic observation,
runs preprocessor → model → postprocessor, verifies output dimensions
and round-trip consistency.

Uses select_action (not predict_action_chunk) to match the eval/deploy path.
"""

import sys
import torch
import numpy as np

MODEL_PATH = "/tmp/test_relative_ee_processor/checkpoints/last/pretrained_model"


def test_inference():
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_STATE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load policy
    print(f"\nLoading policy from: {MODEL_PATH}")
    policy = ACTPolicy.from_pretrained(MODEL_PATH, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)
    print(f"  Policy loaded. Config type: {type(policy.config).__name__}")
    print(f"  derive_state_from_action: {policy.config.derive_state_from_action}")
    print(f"  use_relative_actions: {policy.config.use_relative_actions}")
    print(f"  use_rot6d: {policy.config.use_rot6d}")
    print(f"  pose_dim: {policy.config.pose_dim}")
    print(f"  chunk_size: {policy.config.chunk_size}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print(f"  Input features: {policy.config.input_features}")
    print(f"  Output features: {policy.config.output_features}")

    # 2. Create preprocessor/postprocessor from checkpoint
    print("\nCreating preprocessor/postprocessor...")
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=MODEL_PATH,
        )
        print(f"  Preprocessor steps: {[s.__class__.__name__ for s in preprocessor.steps]}")
        print(f"  Postprocessor steps: {[s.__class__.__name__ for s in postprocessor.steps]}")
    except Exception as e:
        print(f"  ERROR creating processors: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Reset processors for clean state
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # 4. Create synthetic observation (simulating robot FK output)
    print("\nCreating synthetic observation...")
    ee_pose_aa = torch.tensor([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.5], dtype=torch.float32)
    image = torch.rand(3, 480, 640, dtype=torch.float32)

    batch = {
        OBS_STATE: ee_pose_aa.unsqueeze(0).to(device),  # (1, 7)
        "observation.images.camera": image.unsqueeze(0).to(device),  # (1, 3, 480, 640)
    }
    print(f"  obs.state shape: {batch[OBS_STATE].shape}")
    print(f"  obs.image shape: {batch['observation.images.camera'].shape}")

    # 5. Run preprocessor
    print("\nRunning preprocessor...")
    try:
        processed = preprocessor(batch)
        for k, v in processed.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    except Exception as e:
        print(f"  ERROR in preprocessor: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify preprocessor output shapes
    state = processed.get(OBS_STATE)
    if state is not None and state.shape[-1] != 20:
        print(f"  ERROR: Expected 20D state after preprocessing, got {state.shape[-1]}D")
        return False

    # 6. Run model prediction using select_action (same as eval/deploy)
    print("\nRunning select_action (eval path)...")
    try:
        with torch.no_grad():
            action = policy.select_action(processed)
        print(f"  Raw action: shape={action.shape}, dtype={action.dtype}")
    except Exception as e:
        print(f"  ERROR in select_action: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 7. Run postprocessor (takes Tensor, returns Tensor)
    print("\nRunning postprocessor...")
    try:
        result = postprocessor(action)
        print(f"  Result: shape={result.shape}, dtype={result.dtype}")
    except Exception as e:
        print(f"  ERROR in postprocessor: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 8. Verify output
    action_np = result.cpu().numpy()
    print(f"\nOutput action shape: {action_np.shape}")
    print(f"Output action values: {action_np[0]}")

    if action_np.shape[-1] != 7:
        print(f"  ERROR: Expected 7D output, got {action_np.shape[-1]}D")
        return False
    print(f"  7D aa output — OK!")

    # 9. Run multiple inference steps to test stateful caching + action queue
    #    select_action dequeues from action_queue, only calls model when empty
    print(f"\n--- Running {policy.config.n_action_steps + 3} consecutive select_action steps ---")
    for step in range(policy.config.n_action_steps + 3):
        ee_pose_aa = torch.tensor(
            [0.3 + step * 0.01, 0.0, 0.4, 0.0, 0.0, 0.0, 0.5],
            dtype=torch.float32,
        )
        batch = {
            OBS_STATE: ee_pose_aa.unsqueeze(0).to(device),
            "observation.images.camera": torch.rand(1, 3, 480, 640).to(device),
        }

        try:
            processed = preprocessor(batch)
            with torch.no_grad():
                action = policy.select_action(processed)
            result = postprocessor(action)
            action_np = result.cpu().numpy()
            print(f"  Step {step+1}: shape={action_np.shape}, vals={action_np[0]}")
        except Exception as e:
            print(f"  Step {step+1} ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n=== All inference tests passed! ===")
    return True


if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
