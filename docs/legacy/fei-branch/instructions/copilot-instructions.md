# LeRobot Workspace Instructions

LeRobot is a PyTorch-based robotics library (Hugging Face) for imitation learning and real-robot deployment.
This workspace is customized for **SO101 deployment with the SROI gripper**, using a **relative EE action representation** and `camera_link` as the end-effector frame.

## Build and Test

```bash
# Install (choose one)
poetry sync --all-extras          # full install
poetry sync --extra dev --extra test  # minimal dev

# Run tests
python -m pytest -sv ./tests
make test-end-to-end DEVICE=cpu   # end-to-end policy tests

# Lint / Format
ruff format .
ruff check --fix .
pre-commit run --all-files
```

> **Before running tests:** `git lfs install && git lfs pull` (test artifacts use git-lfs)

## Architecture

| Path | Purpose |
|------|---------|
| `src/lerobot/policies/` | Policy models (ACT, Diffusion, …) — each has `configuration_*.py`, `modeling_*.py`, `processor_*.py` |
| `src/lerobot/datasets/` | `LeRobotDataset` + `RelativeEEDataset` (UMI-style relative EE) |
| `src/lerobot/processor/` | Modular processor pipeline: normalization, IK, bounds, temporal flatten |
| `src/lerobot/robots/` | Robot control (`Robot` base: `connect`, `send_action`, `get_observation`, `disconnect`) |
| `src/lerobot/model/kinematics.py` | Placo-based FK/IK solver (`RobotKinematics`) |
| `src/lerobot/configs/` | Draccus dataclass configs — CLI override via dotted path: `--policy.lr=1e-4` |
| `examples/so101_relative_ee/` | Debug & deployment scripts for this workspace |
| `urdf/Simulation/SO101/` | Robot URDF models |

## Conventions

### Adding a Policy
1. Create `src/lerobot/policies/<name>/` with all three files.
2. Register in `src/lerobot/__init__.py` (`available_policies`, `available_policies_per_env`).
3. Set `name` class attribute in config to match registry key.

### Adding a Processor
1. Inherit from `ProcessorStep` (or `RobotActionProcessorStep` / `ObservationProcessorStep`).
2. Decorate with `@ProcessorStepRegistry.register("name")`.
3. Import in `src/lerobot/processor/__init__.py`.

### Config-Driven CLI
All training/eval entry points use `@parser.wrap()` (Draccus). Nested overrides:
```bash
lerobot-train --policy.type=act --policy.learning_rate=1e-4 --env.type=aloha
```

## SO101 Relative EE Conventions (this workspace)

**URDF / EE frame:**
- URDF: `urdf/Simulation/SO101/so101_sroi.urdf`
- EE frame: `camera_link` (fixed to the SROI gripper; NOT `gripper_frame_link`)

**10D Pose Representation:** `[dx, dy, dz, rot6d_0..5, gripper]`
- Translation: along gripper frame axes
- Rotation: first two rows of rotation matrix (6D)
- Gripper: normalized [0, 1]

**Action application (CRITICAL — wrong order is a silent bug):**
```python
# At chunk start:
chunk_base_pose = forward_kinematics(current_joints)

# For EVERY action in the chunk:
T_rel   = pose10d_to_mat(action[:9])
T_target = chunk_base_pose @ T_rel   # all relative to SAME base, NOT chained
target_joints = inverse_kinematics(current_joints, T_target)
```
❌ Do NOT accumulate: `T1 @ T2 @ T3`

**Joint names (SO101, 6-DOF):**
`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`

**Temporal observation encoding:**
- Dataset: `obs_state_horizon` temporal frames stacked → `(T, C, H, W)` per camera
- In ACT: flatten to `(B*T, C, H, W)` → encode independently → aggregate to `(B, T*F)`
- This preserves pretrained ResNet weights (3-channel inputs per timestep)

## Key Pitfalls

- **Relative action chunking:** All chunk actions are relative to `chunk_base_pose`, NOT accumulated.
- **IK validation:** Verify FK(IK result) ≈ target pose (pos < 5mm, rot < 5.7°); fall back to previous joint if IK fails.
- **Processor import order:** Register processors via decorator before creating any pipeline; import in `processor/__init__.py`.
- **Camera frame (SROI):** `camera_link` has a 90° Y-axis rotation from `gripper_link` (`rpy="0 1.5708 0"`). Frame conventions matter for relative action math.
- **Dataset metadata:** After dataset modifications recompute stats; do not hand-edit `meta/` files.
- **git-lfs:** Test weights and sample datasets are stored in git-lfs — pull before running tests.
