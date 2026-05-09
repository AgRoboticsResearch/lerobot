## Paper

https://arxiv.org/abs/2506.01844

## Citation

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```

## UMI EE-Pose Relative Action Support

SmolVLA supports UMI-style relative trajectory actions in EE-pose (end-effector) space, following the same pattern as pi0/pi0.5/pi0_fast (PR #2970).

### What this means

- **Relative actions**: Each action in the predicted chunk is an offset from the robot's current state at prediction time, not an absolute target. This avoids error accumulation and doesn't require a global coordinate frame.
- **EE-pose space**: State and actions use Cartesian end-effector coordinates `[x, y, z, wx, wy, wz, gripper]` instead of joint angles. Requires a URDF kinematics model for FK/IK conversion.
- **1 camera**: SmolVLA's 64 visual tokens per frame make single-camera operation efficient (~113 total prefix tokens).

### Usage

**Step 1**: Precompute relative action statistics:

```bash
lerobot-edit-dataset \
    --repo_id your_ee_dataset \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size 50 \
    --operation.relative_exclude_joints "['gripper']"
```

**Step 2a** — Train from scratch:

```bash
lerobot-train \
    --policy.type=smolvla \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.freeze_vision_encoder=True \
    --policy.train_expert_only=True \
    --policy.train_state_proj=True \
    --dataset.repo_id=your_ee_dataset \
    --batch_size=64 \
    --steps=200000
```

**Step 2b** — Fine-tune from `smolvla_base`:

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.freeze_vision_encoder=True \
    --policy.train_expert_only=False \
    --policy.train_state_proj=True \
    --policy.optimizer_lr=2.5e-5 \
    --policy.optimizer_grad_clip_norm=1.0 \
    --dataset.repo_id=your_ee_dataset \
    --batch_size=32 \
    --steps=30000
```

**Important**: When fine-tuning `smolvla_base` for EE-pose, `train_expert_only` must be `False` so the VLM layers can adapt to the new state representation. Use a lower learning rate (2.5e-5) and grad clip (1.0) to avoid destabilizing the unfrozen VLM.

### Architecture notes

In SmolVLA, state flows through the VLM prefix (not the action expert like in pi0). This means the VLM can ground EE-pose state in visual and language features, which is advantageous for spatial reasoning. However, when fine-tuning from `smolvla_base` (which was trained with joint-space state), the VLM layers must be unfrozen to adapt.

### Code changes

Two files were modified to enable this feature:
1. `configuration_smolvla.py` — Added `use_relative_actions`, `relative_exclude_joints`, `action_feature_names` config fields
2. `processor_smolvla.py` — Wired `RelativeActionsProcessorStep` (preprocessing) and `AbsoluteActionsProcessorStep` (postprocessing) into the processor pipeline
