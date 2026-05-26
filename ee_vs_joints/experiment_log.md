# EE vs Joint Action Representation Experiment

**Date**: 2026-04-27

## Goal

Compare three training modes to understand whether **relative EE actions** improve policy performance over standard **joint actions** for SO101 robot manipulation.

## Hypothesis

Relative EE actions (SE(3) transforms) may generalize better than joint-space actions because:
- EE space is task-centric (pick, place, insert) regardless of robot configuration
- Relative transforms are scale-invariant and decouple position from orientation
- Joint-space actions are robot-specific and can have complex nonlinearities

The experiment also tests whether **joint observations** with **EE actions** (hybrid) gets the best of both worlds: proprioceptive richness from joints + task-centric action space.

## Dataset

- **Source**: `/mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged`
  - 54 episodes, 29517 frames, 30 fps
  - `observation.state`: 6D joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
  - `action`: 6D joints
  - `observation.images.wrist`: wrist camera

- **Converted (v2)**: `/mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged_ee_v2`
  - Same 54 episodes, 29517 frames, 30 fps
  - `observation.state`: 6D joints (unchanged from source)
  - `observation.ee`: 7D EE pose at current frame — `[x, y, z, wx, wy, wz, gripper]`
  - `action`: 7D EE pose at next frame — `[x, y, z, wx, wy, wz, gripper]`
  - `observation.images.wrist`: copied directly from source
  - No `action.ee` column

### Conversion command
```bash
python relative_ee_dataset/convert_joint_to_ee_dataset.py \
  /mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged \
  /mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged_ee_v2
```

## Three Training Modes

### Mode 1: Joint Obs + Joint Action (baseline)

Standard `lerobot-train`. Observation and action are both in joint space.

**Mode 1a (chunk30)** — Completed:
```bash
lerobot-train \
  --dataset.repo_id=red_strawberry_picking_260119_merged \
  --dataset.root=/mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged \
  --policy.type=act \
  --output_dir=~/code/lerobot/outputs/train/ee_vs_joints/joint_obs_joint_action_v2_chunk30 \
  --job_name=act_joint_obs_joint_action_chunk30 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=zfff/act_policy \
  --save_freq=50000 \
  --steps=500000 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30
```

**Mode 1b (chunk100)** — Completed:
```bash
lerobot-train \
  --dataset.repo_id=red_strawberry_picking_260119_merged \
  --dataset.root=/mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged \
  --policy.type=act \
  --output_dir=~/code/lerobot/outputs/train/ee_vs_joints/joint_obs_joint_action_v2_chunk100 \
  --job_name=act_joint_obs_joint_action_chunk100 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=zfff/act_policy \
  --policy.push_to_hub=false \
  --save_freq=50000 \
  --steps=500000 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100
```

- **Observation**: 6D joints
- **Action**: 6D joints
- **Status**: chunk30 completed, chunk100 completed

### Mode 2: EE Identity Obs + Relative EE Action

`train_relative_ee.py` with `use_joint_obs=false`. Observation is a 10D identity (current pose = reference frame). Actions are 10D relative SE(3) transforms computed from `observation.ee` (T_current) and `action` (T_future).

**Mode 2a (chunk30)** — Completed:
```bash
python train_relative_ee.py \
  --dataset.repo_id=red_strawberry_picking_260119_merged_ee_v2 \
  --dataset.root=/mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged_ee_v2 \
  --policy.type=act \
  --output_dir=~/code/lerobot/outputs/train/ee_vs_joints/ee_obs_ee_action_v2_chunk30 \
  --job_name=act_ee_obs_ee_action_chunk30 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --save_freq=50000 \
  --steps=500000 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --num_stat_samples=0 \
  --policy.obs_down_sample_steps=1 \
  --policy.obs_state_horizon=1 \
  --policy.use_joint_obs=false
```

**Mode 2b (chunk100)** — Pending:
```bash
python train_relative_ee.py \
  --dataset.repo_id=red_strawberry_picking_260119_merged_ee_v2 \
  --dataset.root=/mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged_ee_v2 \
  --policy.type=act \
  --output_dir=~/code/lerobot/outputs/train/ee_vs_joints/ee_obs_ee_action_v2_chunk100 \
  --job_name=act_ee_obs_ee_action_chunk100 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --save_freq=50000 \
  --steps=500000 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --num_stat_samples=0 \
  --policy.obs_down_sample_steps=1 \
  --policy.obs_state_horizon=1 \
  --policy.use_joint_obs=false
```

- **Observation**: 10D identity `[0,0,0, 1,0,0,0,1,0, gripper]` (current = reference frame)
- **Action**: 10D relative EE `[delta.xyz(3), rot6d(6), gripper(1)]` — `T_rel = T_current⁻¹ @ T_future`
- **Status**: chunk30 completed, chunk100 pending

### Mode 3: Joint Obs + Relative EE Action (hybrid)

`train_relative_ee.py` with `use_joint_obs=true`. Observation is 6D joints (like baseline). Actions are 10D relative EE (like mode 2).

**Mode 3a (chunk30)** — Completed:
```bash
python train_relative_ee.py \
  --dataset.repo_id=red_strawberry_picking_260119_merged_ee_v2 \
  --dataset.root=/mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged_ee_v2 \
  --policy.type=act \
  --output_dir=~/code/lerobot/outputs/train/ee_vs_joints/joint_obs_ee_action_v2_chunk30 \
  --job_name=act_joint_obs_ee_action_chunk30 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --save_freq=50000 \
  --steps=500000 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --num_stat_samples=0 \
  --policy.obs_down_sample_steps=1 \
  --policy.obs_state_horizon=1 \
  --policy.use_joint_obs=true
```

**Mode 3b (chunk100)** — Pending:
```bash
python train_relative_ee.py \
  --dataset.repo_id=red_strawberry_picking_260119_merged_ee_v2 \
  --dataset.root=/mnt/data0/data/sroi/sroi_lerobot/red_strawberry_picking_260119_merged_ee_v2 \
  --policy.type=act \
  --output_dir=~/code/lerobot/outputs/train/ee_vs_joints/joint_obs_ee_action_v2_chunk100 \
  --job_name=act_joint_obs_ee_action_chunk100 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --save_freq=50000 \
  --steps=500000 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --num_stat_samples=0 \
  --policy.obs_down_sample_steps=1 \
  --policy.obs_state_horizon=1 \
  --policy.use_joint_obs=true
```

- **Observation**: 6D joints
- **Action**: 10D relative EE
- **Status**: chunk30 completed, chunk100 pending

## Summary Table

| Mode | Script | Dataset | Observation | Action | Chunk Size | `use_joint_obs` |
|------|--------|---------|-------------|--------|-----------|-----------------|
| 1a | `lerobot-train` | source | 6D joints | 6D joints | 30 | n/a |
| 1b | `lerobot-train` | source | 6D joints | 6D joints | 100 | n/a |
| 2a | `train_relative_ee.py` | EE v2 | 10D identity | 10D relative EE | 30 | `false` |
| 2b | `train_relative_ee.py` | EE v2 | 10D identity | 10D relative EE | 100 | `false` |
| 3a | `train_relative_ee.py` | EE v2 | 15D joints+EE(rot6d) | 10D relative EE | 30 | `true` |
| 3b | `train_relative_ee.py` | EE v2 | 15D joints+EE(rot6d) | 10D relative EE | 100 | `true` |

## Checkpoints

All checkpoints stored at: `~/code/lerobot/outputs/train/ee_vs_joints/`

| Mode | Directory | Chunk Size | Status |
|------|-----------|-----------|--------|
| 1a | `joint_obs_joint_action_v2_chunk30/` | 30 | Completed 500K |
| 1b | `joint_obs_joint_action_v2_chunk100/` | 100 | Completed 500K |
| 2a | `ee_obs_ee_action_v2_chunk30/` | 30 | Retraining (v3: ee_target_frame) |
| 2b | `ee_obs_ee_action_v2_chunk100/` | 100 | Retraining (v3: ee_target_frame) |
| 3a | `joint_obs_ee_action_v2_chunk30/` | 30 | Retraining (v3: 15D obs + ee_target_frame) |
| 3b | `joint_obs_ee_action_v2_chunk100/` | 100 | Retraining (v3: 15D obs + ee_target_frame) |

## Common Parameters

- Policy: ACT
- chunk_size: 30 or 100
- n_action_steps: same as chunk_size
- steps: 500,000
- save_freq: 50,000
- device: cuda
- Robot: SO101
- Task: red strawberry picking

## Relative EE Action Format (10D)

```
T_rel = T_current⁻¹ @ T_future

action = [delta.x, delta.y, delta.z,  rot6d_0..rot6d_5,  gripper]
           translation(3)               rotation(6)        (1)
```

All 30 actions in a chunk are relative to the **same** base pose (chunk start), NOT chained sequentially.

## Files Modified

- `relative_ee_dataset/convert_joint_to_ee_dataset.py` — conversion script (outputs EE-only format: `observation.ee` + `action` as 7D EE)
- `src/lerobot/datasets/relative_ee_dataset.py` — reads `observation.ee` for T_current, `action` for T_future, no `action.ee` logic
- `train_relative_ee.py` — uses `action` directly via delta_timestamps, no format detection/remapping
- `src/lerobot/policies/act/configuration_act.py` — added `use_joint_obs` config field

## Monitoring

```bash
tmux attach -t train    # attach to tmux session
# Ctrl+B 0/1/2 to switch windows
```

## Results

We tested mode 1 using the following command, however, the performance is not good. the robot is moving to the correct direction but not picking the strawberry well, i would assume the reason might be
1. not having a thrid view camera.  
2. the data was record in another time.  
3. the robot calibration changes.  
4. not using async inference. 
5. the act chunk size is too small for joint control (was 100 now 30)
``` bash
python examples/so101/deploy_act_so101.py \
  --robot_id=oscar_so101_follower \
  --pretrained_path ./outputs/train/ee_vs_joints/joint_obs_joint_action_v2/checkpoints/200000/pretrained_model \
  --robot_port /dev/ttyACM0 \
  --cameras "{ wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG} }" \
  --warm_start
```

## Evaluation Results (2026-05-15)

### Mode 3 (Joint Obs + EE Action) — Performs Very Well

Deployed Mode 3 (chunk30, 200K checkpoint) on the real SO101 robot for red strawberry picking. The robot successfully picks the strawberry with smooth, accurate motions. This is the best-performing mode so far, significantly better than Mode 1 (joint obs + joint action) which struggled with grasp precision.

**Why it works well despite `observation.ee` not being consumed by the model:**
- The 15D `observation.state` (6D joints + 9D EE pose) gives the model both proprioceptive and spatial information as a single concatenated vector
- The relative EE action space (10D SE(3) transforms) is more task-centric than joint-space actions, providing smoother trajectories that generalize better across configurations
- Chunk size 30 provides a good balance between prediction horizon and reactivity

## Verified Findings (2026-04-28)

### Bug Confirmed: `observation.ee` is NEVER consumed by the ACT model

The data flow analysis confirms `observation.ee` is only an internal computation artifact:

1. **Dataset** (`relative_ee_dataset.py`): `__getitem__()` reads `hf_dataset[idx]['observation.ee']` internally to compute `T_current` for relative action labels, but **never places** it in the returned item dict
2. **Normalizer** (`normalize_processor.py`): Since `observation.ee` is not in the batch dict, the `key in new_observation` check skips it
3. **ACT model** (`modeling_act.py`): `forward()` only reads `observation.state`, `observation.images.*`, and `observation.environment_state`
4. **Feature lookup** (`configs/policies.py:129`): `robot_state_feature` requires exact key match `"observation.state"` — `"observation.ee"` fails this check even though both are `FeatureType.STATE`

**Impact on Mode 3**: ~~The model sees only 6D joints + images (same as Mode 1)~~ **FIXED**: `observation.state` is now 15D (6D joints + 9D EE pose: xyz + rot6d) when `use_joint_obs=True`. The model receives proprioceptive joints AND spatial EE awareness as a single concatenated vector. No changes to ACT architecture needed — the input MLP adapts its dimension automatically.

### Q1: What EE frame does the robot learn during training?

**Resolved default: `camera_link` via `so101_sroi.urdf`**

- `convert_joint_to_ee_dataset.py`: `EE_LINK_NAME = "camera_link"`
- Default URDF: `urdf/Simulation/SO101/so101_sroi.urdf`
- Dataset metadata records `ee_target_frame="camera_link"` and the URDF path used for conversion.

### Q2: What EE frame does the robot output/follow at deployment?

**Resolved default: `camera_link` via `so101_sroi.urdf`**

- `deploy_relative_ee_so101.py`: default `--urdf_path=urdf/Simulation/SO101/so101_sroi.urdf`
- Default `--deploy_frame=camera_link`
- `RobotKinematics(target_frame_name=args.deploy_frame)` so FK, IK, chunk bases, and visualization use the same deploy frame.

**Default frame consistency:**

| Aspect | Training | Deployment |
|--------|----------|------------|
| Target frame | `camera_link` | `camera_link` |
| URDF | `so101_sroi.urdf` | `so101_sroi.urdf` |

Legacy checkpoints that have `ee_target_frame="gripper_frame_link"` still use the deploy-time frame adapter.

### Q3: Should we unify the representation?

**Yes, implemented for new SO101 relative-EE experiments:**

1. **Frame consistency**: Training and deployment default to `camera_link` with `so101_sroi.urdf`.

2. **State-action representation**: Mode 2 uses 10D identity obs + 10D relative EE action. Mode 3 uses 15D joints+EE obs + 10D relative EE action.



## Commands
```bash
# ACT (Joint space control) solo inference script (recommended) 
python examples/so101/deploy_act_so101.py \
  --robot_id=oscar_so101_follower \
  --pretrained_path ./outputs/train/ee_vs_joints/joint_obs_joint_action_v2_chunk30/checkpoints/last/pretrained_model \
  --robot_port /dev/ttyACM0 \
  --cameras "{ wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG} }" \
  --warm_start \
  --n_action_steps 30

# ACT record-inference script
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=oscar_so101_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/video4,
   width: 640, height: 480, fps: 25, fourcc: MJPG} }" \
    --display_data=true \
    --dataset.repo_id=tmp/eval_throwaway \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick the red strawberry" \
    --dataset.push_to_hub=false \
    --policy.path=./outputs/train/ee_vs_joints/joint_obs_joint_action_v2/checkpoints/200000/pretrained_model

# Relative Policy
# Mode2
python examples/so101_relative_ee/deploy_relative_ee_so101.py   --robot_id=oscar_so101_follower   --pretrained_path ./outputs/train/ee_vs_joints/ee_obs_ee_action_v2/checkpoints/200000/pretrained_model   --robot_port /dev/ttyACM0   --cameras "{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG} }"   --warm_start   --n_action_steps 30   --cameraview   --delay_chunk 0   --display_data   --chunk_base_ideal

# Mode3
python examples/so101_relative_ee/deploy_relative_ee_so101.py   --robot_id=oscar_so101_follower   --pretrained_path ./outputs/train/ee_vs_joints/joint_obs_ee_action_v2_chunk30/checkpoints/200000/pretrained_model   --robot_port /dev/ttyACM0   --cameras "{ wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG} }"   --warm_start   --n_action_steps 30   --cameraview   --delay_chunk 0   --display_data   --chunk_base_ideal --use_joint_obs

```

---

## Mode 4: UMI-Style Processor Pipeline + rot6d (2026-05-26)

**Date**: 2026-05-26

### Goal

Train ACT with UMI-style relative EE actions using the processor pipeline approach. Unlike Modes 2/3 which use `RelativeEEDataset` + `TemporalACTWrapper`, this mode uses a standard `LeRobotDataset` with processor steps that handle all SE(3) math. State is derived from the action column (`derive_state_from_action`), so the dataset doesn't need an `observation.state` column at all.

### Key Differences from Modes 2/3

| Aspect | Mode 2/3 (Pattern A) | Mode 4 (Processor Pipeline) |
|--------|----------------------|------------------------------|
| Dataset class | `RelativeEEDataset` wrapper | Standard `LeRobotDataset` |
| State source | `observation.ee` column | Derived from action column |
| SE(3) math | In dataset `__getitem__()` | In processor steps (saved in checkpoint) |
| observation.state in dataset | Required | Not needed |
| Normalization | MEAN_STD | MIN_MAX (UMI-style) |

### Dataset

- **Repo**: `lerobot_sroi_v2`
- **Root**: `/mnt/data1/data/lerobot/lerobot_sroi_v2`
- 85 episodes, 13050 frames, 30 fps
- Features: `action` (7D EE `[x,y,z,wx,wy,wz,gripper]`) + `observation.images.camera` (video)
- No `observation.state` column — state derived from action during training

### Training Command

```bash
PYTHONPATH=src python train_relative_ee_processor.py \
  --dataset.repo_id=lerobot_sroi_v2 \
  --dataset.root=/mnt/data1/data/lerobot/lerobot_sroi_v2 \
  --policy.type=act \
  --output_dir=/home/zfei/code/lerobots/lerobot/outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30 \
  --job_name=act_umi_processor_ee_action_chunk30 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=zfff/act_policy \
  --policy.push_to_hub=false \
  --save_freq=50000 \
  --steps=500000 \
  --batch_size=8 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true
```

### Pipeline

```
DeriveStateFromAction → RelativeRot6dActions → RelativeRot6dState → Normalizer → ACT Model

Input:  7D aa absolute from dataset action column
State:  20D (2×10D rot6d relative, derived from action[t-1] and action[t])
Output: 10D rot6d relative to model
Post:   10D rot6d relative → 7D aa absolute (via cached state)
```

### Dimension Flow

| Stage | Action | State |
|--------|--------|-------|
| Dataset on disk | 7D aa | (none) |
| After DeriveState | 7D aa (chunk) | 2×7D aa |
| After RelativeRot6d | 10D rot6d relative | 20D rot6d relative |
| Model | 10D rot6d relative | 20D rot6d relative |
| After Postprocessor | 7D aa absolute | — |

### Status

- **Status**: In progress
- **Loss at 13K steps**: 0.114 (converging well)
- **Speed**: ~34ms/step
- **Output dir**: `umi_processor_ee_action_chunk30/`

### Deploy Command

```bash
python examples/so101_relative_ee/deploy_relative_ee_processor_so101.py \
  --pretrained_path ./outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30/checkpoints/last/pretrained_model \
  --robot_id=oscar_so101_follower \
  --robot_port /dev/ttyACM0 \
  --cameras "{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG}}" \
  --deploy_frame camera_link \
  --n_action_steps 30 \
  --warm_start
```

### Files

- `examples/umi_relative_ee/train_relative_ee_processor.py` — training script
- `examples/umi_relative_ee/deploy_relative_ee_processor_so101.py` — deployment script
- `examples/umi_relative_ee/umi_style_ee_processor_pipeline.md` — detailed documentation
- `src/lerobot/processor/relative_action_processor.py` — SE(3) rot6d math + processor steps
- `src/lerobot/processor/relative_action_processor_act.py` — ACT processor factory
- `src/lerobot/datasets/relative_action_stats.py` — 10D rot6d relative stats
