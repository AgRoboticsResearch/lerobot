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
| 3a | `train_relative_ee.py` | EE v2 | 6D joints + obs.ee | 10D relative EE | 30 | `true` |
| 3b | `train_relative_ee.py` | EE v2 | 6D joints + obs.ee | 10D relative EE | 100 | `true` |

## Checkpoints

All checkpoints stored at: `~/code/lerobot/outputs/train/ee_vs_joints/`

| Mode | Directory | Chunk Size | Status |
|------|-----------|-----------|--------|
| 1a | `joint_obs_joint_action_v2_chunk30/` | 30 | Completed 500K |
| 1b | `joint_obs_joint_action_v2_chunk100/` | 100 | Completed 500K |
| 2a | `ee_obs_ee_action_v2_chunk30/` | 30 | Completed 500K |
| 2b | `ee_obs_ee_action_v2_chunk100/` | 100 | Pending |
| 3a | `joint_obs_ee_action_v2_chunk30/` | 30 | Completed 500K |
| 3b | `joint_obs_ee_action_v2_chunk100/` | 100 | Pending |

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

## Commands
```bash
# ACT Solo script 
python examples/so101/deploy_act_so101.py \
  --robot_id=oscar_so101_follower \
  --pretrained_path ./outputs/train/ee_vs_joints/joint_obs_joint_action_v2/checkpoints/200000/pretrained_model \
  --robot_port /dev/ttyACM0 \
  --cameras "{ wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG} }" \
  --warm_start \
  --n_action_steps 10



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
```