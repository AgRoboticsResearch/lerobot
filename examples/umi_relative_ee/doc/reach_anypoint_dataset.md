# Reach AnyPoint generated motion datasets

Dataset processing and export belong to `/home/wenli/reach-anypoint`. Training
and inference belong to this repository. The adapter has no import or runtime
code dependency on the data engine; only the exported dataset is needed.

## Entry points

The example-local [dataset adapter](../reach_anypoint_dataset.py) recognizes
`annotations/reach_anypoint/manifest.json` under a local dataset root and uses
`ReachAnyPointDataset`. Existing
`train_umi_relative_ee.py`, `train_relative_ee_processor.py`, and
`train_pi05_lora.py` all install this factory for the duration of their training
call, restoring the trainer's original factory afterward, including on failure.
The standard `lerobot-train` command does not recognize these annotations; use
one of the example entrypoints above. No library files under `src/` are changed.
Ordinary SROI datasets without these annotations use the standard factory.
Checkpoint inference
uses the existing saved UMI processors and existing deployment entrypoints.

```bash
cd /home/wenli/lerobot-umi
conda activate py312
PYTHONPATH=src python examples/umi_relative_ee/train_umi_relative_ee.py \
  --dataset.repo_id=local/reachanypoint_30fps_20261013 \
  --dataset.root=/home/wenli/reach-anypoint/generated_datasets/reachanypoint_30fps_20261013 \
  --policy.type=act --policy.use_umi_relative_ee=true \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.device=cuda --output_dir=outputs/train/reachanypoint_act
```

Use the usual policy/checkpoint and optimization arguments for SmolVLA or pi05.
Select a Python 3.12 environment with this checkout's training dependencies.
Non-streaming UMI mode is required. Set `drop_n_last_frames=0` if the policy has
that option: each exported episode contains exactly one observed image.
Validation roots are detected separately and can use either compatible generated
annotations or ordinary SROI data. The existing trainer uses training stats for
normalizing validation inputs.

## Stored contract

The base dataset is LeRobot v3: RGB video, indices, tasks and current raw 7D
camera-link action, one observation per episode. It supplies observation indexing,
video decoding and image transforms. Future planning labels are independent NPZ
files indexed by episode in `annotations/reach_anypoint/manifest.json`.
Manifest schema version is 1 and format is `reach_anypoint_lerobot_motion_chunks`.
The action frame is `observation_camera_link` and FPS matches the base dataset.

Each label contains:

- `action_camera_ref_7d`: `[N,7]` absolute camera-link target poses in a fixed
  reference equal to the observation camera_link, `[xyz, axis-angle, gripper]`.
- `trajectory_time_s`: `[N]`, `arange(N)/fps`, starting at current pose time zero.
- `previous_state_camera_ref_7d`: `[7]`, previous observed camera-link pose in the
  same fixed reference, or an explicitly recorded previous=current assumption.

Positions are meters; rotation vectors are radians; gripper is 0 closed / 1 open.
Current pose must be identity in the local reference. The data engine converts
optical-frame TCP paths into these camera-link targets before export. Do not
reinterpret source XYZ as camera-link XYZ or concatenate paths from neighboring
observations. Source episode/frame/time and label policies remain in the manifest.

## Processor boundary

For H targets the adapter returns raw `[H+1,7]`:

```text
[previous, current, planned_target_1, ...]
  -> UmiDeriveStateFromActionStep
state [2,7] = [previous, current]
action [H,7] = [current, planned_target_1, ...]
  -> existing UMI relative processors
state [20], action [H,10]
```

The chunk starts with current/identity at time zero. Short paths repeat terminal
pose and mark padded targets; long paths use the first H ticks. The adapter
computes action/state statistics from the exact unpadded targets and history.
Training stats require at least two selected observations. Model-facing metadata
is updated in memory to 10D action and 20D state; disk labels are not rewritten.
Split by original source episode/session, not the exported one-frame episodes.
