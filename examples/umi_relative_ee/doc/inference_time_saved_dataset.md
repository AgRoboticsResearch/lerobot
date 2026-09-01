# Inference-time saved dataset (`--save_dataset`)

Both UMI relative-EE Piper deploy entrances can record what happened during a
deployment into a standard **LeRobotDataset**:

| entrance                                                         | mode                          | flag             |
| ---------------------------------------------------------------- | ----------------------------- | ---------------- |
| `examples/umi_relative_ee/deploy_umi_relative_ee_piper.py`       | sync (local inference)        | `--save_dataset` |
| `examples/umi_relative_ee/async_umi_relative_ee_piper_client.py` | async (gRPC remote inference) | `--save_dataset` |

The implementation lives in ONE shared module,
`examples/umi_relative_ee/deploy_dataset_recorder.py` (the same
"shared-module" pattern as `control_logger.py`): both entrances call the same
factory with a different `prefix` (`sync` / `async`), so the dataset schema,
column notation, and episode semantics are **identical across the two deploy
systems** and runs can be diffed directly. The column names deliberately match
the per-tick `--log` ControlLogger fields, so a dataset frame and a CSV row for
the same tick are easy to join (`chunk_id` links them at chunk granularity;
there is no shared per-tick index — the dataset only contains executed ticks).

## Usage

```bash
# sync deploy — bare flag: auto path outputs/deploy_datasets/sync_<timestamp>/
python examples/umi_relative_ee/deploy_umi_relative_ee_piper.py \
    --pretrained_path outputs/.../pretrained_model \
    --cameras "{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
    --save_dataset

# async deploy — custom dataset root (must NOT exist yet; each run is a fresh dataset)
python examples/umi_relative_ee/async_umi_relative_ee_piper_client.py \
    --pretrained_path outputs/.../pretrained_model \
    --cameras "..." --server_address 10.98.19.22:8080 \
    --save_dataset outputs/deploy_datasets/my_run
```

Recommendation: pair it with `--log` — the CSV/NPZ keeps full-fidelity
wall-clock timing of _every_ tick, while the dataset keeps the replayable
robot/policy stream (executed ticks only). Without `--save_dataset` the factory
returns `None` and every hook is a no-op: the deploy loop is untouched.

Ignored (with a warning) in the camera-only dry-run modes (`--dry_run` /
`--dryrun`).

## What gets recorded — and when

- **One frame per EXECUTED control tick**: an action was popped and sent to
  IK/motors while the loop was in the `INFERENCE` state. Ticks where IK
  rejected the action are still recorded (`action.joints` = NaN,
  `action.executed_ok` = False). Paused/holding ticks and async underrun ticks
  (queue empty, nothing popped) are NOT recorded.
- **One episode per engagement**: engaging (`s`, or `.` for single-chunk) opens
  an episode lazily on its first executed tick; the first non-INFERENCE tick
  afterwards (`SPACE`, `q`, `r`, single-chunk completion) or shutdown closes
  it. A `.`-driven session is therefore many short (~`n_action_steps`-frame)
  episodes.
- The dataset directory is created lazily: a run where control is never
  engaged leaves nothing on disk.

## Schema

All float columns are `float32`; images are per-camera `video` features
(h264/mp4, encoded per episode). Auto-added by the writer and also present:
`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`, `task`.

| column                             | shape     | source (control-logger field) | units / notes                                                                                                                                                             |
| ---------------------------------- | --------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observation.state`                | (6,)      | `current_joints_rad`          | arm joints, **radians** — converted here, see units warning below                                                                                                         |
| `observation.ee_pose`              | (7,)      | `current_ee`                  | FK EE pose `[x,y,z,wx,wy,wz,gripper]`: m, axis-angle rad, gripper_norm [0,1]                                                                                              |
| `observation.gripper_position`     | (1,)      | raw gripper read-back         | **DM4310 external: radians** (0=open, ≈−0.91=closed); builtin Piper: mm (0..55). The _normalized_ state is `observation.ee_pose[6]`; the gripper _command_ is `action[6]` |
| `observation.images.<cam>`         | (3,H,W)   | raw camera frame              | the uint8 RGB frame the policy saw (not the viz-annotated copy)                                                                                                           |
| `action`                           | (7,)      | `action_agg`                  | **executed** absolute EE target (post temporal-ensemble in async)                                                                                                         |
| `action.pre_ensemble`              | (7,)      | `action_abs`                  | the popped chunk's pre-blend absolute target (identical to `action` in sync mode)                                                                                         |
| `action.relative`                  | (10,)     | `action_rel`                  | raw **normalized** 10D rot6d relative model output `[t(3), rot6d(6), gripper]` — see note below                                                                           |
| `action.reference_ee`              | (7,)      | `chunk_ref_ee`                | the 7D EE pose that anchored the chunk (T_anchor)                                                                                                                         |
| `action.joints`                    | (6,)      | `ik_joints_rad`               | IK joint command written to the motors, **radians** (NaN if IK failed)                                                                                                    |
| `chunk_id`                         | (1,)      | `chunk_id`                    | stored as float32 for NaN-friendliness (NaN when absent)                                                                                                                  |
| `action.executed_ok`               | (1,) bool | `ik_ok`                       | False on IK-failed / invalid-action ticks                                                                                                                                 |
| `e2e_ms` / `wire_ms` / `server_ms` | (1,)      | same names                    | async-only; NaN in sync runs. `e2e_ms` is set only on the tick a fresh chunk's _first_ action executes (response-weighted, matching the `--log` JSON stats)               |

**Units warning.** `piper.read_joints()` and the IK results flow through the
deploy loop (and the `--log` CSV/NPZ `current_joints_rad` / `ik_joints_rad`
columns — misnomers!) in **degrees**. The dataset converts them to **radians**
(LeRobot convention) in `observation.state` and `action.joints`. When joining
dataset rows against `--log` rows, apply `deg2rad` to the CSV columns.

**About `action.relative`.** This is the model's direct output
(`pred_norm`), i.e. the 10D relative pose in the chunk-start frame _after
min-max normalization_ — identical in sync (in-process `pred_norm`) and async
(server-attached `relative_action`). To interpret it physically, either
unnormalize with the checkpoint's action stats, or reconstruct the relative
transform from the absolute columns: `T_rel = inv(T(action.reference_ee)) @
T(action.pre_ensemble)`.

## Timing impact on the control loop

- Per recorded tick: a handful of numpy casts (µs) + one queue put per camera.
  PNG frame writes run on 4 background `AsyncImageWriter` threads — the same
  path the standard `lerobot-record` loop uses at ≥30 fps.
- **Blocking stall at disengage**: closing an episode runs `save_episode()`,
  which waits for the PNG writer and encodes the episode's mp4s (PyAV/SVT — no
  ffmpeg binary needed). A single chunk (~30 frames) is well under a second; a
  long ~30 s engagement (~900 frames × cameras) can take a few seconds. This
  happens while the arm is PAUSED/holding, the duration is logged
  (`Saved dataset episode N (F frames, video encode X.XXs)`), and it shows up
  as one large `tick_dt_ms` in the companion `--log`.
- At shutdown, `close()` saves any trailing episode and finalizes (parquet
  footers). Press Ctrl+C **once** — a second interrupt during the trailing
  encode can leave the dataset unfinalized.
- If the recorder itself fails at any point (disk full, shape mismatch, …), it
  logs once, disables itself, and the deploy loop continues unrecorded.

## On-disk layout and reloading

Standard v3.0 LeRobotDataset layout under the dataset root:

```
meta/info.json  meta/stats.json  meta/tasks.parquet  meta/episodes/...
data/chunk-XXX/file-XXX.parquet          # all columns above
videos/observation.images.<cam>/chunk-XXX/file-XXX.mp4
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="sync_20260901_120000",          # = dir name
                    root="outputs/deploy_datasets/sync_20260901_120000")
frame = ds[0]          # dict of torch tensors; videos decode to float32 CHW in [0,1]
```

The raw mp4s under `videos/` can also be previewed directly with any player.

## Limitations

- **Timestamps are synthetic** (`frame_index / fps`): async underrun gaps and
  any slower-than-nominal ticks are invisible in the dataset. Wall-clock truth
  lives in the companion `--log`.
- **h264 is lossy**: don't expect pixel-exact frame round-trips (fine for
  analysis/training, not for checksum-style comparisons).
- **No resume**: each run creates a fresh dataset (`--save_dataset PATH` must
  not exist). Use `LeRobotDataset.resume` yourself if you ever need to append.
- Dry-run modes are not recorded; the 20D relative policy _input_ state is not
  stored (it is derivable from consecutive `observation.ee_pose` rows plus the
  preprocessor's buffering rules).

## Smoke test

`examples/umi_relative_ee/test_deploy_dataset_recorder.py` drives the shared
recorder with fake cameras/diag dicts through two episodes (including an
IK-failed tick), reloads the dataset, and asserts the schema, unit conversion,
NaN semantics, latency columns, and lazy-creation behavior:

```bash
uv run python examples/umi_relative_ee/test_deploy_dataset_recorder.py
```
