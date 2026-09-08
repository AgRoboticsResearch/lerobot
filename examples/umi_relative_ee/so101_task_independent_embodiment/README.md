# Task-independent embodiment learning: ACT-R18 + SO-101

SO-101 counterpart of `../task_independent_embodiment` (Piper). Same protocol,
same frozen ACT-R18 and datasets, with the robot swapped for the 5-DOF SO-101
(Feetech STS3215 bus servos) using the `so101_sim` MuJoCo package in the
piper_mujoco repo, built from the in-repo `so101_sroi.urdf` (wrist
`camera_link`, matching the deploy-frame convention; FK verified against placo
to machine precision).

Robot-specific differences from the Piper version:

- 5 arm joints (`shoulder_pan … wrist_roll`), not 6; controller input/output
  widths follow. With only 5 DOF, many UMI orientation constraints are
  infeasible — the feasibility-coverage metric is a first-class result here.
- Workspace box: x [0.02, 0.32], y [-0.20, 0.20], z [0.0, 0.30] metres.
- Servo goal rate limit 180 deg/s (STS3215-class); actuator kp=17.8, force
  limit 3.35 Nm, damping 0.60 / frictionloss 0.052 / armature 0.028 from the
  SO-ARM100 MuJoCo parameters.
- Gripper is a normalized 1-DOF jaw (0=open, 1=closed) mapped onto the URDF
  gripper joint range; scored as `gripper_rmse_norm`.
- Default artifact root: `/mnt/data1/projects/lerobot-embodiment-exp-so101`.
  Everything else (timing, relabeling, models, budgets, evaluation, statistics)
  is identical to the Piper experiment.

This experiment learns an image-free inverse controller from autonomous SO-101
simulation and composes it with a frozen ACT-R18 task policy. The training
target is the **issued joint command**, conditioned on the **achieved future
EE trajectory** and strictly causal proprioceptive/command history.

The initial study uses the existing 1,459-episode strawberry corpus and all
100 separate validation episodes. It tests trajectory execution on one task
and one robot. It does not establish cross-task transfer, manipulation success,
physical backlash compensation, or benefits from external motion tracking.


## Run

From the repository root:

```bash
bash examples/umi_relative_ee/so101_task_independent_embodiment/setup_env.sh

# Complete software smoke: both dynamics, all methods, one iteration, real ACT queries.
bash examples/umi_relative_ee/so101_task_independent_embodiment/run.sh all \
  --preset smoke --device cpu --root /tmp/umi-embodiment-smoke

# Full experiment: one GPU process at a time, seeds 1000/2000/3000.
bash examples/umi_relative_ee/so101_task_independent_embodiment/run.sh all \
  --preset full --device cuda --root /mnt/data1/projects/lerobot-embodiment-exp
```

`all --resume` continues an interrupted experiment with identical inputs. Keep
`--preset smoke` when resuming smoke. Existing artifacts are never silently
overwritten. Source/configuration/model changes require a new experiment root.
Run only one driver per root; concurrent writers are not supported.

The isolated environment defaults to
`/mnt/data1/projects/lerobot-embodiment-env`. Set `UMI_EMB_ENV` to override it,
`UMI_EMB_BASE_PYTHON` to select the existing LeRobot interpreter, and
`PIPER_SIM_SOURCE` to select the Piper simulator source checkout. The setup
records resolved packages in the environment. It does not upgrade the base
LeRobot environment or modify the simulator source.

The full run requires a working CUDA driver. CPU is supported for smoke tests.
Collection is synchronous, headless and independent of a running gRPC server.
There is no hardware backend or physical robot connection in this experiment.

## Fixed data and task policy

- Training: `/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1459_occlusion`.
- Final evaluation: `/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation`.
- Frozen checkpoint:
  `/mnt/data1/projects/lerobot-arch-exp/lerobot-arch-exp/train/act_r18_l1_seed1000_100000steps/checkpoints/100000/pretrained_model`.
- Query manifest: `../act_flow_ablation/repro/query_frames_h10_seed1000.json`.
  Its 500 fixed query locations are reused with the full 30-step ACT chunk;
  the manifest's historical `eval_horizon=10` does not truncate this experiment.

The checkpoint is deterministic ACT-L1, ResNet-18, 100k steps. It consumes an
image plus two recent relative poses and predicts 30 relative actions. Its
processors and normalization are loaded from the checkpoint. No ACT weights
are trained here. ACT caching uses recorded pose history; it is not a claim
that the task policy is image-only.

`init` creates `config.json`. To override data paths or experiment budgets,
edit that file before running any subsequent stage. Preflight verifies the
expected corpus counts, checkpoint family, training corpus, model/processor
hashes, simulator dependencies, and FK/site agreement. A full run requires
the specified 1,459/100 corpus, rather than silently accepting a small subset.

## Experimental protocol

### Time, frames, and observations

The packaged sroiv2 model runs at 500 Hz. Joint commands run at 50 Hz; EE
trajectory samples retain 30 Hz timing. Translation uses linear interpolation
and rotations use SLERP. All geometry refers to `camera_link`, the existing
Piper deployment frame. Dataset poses are transplanted onto the settled robot
pose once per chunk; tracking errors never move that anchor.

Records use metres, radians, seconds, and row-based rot6d (the first two rows
of a rotation matrix). Degrees appear only at the Piper/placo interface.
There are 30 future pose tokens at `t + arange(30)/30`, eleven measured joint
position/velocity samples through `t`, and ten issued commands strictly before
`t`. The model predicts five absolute six-joint position targets; only the
first is executed, then the controller replans after 20 ms.

The direct controller needs no runtime IK. The adapter anchors a trajectory
using current measured pose; feasibility analysis and residual-label creation
are offline operations. No future state, delayed command delivery, rate-limiter
target, simulator parameters, images, or task labels enter the inverse model.

### Collection and relabeling

Source episodes from the training corpus are split 90/10 with seed 1000.
Augmentations and every recollection generation retain that split. All 100
validation episodes remain reserved for final evaluation; the embodiment
development set comes from training-corpus episodes only.

Sample approximately one-second motion chunks. Scale translation and rotation
amplitude independently in `[0.8, 1.2]`, and speed in `[0.8, 1.2]`. Sample start
configurations within model joint limits with a 5-degree margin, retaining
positions inside the existing deployment workspace
`[-.5, -.5, -.1]` to `[.5, .5, .6]` metres. Rotational workspace variation comes
from the sampled configurations. Reject only source episodes too short for a
full chunk, and list them in collection metadata.

Bootstrap collection uses the existing `EEBoundsAndSafety` and
current-joint-seeded `InverseKinematicsEEToJoints` pipeline. Store raw queries,
safety-adjusted targets, issued commands, delayed deliveries, actual site
poses, measured joints, timestamps, invalid-output holds and joint clamping.
An IK safety exception holds the measured pose and is recorded; it is not
silently replaced with a successful trial. Joint-bound guards are shared by
all methods, and the simulator retains its native 90-degree/s target limiter.

Native and `delay40` are separate experiments. The latter adds a deterministic
two-tick command FIFO before the native controller. Each reset clears the
queue; 0.5 seconds of settling and 0.2 seconds of measured command history
precede each motion. Labels pair future **actual** poses with **issued**
commands, not delayed deliveries. Padding masks mark unavailable future poses
and commands. No example crosses a rollout boundary.

### Matched methods and iteration

| Run | Input trajectory / command target |
|---|---|
| `C0/ik` | Runtime safety + IK, no learning |
| `C0/query` | Requested trajectory → issued commands |
| `C0/hindsight` | Achieved trajectory → issued commands |
| `C0/residual` | Achieved trajectory → issued command minus nominal IK |
| `C1/hindsight` | Same model trained on D0 + C0 recollection D1 |
| `bootstrap20k/hindsight` | Same model trained on D0 + additional bootstrap data |

Residual targets recompute nominal IK on achieved poses with the measured
joint state at each recorded tick. Runtime residual execution adds the
predicted correction to IK for the requested pose. All methods share command
guards; the direct models never invoke an IK fallback.

Each dynamics condition collects 10,000 D0 training and 1,000 fixed development
rollouts. C1 adds 10,000 recollected rollouts for each C0 training seed. Its
equal-data bootstrap comparator receives 10,000 new IK rollouts with identical
new motion/start seeds. Both train from scratch for the same 30k updates.

All learned methods use an image-free transformer: width 256, eight heads,
four encoder layers, two decoder layers, feedforward width 1024, dropout 0.1.
Train with masked normalized L1, AdamW `lr=1e-4`, weight decay `1e-4`, batch
256, and gradient clipping at 1.0. Statistics use training windows only.
Checkpoint selection uses development execution position RMSE, then rotation
error, every 5k updates. Save optimizer and RNG states for restart.

The smoke preset uses eight training/two development rollouts, two ACT queries,
one start, one seed, width 32, and four optimizer steps. Its results verify
software paths and must not be interpreted as method performance.

### Feasibility and metrics

Classify each original query using method-independent sequential IK on the
50 Hz grid. `ik_feasible` requires workspace and joint bounds, ≤5 mm position
residual, ≤3 degrees rotation residual, and ≤90 degrees/s nominal joint
velocity. This label describes that IK procedure, not all physically feasible
motions. Keep rejected/infeasible queries in all-query metrics and failure
statistics. The simulator has no self-collision/contact constraints, so this
workspace procedure is not a physical robot safety certificate.

Evaluate recorded UMI and frozen ACT trajectories separately, using the same
three deterministic starts for every query and controller. The arm runs with
proprioceptive feedback, but the recorded task images do not react to simulated
motion. This is offline ACT inference followed by trajectory execution.

Report ACT-versus-demonstration error, execution-versus-request error, and
execution-versus-demonstration error separately. Other metrics include
rotation geodesic error, endpoint error, command acceleration, joint clamping,
invalid outputs, p95 latency, 20 ms deadline misses, and tracking failures.
A tracking failure means any execution error, position error above 5 cm, or
orientation error above 15 degrees during the chunk. Gripper targets use the
same `-0.91 * normalized_position` mapping for every method and are scored
separately; no grasp success is inferred.

Reports show all queries and the shared IK-feasible subset, along with coverage.
Paired 95% bootstrap intervals resample source episodes, retaining all starts
and frames in the same cluster. Training seeds are displayed separately.
The report refuses to compare different trial sets or inconsistent feasibility
labels. Native rigid-model FK/site agreement is a consistency test, not an
encoder-versus-external-tracker experiment.

## Individual stages and artifacts

```bash
RUN=examples/umi_relative_ee/so101_task_independent_embodiment/run.sh
bash "$RUN" init --preset full
bash "$RUN" preflight --device cuda
bash "$RUN" extract
bash "$RUN" cache-act --device cuda
bash "$RUN" collect --condition native --generation D0 --split train
bash "$RUN" collect --condition native --generation D0 --split dev
bash "$RUN" train --condition native --generation C0 --method hindsight --seed 1000
bash "$RUN" collect --condition native --generation D1 --split train --seed 1000
bash "$RUN" train --condition native --generation C1 --method hindsight --seed 1000
bash "$RUN" evaluate --condition native --generation C1 --method hindsight --seed 1000
bash "$RUN" report
```

Use `all` for the complete comparison matrix. Per-stage commands above illustrate
one route, rather than the full baseline matrix. Supply the same `--root` to
each command when overriding the default. JSON manifests, not directory names,
bind stages to their inputs. Prepared training windows are memory-mapped to
bound RAM use. Rollouts and reports are written atomically; completed stages
and individual rollouts are reused only with `--resume` and matching provenance.

The launcher saves timestamped console logs and exit codes under `logs/`.
The artifact root also contains `config.json`, `preflight.json`, `motions/`,
`collections/`, `prepared/`, `act_cache/`, `train/`, `eval/`, and `report/`.
The report exports `REPORT.md`, `summary.json` (confidence intervals and paired
differences), `summary.csv`, `trials.csv`, `tracking.png`, and `tracking.pdf`.

Run contract tests with the experiment interpreter:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/umi-emb-uv-cache uv run --no-project \
  --python /mnt/data1/projects/lerobot-embodiment-env/bin/python python -m pytest \
  tests/scripts/test_so101_task_independent_embodiment.py -q
```
