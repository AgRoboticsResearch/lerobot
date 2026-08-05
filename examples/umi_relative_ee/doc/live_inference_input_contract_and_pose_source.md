---
created: 2026-08-04
updated: 2026-08-04
workspace: /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
branch: fei-v5.0-umi-unified
status: technical note
tags:
  - pi0.5
  - live-inference
  - pose-source
  - umi
  - end-to-end
  - visualization
---

# π0.5 Live-Inference Input Contract & Where the Pose Comes From

This note records a verification pass (2026-08-04) over four questions that came
up while planning a *live, camera-only* π0.5 prediction visualizer:

1. What does π0.5 actually consume at inference, and what does the working
   offline visualizer feed it?
2. Is the temporal/motion information carried by a second image `(t, t-1)`, or by
   the state?
3. Does the policy "need a previous pose"?
4. Is the relative-EE pose an *external* (non-end-to-end) dependency inherent to
   UMI — i.e. does canonical UMI run SLAM live at deploy?

Every claim below is backed by the cited source. Two earlier assumptions were
corrected during this pass and are called out explicitly.

## TL;DR

- **π0.5 inference input = 1 image + language tokens.** The 20D relative state is
  *not* a separate tensor; it is discretized and baked into the PaliGemma prompt
  (`Task: ..., State: <20 bins>;\nAction:`).
- **One image.** Temporal motion lives in the 20D state (`[prev, current]` poses),
  not in image history. This is true for π0.5, unified ACT, and the `fei`-branch
  ACT (`n_obs_steps = 1`).
- **No explicit previous pose is required.** A single 7D current pose suffices;
  the processor fills `prev = current`. Held still, the state collapses to
  ≈ identity and the image alone drives the chunk.
- **Image slots are spatial (multi-camera), not temporal.** Using a 2nd image slot
  for `t-1` feeds the pixels but the model reads it as another camera view (no
  temporal positional encoding). Genuine `(t, t-1)` needs a retrain.
- **Canonical UMI does NOT run SLAM at deploy.** Deploy pose comes from the robot
  arm's FK (`ActualTCPPose`). SLAM is offline data-collection only. SROI matches
  this split (SLAM data, Piper FK deploy). The "external pose computation"
  concern applies *only* to a no-robot, camera-only live setting — not to UMI
  deploy.

## 0. Two visualizers, two branches

There are two `visualize_predictions.py` scripts. They are **different tools on
different branches**, not copies.

| | Unified (this workspace) | `fei` checkout |
| --- | --- | --- |
| Path | `examples/umi_relative_ee/visualize_predictions.py` | `~/code/lerobots/lerobot/examples/umi_relative_ee/visualize_predictions.py` |
| Mode | **Offline** on a recorded dataset | **Live** RealSense capture + offline dataset mode |
| RealSense | none | `pyrealsense2`, `auto_detect_realsense_serial()`, intrinsics from the live pipeline |
| Branch | `fei-v5.0-umi-unified` | `fei` |

**Load incompatibility (important):** the `fei` live script **cannot load unified
π0.5 checkpoints.** Unified checkpoints store the preprocessor steps
`umi_derive_state_from_action` / `umi_relative_actions` / `umi_relative_state`
(defined in `src/lerobot/processor/umi_relative_ee_processor.py`), which do not
exist in the `fei` checkout (it has the older
`relative_action_processor*.py` pipeline with different registry names).
Deserialization of `policy_preprocessor.json` would fail on the registry lookup.
The `fei` live script was built for `fei`-branch ACT/SmolVLA models.

So there is **no existing script** that does live-RealSense + unified-π0.5
together; that requires porting the capture loop into the unified visualizer.

## 1. π0.5 inference input contract

From `src/lerobot/policies/pi05/modeling_pi05.py:1227-1242`
(`predict_action_chunk`):

```python
images, img_masks = self._preprocess_images(batch)                 # observation.images.camera
tokens,  masks   = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
actions = self.model.sample_actions(images, img_masks, tokens, masks, ...)
```

The model forward consumes only **image(s)** and **language tokens**. The code
comment says it explicitly: *"no separate state needed for PI05"* — because the
state is baked into the text prompt.

`Pi05PrepareStateTokenizerProcessorStep`
(`src/lerobot/policies/pi05/processor_pi05.py:66-94`) builds

```text
Task: pick the strawberry, State: <20 integer bins>;
Action:
```

and **raises** if state is absent (`processor_pi05.py:70-71`,
`"State is required for PI05"`). The 20D state is normalized to `[-1, 1]` then
discretized into 256 bins before tokenization.

### Checkpoint `input_features` (verified, step `087000`→`087500`)

```text
observation.images.camera   shape=[3, 480, 640]   type=VISUAL
observation.state           shape=[20]            type=STATE
action                      shape=[10]            type=ACTION
use_umi_relative_ee = True
chunk_size = 30
```

Exactly **one** VISUAL key.

### What the working offline visualizer feeds

The unified `visualize_predictions.py` gets a recorded batch and the preprocessor
derives everything. Raw keys that must be present and meaningful:

| Raw key | Why |
| --- | --- |
| `observation.images.camera` | the image → model input |
| `action` window `[t-1, t, … t+chunk-1]` | **only `action[t-1]` and `action[t]` are used** — `UmiDeriveStateFromActionStep` (`umi_relative_ee_processor.py:156`) peels off `action[..., :2, :]` as the `[prev, current]` state. The future targets are GT, not model input. |
| `task` string | injected (`visualize_predictions.py`) → prompt |

So the offline path reads `action[t-1:t+1]` from the recording to build the
relative state; it passes no separate "current camera pose" argument and no FK.

## 2. One image; temporal info is in the state

Verified across the three policies:

| Policy | Images | Temporal info |
| --- | --- | --- |
| π0.5 UMI | **1** | `[prev, current]` poses → 20D state (in the prompt) |
| unified ACT | **1** (`n_obs_steps = 1`, `configuration_act.py:84`) | `obs_state_horizon = 2` (`:99`) → `[prev, current]` poses |
| `fei` ACT | **1** (`n_obs_steps = 1`, `:94`; raises if `!= 1`, `:172-174`) | `obs_state_horizon = 2` (`:97`) |

The `fei` dataset-mode visualizer confirms the same: its
`delta_timestamps = {"action": [...]}` (`visualize_predictions.py:571`) applies
deltas **only to `action`**, never to the image key — so the dataset returns a
single current frame.

The "2" is always in the **state horizon** (poses), never in the images. Motion
history lives entirely in the 20D relative state; the policies are single-frame
on vision.

## 3. No explicit previous pose is required

`UmiRelativeStateStep` (`umi_relative_ee_processor.py:228-247`) accepts a single
`[B, 7]` state and synthesizes the pair itself:

```python
if state.ndim == 2:                                   # single [B,7] pose
    previous = state if self._previous_state is None else self._previous_state.to(state)
    state_pair = torch.stack([previous, state], dim=1)
```

On the first tick `prev = current`, so a single current pose is enough. And
`UmiDeriveStateFromActionStep` no-ops when `action` is absent
(`umi_relative_ee_processor.py:152`), so feeding `observation.state` directly
(without the recorded `action`) works for live inference.

The `fei` live camera mode exploits exactly this (`visualize_predictions.py:425-445`):

```python
current_state = np.array([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)   # origin + half gripper
...
batch = {OBS_STATE: torch.from_numpy(current_state).unsqueeze(0).to(device)}
```

One static pose, defaulted to identity. **Held still, the 20D state collapses to
≈ `[identity, identity]` and the image alone drives the predicted chunk.**
(Optional `--update_state` auto-chains the last prediction as the next state.)

## 4. Multi-camera slot vs. multi-time image `(t, t-1)`

The distinction is *mechanical* vs. *semantic*.

**Mechanically** you can put a `t-1` frame into a second image slot and the
pixels flow in. **Semantically** both architectures treat image slots as
**spatial views (multiple cameras), not temporal frames** — there is no
positional signal marking "this is the past," so the model cannot distinguish a
second camera from a previous timestep.

- **ACT**: a shared ResNet encodes each `observation.images.*` key independently;
  multiple keys "are treated as multiple camera views," and `n_obs_steps = 1` is
  enforced. A `t-1`-in-a-camera-slot hack therefore yields two viewpoints with
  **zero temporal awareness**. Real temporal images require the
  `TemporalACTWrapper` (`src/lerobot/policies/act/temporal_wrapper.py:31`, which
  *does* add temporal position embeddings and "UMI-style batching") — but it is
  `disable_temporal_wrapper = True` and was never trained.
- **π0.5**: PaliGemma turns each image into its own block of patch tokens in the
  VLM context. Multiple images = multiple views; no temporal marker, and the UMI
  fine-tune trained on one image.

A second *camera* buys geometry (stereo/parallax/extra viewpoint); a second
*time* buys motion/ego-motion. They are not interchangeable. Genuine end-to-end
`(t, t-1)` is a **retrain**, not a slot rename: ACT → enable+train
`TemporalACTWrapper`; π0.5 → add a temporal image slot with explicit time
encoding and LoRA from `lerobot/pi05_base`.

## 5. Canonical UMI does NOT run SLAM at deploy  *(correction)*

An earlier claim — "needing external pose computation is inherent to the UMI
paradigm; canonical UMI runs ORB-SLAM3 live at deploy" — is **false**. Verified
against `~/code/universal_manipulation_interface`:

- **Deploy loop** `scripts_real/eval_real_umi.py:238-239` runs on a real arm
  (`UmiEnv(robot_ip=..., gripper_ip=...)`, UR5/Franka + WSG) and reads the pose
  from the **robot**: `state = env.get_robot_state(); target_pose = state['ActualTCPPose']`.
- **Obs builder** `umi/real_world/real_inference_util.py`
  (`get_real_umi_obs_dict`) builds the state from `env_obs['robot0_eef_pos']` /
  `env_obs['robot0_eef_rot_axis_angle']` — **robot forward kinematics** — then
  relativizes it (`base_pose_mat = pose_mat[-1]`).
- **SLAM is offline data-collection only**: `run_slam_pipeline.py` +
  `scripts_slam_pipeline/00_process_videos … 07_generate_replay_buffer` turn the
  handheld GoPro recordings into the Zarr training set. The README calls ORB-SLAM3
  "the most fragile part of UMI pipeline" — a *data-collection* burden, not a
  deploy one.

So at deploy, UMI's `[prev, current]` state is just **robot proprioception** (the
arm's own TCP pose), like any robot-arm policy. It is effectively end-to-end at
deploy: *image + robot-state → action*. There is no live SLAM tracker in the
control loop.

**SROI is faithful to this split:** SLAM for data (RealSense + ORB-SLAM3 +
AprilTag gripper), Piper FK/IK for deploy. Neither runs SLAM at deploy.

Consequence: the "needs an external pose tracker / not end-to-end" concern applies
**only to a no-robot, camera-only live setting** (e.g. the handheld live
visualizer this note started from). With a robot in the loop the pose is free
proprioception.

## 6. On-disk action frame semantics  *(verified on validation data)*

The stored `action` column is **not** a robot-base-frame "absolute EE pose."
Measured on `sroiv2_strawberry_picking_lab_validation` episode 0:

```text
raw action xyz (m):   t0 ≈ [0, 0.0006, -0.0004]   t1 ≈ [0, 0.001, -0.0007]
whole-episode xyz span (m): [0.092, 0.009, 0.065]      (~9 cm total)
relative step inv(T[t])@T[t+1]: 0.3–0.6 mm, 0.06–0.33 deg
```

mm-scale, starting at origin → the SLAM **camera trajectory in a tracking frame
re-initialized at episode start**. `umi_relative_ee_processor.py` labels it
"absolute" only to mean *fixed tracking-frame reference* (vs. the per-chunk
relative target); `absolute_aa_to_relative_rot6d` computes `inv(T_base) @ T_target`
and `to_umi_relative_state` computes `inv(T_current) @ [prev, current]`.
Everything the model consumes is relative to the current (chunk-start) camera
frame. No robot EE / FK anywhere in the data.

## 7. Implications for a live, no-robot camera-only π0.5 visualizer

Because the model is single-image, single-(static)-pose, and image-driven:

- **Feasible as-is:** feed the live RealSense frame + a static identity pose
  (`[0,0,0,0,0,0,0.5]`) + task string → 30-step predicted chunk overlaid on the
  camera view. This is a valid "predict from the current camera view" peek.
- **What you lose:** the prev→current motion cue (state ≈ identity when held
  still). For a real rollout you'd want the live camera pose from a VO/SLAM
  tracker (the one place an *external* pose computation is actually needed), or a
  robot (FK pose).
- **To make it truly end-to-end on images** (drop the pose-state): retrain with
  `(t, t-1)` image history and temporal encoding (§4) — not achievable by
  repointing the current checkpoints.

### Build plan (port the live loop into the unified visualizer)

The `fei` live script's camera loop is the template; it must run against the
**unified** source so the π0.5 checkpoint loads:

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"
```

Per-tick contract (single image + single static 7D state + task):

```text
observation.images.camera  ← RealSense D405 frame (uint8 → /255)
observation.state          ← [0,0,0,0,0,0,0.5]  (identity + half gripper; or tracked pose)
task                       ← "pick the strawberry"
```

Then `preprocessor.reset()` → `processed = preprocessor(batch)` →
`policy.predict_action_chunk(processed)` → unnormalize → overlay. The projection
onto the camera image uses the existing D405 hand-eye
(`camera_gripper_extrinsics_sroi_v2_d405.json`) + intrinsics K (from the live
RealSense profile or a saved `camera_info_color.json`).

## References

- `src/lerobot/policies/pi05/modeling_pi05.py:1227-1242` — predict path
- `src/lerobot/policies/pi05/processor_pi05.py:66-94` — state → prompt tokenizer
- `src/lerobot/processor/umi_relative_ee_processor.py:143-263` — derive/relative state steps
- `src/lerobot/policies/act/configuration_act.py` — `n_obs_steps = 1`, `obs_state_horizon = 2`
- `src/lerobot/policies/act/temporal_wrapper.py` — disabled temporal image support
- `examples/umi_relative_ee/visualize_predictions.py` — unified offline visualizer
- `~/code/lerobots/lerobot/examples/umi_relative_ee/visualize_predictions.py` — `fei` live visualizer (incompatible with unified π0.5 checkpoints)
- `~/code/universal_manipulation_interface/scripts_real/eval_real_umi.py` — canonical UMI deploy (robot FK)
- `~/code/universal_manipulation_interface/run_slam_pipeline.py` + `scripts_slam_pipeline/` — offline SLAM data prep
- See also `pi0.5_finetunning.md`, `umi_style_ee_processor_pipeline.md`
