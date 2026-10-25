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

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# π0.5 实时推理输入契约与位姿来源

本笔记记录了一次验证工作（2026-08-04），针对在规划一个*实时、仅相机*的 π0.5 预测可视化器时出现的四个问题：

1. π0.5 在推理时实际消费什么，而可用的离线可视化器向它喂入什么？
2. 时间/运动信息是由第二张图像 `(t, t-1)` 承载，还是由 state 承载？
3. 策略是否“需要上一帧位姿”？
4. 相对 EE 位姿是否是 UMI 固有的*外部*（非端到端）依赖——即规范 UMI 是否在部署时实时运行 SLAM？

下文每个论断都有所引用的源码支撑。本次验证中修正了先前两个假设，并明确标出。

## 摘要（TL;DR）

- **π0.5 推理输入 = 1 张图像 + 语言 token。** 20D 相对 state *不是*单独的 tensor；它被离散化并固化进 PaliGemma prompt（`Task: ..., State: <20 bins>;\nAction:`）。
- **一张图像。** 时间运动信息存在于 20D state（`[prev, current]` 位姿）中，而不是图像历史中。对 π0.5、unified ACT 以及 `fei` 分支 ACT（`n_obs_steps = 1`）都是如此。
- **不需要显式的上一位姿。** 单个 7D 当前位姿就足够；processor 会填充 `prev = current`。保持静止时，state 退化为 ≈ 单位变换，仅由图像驱动 chunk。
- **图像槽位是空间性的（多相机），不是时间性的。** 将第二个图像槽位用于 `t-1` 会把像素喂进去，但模型将其读作另一个相机视角（没有时间位置编码）。真正的 `(t, t-1)` 需要重新训练。
- **规范 UMI 在部署时并不运行 SLAM。** 部署位姿来自机器人手臂的 FK（`ActualTCPPose`）。SLAM 仅用于离线数据采集。SROI 符合这一分工（SLAM 用于数据，Piper FK 用于部署）。“外部位姿计算”的顾虑*仅*适用于无机器人、仅相机的实时场景——不适用于 UMI 部署。

## 0. 两个可视化器，两个分支

有两个 `visualize_predictions.py` 脚本。它们是**不同分支上的不同工具**，不是副本。

| | 统一版（本工作区） | `fei` 检出 |
| --- | --- | --- |
| 路径 | `examples/umi_relative_ee/visualize_predictions.py` | `~/code/lerobots/lerobot/examples/umi_relative_ee/visualize_predictions.py` |
| 模式 | 在录制数据集上**离线** | **实时** RealSense 采集 + 离线数据集模式 |
| RealSense | 无 | `pyrealsense2`、`auto_detect_realsense_serial()`、内参来自实时流水线 |
| 分支 | `fei-v5.0-umi-unified` | `fei` |

**加载不兼容（重要）：** `fei` 实时脚本**无法加载统一版 π0.5 checkpoint。** 统一版 checkpoint 保存了预处理步骤 `umi_derive_state_from_action` / `umi_relative_actions` / `umi_relative_state`（定义于 `src/lerobot/processor/umi_relative_ee_processor.py`），这些在 `fei` 检出中不存在（它有的是较旧的 `relative_action_processor*.py` 流水线，registry 名称不同）。`policy_preprocessor.json` 的反序列化会在 registry 查找时失败。`fei` 实时脚本是为 `fei` 分支的 ACT/SmolVLA 模型构建的。

因此，**不存在现有脚本**能同时完成实时 RealSense + 统一版 π0.5；这需要将采集循环移植进统一版可视化器。

## 1. π0.5 推理输入契约

来自 `src/lerobot/policies/pi05/modeling_pi05.py:1227-1242`（`predict_action_chunk`）：

```python
images, img_masks = self._preprocess_images(batch)                 # observation.images.camera
tokens,  masks   = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
actions = self.model.sample_actions(images, img_masks, tokens, masks, ...)
```

模型 forward 仅消费**图像**与**语言 token**。代码注释明确写道：*"no separate state needed for PI05"*——因为 state 已被固化进文本 prompt。

`Pi05PrepareStateTokenizerProcessorStep`（`src/lerobot/policies/pi05/processor_pi05.py:66-94`）构建

```text
Task: pick the strawberry, State: <20 integer bins>;
Action:
```

并在 state 缺失时**抛出异常**（`processor_pi05.py:70-71`，`"State is required for PI05"`）。20D state 先归一化到 `[-1, 1]`，然后在 tokenization 之前离散化为 256 个 bin。

### Checkpoint 的 `input_features`（已验证，步 `087000`→`087500`）

```text
observation.images.camera   shape=[3, 480, 640]   type=VISUAL
observation.state           shape=[20]            type=STATE
action                      shape=[10]            type=ACTION
use_umi_relative_ee = True
chunk_size = 30
```

恰好**一个** VISUAL key。

### 可用的离线可视化器喂入什么

统一版 `visualize_predictions.py` 获取一个录制 batch，预处理会派生一切。必须存在且有意义的原始 key：

| 原始 key | 原因 |
| --- | --- |
| `observation.images.camera` | 图像 → 模型输入 |
| `action` 窗口 `[t-1, t, … t+chunk-1]` | **仅 `action[t-1]` 和 `action[t]` 被使用**——`UmiDeriveStateFromActionStep`（`umi_relative_ee_processor.py:156`）剥出 `action[..., :2, :]` 作为 `[prev, current]` state。未来目标是 GT，不是模型输入。 |
| `task` 字符串 | 注入（`visualize_predictions.py`）→ prompt |

因此离线路径从录制中读取 `action[t-1:t+1]` 来构建相对 state；它不传入单独的“当前相机位姿”参数，也不传入 FK。

## 2. 一张图像；时间信息在 state 中

在三种策略上已验证：

| 策略 | 图像数 | 时间信息 |
| --- | --- | --- |
| π0.5 UMI | **1** | `[prev, current]` 位姿 → 20D state（在 prompt 中） |
| unified ACT | **1**（`n_obs_steps = 1`，`configuration_act.py:84`） | `obs_state_horizon = 2`（`:99`）→ `[prev, current]` 位姿 |
| `fei` ACT | **1**（`n_obs_steps = 1`，`:94`；若 `!= 1` 则抛出异常，`:172-174`） | `obs_state_horizon = 2`（`:97`） |

`fei` 数据集模式可视化器印证了同一点：其 `delta_timestamps = {"action": [...]}`（`visualize_predictions.py:571`）**仅对 `action`** 应用 delta，从不对图像 key 应用——因此数据集返回单个当前帧。

这个“2”始终在 **state horizon**（位姿）中，从不在图像中。运动历史完全存在于 20D 相对 state 中；这些策略在视觉上是单帧的。

## 3. 不需要显式的上一位姿

`UmiRelativeStateStep`（`umi_relative_ee_processor.py:228-247`）接受单个 `[B, 7]` state，并自行合成这一对：

```python
if state.ndim == 2:                                   # single [B,7] pose
    previous = state if self._previous_state is None else self._previous_state.to(state)
    state_pair = torch.stack([previous, state], dim=1)
```

在第一个 tick `prev = current`，因此单个当前位姿就足够。并且 `UmiDeriveStateFromActionStep` 在 `action` 缺失时是无操作（`umi_relative_ee_processor.py:152`），所以直接喂入 `observation.state`（不带录制的 `action`）可用于实时推理。

`fei` 实时相机模式正是利用了这一点（`visualize_predictions.py:425-445`）：

```python
current_state = np.array([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)   # origin + half gripper
...
batch = {OBS_STATE: torch.from_numpy(current_state).unsqueeze(0).to(device)}
```

一个静态位姿，默认为单位变换。**保持静止时，20D state 退化为 ≈ `[identity, identity]`，仅由图像驱动预测 chunk。**（可选的 `--update_state` 会将上一次预测自动串为下一个 state。）

## 4. 多相机槽位 vs. 多时间图像 `(t, t-1)`

区别在于*机械层面*与*语义层面*。

**机械上**，你可以把 `t-1` 帧放进第二个图像槽位，像素会流入。**语义上**，两种架构都将图像槽位视为**空间视角（多相机），而非时间帧**——没有位置信号标记“这是过去”，因此模型无法区分第二台相机和上一个时间步。

- **ACT**：共享 ResNet 独立编码每个 `observation.images.*` key；多个 key “被视为多个相机视角”，且强制 `n_obs_steps = 1`。因此把 `t-1` 塞进相机槽位的 hack 会产生两个视角，**没有任何时间感知**。真正的时间图像需要 `TemporalACTWrapper`（`src/lerobot/policies/act/temporal_wrapper.py:31`，它*确实*添加了时间位置嵌入和 "UMI-style batching"）——但它是 `disable_temporal_wrapper = True`，且从未训练过。
- **π0.5**：PaliGemma 将每张图像变成 VLM 上下文中它自己的 patch token 块。多张图像 = 多个视角；没有时间标记，且 UMI 微调只在一张图像上训练。

第二台*相机*带来几何信息（立体/视差/额外视角）；第二个*时间*带来运动/自运动信息。二者不可互换。真正的端到端 `(t, t-1)` 是一次**重新训练**，而不是槽位重命名：ACT → 启用并训练 `TemporalACTWrapper`；π0.5 → 添加带显式时间编码的时间图像槽位，并从 `lerobot/pi05_base` 做 LoRA。

## 5. 规范 UMI 在部署时不运行 SLAM *（更正）*

先前的论断——“需要外部位姿计算是 UMI 范式固有的；规范 UMI 在部署时实时运行 ORB-SLAM3”——是**错误的**。已对照 `~/code/universal_manipulation_interface` 验证：

- **部署循环** `scripts_real/eval_real_umi.py:238-239` 在真实机械臂上运行（`UmiEnv(robot_ip=..., gripper_ip=...)`，UR5/Franka + WSG），并从**机器人**读取位姿：`state = env.get_robot_state(); target_pose = state['ActualTCPPose']`。
- **Obs 构建器** `umi/real_world/real_inference_util.py`（`get_real_umi_obs_dict`）从 `env_obs['robot0_eef_pos']` / `env_obs['robot0_eef_rot_axis_angle']` 构建 state——**机器人正运动学**——然后将其相对化（`base_pose_mat = pose_mat[-1]`）。
- **SLAM 仅用于离线数据采集**：`run_slam_pipeline.py` + `scripts_slam_pipeline/00_process_videos … 07_generate_replay_buffer` 将手持 GoPro 录制转换为 Zarr 训练集。README 称 ORB-SLAM3 是“UMI 流水线中最脆弱的部分”——这是*数据采集*负担，不是部署负担。

因此在部署时，UMI 的 `[prev, current]` state 只是**机器人本体感知**（机械臂自身的 TCP 位姿），与任何机械臂策略一样。它在部署时实际上是端到端的：*图像 + 机器人状态 → 动作*。控制循环中没有实时 SLAM 跟踪器。

**SROI 忠实于这一分工：** 数据用 SLAM（RealSense + ORB-SLAM3 + AprilTag gripper），部署用 Piper FK/IK。两者在部署时都不运行 SLAM。

推论：“需要外部位姿跟踪器 / 非端到端”的顾虑**仅适用于无机器人、仅相机的实时场景**（例如本笔记起初关注的手持实时可视化器）。有机器人在环时，位姿是免费的本体感知。

## 6. 磁盘上动作帧的语义 *（在验证数据上已验证）*

存储的 `action` 列**不是**机器人基座坐标系中的“绝对 EE 位姿”。在 `sroiv2_strawberry_picking_lab_validation` 第 0 回合上测得：

```text
raw action xyz (m):   t0 ≈ [0, 0.0006, -0.0004]   t1 ≈ [0, 0.001, -0.0007]
whole-episode xyz span (m): [0.092, 0.009, 0.065]      (~9 cm total)
relative step inv(T[t])@T[t+1]: 0.3–0.6 mm, 0.06–0.33 deg
```

毫米级、从原点开始 → 这是 SLAM **相机轨迹，位于一个在回合开始时重新初始化的跟踪坐标系**。`umi_relative_ee_processor.py` 将其标记为 "absolute" 仅表示*固定跟踪坐标系参考*（相对于逐 chunk 的相对目标）；`absolute_aa_to_relative_rot6d` 计算 `inv(T_base) @ T_target`，`to_umi_relative_state` 计算 `inv(T_current) @ [prev, current]`。模型消费的一切都相对于当前（chunk 起始）相机帧。数据中任何地方都没有机器人 EE / FK。

## 7. 对实时、无机器人、仅相机的 π0.5 可视化器的影响

由于模型是单图像、单（静态）位姿、图像驱动的：

- **按原样可行：** 喂入实时 RealSense 帧 + 静态单位位姿（`[0,0,0,0,0,0,0.5]`）+ task 字符串 → 在相机视图上叠加 30 步预测 chunk。这是一个有效的“从当前相机视角预测”预览。
- **你会失去什么：** prev→current 运动线索（静止时 state ≈ 单位变换）。对于真实 rollout，你会想要来自 VO/SLAM 跟踪器的实时相机位姿（这是真正需要*外部*位姿计算的唯一场景），或者一台机器人（FK 位姿）。
- **要使其在图像上真正端到端**（去掉位姿 state）：用 `(t, t-1)` 图像历史和时间编码（§4）重新训练——无法通过重新指向当前 checkpoint 实现。

### 构建计划（将实时循环移植进统一版可视化器）

`fei` 实时脚本的相机循环是模板；它必须针对**统一版**源码运行，以便 π0.5 checkpoint 能加载：

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"
```

每 tick 契约（单图像 + 单个静态 7D state + task）：

```text
observation.images.camera  ← RealSense D405 frame (uint8 → /255)
observation.state          ← [0,0,0,0,0,0,0.5]  (identity + half gripper; or tracked pose)
task                       ← "pick the strawberry"
```

然后 `preprocessor.reset()` → `processed = preprocessor(batch)` → `policy.predict_action_chunk(processed)` → 反归一化 → 叠加。投影到相机图像使用现有 D405 手眼（`camera_gripper_extrinsics_sroi_v2_d405.json`）+ 内参 K（来自实时 RealSense profile 或保存的 `camera_info_color.json`）。

## 参考文献

- `src/lerobot/policies/pi05/modeling_pi05.py:1227-1242` — 预测路径
- `src/lerobot/policies/pi05/processor_pi05.py:66-94` — state → prompt tokenizer
- `src/lerobot/processor/umi_relative_ee_processor.py:143-263` — derive/relative state 步骤
- `src/lerobot/policies/act/configuration_act.py` — `n_obs_steps = 1`、`obs_state_horizon = 2`
- `src/lerobot/policies/act/temporal_wrapper.py` — 已禁用的时间图像支持
- `examples/umi_relative_ee/visualize_predictions.py` — 统一版离线可视化器
- `~/code/lerobots/lerobot/examples/umi_relative_ee/visualize_predictions.py` — `fei` 实时可视化器（与统一版 π0.5 checkpoint 不兼容）
- `~/code/universal_manipulation_interface/scripts_real/eval_real_umi.py` — 规范 UMI 部署（机器人 FK）
- `~/code/universal_manipulation_interface/run_slam_pipeline.py` + `scripts_slam_pipeline/` — 离线 SLAM 数据准备
- 另见 `pi0.5_finetunning.md`、`umi_style_ee_processor_pipeline.md`
