# UMI Relative-EE Trajectory Interpolation

## Status and recommendation

This is the recommended design for adding trajectory interpolation to the Piper
UMI relative-EE deployment. It is a design specification, not yet an implemented
deployment feature.

Use a **commanded-trajectory splice with a bounded, decaying anchor correction**:

1. Keep policy state and safety feedback based on measured FK.
2. Start every replacement trajectory at the active commanded trajectory value.
3. Discard policy waypoints whose timestamps are already stale.
4. Shift the first future targets toward the command frame only by a bounded amount.
5. Decay that shift to zero across the beginning of the chunk, restoring measured-frame
   targets.
6. Hold and require recovery when command-versus-measured error exceeds a hard limit.

Do not use unconditional full command re-anchoring as the default. It is smooth in the
pure-delay offline experiment, but it removes tracking-error correction and can keep
advancing a plan when the physical arm is blocked.

The corrected experiment and figures are in
`outputs/research_report/low_level_control_debug/umi_interp_experiment/umi_interp_experiment.md`.

## Why interpolation is needed

The policy predicts one chunk of relative poses, all expressed from the measured pose at
the chunk start. The current synchronous deployment converts that chunk to absolute poses
and replays it verbatim. When the robot lags its command, the next measured-anchored chunk
can begin behind the last command from the previous chunk. Replacing the queue therefore
creates a Cartesian command discontinuity.

UMI avoids a position discontinuity by retaining a timestamped commanded trajectory. A
new plan replaces only its future; the connection begins at the value of the active
commanded trajectory at the splice time, not by teleporting the command to measured FK.

That mechanism guarantees position continuity (C0), but linear translation and SLERP do
not guarantee continuous velocity (C1). If the first new absolute target lies far behind
the command, the controller can still make a sharp correction during the following tick.
The bounded anchor correction addresses that transition without abandoning measured
feedback for the whole chunk.

## Pose and action contract

Absolute EE poses are:

```text
T = [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
```

The rigid part is an SE(3) transform. Gripper is a separate scalar.

Policy actions are:

```text
ΔT[i] = [dx, dy, dz, rot6d(6), gripper], i = 0 ... horizon-1
```

Every `ΔT[i]` is relative to the same chunk-start measured pose `T_m`:

```text
T_measured_target[i] = T_m · ΔT[i]
```

Targets are not integrated from the preceding predicted action.

For the current UMI ACT checkpoint, the dataset action window is
`[-1, 0, ..., horizon-1]`; the processor consumes `-1`. Consequently, model action index
0 corresponds to the current pose at observation time, not the next control step.

## Recommended algorithm

### 1. Maintain the active commanded trajectory

Keep a `PoseTrajectoryInterpolator7D` containing timestamped absolute poses:

- XYZ: linear interpolation;
- rotation: quaternion SLERP, exposed as axis-angle at the output;
- gripper: linear interpolation with a gripper rate limit;
- time outside the trajectory: clamp and hold its first/last pose.

At control time `now`, define:

```text
T_c = active_trajectory(now)
```

`T_c` is the command that would be emitted if no new policy result arrived. It is not
measured FK and not merely the last chunk endpoint.

### 2. Timestamp actions and remove stale waypoints

For an observation captured at `observation_time`, action index `i` has timestamp:

```text
action_time[i] = observation_time + i · action_dt
```

Accept only waypoints satisfying:

```text
action_time[i] > now + execution_latency_margin
```

With immediate 30 Hz inference this normally removes action 0. With real inference
latency it may remove additional actions. Do not hard-code “always remove exactly one” in
the live implementation.

If no future waypoint survives, hold the current trajectory and request a new plan. Do
not schedule the last stale target as an immediate jump.

### 3. Measure and bound the anchor gap

Let the measured chunk-start pose be `T_m` and active command be `T_c`. Compute their
translation and geodesic rotation gaps:

```text
d = ||p_c - p_m||
θ = angle(R_m⁻¹ R_c)
```

Construct a bounded command reference `T_b`:

```text
p_b = p_m + min(1, position_cap / d) · (p_c - p_m)
R_b = R_m · Exp(min(1, rotation_cap / θ) · Log(R_m⁻¹ R_c))
```

Handle zero `d` or `θ` as identity corrections. Translation must be capped by Euclidean
norm, and rotation must be capped by SO(3) angle—not by clipping axis-angle components.

The cap parameters must be tuned from simulation and robot logs. They should be much
smaller than the existing 50 mm emergency EE-step rejection threshold; that threshold is
a last-resort safety check, not a smoothing target.

### 4. Decay the correction across future targets

For each surviving future index `i`, choose a weight `w[i]` that starts at 1 and decreases
monotonically to 0 over `anchor_decay_steps`:

```text
w[i] = max(0, 1 - (i - first_future_index) / anchor_decay_steps)
```

Interpolate the reference pose on SE(3):

```text
p_ref[i] = lerp(p_m, p_b, w[i])
R_ref[i] = slerp(R_m, R_b, w[i])
T_target[i] = T_ref[i] · ΔT[i]
```

This gives the first future targets some of the smooth full-command-anchor behavior, then
returns later targets to the measured reference intended by the policy. The gripper value
comes directly from `ΔT[i]`; it is not affected by EE anchoring.

Useful A/B endpoints are:

- `position_cap = rotation_cap = 0`: unchanged measured-decoded targets, equivalent to
  UMI-style splicing without command re-anchoring;
- infinite caps with no decay: full command re-anchoring, an experimental upper bound;
- finite caps plus decay: recommended deploy candidate.

### 5. Splice targets onto the commanded future

Schedule every accepted `T_target[i]` at its original `action_time[i]` using
`schedule_waypoint` with:

```text
curr_time = now
last_waypoint_time = last accepted/scheduled waypoint time
```

The scheduler must:

- preserve the active command at `now`;
- discard superseded future waypoints;
- retain accepted earlier waypoints when appropriate;
- append new targets in strictly increasing timestamp order;
- keep the newest target if two targets share a timestamp;
- apply finite Cartesian translation and rotation speed limits.

Evaluate the interpolator once per control tick and pass the resulting absolute EE target
through the existing `EEBoundsAndSafety` and IK pipeline.

## Pseudocode

```python
def accept_policy_chunk(relative_chunk, observation_time, now, measured_pose):
    command_pose = trajectory(now)

    gap = se3_difference(measured_pose, command_pose)
    if gap.position > hard_position_gap or gap.rotation > hard_rotation_gap:
        hold_and_require_recovery("tracking gap exceeded")
        return

    bounded_command_ref = cap_se3_reference(
        measured_pose,
        command_pose,
        position_cap=anchor_position_cap,
        rotation_cap=anchor_rotation_cap,
    )

    future = []
    for index, relative in enumerate(relative_chunk):
        target_time = observation_time + index * action_dt
        if target_time <= now + execution_latency_margin:
            continue

        weight = anchor_decay_weight(index, first_future_index, anchor_decay_steps)
        reference = interpolate_se3(measured_pose, bounded_command_ref, weight)
        target = compose_relative(reference, relative)
        future.append((target_time, target))

    if not future:
        request_replan_and_hold()
        return

    for target_time, target in future:
        trajectory = trajectory.schedule_waypoint(
            target,
            target_time,
            curr_time=now,
            last_waypoint_time=last_waypoint_time,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
        )
        last_waypoint_time = target_time
```

## Control-loop rules

- **Measured state stays measured.** Always construct policy state from consecutive FK
  measurements. Never feed commanded pose into the observation state to hide lag.
- **One chunk, one measured base.** Decode all relative targets from the measured pose
  captured for that inference request before applying the bounded correction.
- **Use monotonic time.** Scheduling and latency checks must use one monotonic clock.
- **No extrapolation during inference stalls.** Hold the last trajectory pose.
- **Reset on direct/manual motion.** Pause, start-pose, safe-pose, or IK-resync operations
  must clear pending waypoints and initialize a one-pose trajectory from the new measured
  state.
- **Reject stale async results.** A result generated before pause/resume or direct motion
  must never be accepted afterward.
- **Do not average axis-angle vectors.** Blend rotations with SLERP or Lie-group
  interpolation.

## Safety behavior

The smoothing mechanism must not override existing safety checks.

| Condition | Required behavior |
|---|---|
| Non-finite or malformed target | Reject chunk and hold |
| No non-stale waypoint | Hold and request another inference |
| Moderate command–measurement gap | Apply bounded, decaying correction |
| Gap over hard position/rotation limit | Hold, clear future, require recovery/resync |
| IK failure | Hold last valid joint/EE command; do not advance provenance as executed |
| EE workspace or step violation | Reject/hold through existing safety pipeline |
| Pause or direct move | Invalidate queued and in-flight trajectories |

Full command anchoring must not be used to mask a growing tracking gap. Log both the raw
gap and the bounded correction applied at every replan.

## Parameters to expose

```text
trajectory_interpolation: bool
anchor_mode: measured | command | bounded_decay
anchor_position_cap_m: float
anchor_rotation_cap_rad: float
anchor_decay_steps: int
execution_latency_margin_s: float
max_cartesian_speed_m_s: float
max_rotation_speed_rad_s: float
hard_tracking_gap_m: float
hard_tracking_rotation_gap_rad: float
```

`bounded_decay` should become the recommended mode only after the validation gates below
pass. Preserve `measured` and `command` modes for controlled A/B experiments.

## Required logging

Per tick:

- measured FK pose;
- interpolated command pose;
- pre-correction measured-decoded target;
- post-correction scheduled target;
- command–measurement translation and rotation gap;
- correction translation and rotation actually applied;
- source chunk ID and action index;
- action timestamp, observation timestamp, and execution time;
- IK result, joint command, skip/hold reason, and safety status.

At chunk boundaries, report:

- position step `||cmd[s] - cmd[s-1]||`;
- boundary velocity change
  `||(cmd[k]-cmd[k-1]) - (cmd[k-1]-cmd[k-2])||` near the splice;
- command–measurement gap before and after the transition;
- maximum joint velocity and acceleration;
- IK failures and safety rejections.

Do not use a global median-step ratio as proof that a discontinuity was eliminated.

## Validation gates

### 1. Offline fixed-chunk study

The corrected study must remain reproducible and compare identical policy chunks across
execution strategies. This gate establishes command-space behavior only.

### 2. Simulated robot control

Replay dataset images while generating state from simulated Piper FK. Test nominal
tracking, delay, slow servo response, joint saturation, and temporary blockage. Full and
bounded command anchoring must not allow unbounded command–measurement gap growth.

### 3. Real robot, no-contact workspace

Run at reduced Cartesian speed with conservative caps. Compare measured, full-command,
and bounded-decay modes using per-tick logs. Pause automatically on the hard gap limit.

### 4. Task trial

Only after the no-contact gate passes, evaluate task completion and contact behavior.
Interpolation is accepted when it reduces boundary acceleration without increasing IK
failures, hard-gap events, workspace violations, or task failure rate.

## Current evidence

In the corrected five-episode pure-delay study at six ticks of lag:

| strategy | boundary velocity change | boundary position step | GT drift |
|---|---:|---:|---:|
| SYNC replay | 16.83 mm/tick | 14.07 mm | 42.17 mm |
| UMI splice, unchanged targets | 14.10 mm/tick | 0.00 mm | 42.20 mm |
| full command re-anchor | **2.86 mm/tick** | **0.00 mm** | **35.32 mm** |

Full command re-anchoring is therefore a useful smoothness upper bound, but the test does
not contain blockage/contact dynamics and cannot establish it as the safe deploy mode.
The bounded-decay design is the next implementation and simulation candidate.

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# UMI 相对末端执行器轨迹插值

## 现状与建议

这是为 Piper UMI 相对末端执行器部署添加轨迹插值的推荐设计。它是一份设计规范，尚未成为已实现的部署功能。

使用**带边界、衰减式锚点修正的命令轨迹拼接**：

1. 策略状态与安全反馈继续基于实测 FK。
2. 每条替换轨迹都从当前已命令轨迹的值开始。
3. 丢弃时间戳已经过期的策略路点。
4. 仅以有界幅度将首批未来目标移向命令帧。
5. 在该 chunk 起始部分内将这一偏移衰减到零，恢复实测帧目标。
6. 当命令与实测误差超过硬性限制时，保持并请求恢复。

不要将无条件全命令重锚定作为默认方案。它在纯延迟离线实验中很平滑，但会移除跟踪误差修正，并且可能在物理机械臂被阻挡时仍继续推进计划。

修正后的实验与图表位于
`outputs/research_report/low_level_control_debug/umi_interp_experiment/umi_interp_experiment.md`。

## 为什么需要插值

策略预测一个相对位姿 chunk，全部以 chunk 起始时刻的实测位姿为基准表达。当前同步部署会将该 chunk 转换为绝对位姿并原样回放。当机器人滞后于其命令时，下一个以实测为锚的 chunk 可能起始于上一 chunk 最后一条命令之后（即落后于它）。因此替换队列会造成笛卡尔命令不连续。

UMI 通过保留带时间戳的命令轨迹来避免位置不连续。新计划只替换其未来部分；连接从拼接时刻当前命令轨迹的值开始，而不是将命令瞬移到实测 FK。

该机制保证位置连续（C0），但线性平移与 SLERP 不保证速度连续（C1）。如果第一个新的绝对目标远落后于命令，控制器仍可能在下一个 tick 中做出剧烈修正。有界锚点修正可以在不放弃整个 chunk 实测反馈的情况下处理该过渡。

## 位姿与动作契约

绝对 EE 位姿为：

```text
T = [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
```

刚体部分是一个 SE(3) 变换。Gripper 是单独的标量。

策略动作为：

```text
ΔT[i] = [dx, dy, dz, rot6d(6), gripper], i = 0 ... horizon-1
```

每个 `ΔT[i]` 都相对于同一个 chunk 起始实测位姿 `T_m`：

```text
T_measured_target[i] = T_m · ΔT[i]
```

目标并非由前一个预测动作积分得到。

对于当前的 UMI ACT checkpoint，数据集动作窗口为
`[-1, 0, ..., horizon-1]`；processor 消费 `-1`。因此，模型动作索引 0 对应观测时刻的当前位姿，而不是下一个控制步。

## 推荐算法

### 1. 维护当前命令轨迹

维护一个包含带时间戳绝对位姿的 `PoseTrajectoryInterpolator7D`：

- XYZ：线性插值；
- 旋转：四元数 SLERP，输出时以 axis-angle 表示；
- gripper：线性插值，并施加 gripper 速率限制；
- 轨迹之外的时间：钳制并保持其首/末位姿。

在控制时刻 `now` 定义：

```text
T_c = active_trajectory(now)
```

`T_c` 是在没有新策略结果到达时会发出的命令。它不是实测 FK，也不只是上一个 chunk 端点。

### 2. 为动作打时间戳并移除过期路点

对于在 `observation_time` 采集的观测，动作索引 `i` 的时间戳为：

```text
action_time[i] = observation_time + i · action_dt
```

仅接受满足以下条件的路点：

```text
action_time[i] > now + execution_latency_margin
```

在即时 30 Hz 推理下，这通常会移除动作 0。在存在真实推理延迟时，它可能移除更多动作。不要在实时实现中硬编码“总是恰好移除一个”。

如果没有未来路点保留下来，保持当前轨迹并请求新的计划。不要将最后一个过期目标安排为立即跳变。

### 3. 测量并约束锚点间隙

设实测 chunk 起始位姿为 `T_m`，当前命令为 `T_c`。计算它们的平移与测地旋转间隙：

```text
d = ||p_c - p_m||
θ = angle(R_m⁻¹ R_c)
```

构造有界命令参考 `T_b`：

```text
p_b = p_m + min(1, position_cap / d) · (p_c - p_m)
R_b = R_m · Exp(min(1, rotation_cap / θ) · Log(R_m⁻¹ R_c))
```

将 `d` 或 `θ` 为零的情况作为单位修正处理。平移必须按欧几里得范数设上限，旋转必须按 SO(3) 角度设上限——而不是裁剪 axis-angle 分量。

上限参数必须通过仿真与机器人日志调参。它们应远小于现有的 50 mm 紧急 EE 步长拒绝阈值；该阈值是最后手段的安全检查，而不是平滑目标。

### 4. 在未来目标上衰减修正

对于每个保留的未来索引 `i`，选择权重 `w[i]`，使其从 1 开始并在 `anchor_decay_steps` 内单调递减到 0：

```text
w[i] = max(0, 1 - (i - first_future_index) / anchor_decay_steps)
```

在 SE(3) 上插值参考位姿：

```text
p_ref[i] = lerp(p_m, p_b, w[i])
R_ref[i] = slerp(R_m, R_b, w[i])
T_target[i] = T_ref[i] · ΔT[i]
```

这让首批未来目标获得一些平滑的全命令锚定行为，然后让后续目标回到策略所预期的实测参考。gripper 值直接来自 `ΔT[i]`；它不受 EE 锚定的影响。

有用的 A/B 端点：

- `position_cap = rotation_cap = 0`：未改变的实测解码目标，等价于 UMI 风格拼接但不做命令重锚定；
- 无穷大上限且无衰减：完全命令重锚定，一个实验性上界；
- 有限上限加衰减：推荐的部署候选。

### 5. 将目标拼接到命令未来

使用 `schedule_waypoint`，按每个被接受的 `T_target[i]` 的原始 `action_time[i]` 进行安排，参数为：

```text
curr_time = now
last_waypoint_time = last accepted/scheduled waypoint time
```

调度器必须：

- 在 `now` 保持当前命令；
- 丢弃被取代的未来路点；
- 在适当情况下保留已接受的较早路点；
- 按严格递增的时间戳顺序追加新目标；
- 若两个目标共享同一时间戳，保留最新目标；
- 施加有限的笛卡尔平移与旋转速度限制。

每个控制 tick 对插值器求值一次，并将得到的绝对 EE 目标传入现有的 `EEBoundsAndSafety` 与 IK 流水线。

## 伪代码

```python
def accept_policy_chunk(relative_chunk, observation_time, now, measured_pose):
    command_pose = trajectory(now)

    gap = se3_difference(measured_pose, command_pose)
    if gap.position > hard_position_gap or gap.rotation > hard_rotation_gap:
        hold_and_require_recovery("tracking gap exceeded")
        return

    bounded_command_ref = cap_se3_reference(
        measured_pose,
        command_pose,
        position_cap=anchor_position_cap,
        rotation_cap=anchor_rotation_cap,
    )

    future = []
    for index, relative in enumerate(relative_chunk):
        target_time = observation_time + index * action_dt
        if target_time <= now + execution_latency_margin:
            continue

        weight = anchor_decay_weight(index, first_future_index, anchor_decay_steps)
        reference = interpolate_se3(measured_pose, bounded_command_ref, weight)
        target = compose_relative(reference, relative)
        future.append((target_time, target))

    if not future:
        request_replan_and_hold()
        return

    for target_time, target in future:
        trajectory = trajectory.schedule_waypoint(
            target,
            target_time,
            curr_time=now,
            last_waypoint_time=last_waypoint_time,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
        )
        last_waypoint_time = target_time
```

## 控制循环规则

- **实测状态保持实测。** 始终使用连续 FK 测量构造策略状态。绝不要将命令位姿喂入观测状态以掩盖滞后。
- **一个 chunk，一个实测基准。** 在应用有界修正之前，从该推理请求所采集的实测位姿解码所有相对目标。
- **使用单调时间。** 调度与延迟检查必须使用同一个单调时钟。
- **推理停顿期间不进行外推。** 保持最后一个轨迹位姿。
- **直接/手动运动时重置。** 暂停、起始位姿、安全位姿或 IK 重同步操作必须清除待处理路点，并从新的实测状态初始化单一位姿轨迹。
- **拒绝过期异步结果。** 在暂停/恢复或直接运动之前生成的结果，之后绝不能被接受。
- **不要对 axis-angle 向量求平均。** 使用 SLERP 或李群插值来混合旋转。

## 安全行为

平滑机制不得覆盖现有安全检查。

| 条件 | 要求行为 |
|---|---|
| 非有限或格式错误的目标 | 拒绝 chunk 并保持 |
| 没有非过期路点 | 保持并请求另一次推理 |
| 中等命令–测量间隙 | 应用有界、衰减式修正 |
| 间隙超过硬性位置/旋转限制 | 保持、清除未来、要求恢复/重同步 |
| IK 失败 | 保持最后有效的关节/EE 命令；不要将 provenance 推进为已执行 |
| EE 工作空间或步长违规 | 通过现有安全流水线拒绝/保持 |
| 暂停或直接移动 | 使排队中与进行中的轨迹失效 |

不得使用完全命令锚定来掩盖不断增大的跟踪间隙。在每次重规划时记录原始间隙与所应用的有界修正。

## 需要暴露的参数

```text
trajectory_interpolation: bool
anchor_mode: measured | command | bounded_decay
anchor_position_cap_m: float
anchor_rotation_cap_rad: float
anchor_decay_steps: int
execution_latency_margin_s: float
max_cartesian_speed_m_s: float
max_rotation_speed_rad_s: float
hard_tracking_gap_m: float
hard_tracking_rotation_gap_rad: float
```

`bounded_decay` 应只在下列验证门通过后才成为推荐模式。保留 `measured` 与 `command` 模式用于受控 A/B 实验。

## 必需的日志记录

每个 tick：

- 实测 FK 位姿；
- 插值后的命令位姿；
- 修正前的实测解码目标；
- 修正后的已调度目标；
- 命令–测量平移与旋转间隙；
- 实际应用的修正平移与旋转；
- 源 chunk ID 与动作索引；
- 动作时间戳、观测时间戳与执行时间；
- IK 结果、关节命令、跳过/保持原因与安全状态。

在 chunk 边界处报告：

- 位置步长 `||cmd[s] - cmd[s-1]||`；
- 拼接附近的边界速度变化
  `||(cmd[k]-cmd[k-1]) - (cmd[k-1]-cmd[k-2])||`；
- 过渡前后的命令–测量间隙；
- 最大关节速度与加速度；
- IK 失败与安全拒绝。

不要用全局中值步长比作为不连续已被消除的证据。

## 验证门

### 1. 离线固定 chunk 研究

修正后的研究必须保持可复现，并在不同执行策略之间比较相同的策略 chunk。该门仅确立命令空间行为。

### 2. 仿真机器人控制

回放数据集图像，同时从仿真 Piper FK 生成状态。测试标称跟踪、延迟、慢伺服响应、关节饱和与临时阻塞。完全与有界命令锚定都不得允许命令–测量间隙无界增长。

### 3. 真实机器人，无接触工作空间

以降低的笛卡尔速度与保守上限运行。使用逐 tick 日志比较 measured、full-command 与 bounded-decay 模式。在硬间隙限制处自动暂停。

### 4. 任务试验

只有在无接触门通过后，才评估任务完成度与接触行为。当插值在不增加 IK 失败、硬间隙事件、工作空间违规或任务失败率的情况下降低边界加速度时，才接受插值。

## 当前证据

在修正后的五回合纯延迟研究中，滞后为六个 tick：

| 策略 | 边界速度变化 | 边界位置步长 | GT 漂移 |
|---|---:|---:|---:|
| SYNC 回放 | 16.83 mm/tick | 14.07 mm | 42.17 mm |
| UMI 拼接，目标不变 | 14.10 mm/tick | 0.00 mm | 42.20 mm |
| 完全命令重锚定 | **2.86 mm/tick** | **0.00 mm** | **35.32 mm** |

因此，完全命令重锚定是一个有用的平滑上界，但该测试不包含阻塞/接触动力学，无法确立其作为安全部署模式。有界衰减设计是下一个实现与仿真候选。
