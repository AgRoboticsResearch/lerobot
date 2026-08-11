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
