# SmolVLA UMI EE-Pose Preprocessor Pipeline

> Dataset: `test_ee_dataset` (334 frames, 2 episodes)
> Verified: 2026-05-03

---

## 1. Architecture Overview

```
Dataset (parquet)
  │ action: 51-timestep absolute EE-pose [t=-1, t=0, ..., t=49]
  │ observation.state: 2-timestep absolute EE-pose [t=-1, t=0]
  │ observation.images.color: 1 frame [t=0] 480×640
  │
  ▼ Preprocessor (8 steps)
  │  1. RenameObservations        — identity
  │  2. AddBatchDimension         — unsqueeze(0)
  │  3. NewLineTask               — append '\n' to task string
  │  4. Tokenizer                 — tokenize language instruction
  │  5. DeviceProcessor           — move to GPU
  │  6. DeriveStateFromAction     — extract state, strip action (51→50)
  │  7. RelativeActions           — action -= current_state (pos+rot only)
  │  8. RelativeState             — state -= current_state, flatten (2,7)→(14,)
  │  9. Normalizer                — (x - mean) / std
  │
  ▼ Model
     state: 14D → pad to 32D
     action: 7D → pad to 32D
     images: 480×640 → pad to 512×512, [0,1]→[-1,1]
```

## 2. Dim Names (7D EE-pose)

| Index | Name | Type | Unit |
|---|---|---|---|
| 0 | `ee.x` | position | meters |
| 1 | `ee.y` | position | meters |
| 2 | `ee.z` | position | meters |
| 3 | `ee.wx` | rotation (axis-angle) | radians |
| 4 | `ee.wy` | rotation (axis-angle) | radians |
| 5 | `ee.wz` | rotation (axis-angle) | radians |
| 6 | `ee.gripper_pos` | gripper | normalized [0,1] |

## 3. Stage-by-Stage Data Flow

### Stage 0: Raw Dataset Load

Dataset loaded with `delta_timestamps`:
- `observation.state`: [-0.033s, 0.0s] → 2 timesteps
- `action`: [-0.033s, 0.0s, 0.033s, ..., 1.6s] → 51 timesteps (includes t=-1 lead-in)
- `observation.images.color`: [0.0s] → 1 frame

### Stage 1: DeriveStateFromAction

```python
# Input:  action (1, 51, 7)
# Output: state (1, 2, 7) = action[:, :2, :]       — extracted from action column
#         action (1, 50, 7) = action[:, 1:, :]       — leading timestep stripped
```

Purpose: In UMI-style datasets, the action column serves double duty — the first 2 timesteps encode the current + previous EE-pose (state), the remaining 50 timesteps encode the action chunk.

`action_is_pad` is also stripped from both `complementary_data` and top-level to maintain consistent shapes.

### Stage 2: RelativeActions

```python
# Subtract current EE-pose from all action timesteps (pos+rot only, gripper excluded):
action[:, :, :6] -= current_state[:, :6]       # current_state = state[:, -1, :]

# mask: [True, True, True, True, True, True, False]
#       pos+rot → relative    gripper → absolute (unchanged)
```

**Key invariant: t=0 pos+rot is always ZERO.**
This is the mathematical identity — at the start of prediction, the first action equals the current state, so the relative offset is zero.

**Gripper is excluded** from relative conversion. Gripper values are absolute position commands in [0,1].

### Stage 3: RelativeState

```python
# Convert state to offsets from current timestep:
state[:, :, :6] -= current_state[:, :6]        # same reference as action

# Flatten: (1, 2, 7) → (1, 14)
# [t=-1 xyz+rot(6) + t=-1 gripper(1)] + [t=0 xyz+rot(6) + t=0 gripper(1)]
```

**Key invariant: t=0 pos+rot is always ZERO in relative space.** (state[t=0] - current = 0)

**At inference time**: `RelativeStateProcessorStep` buffers the previous observation and stacks `[prev, cur]` to produce the same 2-timestep format the model was trained on.

### Stage 4: Normalize

```python
# Z-score normalization:
action_norm = (action_rel - action_mean) / (action_std + eps)
state_norm  = (state_rel - state_mean) / (state_std + eps)
```

**Normalization mode: MEAN_STD** (z-score, NOT min-max bounded).

Values typically fall in `[-2, 2]` with occasional excursions to ±3. This is correct and expected — neural networks handle this fine.

The t=0 state dims (indices 7-12) have `mean=0, std≈0`. The normalizer handles this via `denom = std + eps` (eps=1e-8), producing `0/1e-8 = 0` for those dims.

## 4. Example Data (Frame 100)

### Current State

```
ee.x=0.0784  ee.y=-0.1969  ee.z=0.1754  ee.wx=0.2695  ee.wy=0.0271  ee.wz=0.0514  gripper=0.9953
```

### Action — Selected Timesteps

| t | Stage | ee.x | ee.y | ee.z | ee.wx | ee.wy | ee.wz | gripper |
|---|-------|-------|-------|-------|-------|-------|-------|---------|
| 0 | RAW | 0.0784 | -0.1969 | 0.1754 | 0.2695 | 0.0271 | 0.0514 | 0.9953 |
| 0 | REL | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.9953 |
| 0 | NORM | -0.8133 | 0.9819 | -0.9525 | -0.7266 | -0.2784 | -0.2067 | 0.5514 |
| 5 | RAW | 0.0749 | -0.2033 | 0.1854 | 0.2804 | 0.0284 | 0.0100 | 0.9707 |
| 5 | REL | -0.0035 | -0.0065 | 0.0100 | 0.0109 | 0.0014 | -0.0414 | 0.9707 |
| 5 | NORM | -0.9963 | 0.8362 | -0.7683 | -0.4723 | -0.2542 | -1.7141 | 0.4798 |
| 25 | RAW | 0.0659 | -0.2130 | 0.1979 | 0.2911 | 0.0578 | -0.0277 | 0.3597 |
| 25 | REL | -0.0125 | -0.0162 | 0.0224 | 0.0216 | 0.0307 | -0.0791 | 0.3597 |
| 25 | NORM | -1.4673 | 0.6168 | -0.5381 | -0.2236 | 0.2632 | -3.0866 | -1.3004 |
| 49 | RAW | 0.0635 | -0.2141 | 0.1985 | 0.3133 | 0.0697 | -0.0146 | 0.0000 |
| 49 | REL | -0.0149 | -0.0173 | 0.0231 | 0.0438 | 0.0426 | -0.0659 | 0.0000 |
| 49 | NORM | -1.5898 | 0.5921 | -0.5264 | 0.2918 | 0.4720 | -2.6083 | -2.3485 |

### Statistics (all 50 timesteps of this frame)

| Dim | RAW mean | RAW range | REL mean | NORM mean | NORM std |
|-----|----------|-----------|----------|-----------|----------|
| ee.x | 0.0687 | [0.0635, 0.0784] | -0.0097 | -1.3209 | 0.2272 |
| ee.y | -0.2109 | [-0.2151, -0.1969] | -0.0141 | 0.6650 | 0.1101 |
| ee.z | 0.1941 | [0.1754, 0.1997] | 0.0187 | -0.6078 | 0.1120 |
| ee.wx | 0.2978 | [0.2695, 0.3208] | 0.0283 | -0.0688 | 0.3550 |
| ee.wy | 0.0538 | [0.0264, 0.0757] | 0.0268 | 0.1929 | 0.2965 |
| ee.wz | -0.0124 | [-0.0302, 0.0514] | -0.0638 | -2.5290 | 0.6642 |
| gripper | 0.4426 | [0.0000, 0.9953] | 0.4426 | -1.0590 | 1.1463 |

## 5. Postprocessor (Inference Reverse Path)

```
Model output: normalized relative action (7D)
  │
  ▼ Postprocessor
  │  1. Unnormalizer   — action = action * std + mean
  │  2. AbsoluteActions — action[:,:6] += cached_current_state[:,:6]
  │  3. DeviceProcessor — move to CPU
  │
  ▼ Output: absolute EE-pose (7D)
```

`AbsoluteActionsProcessorStep` reads the cached state from its paired `RelativeActionsProcessorStep`. At inference, `RelativeActionsProcessorStep` caches the current observation.state during preprocessing.

## 6. Round-Trip Verification

### Test A: Action Round-Trip

```
ABS action → preprocess(rel+norm) → postprocess(unnorm+abs) → ABS action'
```

- 9 frames tested (0, 5, 10, 50, 100, 150, 200, 250, 300)
- Max error: **8.75e-08** (float32 precision)
- Result: **ALL PASSED** ✓

### Test B: State Round-Trip

```
ABS state → relative → flatten → normalize → unnormalize → unflatten → absolute
```

- 9 frames tested
- Max error: **4.47e-08**
- Result: **ALL PASSED** ✓

## 7. Config Requirements

### SmolVLAConfig

```python
SmolVLAConfig(
    derive_state_from_action=True,       # UMI-style state from action column
    use_relative_actions=True,           # ← MUST be set explicitly!
    use_relative_state=True,             # auto-set by derive_state_from_action
    state_obs_steps=2,                   # auto-set by derive_state_from_action
    relative_exclude_joints=["gripper"],        # gripper stays absolute in actions
    relative_exclude_state_joints=["gripper"],  # gripper stays absolute in state
)
```

**Common pitfall**: `derive_state_from_action=True` auto-sets `use_relative_state=True` and `state_obs_steps=2`, but does NOT auto-set `use_relative_actions=True`. Without it, `RelativeActionsProcessorStep` is disabled and actions remain absolute while stats are computed in relative space → normalization mismatch.

### Stats (computed via `recompute_stats`)

```python
recompute_stats(
    ds, num_workers=2,
    relative_action=True,              # compute action stats in relative space
    relative_exclude_joints=["gripper"],
    relative_state=True,               # compute state stats in relative space
    relative_exclude_state_joints=["gripper"],
    state_obs_steps=2,
    derive_state_from_action=True,     # use action column as source for state stats
)
```

Stats output:
- `action`: 7D relative (mean centered near 0 for pos+rot)
- `observation.state`: 14D relative (first 7 = t=-1 offset, last 7 = t=0 which is all zeros for pos+rot)

## 8. Key Invariants

1. **t=0 action REL = ZERO** for pos+rot dims (by definition: `action[t=0] == current_state`)
2. **t=0 state REL = ZERO** for pos+rot dims (by definition: `state[t=0] - current = 0`)
3. **Gripper dim** is excluded from all relative conversions — always absolute [0,1]
4. **State 14D format**: `[t=-1[7], t=0[7]]` where t=0 pos+rot (dims 7-12) have mean=0, std≈0
5. **Normalize is z-score**, not min-max — values naturally range [-3, 3], not [-1, 1]
6. **Zero-std handling**: `denom = std + 1e-8` prevents division by zero for t=0 dims

## 9. Trace Command

```bash
cd /home/hls/codes/lerobot_piper_sroi
uv run --directory lerobot python /home/hls/codes/lerobot_piper_sroi/lerobot/trace_preprocessor.py
```

## 10. Round-Trip Test Command

```bash
cd /home/hls/codes/lerobot_piper_sroi
uv run --directory lerobot python /home/hls/codes/lerobot_piper_sroi/lerobot/test_preprocessor_roundtrip.py
```
