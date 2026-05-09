# pi0 vs SmolVLA: Observation Handling & Relative Actions Design

## What is "Multi-step Observation"?

Most robot policies look at multiple past frames at each decision step, not just the current frame:

```
Timeline:  ── t-2 ── t-1 ── t ── t+1 ── t+2 ── ...
                        ↑
                   current decision point

Single-step obs (n_obs_steps=1):
  Input: [state_t]

Multi-step obs (n_obs_steps=3):
  Input: [state_{t-2}, state_{t-1}, state_t]
  Model can infer velocity, acceleration, motion trends
```

## Policy Comparison — Who Actually Uses Multi-step Obs?

```
Policy                n_obs_steps  observation_delta_indices     state shape
────────────────────  ───────────  ──────────────────────────   ──────────
DiffusionPolicy       2            [1-n_obs, ..., -1] = [-1, 0]  (B, 2, dim)
ACT                   1            None                          (B, dim)
pi0                   1            None                          (B, dim)
SmolVLA               1            [0]                           (B, 1, dim)
```

## Why pi0 Returns None, SmolVLA Returns [0]

Both use n_obs_steps=1, but different code paths for data loading.

### The Critical Code: `dataset_reader.py:228`

```python
result[key] = torch.stack(self.hf_dataset[key][relative_indices])
```

`torch.stack` always adds a dimension at dim 0:
- Query 1 frame → `stack([tensor(7)])` → shape `(1, 7)`
- Query 50 frames → `stack([tensor(7), ...])` → shape `(50, 7)`
- No query → direct `tensor(7)` → shape `(7,)`

### pi0: Direct Path (no stack)

```python
# configuration_pi0.py
@property
def observation_delta_indices(self) -> None:
    return None    # ← No delta_timestamps for observations

# Result: dataset returns raw state shape (7,)
# After DataLoader batch: (B, 7) — clean 2D
```

### SmolVLA: Unified Path (always stack)

```python
# configuration_smolvla.py
@property
def observation_delta_indices(self) -> list:
    return [0]     # ← Goes through delta_timestamps path

# Result: dataset stacks 1 frame → (1, 7)
# After DataLoader batch: (B, 1, 7) — with obs_steps dimension
```

### Why SmolVLA Uses [0]

SmolVLA's design supports multi-step observations (n_obs_steps parameter). Returning `[0]` ensures the dataset always produces `(obs_steps, dim)` shaped observations. The model internally uses `[:, -1, :]` to extract the last obs step, keeping code consistent for any n_obs_steps value.

pi0 doesn't need this because its architecture directly assumes `(B, dim)` state — no obs_steps concept.

## Full Data Flow: pi0 UMI EE-pose Mode

### Step 1: resolve_delta_timestamps

```
pi0 observation_delta_indices = None
    → No delta_timestamps for observations
    → delta_timestamps = {"action": [0.0, 0.033, ...]} only

SmolVLA observation_delta_indices = [0]
    → delta_timestamps = {
        "observation.images.camera": [0.0],
        "observation.state":         [0.0],
        "action": [0.0, 0.033, ...]
    }
```

### Step 2: Dataset.__getitem__

```
pi0 — no delta_timestamps for obs:
    state = item["observation.state"]     → tensor(7,)     ← direct

SmolVLA — delta_timestamps [0.0]:
    state = torch.stack([item["observation.state"]])
                                       → tensor(1, 7)   ← stacked
```

### Step 3: After DataLoader batch

```
pi0:     state = (B, 7)       ← 2D
SmolVLA: state = (B, 1, 7)    ← 3D with obs_steps

Both:    action = (B, 50, 7)  ← both stack 50 frames
```

### Step 4: Model prepare_state

```python
# pi0 — modeling_pi0.py:1236
def prepare_state(self, batch):
    state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
    #                  ↑ already (B, 7), pad to (B, 32)
    return state

# SmolVLA — modeling_smolvla.py:486
def prepare_state(self, batch):
    state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
    #                                  ↑ (B, 1, 7) → (B, 7), then pad to (B, 32)
    state = pad_vector(state, self.config.max_state_dim)
    return state
```

### Step 5: Where State Goes in the Model

```
                pi0                              SmolVLA
        ──────────────────                ──────────────────

state → Action Expert (suffix)          state → VLM (prefix)

        [state, action,                  [state, image,
         timestep]                        language]

        VLM NEVER sees state ← key       VLM CAN see state ← key
        architectural difference          architectural difference
```

## The Bug We Hit

```
Training batch with SmolVLA:
    action = (8, 50, 7)    ← 50-step action chunk
    state  = (8, 1, 7)     ← SmolVLA obs_steps=1

to_relative_actions(action, state, mask):
    state_offset = state[..., :dims] * mask     → (8, 1, 7)
    state_offset = state_offset.unsqueeze(-2)    → (8, 1, 1, 7)  ← extra dim!

    actions[..., :dims] -= state_offset
    #  (8, 50, 7)          (8, 1, 1, 7) → broadcast fails → [8, 8, 50, 7]
```

Why pi0 never hits this: state is `(B, 7)` (2D), `unsqueeze(-2)` → `(B, 1, 7)`, broadcasts correctly with `(B, 50, 7)`.

## The Fix

```python
# relative_action_processor.py — squeeze 3D state before relative conversion
if state.ndim == 3:
    state = state[:, -1, :]   # (B, 1, 7) → (B, 7), take last obs step
```

## Summary Table

```
                        pi0                            SmolVLA
──────────────────────  ──────────────────────────     ─────────────────────────
obs_delta_indices       None                           [0]
delta_timestamps        No observation                 Has observation
Dataset state           Direct item[key]               torch.stack([item[key]])
Single sample shape     (7,)                           (1, 7)
Batch shape             (B, 7)                         (B, 1, 7)
prepare_state           Direct pad                     Squeeze then pad
Final model input       (B, 32)                        (B, 32) ← same!
State destination       Action Expert (suffix)         VLM (prefix)
```

Core conclusion: Both models receive the same `(B, 32)` padded state tensor. The difference is only in the intermediate data loading path — pi0 skips stack, SmolVLA stacks then squeezes. The real architectural difference is where state goes inside the model (expert vs VLM).
