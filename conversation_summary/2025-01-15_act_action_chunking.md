---
conversation_id: act_action_chunking_2025-01-15
created: 2025-01-15
tags: #lerobot #act #action-chunking #robotics
---

# Conversation Summary - 2025-01-15

**Topic**: ACT Policy Action Chunking in LeRobot

## Overview
Deep-dived into how action chunking works in LeRobot's ACT policy, comparing the standard `lerobot_eval.py` with the custom `deploy_relative_ee_so101.py` deployment script. Explored `policy.select_action()` internals, the relationship between `chunk_size` and `n_action_steps`, temporal ensembling, and the `actions_processed_in_chunk` tracking mechanism.

## Topics
- Action chunking in ACT policy
- `policy.select_action()` internal queue management
- Difference between `chunk_size` and `n_action_steps`
- Temporal ensembling (optional, off by default)
- UMI-style relative end-effector action handling

## Key Concepts

### How `select_action()` Works
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Every step:  select_action() called                                     │
│ Inside:  Queue management determines if inference happens               │
└─────────────────────────────────────────────────────────────────────────┘

Step 1:  Queue empty → INFERENCE (predict chunk_size actions)
         → Queue first n_action_steps, discard rest
         → popleft() → return action[0]

Step 2-N:  Queue not empty → NO INFERENCE
           → popleft() → return next action

Step N+1:  Queue empty → INFERENCE (new chunk)
```

### Parameter Relationship
```
chunk_size = 100  # What model predicts (fixed by architecture)
n_action_steps = 50  # How many are executed/queued (can be smaller)

Inference frequency: Every n_action_steps steps
Actions discarded: chunk_size - n_action_steps
```

### Temporal Ensembling
- **Implemented**: Yes, in `ACTTemporalEnsembler` class
- **Default**: OFF (`temporal_ensemble_coeff = None`)
- **When enabled**: Requires `n_action_steps = 1` (inference every step)
- **Formula**: `w_i = exp(-coeff * i)` - exponential weighting

### Deployment Script Tracking
The `actions_processed_in_chunk` variable tracks chunk boundaries because:
1. Policy's `_action_queue` is private
2. Need to capture `chunk_base_pose` at chunk start for UMI-style relative EE
3. All actions in a chunk use same base pose (not chained)

```
UMI-style semantics:
  Step 0: chunk_base_pose = FK(actual_joints)
  Step 0-49: target = chunk_base_pose @ rel_action[i]
  Step 50: NEW chunk_base_pose captured (reset)
```

## Files Referenced

| File | Lines | Description |
|------|-------|-------------|
| `src/lerobot/policies/act/modeling_act.py` | 99-122 | `select_action()` implementation with queue logic |
| `src/lerobot/policies/act/modeling_act.py` | 165-253 | `ACTTemporalEnsembler` class |
| `src/lerobot/policies/act/configuration_act.py` | 128-129 | `temporal_ensemble_coeff` default value |
| `src/lerobot/policies/act/configuration_act.py` | 46-50 | `chunk_size` vs `n_action_steps` documentation |
| `examples/so101_relative_ee/deploy_relative_ee_so101.py` | 406-523 | Action handling loop with chunk tracking |
| `examples/so101_relative_ee/deploy_relative_ee_so101.py` | 426-429 | Chunk boundary detection |
| `src/lerobot/scripts/lerobot_eval.py` | 176-177 | Standard eval pattern |

## Key Commands

### Training with Custom Action Steps
```bash
# Default: chunk_size=100, n_action_steps=100
lerobot-train --policy.type=act --policy.n_action_steps=50 --policy.chunk_size=100

# Enable temporal ensembling (requires n_action_steps=1)
lerobot-train --policy.type=act --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1
```

### Evaluation
```bash
# Standard eval
lerobot-eval --policy.path=outputs/train/checkpoints/last/pretrained_model --env.type=aloha

# With custom action steps
lerobot-eval --policy.path=... --eval.n_action_steps=50
```

## Decisions & Insights

1. **Calling `select_action()` every step is correct** - The policy's internal queue manages whether inference happens. Don't skip calls.

2. **`chunk_size` vs `n_action_steps`**:
   - `chunk_size`: Model output dimension (fixed by trained architecture)
   - `n_action_steps`: How many predicted actions to execute (adjustable at inference)
   - Setting `n_action_steps < chunk_size` = more frequent re-prediction for better closed-loop control

3. **UMI-style action semantics** - All actions in a chunk are relative to the chunk's starting pose, not chained sequentially. This prevents error accumulation within a chunk.

4. **Temporal ensembling not needed by default** - The simple queue-based approach works well; temporal ensembling adds computational overhead (inference every step) and may not improve performance.

## Comparison: Your Script vs Standard ACT

| Aspect | Standard ACT (`lerobot_eval.py`) | Your Approach |
|--------|----------------------------------|---------------|
| Action type | Absolute joint positions | Relative EE pose |
| Chunk handling | Policy internal (transparent) | Manual tracking with `actions_processed_in_chunk` |
| Base reference | Not needed | `chunk_base_pose` captured at chunk start |
| Action chaining | Each action independent | All actions relative to same `chunk_base_pose` |

## Tags
#lerobot #act #action-chunking #robotics #umi #temporal-ensembling
