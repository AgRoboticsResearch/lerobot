---
conversation_id: relative-ee-fps-fix-2025-01-24
created: 2025-01-24
duration: ~2 hours
tags: #lerobot #fps-mismatch #relative-ee #act #umi #debugging
---

# Conversation Summary - 2025-01-24

## Overview

Debugged and fixed FPS mismatch issues in the Relative EE training and inference pipeline for LeRobot. The core issue was that training used `ds_meta.fps=30Hz` while inference scripts defaulted to 10Hz, causing a mismatch in action horizons. This led to incorrect evaluation (model was evaluated on 9.9s predictions vs 3.3s training horizon).

## Topics
- Relative EE (End-Effector) dataset implementation
- FPS mismatch between training and inference
- UMI (Universal Manipulation Interface) style relative pose transformations
- ACT policy training and debugging

## Tasks Completed

### 1. Identified FPS Mismatch Bug
- **Problem**: Debug script used `fps = getattr(policy.config, 'fps', 10)` defaulting to 10Hz
- **Training**: Actually used `ds_meta.fps=30Hz` via `resolve_delta_timestamps()`
- **Impact**: Model trained for 3.3s horizon, but debug evaluated on 9.9s horizon → 107mm vs 30mm displacement mismatch

### 2. Fixed Training Script (`train_relative_ee.py:71-76`)
```python
# Store fps in policy config for inference use
if not hasattr(cfg.policy, 'fps'):
    cfg.policy.fps = ds_meta.fps
    logging.info(f"Set policy.fps = {cfg.policy.fps} Hz (from dataset metadata)")
```

### 3. Fixed Debug Inference Script (`debug_relative_ee_inference.py:475-479`)
```python
fps = getattr(policy.config, 'fps', 30)  # Changed from 10 to 30
logger.info(f"  Using fps={fps} Hz ({'from policy config' if hasattr(policy.config, 'fps') else 'default'})")
```

### 4. Fixed Deployment Scripts (4 files)
Changed `FPS = 10` to `FPS = 30` in all deployment scripts:
- `examples/so101_relative_ee/deploy_relative_ee_so101.py:91`
- `examples/so101_relative_ee/deploy_relative_ee_so101_static.py:94`
- `placo_sim/so101_deploy_real.py:73`
- `placo_sim/so101_deploy_sim.py:77`

## Files Modified

| File | Change |
|------|--------|
| `train_relative_ee.py` | Added `cfg.policy.fps = ds_meta.fps` to save training fps |
| `debug_relative_ee_inference.py` | Changed default fps from 10 to 30 |
| `deploy_relative_ee_so101.py` | FPS: 10 → 30 |
| `deploy_relative_ee_so101_static.py` | FPS: 10 → 30 |
| `so101_deploy_real.py` | FPS: 10 → 30 |
| `so101_deploy_sim.py` | FPS: 10 → 30 |

## Key Analysis Insights

### Camera fps vs Control fps
| FPS | Value | Purpose |
|-----|-------|---------|
| Camera fps | 25 Hz | Image capture rate (from `lerobot-record`) |
| Dataset/control fps | 30 Hz | State-action pair frequency (what model actually uses) |

**Key point**: The model learns temporal dynamics based on control fps (30Hz), not camera fps (25Hz). Using wrong fps creates mismatched action horizons.

### Why Joint Script Didn't Have This Bug
The joint-based debug script (`debug_act_so101_inference.py:625`) hardcodes `fps = 30` which happened to match the control frequency.

### Left for Later
- `obs_state_horizon` parameter is collected but never used (due to ACT's `n_obs_steps=1` limitation)
- This parameter should either be removed or properly documented

## Decisions & Insights
- **Always use `ds_meta.fps` for consistency** - this is what the model was trained with
- **Camera fps doesn't affect model predictions** - only affects how many images are stored
- **Existing checkpoints** need to be retrained to have `policy.fps` saved, but will work with 30Hz fallback

## Tags
#lerobot #fps-mismatch #relative-ee #act #umi #debugging
