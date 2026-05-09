# SmolVLA UMI EE-Pose Training Pipeline

> Dataset: `test_ee_dataset` (334 frames, 2 episodes, 30fps)
> Verified: 2026-05-03

---

## 1. Training Pipeline

```
                              ┌──────────────────────┐
                              │   test_ee_dataset     │
                              │   2 episodes          │
                              │   334 frames          │
                              │   30 fps              │
                              └──────────┬───────────┘
                                         │
                    delta_timestamps:    │
                    ┌────────────────────┼────────────────────┐
                    │                                                    │
                    ▼                                                    ▼
     ┌──────────────────────────┐                   ┌──────────────────────────┐
     │  observation.state        │                   │  action                   │
     │  [-0.033s, 0.0s]         │                   │  [-0.033s, 0.0s,          │
     │  2 timesteps             │                   │    0.033s ... 1.6s]       │
     │  shape: (2, 7)           │                   │  51 timesteps             │
     │                          │                   │  shape: (51, 7)           │
     │  ┌───────────────────┐   │                   │                           │
     │  │ t=-1: [x,y,z,wx, │   │                   │  ┌──────────────────────┐  │
     │  │        wy,wz,grip]│   │                   │  │ t=-1: [ee-pose @-1]  │  │
     │  │ t=0:  [x,y,z,wx, │   │                   │  │ t=0:  [ee-pose @0]   │  │
     │  │        wy,wz,grip]│   │                   │  │ t=1:  [ee-pose @1]   │  │
     │  └───────────────────┘   │                   │  │ ...                   │  │
     └──────────┬───────────────┘                   │  │ t=49: [ee-pose @49]  │  │
                │                                   │  └──────────────────────┘  │
                │                                   └──────────┬───────────────┘
                │                                              │
                │  ┌───────────────────────────────────────────┘
                │  │
                ▼  ▼
     ╔══════════════════════════════════════════════════════════════════╗
     ║                     P R E P R O C E S S O R                      ║
     ╠══════════════════════════════════════════════════════════════════╣
     ║                                                                  ║
     ║  ① RenameObservations  ── identity (no-op)                      ║
     ║                                                                  ║
     ║  ② AddBatchDimension   ── unsqueeze(0): add batch dim           ║
     ║                                                                  ║
     ║  ③ NewLineTask         ── task += '\n'                          ║
     ║                                                                  ║
     ║  ④ Tokenizer           ── tokenize language instruction         ║
     ║                                                                  ║
     ║  ⑤ DeviceProcessor     ── CPU → CUDA                            ║
     ║                                                                  ║
     ║  ┌───────────────────────────────────────────────────────────┐  ║
     ║  │ ⑥ DeriveStateFromAction                                    │  ║
     ║  │                                                            │  ║
     ║  │   action[:, :2, :]  ──────────────►  observation.state     │  ║
     ║  │   (1, 51, 7)                         (1, 2, 7) ABS         │  ║
     ║  │                                                            │  ║
     ║  │   action[:, 1:, :] ──────────────►  action                 │  ║
     ║  │   (1, 51, 7)                         (1, 50, 7) ABS        │  ║
     ║  │                                                            │  ║
     ║  │   action_is_pad[:, 1:] ─────────►  action_is_pad           │  ║
     ║  │   (1, 51)                            (1, 50)               │  ║
     ║  └───────────────────────────────────────────────────────────┘  ║
     ║                                                                  ║
     ║  ┌───────────────────────────────────────────────────────────┐  ║
     ║  │ ⑦ RelativeActionsProcessorStep                            │  ║
     ║  │                                                            │  ║
     ║  │   current_state = state[:, -1, :]    # (1, 7)             │  ║
     ║  │                                                            │  ║
     ║  │   action[:, :, :6] -= current_state[:, :6]                 │  ║
     ║  │   ┌──────────────────────────────────────┐                │  ║
     ║  │   │  mask = [T,T,T,T,T,T, F]             │ ← gripper      │  ║
     ║  │   │  pos+rot → relative    gripper stays │   excluded     │  ║
     ║  │   │  action[t=0] ≡ [0,0,0,0,0,0, grip]  │   from rel     │  ║
     ║  │   └──────────────────────────────────────┘                │  ║
     ║  │                                                            │  ║
     ║  │   CACHES _last_state for AbsoluteActions step              │  ║
     ║  └───────────────────────────────────────────────────────────┘  ║
     ║                                                                  ║
     ║  ┌───────────────────────────────────────────────────────────┐  ║
     ║  │ ⑧ RelativeStateProcessorStep                              │  ║
     ║  │                                                            │  ║
     ║  │   state[:, :, :6] -= current_state[:, :6]                 │  ║
     ║  │                                                            │  ║
     ║  │   state[t=-1]: [Δx, Δy, Δz, Δwx, Δwy, Δwz, grip_abs]     │  ║
     ║  │   state[t=0]:  [ 0,  0,  0,   0,   0,   0,  grip_abs]    │  ║
     ║  │                                                            │  ║
     ║  │   flatten: (1, 2, 7) ──────────► (1, 14)                  │  ║
     ║  │   ┌──────────────────────────────────────────────────┐    │  ║
     ║  │   │ [t=-1[0..6], t=0[0..6]]                         │    │  ║
     ║  │   │  first 7D = velocity     last 7D = mostly zeros │    │  ║
     ║  │   └──────────────────────────────────────────────────┘    │  ║
     ║  └───────────────────────────────────────────────────────────┘  ║
     ║                                                                  ║
     ║  ┌───────────────────────────────────────────────────────────┐  ║
     ║  │ ⑨ NormalizerProcessorStep                                 │  ║
     ║  │                                                            │  ║
     ║  │   STATE:  (state_14d - state_mean) / (state_std + 1e-8)   │  ║
     ║  │   ACTION: (action_7d - action_mean) / (action_std + 1e-8) │  ║
     ║  │                                                            │  ║
     ║  │   Stats from recompute_stats():                           │  ║
     ║  │   • action: 7D relative stats (mean≈0 for pos+rot)        │  ║
     ║  │   • state:  14D relative stats (source=action column)     │  ║
     ║  │   • t=0 pos+rot: mean=0, std≈0 → normalized to 0.0       │  ║
     ║  └───────────────────────────────────────────────────────────┘  ║
     ║                                                                  ║
     ╚══════════════════════════════════════════════════════════════════╝
                                         │
                                         │ preprocessed batch
                                         ▼
     ╔══════════════════════════════════════════════════════════════════╗
     ║                        M O D E L                                 ║
     ╠══════════════════════════════════════════════════════════════════╣
     ║                                                                  ║
     ║  prepare_images()          prepare_state()      prepare_action() ║
     ║  ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐    ║
     ║  │ 480×640 → 512²  │    │ 14D → 32D    │    │ 7D → 32D     │    ║
     ║  │ [0,1] → [-1,1]   │    │ (zero-pad)   │    │ (zero-pad)   │    ║
     ║  │ (SigLIP norm)    │    └──────────────┘    └──────────────┘    ║
     ║  └─────────────────┘                                            ║
     ║                                                                  ║
     ║        ┌─────────────────────────────────────────────────┐      ║
     ║        │  SmolVLM2-500M-Video-Instruct Backbone           │      ║
     ║        │  ┌──────────────┐     ┌────────────────────┐     │      ║
     ║        │  │ Vision       │     │ Language Model     │     │      ║
     ║        │  │ Encoder      │     │                    │     │      ║
     ║        │  │ (frozen)     │     │ lang_tokens        │     │      ║
     ║        │  │              │     │ + state_proj       │     │      ║
     ║        │  │ image ──►    │     │ ────► prefix       │     │      ║
     ║        │  │ features     │     │                    │     │      ║
     ║        │  └──────────────┘     └─────────┬──────────┘     │      ║
     ║        │                                  │                │      ║
     ║        │                      ┌───────────▼───────────┐   │      ║
     ║        │                      │  Action Expert         │   │      ║
     ║        │                      │  (trained from scratch)│   │      ║
     ║        │                      │                        │   │      ║
     ║        │                      │  cross-attn to         │   │      ║
     ║        │                      │  VLM prefix            │   │      ║
     ║        │                      │           │            │   │      ║
     ║        │                      │  Flow Matching         │   │      ║
     ║        │                      │  denoising             │   │      ║
     ║        │                      │  10 Euler steps        │   │      ║
     ║        │                      │           │            │   │      ║
     ║        │                      │  32D → 7D              │   │      ║
     ║        │                      │  (unpad)               │   │      ║
     ║        │                      └───────────┬───────────┘   │      ║
     ║        └──────────────────────────────────┼───────────────┘      ║
     ║                                           │                      ║
     ║                                   predicted actions               ║
     ║                                   (1, 50, 7) RELATIVE+NORM       ║
     ╚══════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
     ╔══════════════════════════════════════════════════════════════════╗
     ║                     L O S S                                      ║
     ╠══════════════════════════════════════════════════════════════════╣
     ║                                                                  ║
     ║   Flow Matching MSE:  ‖predicted - ground_truth‖²               ║
     ║                                                                  ║
     ║   Masked by action_is_pad:  loss[~in_episode_bound] = 0         ║
     ║                                                                  ║
     ║   Both prediction and GT are in RELATIVE space                   ║
     ║   Both prediction and GT are NORMALIZED                          ║
     ║   → Model learns to predict normalized offsets                   ║
     ║                                                                  ║
     ╚══════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │    BACKWARD PASS      │
                              │    ┌──────────────┐   │
                              │    │ AdamW        │   │
                              │    │ lr=1e-4      │   │
                              │    │ grad_clip=10 │   │
                              │    │              │   │
                              │    │ CosineDecay  │   │
                              │    │ warmup=100   │   │
                              │    │ decay→2.5e-6 │   │
                              │    └──────────────┘   │
                              │                       │
                              │  Trainable:           │
                              │  • Action Expert      │
                              │  • state_proj         │
                              │  • action_in/out_proj │
                              │                       │
                              │  Frozen:              │
                              │  • Vision Encoder     │
                              │  • VLM backbone       │
                              └──────────────────────┘
```

## 2. Data Dimensions Through the Pipeline

| Stage | Action | State | Image |
|-------|--------|-------|-------|
| Dataset load | (51, 7) | (2, 7) | (480, 640, 3) |
| DeriveStateFromAction | (50, 7) | (2, 7) | (480, 640, 3) |
| RelativeActions | (50, 7)\* | (2, 7) | (480, 640, 3) |
| RelativeState | (50, 7)\* | (14,)\* | (480, 640, 3) |
| Normalize | (50, 7)\* | (14,)\* | (3, 480, 640) |
| Model prepare | (50, 32) | (32,) | (3, 512, 512) |
| Model output | (50, 7)\* | — | — |
| Loss (MSE) | scalar | — | — |

\* = relative space (offsets from current state, gripper excluded)

## 3. SmolVLAConfig

```python
SmolVLAConfig(
    # UMI EE-pose pipeline
    derive_state_from_action=True,        # extract state from action column
    use_relative_actions=True,            # REQUIRED — not auto-set!
    use_relative_state=True,              # auto-set by derive_state_from_action
    state_obs_steps=2,                    # auto-set by derive_state_from_action
    relative_exclude_joints=["gripper"],         # gripper stays absolute in actions
    relative_exclude_state_joints=["gripper"],   # gripper stays absolute in state

    # Training mode
    freeze_vision_encoder=True,           # frozen
    train_expert_only=True,               # only train action expert
    train_state_proj=True,                # train state projection layer
    load_vlm_weights=False,               # train from scratch (no pretrained)
    push_to_hub=False,                    # local only

    # Image
    resize_imgs_with_padding=(512, 512),  # 480×640 → pad to 512×512

    # Optimizer
    optimizer_lr=1e-4,
    optimizer_weight_decay=1e-10,
    optimizer_grad_clip_norm=10,

    # Scheduler
    scheduler_warmup_steps=100,
    scheduler_decay_steps=10000,
    scheduler_decay_lr=2.5e-6,
)
```

**Common pitfall**: `derive_state_from_action=True` auto-sets `use_relative_state=True` and `state_obs_steps=2`, but does NOT auto-set `use_relative_actions=True`. Without it, `RelativeActionsProcessorStep` is disabled and actions remain absolute while stats are computed in relative space → normalization mismatch.

## 4. Stats Computation

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
- `action`: 7D relative (mean centered near 0 for pos+rot, 236 chunks × 50 = 11800 frames)
- `observation.state`: 14D relative (332 windows, source=action column)
  - First 7D: t=-1 offset from current → velocity information
  - Last 7D: t=0 offset from current → all zeros for pos+rot, gripper value

## 5. Inference Differenences

| Aspect | Training | Inference |
|--------|----------|-----------|
| State source | `DeriveStateFromAction` from action column | FK from robot joints |
| State format | 2-timestep from delta_indices | `RelativeStateProcessorStep` buffers previous, stacks [prev, cur] |
| Action chunk | Full 50-step ground truth | Model predicts 50-step chunk |
| Action execution | Not executed | One-at-a-time via RTC |
| Postprocessor | Not applied to GT | Unnormalize → AbsoluteActions → IK → joints |

## 6. Dimension Names (7D EE-pose)

| Index | Name | Type | Unit | Relative? |
|-------|------|------|------|-----------|
| 0 | `ee.x` | position | meters | YES |
| 1 | `ee.y` | position | meters | YES |
| 2 | `ee.z` | position | meters | YES |
| 3 | `ee.wx` | rotation (axis-angle) | radians | YES |
| 4 | `ee.wy` | rotation (axis-angle) | radians | YES |
| 5 | `ee.wz` | rotation (axis-angle) | radians | YES |
| 6 | `ee.gripper_pos` | gripper | normalized [0,1] | NO (excluded) |

## 7. Key Invariants

1. **t=0 action REL = ZERO** for pos+rot dims (by definition: `action[t=0] == current_state`)
2. **t=0 state REL = ZERO** for pos+rot dims (by definition: `state[t=0] - current = 0`)
3. **Gripper dim** is excluded from all relative conversions — always absolute [0,1]
4. **State 14D format**: `[t=-1[7], t=0[7]]` where t=0 pos+rot (dims 7-12) have mean=0, std≈0
5. **Normalize is z-score**, not min-max — values naturally range [-3, 3], not [-1, 1]
6. **Zero-std handling**: `denom = std + 1e-8` prevents division by zero for t=0 dims

## 8. Training Verification

### Smoke Test

```bash
cd /home/hls/codes/lerobot_piper_sroi
uv run --directory lerobot python /home/hls/codes/lerobot_piper_sroi/lerobot/train_smoke_test.py
```

Results:
- Level 1 (10 steps): No crashes, no NaNs — PASSED
- Level 2 (30-step overfitting): Loss 3.37 → 1.06 (68% reduction), gradients converging — PASSED

### Full Training

```bash
cd /home/hls/codes/lerobot_piper_sroi
uv run --directory lerobot python /home/hls/codes/lerobot_piper_sroi/lerobot/train_smolvla_umi_ee.py
```

Config: 1000 steps, batch_size=8, output to `outputs/smolvla_umi_ee_test/`

### Trace Preprocessor

```bash
cd /home/hls/codes/lerobot_piper_sroi
uv run --directory lerobot python /home/hls/codes/lerobot_piper_sroi/lerobot/trace_preprocessor.py
```

### Round-Trip Test

```bash
cd /home/hls/codes/lerobot_piper_sroi
uv run --directory lerobot python /home/hls/codes/lerobot_piper_sroi/lerobot/test_preprocessor_roundtrip.py
```

## 9. Files Reference

| File | Purpose |
|------|---------|
| `lerobot/train_smolvla_umi_ee.py` | Full training script |
| `lerobot/train_smoke_test.py` | Quick smoke test (10 + 30 steps) |
| `lerobot/trace_preprocessor.py` | Print full data flow through preprocessor |
| `lerobot/test_preprocessor_roundtrip.py` | Verify preprocess→postprocess lossless |
| `lerobot/UMI_EE_POSE_PIPELINE.md` | Preprocessor pipeline documentation |
| `lerobot/src/lerobot/processor/relative_action_processor.py` | Core processor steps |
| `lerobot/src/lerobot/policies/smolvla/processor_smolvla.py` | SmolVLA pre/post processor factory |
| `lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py` | SmolVLA config |
| `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` | SmolVLA model forward |
| `Datasets/test_ee_dataset/` | Training dataset |
