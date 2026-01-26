---
conversation_id: 2025-01-26_umi_temporal_batching
created: 2025-01-26 22:30
duration: ~45 minutes
tags: #umi #act #temporal-batching #resnet #normalization
---

# Conversation Summary - UMI-Style Temporal Batching Implementation

**Date**: 2025-01-26
**Topic**: Implementing UMI-style temporal batching for ACT policy training

## Overview

Implemented UMI (Universal Manipulation Interface) style temporal batching for ACT policy training. This approach processes each timestep independently through a standard 3-channel ResNet backbone, then concatenates features. This preserves pretrained ResNet weights (trained on 3-channel RGB) instead of using channel concatenation which would require modifying the backbone's first convolutional layer.

The key difference from channel concatenation:
- **Channel concat**: `(B, T, C, H, W) → (B, T*C, H, W)` - modifies backbone
- **UMI batch flatten**: `(B, T, C, H, W) → (B*T, C, H, W)` - standard backbone

## Tasks Completed

1. **Fixed positional embedding shapes in TemporalACTWrapper** `train_relative_ee.py:159,173,178,225`
   - Changed from `(1, 1, dim)` to `(1, dim)` for proper stacking
   - Fixed latent, state, env_state, and image positional embeddings
   - All embeddings now stack correctly to `(seq, batch, dim)`

2. **Verified training runs successfully**
   - Loss decreasing: 6.060 → 2.672 → 2.248 over 600 steps
   - No shape mismatches or runtime errors
   - Gradient norms healthy (~150-80 → ~70)

## Files Modified

| File | Change |
|------|--------|
| `train_relative_ee.py:159` | Fixed latent positional embedding shape: removed `.unsqueeze(1)` |
| `train_relative_ee.py:173` | Fixed state positional embedding: removed `.unsqueeze(1)` |
| `train_relative_ee.py:178` | Fixed env_state positional embedding: removed `.unsqueeze(1)` |
| `train_relative_ee.py:225` | Fixed spatial positional embeddings: use `x[:1]` instead of `x[:1].unsqueeze(0)` |

## Key Commands

### Training Test
```bash
python train_relative_ee.py \
    --dataset.repo_id red_strawberry_picking_260119_merged_ee \
    --dataset.root /mnt/ldata/sroi_lerobot/red_strawberry_picking_260119_merged_ee \
    --policy.type=act \
    --policy.obs_state_horizon=2 \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir outputs/train/red_strawberry_picking_260119_merged_hist2_resbatch \
    --num_workers=4 \
    --batch_size=8 \
    --steps=200000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_stat_samples=1000
```

## Technical Details

### Temporal Batch Processing Flow

1. **Dataset output**: `(B, T, ...)` where `T=obs_state_horizon`
2. **Flatten for encoding**: `(B, T, ...) → (B*T, ...)`
3. **Encode independently**: Standard 3-channel backbone processes each frame
4. **Aggregate features**: `(B*T, F) → (B, T*F)`

### Shape Transformations

- **State**: `(B, T, 10) → (B*T, 10) → encode → (B*T, 512) → (B, T, 512)` → each timestep as separate token
- **Images**: `(B, T, 3, H, W) → (B*T, 3, H, W) → backbone → (B*T, 512, H', W')` → spatial tokens per timestep
- **Actions**: `(B, action_horizon, 10)` - unchanged, no repetition

### Normalization

- MIN_MAX normalization for position and gripper (UMI-style: maps to [-1, 1])
- MEAN_STD normalization for visual features
- TemporalNormalizeProcessor handles `(B, T, D) → (B*T, D) → normalize → (B, T, D)`

## Decisions & Insights

1. **Positional embedding format**: PyTorch transformer expects `(seq, batch, dim)` format. When stacking a list of embeddings, each should be `(1, dim)` not `(1, 1, dim)`.

2. **No action repetition needed**: Unlike initial flattening approach that would repeat actions T times, UMI-style keeps one action target per sample. The temporal information is preserved in the observation features, not by repeating targets.

3. **VAE encoding uses current timestep only**: The VAE encoder processes only the current state `(B, D)` not the aggregated temporal state, avoiding shape mismatches.

## Tags

#umi #act #temporal-batching #resnet #normalization #robotics #imitation-learning
