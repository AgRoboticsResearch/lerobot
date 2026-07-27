# Debugging UMI relative-EE rotation predictions

This guide documents the numeric GT-versus-prediction rotation diagnostic in
`generate_rotation_comparison.py`. It complements the projected episode videos
described in `prediction_visualization.md`:

- Use the videos to inspect where a predicted trajectory appears in the camera
  image.
- Use this diagnostic to inspect the model's raw relative-pose outputs and
  measure rotation error directly on SO(3).

The commands below are for the `fei-v5.0-umi-unified` branch in:

```text
/home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
```

## What is being compared

For every valid dataset frame, the script asks the policy for one 30-step
action chunk. Ground truth and prediction use the same UMI convention:

```text
T_rel[k] = inverse(T_base) @ T_target[k]
```

All 30 targets are relative to the same chunk-start `T_base`. They are not
chained from one predicted action to the next.

The model's relative pose is 10D:

```text
[dx, dy, dz, r00, r01, r02, r10, r11, r12, gripper]
```

The diagnostic calls the first nine values **raw 9D**:

```text
[dx, dy, dz, r00, r01, r02, r10, r11, r12]
```

- `dx, dy, dz` are relative translation in meters.
- The six `r` values are the first two rows of the relative rotation matrix.
- `gripper` is excluded because this diagnostic is about pose and rotation.

The six stored rotation values are converted to an orthonormal 3×3 rotation
matrix before computing angular error. Thus, the plotted SO(3) error is not a
per-channel rot6d difference.

## Data and normalization path

The diagnostic intentionally follows the training/inference pipeline:

1. `resolve_delta_timestamps()` requests the observation history and action
   horizon required by the saved policy configuration.
2. The checkpoint's saved preprocessor derives the temporal state and converts
   absolute 7D EE actions to same-base relative 10D actions.
3. The saved checkpoint statistics normalize the input.
4. `policy.predict_action_chunk()` predicts a normalized relative-action chunk.
5. The prediction is unnormalized with the checkpoint's action statistics.
6. Raw 9D values and SO(3) rotations are compared in physical, unnormalized
   units.

For ACT, the action field is removed after preprocessing so inference uses the
VAE prior rather than the training-only posterior.

Frames whose 30-step ground-truth action horizon contains padding are skipped.
This avoids treating repeated end-of-episode padding as real motion.

## Outputs

The output directory contains:

### `rotation_gt_vs_prediction_summary.png`

For each selected episode:

- rows 1–3 plot the endpoint rotation vector's x, y, and z components;
- dashed black is ground truth;
- blue is the prediction;
- row 4 plots endpoint geodesic SO(3) error in degrees;
- the dashed red line and label report the episode mean and median.

The endpoint is action-chunk step 29 for a 30-step chunk.

Rotation-vector components are useful for seeing directional bias, but their
coordinates can be discontinuous near the angle-axis branch boundary. Use the
geodesic SO(3) error as the primary accuracy measure.

### `raw_9d_gt_vs_prediction_episode_<N>.png`

One figure is generated per episode. It shows all nine unnormalized channels
across the complete action chunk:

```text
dx, dy, dz, r00, r01, r02, r10, r11, r12
```

Rather than selecting an unusually easy or hard frame, the script chooses the
frame whose endpoint rotation error is closest to that episode's median.

The rotation entries are matrix components and therefore unitless. Values such
as `r00` and `r11` may remain close to `+1`; Matplotlib can show an axis offset
such as `+1` in those panels.

### `raw_9d_gt_vs_prediction.csv`

This contains every retained dataset frame and every action-chunk step. Columns:

```text
episode
frame
chunk_step
rotation_error_deg
gt_dx ... gt_r12
pred_dx ... pred_r12
```

Use the CSV for per-step distributions, per-channel bias, or comparisons
between checkpoints.

## Reproduce the current ACT diagnostics

Activate the unified worktree and make sure its source tree is imported:

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The most recent complete checkpoint from the stopped ACT run is:

```text
outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model
```

### Training episodes 0, 1, and 2

```bash
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model \
  --dataset-root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb_1125 \
  --episodes 0 1 2 \
  --output-dir outputs/debug/rotation_act_unified_ckpt_1p1m_train_ep0_1_2 \
  --device cuda
```

### Validation episodes 0, 1, and 2

```bash
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint outputs/train/ee_vs_joints/umi_unified_ee_action_chunk30_sroi_v2_masked_1125train_100val/checkpoints/1100000/pretrained_model \
  --dataset-root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episodes 0 1 2 \
  --output-dir outputs/debug/rotation_act_unified_ckpt_1p1m_validation_ep0_1_2 \
  --device cuda
```

Change `--episodes` to inspect other episodes. `--device cpu` is supported but
will be slower.

Camera intrinsics and the D405 hand-eye calibration are not required. Those are
only needed when projecting trajectories onto camera images.

## Results from checkpoint 1,100,000

The commands above produced:

| split | retained frames | mean error over all chunk steps | endpoint mean | endpoint median |
|---|---:|---:|---:|---:|
| train, episodes 0–2 | 215 | 0.594° | 0.758° | 0.673° |
| validation, episodes 0–2 | 161 | 2.820° | 4.797° | 3.679° |

Validation endpoint means by episode:

| episode | frames | endpoint mean |
|---:|---:|---:|
| 0 | 53 | 3.039° |
| 1 | 54 | 2.734° |
| 2 | 54 | 8.586° |

Episode 2 is substantially harder than episodes 0 and 1. Inspecting more
validation episodes is necessary before treating these three episodes as an
overall validation estimate.

## How to interpret a failure

### Raw rot6d differs but SO(3) error is small

Do not diagnose rotation quality from a single raw channel alone. The six
values jointly form a rotation representation and are projected to a valid
rotation matrix. Prefer the geodesic error.

### Translation is good but rotation is consistently offset

Check:

- that GT and prediction use the same chunk-start base pose;
- that rotation rows/columns use the same convention;
- that checkpoint statistics belong to the same dataset/action representation;
- whether the error is concentrated in particular episodes or poses.

### Errors grow with chunk step

This usually means long-horizon endpoint prediction is harder. Compare error by
`chunk_step` in the CSV rather than looking only at step 29.

### Validation is much worse than training

Possible causes include domain shift, different motion distributions, camera
or scene differences that affect visual inference, and rare rotations in the
training data. First expand the episode sample and compare the per-episode
distribution.

### Very large normalized values are suspected

Inspect the checkpoint/dataset normalization statistics for narrow ranges or
very small standard deviations. The figures and CSV here contain unnormalized
outputs, so they reveal the physical result but not the internal normalized
magnitude.

## Scope and limitations

- This is recorded-observation, open-loop evaluation. It is not a robot rollout.
- GT observations condition every frame independently, so the diagnostic does
  not measure closed-loop compounding error.
- The representative raw-9D plot shows one frame per episode; use the CSV for
  the full distribution.
- Three episodes are useful for debugging, not for reporting a final benchmark.
- The current script has been exercised with the ACT checkpoint above. Other
  unified UMI policies can use the same numeric approach when their saved
  processor and `predict_action_chunk()` interface are compatible.

