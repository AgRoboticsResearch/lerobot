# Ground Truth EE Trajectory Extraction

This directory contains tools for extracting ground truth end-effector trajectories from datasets for robot EE pose control testing.

## extract_gt_ee_trajectories.py

Extracts full episode EE trajectories from a dataset and saves them as absolute poses.

### How It Works

1. **Loads absolute poses directly** from LeRobotDataset (not RelativeEEDataset)
2. **Transforms to first pose frame** so the trajectory starts at origin
3. **Saves as CSV** with absolute poses [x, y, z, qx, qy, qz, qw, gripper]

The transformation ensures:
- First pose is always at origin: `[0, 0, 0, 0, 0, 0, 1, gripper]`
- Subsequent poses are relative to the first pose
- Trajectory shape/orientation is preserved

### Output Format

For each episode, saves a CSV file to `output/eetrajs/{dataset_name}/`:

- **episode_{idx:04d}.csv**: CSV with absolute pose columns
  - x, y, z: Position in meters (relative to first pose)
  - qx, qy, qz, qw: Orientation as quaternion (relative to first pose)
  - gripper: Gripper state [0=closed, 1=open]

Each row = one timestep's absolute pose, with the first row at origin.

### Usage

```bash
# Extract specific episodes
python piper/eetrajs/extract_gt_ee_trajectories.py \
    --dataset_root /mnt/ldata/sroi/sroi_lab_picking \
    --episode_indices 0 1 2

# Extract all episodes with trajectory plots
python piper/eetrajs/extract_gt_ee_trajectories.py \
    --dataset_root /mnt/ldata/sroi/sroi_lab_picking \
    --plot
```

### Arguments

- `--dataset_root`: Root directory of the dataset (required)
- `--episode_indices`: Episode indices to extract (default: all episodes)
- `--output_dir`: Base output directory (default: output/eetrajs)
- `--plot`: Generate trajectory plots (XY, XZ, YZ projections) and save as PNG

### Notes

- Uses LeRobotDataset directly (not RelativeEEDataset) to get absolute poses
- First pose is transformed to origin: `T_transformed = T_first^{-1} @ T`
- The output directory is already added to .gitignore via the `outputs/` entry
- Original poses are in base frame; saved poses are in first-pose frame
