---
conversation_id: rot6d-umi-fix
created: 2025-01-20
tags: #lerobot #umi #rotation #rot6d #bug-fix #compatibility
---

# Conversation Summary - 2025-01-20

**Duration**: ~30 minutes

## Overview
Investigated a rotation representation bug in LeRobot's `RelativeEEDataset`. Discovered that commit 94111802 incorrectly changed from row-based to column-based rot6d representation, breaking compatibility with UMI. Fixed the code to follow UMI's actual implementation (rows) rather than the Zhou et al. paper's documented convention (columns).

## Topics
- 6D rotation representation (rot6d) conventions
- UMI (Universal Manipulation Interface) codebase compatibility
- Row-major vs column-major rotation matrix encoding
- NumPy stacking behavior (`axis=-1` vs `axis=-2`)

## Tasks Completed
1. **Identified rot6d convention mismatch** - UMI's code uses rows but commit 94111802 changed to columns
2. **Fixed `rot6d_to_mat`** - Changed `axis=-1` to `axis=-2` in two files to stack vectors as rows
3. **Fixed `mat_to_pose10d`** - Changed from column extraction (`rotmat[:, :2].T`) to row extraction (`rotmat[..., :2, :]`)
4. **Updated documentation** - Clarified row vs column convention in docstrings

## Files Modified

| File | Change |
|------|--------|
| `src/lerobot/datasets/relative_ee_dataset.py` | `rot6d_to_mat`: `axis=-1` → `axis=-2`; `mat_to_pose10d`: column extraction → row extraction |
| `src/lerobot/robots/so101_follower/relative_ee_processor.py` | `rot6d_to_mat`: `axis=-1` → `axis=-2` |

## Key Commands

### Investigation
```bash
git show 94111802 --stat
git show 94111802 -- src/lerobot/datasets/relative_ee_dataset.py
```

### Verification
```bash
python3 -c "from umi.common.pose_util import rot6d_to_mat, mat_to_rot6d; ..."
python3 -c "from lerobot.datasets.relative_ee_dataset import rot6d_to_mat; ..."
```

## Decisions & Insights

- **UMI uses rows, not columns**: Despite the Zhou et al. 2019 paper documenting columns, UMI's actual code (`np.stack(..., axis=-2)`) uses rows
- **Commit 94111802 introduced a bug**: The "fix" was actually breaking UMI compatibility by changing from rows to columns
- **The original code was correct for UMI**: `mat[..., :2, :]` extracts rows, matching UMI's `mat_to_rot6d`
- **Verification test**: Identity rot6d is `[1, 0, 0, 0, 1, 0]` in UMI's row-based convention

## Convention Summary

| Implementation | `mat_to_rot6d` | `rot6d_to_mat` |
|----------------|----------------|----------------|
| UMI code | `mat[..., :2, :]` (rows) | `axis=-2` |
| Zhou paper | columns | columns |
| LeRobot (after fix) | `mat[..., :2, :]` (rows) | `axis=-2` |

## Tags
#lerobot #umi #rotation #rot6d #bug-fix #compatibility
