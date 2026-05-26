"""ACT config subclass with UMI-style relative EE action support.

Extends ACTConfig with derive_state_from_action, use_relative_actions,
pose_dim, and use_rot6d fields. Does NOT modify the existing ACTConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.policies.act.configuration_act import ACTConfig


@dataclass
class ACTRelativeEEConfig(ACTConfig):
    """ACT config with UMI-style relative EE action support via processor pipeline.

    This config adds UMI-style processor pipeline approach to ACT:
    - derive_state_from_action: Derives observation.state from action column
    - use_relative_actions: Converts actions to relative SE(3) with rot6d
    - pose_dim: Number of leading action dims forming SE(3) pose (6 = xyz + aa)
    - use_rot6d: Output 10D rot6d instead of 7D axis-angle

    When derive_state_from_action=True, action_delta_indices includes an extra
    leading timestep [-1, 0, 1, ..., chunk_size-1] so DeriveStateFromActionStep
    can extract [action[t-1], action[t]] as a 2-step state.

    NOTE: Not registered as a separate policy type. The training script overrides
    the config class on the standard ACT policy via monkey-patching.
    """

    # Relative EE action fields
    derive_state_from_action: bool = False
    use_relative_actions: bool = False
    pose_dim: int = 0  # 6 = xyz + axis-angle → triggers SE(3) mode
    use_rot6d: bool = False  # True → 10D rot6d output
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])

    # Internal: set by training script from dataset metadata
    _action_names: list[str] | None = None
    _state_names: list[str] | None = None

    @property
    def action_delta_indices(self) -> list:
        if self.derive_state_from_action:
            return [-1] + list(range(self.chunk_size))
        return list(range(self.chunk_size))
