"""UMI-style relative end-effector processor steps.

The on-disk action is an absolute 7D pose
``[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]``.  The policy
uses a 10D relative pose ``[dx, dy, dz, rot6d(6), gripper]``.  Every target in
an action chunk is expressed from the same chunk-start pose.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

# 轴角意思是rpy的旋转角？用轴角描述坐标系旋转应该是需要指定顺序的吧？而3x3的旋转矩阵对于任何一个确定的旋转都是唯一的？
def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to rotation matrices with Rodrigues' formula."""
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    axis = axis_angle / theta
    kx, ky, kz = axis.unbind(dim=-1)
    zeros = torch.zeros_like(kx)
    skew = torch.stack(
        [zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], dim=-1
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return identity + torch.sin(theta).unsqueeze(-1) * skew + (
        1 - torch.cos(theta).unsqueeze(-1)
    ) * (skew @ skew)

# 这两个轴角和旋转矩阵之间的互换函数应该都是现成的吧？
# 我想做几个小的输入输出测试来验证它的正确性
def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to axis-angle vectors."""
    vector = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta_twice = vector.norm(dim=-1)
    cos_theta = (matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2
    theta = torch.atan2(sin_theta_twice / 2, cos_theta)
    result = vector / (sin_theta_twice.unsqueeze(-1) + 1e-8) * theta.unsqueeze(-1)
    return torch.where((sin_theta_twice < 1e-7).unsqueeze(-1), vector / 2, result)

# 旋转矩阵到6d数的正逆变换
# 这个变换是其他论文中已经采用过的是吗？
# 之所以用六个数是因为六个数相比于九个数更容易被学习到吗？
# 但是我们当前的例子中，不存在那种角度跳变的极端情况，单个chunk内收尾坐标系的变化不大，这种情况下会不会采用传统一点的方法更容易获得好结果？
# 是否有此前的实验记录？已经给出了不同方法尝试之后的结论？
def matrix_to_rot6d(matrix: Tensor) -> Tensor:
    """Return the first two matrix rows (the UMI row-based rot6d convention)."""
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """Reconstruct a rotation matrix from UMI's row-based rot6d representation."""
    row0 = torch.nn.functional.normalize(rot6d[..., :3], dim=-1)
    row1 = rot6d[..., 3:] - (row0 * rot6d[..., 3:]).sum(dim=-1, keepdim=True) * row0
    row1 = torch.nn.functional.normalize(row1, dim=-1)
    row2 = torch.cross(row0, row1, dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

# 正向和逆向的主推理公式具体的输入输出是什么？
def absolute_aa_to_relative_rot6d(reference: Tensor, target: Tensor) -> Tensor:
    """Compute ``inverse(T_reference) @ T_target`` for 7D absolute poses."""
    reference_rotation = axis_angle_to_matrix(reference[..., 3:6])
    target_rotation = axis_angle_to_matrix(target[..., 3:6])
    reference_rotation_t = reference_rotation.transpose(-2, -1)
    relative_rotation = reference_rotation_t @ target_rotation
    relative_translation = (
        reference_rotation_t @ (target[..., :3] - reference[..., :3]).unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [relative_translation, matrix_to_rot6d(relative_rotation), target[..., 6:7]], dim=-1
    )


def relative_rot6d_to_absolute_aa(relative: Tensor, reference: Tensor) -> Tensor:
    """Compute ``T_reference @ T_relative`` and return a 7D absolute pose."""
    reference_rotation = axis_angle_to_matrix(reference[..., 3:6])
    absolute_rotation = reference_rotation @ rot6d_to_matrix(relative[..., 3:9])
    absolute_translation = reference[..., :3] + (
        reference_rotation @ relative[..., :3].unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [absolute_translation, matrix_to_axis_angle(absolute_rotation), relative[..., 9:10]], dim=-1
    )

# to_umi_relative_actions
# to_umi_absolute_actions
# to_umi_relative_state
# 这三个函数具体在做什么？我想知道它和其他函数的关系，以及精确的输入和输出是什么
def to_umi_relative_actions(actions: Tensor, state: Tensor) -> Tensor:
    """
    Convert a batch of absolute 7D action chunks to relative 10D chunks.
    
    把整个 Batch 的 7D 绝对动作块 转变为 相对起始状态的 10D 动作块
    """
    if actions.shape[-1] != 7 or state.shape[-1] != 7:
        raise ValueError(
            "UMI relative EE requires 7D [xyz, axis-angle, gripper] input; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    if state.ndim == 3:
        state = state[:, -1]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == 3:
        state = state.unsqueeze(1).expand(*actions.shape[:-1], 7)
    return absolute_aa_to_relative_rot6d(state, actions)


def to_umi_absolute_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert a batch of relative 10D action chunks to absolute 7D chunks."""
    if actions.shape[-1] != 10 or state.shape[-1] != 7:
        raise ValueError(
            "UMI relative EE requires 10D actions and a 7D reference state; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    if state.ndim == 3:
        state = state[:, -1]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == 3:
        state = state.unsqueeze(1).expand(*actions.shape[:-1], 7)
    return relative_rot6d_to_absolute_aa(actions, state)


def to_umi_relative_state(state: Tensor) -> Tensor:
    """Convert ``[previous, current]`` 7D poses to a flattened 20D state."""
    if state.ndim != 3 or state.shape[-2:] != (2, 7):
        raise ValueError(f"UMI state must have shape [batch, 2, 7], got {tuple(state.shape)}")
    current = state[:, -1:].expand_as(state)
    return absolute_aa_to_relative_rot6d(current, state).flatten(start_dim=-2)


_STATE_CACHE: dict[str, Tensor] = {}


def make_umi_cache_key() -> str:
    return f"umi_relative_ee_{uuid4().hex}"


@ProcessorStepRegistry.register("umi_derive_state_from_action")
# ProcessorStepRegistry = ProcessorStep 类的注册表
# 把字符串名映射到类，允许通过字符串名称进行实例化
# 装饰器自动把下面形成的类注册进去，类和 umi_derive_state_from_action 这个名字绑定
@dataclass
# @class 这种写法叫做装饰器
# 从下往上执行，下面的类构造完成之后，这个类经过装饰器变成 dataclass
# 这个 dataclass 是 python 自带的，专门为数据类设计
# dataclass 会自动生成 __init__ __repr__ __eq__ 这些特殊方法
# __init__ 就是在实例化的时候会自动调用的特殊方法
# __repr__ 用于给开发者返回可读的字符串（打印这个类的时候打印什么东西）
# __eq__ 定义 == 行为（两个实例判等的时候怎么才算相等）
class UmiDeriveStateFromActionStep(ProcessorStep):
# ProcessorStep 单步数据处理模块的基类，要求补全其中的基本功能
# UmiDeriveStateFromActionStep = UMI 风格的 从动作推导状态的 单步数据处理模块
    """
    Build state from ``action[t-1:t+1]`` and remove the leading action.
        
    从 action[t-1] 和 action[t] 推导出状态,本代码中直接取时间步0和1当作状态
    然后把 action 序列中的时间步0的值删掉,保留后续部分
    """
    enabled: bool = True
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
    # 特殊方法 __call__ 让实例可以像函数一样被调用，比如：
    # step = UmiDeriveStateFromActionStep() 创建实例
    # new_transition = step(transition) 调用实例，传入旧的 EnvTransition 获得新的 EnvTransition
    # EnvTransition Lerobot框架中的数据容器，里面装着观测值、状态值、奖励之类的东西
        action = transition.get(TransitionKey.ACTION)
        # 从传入的 transition 中取其中 action 键的值，值的类型是 Tensor
        # TransitionKey 来自Lerobot框架，通过 TransitionKey.ACTION 这种形式存放 原始键 的名字
        if not self.enabled or action is None or action.ndim < 3:
        # 三个不处理的判断条件
        # 重点解释为什么 action 的维度需要大于等于3
        # 对于 action 要求以 action[B,T,D] 的形式存在
        # B：Batch数，T：单条动作的时间步，D：单条动作的动作维度
            return transition
            # 对于不满足条件的情况直接返回原始值
        result = deepcopy(transition)
        # 对于符合条件的情况，深拷贝整个 transition 对象，是 ProcessorStep 进行数据处理的过程中约定俗成的做法
        
        # 取出来的东西套一层字典，相当于拷贝一份，不去修改 result 里面的值
        observation = dict(result.get(TransitionKey.OBSERVATION) or {})
        # 取 result 中 TransitionKey.OBSERVATION 键的值
        # 这个值也是一个字典，但是为了避免修改原对象，所以这里套 dict() 拷贝一份
        observation[OBS_STATE] = action[..., :2, :]
        # 使用 TransitionKey.OBSERVATION 键取出的值是一个字典，里面还有类似于 observation.state observation.image 这样的键
        # 把其中的 observation.state 键用 action 中的部分片段来赋值
        # ... 表示取前面的所有维度，此处如果 action 是3维，那么写 ：也一样
        # 这里表示取全部 Batch维，取时间维度的0和1,取动作维度的所有维
        result[TransitionKey.OBSERVATION] = observation
        result[TransitionKey.ACTION] = action[..., 1:, :]
        # 这里把 result 中的两个键换了新的值
        # TransitionKey.OBSERVATION 键是整体替换，因为值本身就是一个字典
        # TransitionKey.ACTION 键在原值的基础上去掉 T=0 那些，保留其他的

        complementary = dict(result.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        # TransitionKey.COMPLEMENTARY_DATA 键用于提供附加信息，其值也是一个字典
        # 可能 TransitionKey.COMPLEMENTARY_DATA 的值当中有 action_is_pad 键
        for container in (result, complementary):
        # container 在 result 和 complementary 两个元素中遍历
        # action_is_pad 键可能直接放在 result 中，也可能放在 complementary 中
            padding = container.get("action_is_pad")
            # 查看是否存在 action_is_pad 键
            # action_is_pad 键的值标记了哪些时间步是 padding 的（也就是自动填充的）
            if isinstance(padding, Tensor) and padding.ndim >= 2:
            # action_is_pad 通常是 [B, T]，和 action 的时间维对齐
            # 这里检查它是 Tensor 且至少有 2 维，才能按时间维裁
                container["action_is_pad"] = padding[..., 1:]
                # 前面去掉了动作当中的 T=0,那么 action_is_pad 键的值也要相应修改
        if complementary:
            result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        # 如果 result 中的 TransitionKey.COMPLEMENTARY_DATA 存在，那么经过上述处理之后应当更新一下
        # 因为动作被去掉了 T=0 的列，那么标记 padding 的时间步也要前移一位
        return result
    
    def get_config(self) -> dict[str, Any]:
    # 该方法用于让整个框架了解该 Step 是怎么构造的
    # 由于该 Step 中只增加了 enabled 这个成员，所以返回这个字典
        return {"enabled": self.enabled}

    def transform_features(self, features):
    # features 是 pipeline 的特征描述表，告诉框架每个键的类型和形状
    # 这里没有改变特征的定义，所以直接返回
        return features


@ProcessorStepRegistry.register("umi_relative_actions")
@dataclass
class UmiRelativeActionsStep(ProcessorStep):
    """
    Convert 7D absolute chunks to 10D UMI-relative chunks and cache the base.
    
    把 7D 绝对动作的 chunk 转变成 10D 相对动作的 chunk
    缓存每个动作块的起始位姿作为计算的基准
    """
    enabled: bool = True
    cache_key: str = "umi_relative_ee"
    _last_state: Tensor | None = field(default=None, init=False, repr=False)
    # 这三个都叫做实例变量

    def __call__(self, transition: EnvTransition) -> EnvTransition:
    # 传入的是完整的 EnvTransition 对象
    # 经过条件判断后，对其中的 action 做相对变化的操作
        observation = transition.get(TransitionKey.OBSERVATION, {})
        raw_state = observation.get(OBS_STATE) if observation else None 
        state = raw_state[..., -1, :] if raw_state is not None and raw_state.ndim >= 3 else raw_state
        if state is not None:
            self._last_state = state.detach().cpu().clone()
            _STATE_CACHE[self.cache_key] = self._last_state
        action = transition.get(TransitionKey.ACTION)
        if not self.enabled or action is None or state is None:
            return transition
        result = deepcopy(transition)
        result[TransitionKey.ACTION] = to_umi_relative_actions(action, state)
        return result

    def get_cached_state(self) -> Tensor | None:
    # 新的方法，获取缓存的状态（7D绝对动作）
        return self._last_state

    def reset(self) -> None:
        self._last_state = None
        _STATE_CACHE.pop(self.cache_key, None)

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "cache_key": self.cache_key}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        action_features = result.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            feature = action_features[ACTION]
            action_features[ACTION] = PolicyFeature(type=feature.type, shape=(10,))
        return result


@ProcessorStepRegistry.register("umi_relative_state")
@dataclass
class UmiRelativeStateStep(ProcessorStep):
    """
    Convert two absolute 7D poses to a flattened relative 20D state.
    
    将两个绝对7D位姿转换为展平的20D相对状态
    """

    enabled: bool = True
    _previous_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is None:
            return transition
        if state.ndim == 2:
            previous = state if self._previous_state is None else self._previous_state.to(state)
            state_pair = torch.stack([previous, state], dim=1)
            self._previous_state = state.detach().clone()
        elif state.ndim == 3 and state.shape[1] == 2:
            state_pair = state
        else:
            raise ValueError(f"UMI state must be [B, 7] or [B, 2, 7], got {tuple(state.shape)}")
        result = deepcopy(transition)
        result_observation = dict(observation)
        result_observation[OBS_STATE] = to_umi_relative_state(state_pair)
        result[TransitionKey.OBSERVATION] = result_observation
        return result

    def reset(self) -> None:
        self._previous_state = None

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        observation_features = result.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            feature = observation_features[OBS_STATE]
            observation_features[OBS_STATE] = PolicyFeature(type=feature.type, shape=(20,))
        return result


@ProcessorStepRegistry.register("umi_absolute_actions")
@dataclass
class UmiAbsoluteActionsStep(ProcessorStep):
    """Convert model-relative 10D chunks back to absolute 7D EE targets."""

    enabled: bool = True
    cache_key: str = "umi_relative_ee"
    relative_step: UmiRelativeActionsStep | None = field(default=None, repr=False)
    single_action_reference_steps: int = 1
    _single_action_reference: Tensor | None = field(default=None, init=False, repr=False)
    _single_action_steps_remaining: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.single_action_reference_steps < 1:
            raise ValueError("single_action_reference_steps must be at least 1")

    def set_single_action_reference_steps(self, steps: int) -> None:
        """Set how many queued single actions share one chunk-start reference."""
        if steps < 1:
            raise ValueError("single_action_reference_steps must be at least 1")
        self.single_action_reference_steps = steps
        self._clear_single_action_reference()

    def _clear_single_action_reference(self) -> None:
        self._single_action_reference = None
        self._single_action_steps_remaining = 0

    def _reference_for_action(self, action: Tensor, current_state: Tensor) -> Tensor:
        # predict_action_chunk returns [batch, time, action_dim] and is
        # postprocessed immediately, so it always uses the latest observation.
        # select_action returns [batch, action_dim] one item at a time; all items
        # in its internal queue must instead retain the frame that generated the
        # chunk, even though preprocessing refreshes current_state every tick.
        if action.ndim >= 3 or self.single_action_reference_steps == 1:
            self._clear_single_action_reference()
            return current_state

        if self._single_action_steps_remaining == 0:
            self._single_action_reference = current_state.detach().clone()
            self._single_action_steps_remaining = self.single_action_reference_steps

        if self._single_action_reference is None:  # Defensive invariant for type checkers.
            raise RuntimeError("UMI single-action reference was not initialized")
        reference = self._single_action_reference
        self._single_action_steps_remaining -= 1
        if self._single_action_steps_remaining == 0:
            self._single_action_reference = None
        return reference

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        state = self.relative_step.get_cached_state() if self.relative_step is not None else None
        if state is None:
            state = _STATE_CACHE.get(self.cache_key)
        if state is None:
            raise RuntimeError("UMI postprocessor has no chunk-base state; run the preprocessor first")
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        state = self._reference_for_action(action, state)
        result = deepcopy(transition)
        result[TransitionKey.ACTION] = to_umi_absolute_actions(action, state)
        return result

    def reset(self) -> None:
        self._clear_single_action_reference()
        _STATE_CACHE.pop(self.cache_key, None)

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cache_key": self.cache_key,
            "single_action_reference_steps": self.single_action_reference_steps,
        }

    def transform_features(self, features):
        return features


EE_KEY = "observation.ee"


def umi_chain_kind(model_path: str | Path) -> str:
    """Classify a checkpoint's UMI convention from its saved preprocessor.

    Returns "ee_pair" for the ee-based chain (20D state and chunk both from the
    observation.ee window, past-anchored on ee[t]) or "action_derived" for the
    legacy chain (state derived from the action column, future-anchored, with
    the deployment cache in UmiRelativeActionsStep/UmiAbsoluteActionsStep).
    Deployment/eval code branches on this to decide what to feed and whether
    the postprocessor already restored absolute actions.
    """
    preprocessor_path = Path(model_path) / "policy_preprocessor.json"
    steps = json.loads(preprocessor_path.read_text())["steps"]
    names = {step.get("registry_name") for step in steps}
    return "ee_pair" if "umi_ee_state" in names else "action_derived"


@ProcessorStepRegistry.register("umi_ee_state")
@dataclass
class UmiEEStateStep(ProcessorStep):
    """ee 采样窗前两帧 [t-1, t] → 20D UMI 相对状态,整形 [B, 1, 20]。

    前半 = 上一帧位姿在当前帧坐标系下的表达(速度),后半 = 恒等+当前夹爪。
    头样本(t=0)时 [t-1] 越界由 dataloader 用帧 0 自复制填充 = 零速,符合
    "episode 从静止开始"的物理事实。ee 数据源为 observation.ee 列
    (action 列在 ee 基准口径下不再被读取)。

    flat_output=False → [B, 1, 20](DP 的 (B, n_obs_steps, state_dim) 合同);
    flat_output=True → [B, 20](ACT 的 (B, state_dim) 合同)。"""

    flat_output: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        ee = observation.get(EE_KEY)
        if ee is None or ee.ndim != 3 or ee.shape[1] < 2 or ee.shape[2] != 7:
            return transition
        result = deepcopy(transition)
        result_observation = dict(observation)
        flat = to_umi_relative_state(ee[:, :2, :]).reshape(ee.shape[0], 20)
        result_observation[OBS_STATE] = flat if self.flat_output else flat.unsqueeze(1)
        result[TransitionKey.OBSERVATION] = result_observation
        return result

    def get_config(self) -> dict[str, Any]:
        return {"flat_output": self.flat_output}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        observation_features = result.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            observation_features[OBS_STATE] = PolicyFeature(
                type=observation_features[OBS_STATE].type,
                shape=(20,) if self.flat_output else (1, 20),
            )
        return result


@ProcessorStepRegistry.register("umi_ee_relative_chunk")
@dataclass
class UmiEERelativeChunkStep(ProcessorStep):
    """ee 采样窗第 2 位起的绝对目标 ee[t+1..t+chunk] → 以 ee[t](观测帧自身
    位姿,窗口第 1 位)为基准的相对动作,写入 action。

    chunk[i] = ee[t]⁻¹ · ee[t+1+i]:第一个元素就是到下一帧的真实运动,
    不再是恒等;部署端以实时观测位姿为基准,与训练完全同源。
    ee 窗长 = chunk_length + 2(前 2 帧供 UmiEEStateStep 用)。"""

    chunk_length: int

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        ee = observation.get(EE_KEY)
        if ee is None or ee.ndim != 3 or ee.shape[1] != self.chunk_length + 2 or ee.shape[2] != 7:
            return transition
        base = ee[:, 1, :]  # ee[t] = 观测帧自身位姿
        result = deepcopy(transition)
        result[TransitionKey.ACTION] = to_umi_relative_actions(ee[:, 2:, :], base)
        # action 列不再经 dataloader 取窗,也就没有 action_is_pad;
        # 由 ee 窗目标段的 pad 接管(drop=chunk 下训练时恒为 False)。
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        ee_pad = complementary.get(f"{EE_KEY}_is_pad")
        if ee_pad is not None:
            complementary["action_is_pad"] = ee_pad[:, 2:]
        else:
            complementary["action_is_pad"] = torch.zeros(
                (ee.shape[0], self.chunk_length), dtype=torch.bool, device=ee.device
            )
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return result

    def get_config(self) -> dict[str, Any]:
        return {"chunk_length": self.chunk_length}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        action_features = result.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            action_features[ACTION] = PolicyFeature(
                type=action_features[ACTION].type, shape=(self.chunk_length, 10)
            )
        return result


@ProcessorStepRegistry.register("umi_state_time_dim")
@dataclass
class UmiStateTimeDimStep(ProcessorStep):
    """恢复 state 的时间维: DP 的 compute_loss 要求 [B, n_obs_steps, D]。

    兼容中间代(8f80a1eb..aee0421f)checkpoint 的保存管线——它们的链末尾带这
    一步(拍平 20D → [B,1,20])。当前代码不再接线它,但注册名必须保留,
    否则那批 checkpoint 的处理器无法加载(如 1M 旧口径基线的开环评估)。"""

    enabled: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None or state.ndim != 2:
            return transition
        result = deepcopy(transition)
        result_observation = dict(observation)
        result_observation[OBS_STATE] = state.unsqueeze(1)
        result[TransitionKey.OBSERVATION] = result_observation
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        observation_features = result.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            feature = observation_features[OBS_STATE]
            observation_features[OBS_STATE] = PolicyFeature(type=feature.type, shape=(1, *feature.shape))
        return result
