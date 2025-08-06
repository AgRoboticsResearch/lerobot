# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
UMI Policy integration for LeRobot.

This module provides integration with Universal Manipulation Interface (UMI) policies,
allowing LeRobot to use UMI-trained diffusion policies for robot control.
"""

from .configuration_umi import UmiConfig
from .modeling_umi import UmiPolicy

__all__ = ["UmiConfig", "UmiPolicy"] 