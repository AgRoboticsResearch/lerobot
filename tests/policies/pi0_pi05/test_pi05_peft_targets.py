#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import re

import pytest

pytest.importorskip("transformers")

from lerobot.policies.pi05 import PI05Policy  # noqa: E402


def test_default_peft_targets_match_pi05_projection_modules():
    target_pattern = PI05Policy._get_default_peft_targets(None)["target_modules"]

    for module_name in (
        "model.action_in_proj",
        "model.action_out_proj",
        "model.time_mlp_in",
        "model.time_mlp_out",
    ):
        assert re.fullmatch(target_pattern, module_name), f"Default PEFT targets do not match {module_name}"

    # These names belong to PI0 or never existed on PI0.5. Matching them would
    # silently leave PI0.5's real timestep-conditioning projections frozen.
    assert not re.fullmatch(target_pattern, "model.action_time_mlp_in")
    assert not re.fullmatch(target_pattern, "model.action_time_mlp_out")
    assert not re.fullmatch(target_pattern, "model.state_proj")
