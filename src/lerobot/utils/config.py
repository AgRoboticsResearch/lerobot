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
Base configuration utilities for LeRobot.

This module provides base configuration classes and utilities used across
the LeRobot framework.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class BaseConfig:
    """
    Base configuration class for LeRobot components.
    
    This class provides common functionality for configuration classes
    throughout the LeRobot framework.
    """
    
    def __post_init__(self):
        """Post-initialization processing."""
        pass
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid
        """
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            Configuration instance
        """
        return cls(**config_dict) 