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
UMI Dataset integration for LeRobot.

This module provides integration with the Universal Manipulation Interface (UMI)
dataset format, allowing LeRobot to load and process UMI-style datasets.
"""

import copy
from typing import Dict, Optional, Any, List
import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil

from ..datasets.lerobot_dataset import LeRobotDataset
from ..datasets.utils import get_dataset_stats
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    from diffusion_policy.common.normalize_util import (
        array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
        get_image_identity_normalizer, get_range_normalizer_from_stat)
    from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
    from diffusion_policy.dataset.base_dataset import BaseDataset
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from umi.common.pose_util import pose_to_mat, mat_to_pose10d
    UMI_AVAILABLE = True
    register_codecs()
except ImportError:
    UMI_AVAILABLE = False
    logger.warning("UMI dependencies not available. UMI dataset support will be limited.")


class UmiDataset(LeRobotDataset):
    """
    UMI Dataset loader for LeRobot.
    
    This class provides integration with UMI datasets, supporting:
    - Zarr-based dataset format
    - Multi-camera setups
    - Pose representations
    - Real-time data loading
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        pose_repr: Dict[str, str] = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize UMI dataset.
        
        Args:
            dataset_path: Path to UMI dataset (Zarr format)
            split: Dataset split ("train", "val", "test")
            cache_dir: Directory for caching processed data
            pose_repr: Pose representation configuration
            action_padding: Whether to pad actions
            temporally_independent_normalization: Whether to normalize independently
            repeat_frame_prob: Probability of repeating frames
            seed: Random seed
            val_ratio: Validation ratio
            max_duration: Maximum duration for episodes
        """
        if not UMI_AVAILABLE:
            raise ImportError("UMI dependencies not available. Please install UMI first.")
            
        self.dataset_path = dataset_path
        self.split = split
        self.cache_dir = cache_dir
        self.pose_repr = pose_repr
        self.action_padding = action_padding
        self.temporally_independent_normalization = temporally_independent_normalization
        self.repeat_frame_prob = repeat_frame_prob
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_duration = max_duration
        
        # Initialize pose representation
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        # Load dataset
        self._load_dataset()
        
        # Initialize parent class
        super().__init__(
            dataset_path=dataset_path,
            split=split,
            **kwargs
        )
    
    def _load_dataset(self):
        """Load UMI dataset from Zarr format."""
        if self.cache_dir is None:
            # Load into memory store
            with zarr.ZipStore(self.dataset_path, mode='r') as zip_store:
                self.replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, 
                    store=zarr.MemoryStore()
                )
        else:
            # Use disk cache
            mod_time = os.path.getmtime(self.dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(self.dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(self.cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            # Load cached file
            logger.info('Acquiring lock on cache.')
            with FileLock(lock_path):
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(self.dataset_path, mode='r') as zip_store:
                                logger.info(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        logger.info("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # Open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            self.replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        # Process dataset structure
        self._process_dataset_structure()
    
    def _process_dataset_structure(self):
        """Process and analyze dataset structure."""
        self.num_robot = 0
        self.rgb_keys = []
        self.lowdim_keys = []
        self.key_horizon = {}
        self.key_down_sample_steps = {}
        self.key_latency_steps = {}
        
        # Analyze observation structure
        obs_group = self.replay_buffer.root['data']['obs']
        for key in obs_group.keys():
            if key.startswith('rgb'):
                self.rgb_keys.append(key)
            else:
                self.lowdim_keys.append(key)
        
        # Analyze action structure
        action_group = self.replay_buffer.root['data']['action']
        self.action_keys = list(action_group.keys())
        
        # Get episode information
        self.episode_ends = self.replay_buffer.root['meta']['episode_ends'][:]
        self.num_episodes = len(self.episode_ends)
        
        logger.info(f"Loaded UMI dataset with {self.num_episodes} episodes")
        logger.info(f"RGB keys: {self.rgb_keys}")
        logger.info(f"Low-dim keys: {self.lowdim_keys}")
        logger.info(f"Action keys: {self.action_keys}")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = super().get_dataset_stats()
        
        # Add UMI-specific stats
        stats.update({
            'num_episodes': self.num_episodes,
            'rgb_keys': self.rgb_keys,
            'lowdim_keys': self.lowdim_keys,
            'action_keys': self.action_keys,
            'pose_repr': self.pose_repr,
        })
        
        return stats
    
    def __len__(self) -> int:
        """Get dataset length."""
        return self.num_episodes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        if idx >= self.num_episodes:
            raise IndexError(f"Index {idx} out of range for {self.num_episodes} episodes")
        
        # Get episode data
        episode_data = self._get_episode_data(idx)
        
        # Convert to torch tensors
        episode_data = dict_apply(episode_data, torch.from_numpy)
        
        return episode_data
    
    def _get_episode_data(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """Get raw episode data."""
        # Calculate episode boundaries
        if episode_idx == 0:
            start_idx = 0
        else:
            start_idx = self.episode_ends[episode_idx - 1]
        end_idx = self.episode_ends[episode_idx]
        
        # Get episode data
        episode_data = {}
        
        # Get observations
        obs_data = {}
        for key in self.rgb_keys + self.lowdim_keys:
            obs_data[key] = self.replay_buffer.root['data']['obs'][key][start_idx:end_idx]
        episode_data['obs'] = obs_data
        
        # Get actions
        action_data = {}
        for key in self.action_keys:
            action_data[key] = self.replay_buffer.root['data']['action'][key][start_idx:end_idx]
        episode_data['action'] = action_data
        
        return episode_data
    
    def get_validation_dataset(self):
        """Get validation dataset."""
        if self.val_ratio <= 0:
            return None
        
        # Create validation dataset with different split
        val_dataset = copy.deepcopy(self)
        val_dataset.split = "val"
        return val_dataset


def create_umi_dataset(
    dataset_path: str,
    split: str = "train",
    **kwargs
) -> UmiDataset:
    """
    Factory function to create UMI dataset.
    
    Args:
        dataset_path: Path to UMI dataset
        split: Dataset split
        **kwargs: Additional arguments for UmiDataset
        
    Returns:
        UmiDataset instance
    """
    return UmiDataset(
        dataset_path=dataset_path,
        split=split,
        **kwargs
    ) 