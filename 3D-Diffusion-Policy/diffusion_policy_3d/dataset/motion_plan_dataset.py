from typing import Dict
import torch
import numpy as np
import copy
import zarr
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class MotionPlanDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.zarr_path = zarr_path
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        self.zarr_root = zarr.open(zarr_path, mode='r')
        self.start_configuration = self.zarr_root['data']['start_configuration'][:]
        self.end_configuration = self.zarr_root['data']['end_configuration'][:]
        self.episode_ends = self.replay_buffer.episode_ends[:]
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        goal_eef_xyz = self.end_configuration[:, 6:9]
        data = {
            'action': self.replay_buffer['action'][..., :7],
            'agent_pos': self.replay_buffer['state'][..., :7],
            'point_cloud': self.replay_buffer['point_cloud'][..., :3],
            'goal_eef_xyz': goal_eef_xyz,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    def _get_episode_idx(self, buffer_start_idx:int)->int:
        return int(np.searchsorted(self.episode_ends,buffer_start_idx,side='right'))

    def _sample_to_data(self, sample,idx:int):
        agent_pos = sample['state'][:, :7].astype(np.float32) # (T, 7)
        # Slice to match shape_meta: [2500, 3] from potentially 5000 points
        point_cloud = sample['point_cloud'][:, :2500, :3].astype(np.float32) # (T, 2500, 3)
        buffer_start_idx = int(self.sampler.indices[idx,0])
        episode_idx = self._get_episode_idx(buffer_start_idx)
        goal_eef_xyz = self.end_configuration[episode_idx, 6:9].astype(np.float32)
        goal_eef_xyz = np.repeat(goal_eef_xyz[None,:],agent_pos.shape[0],axis=0) # (T, 3)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 2500, 3
                'agent_pos': agent_pos, # T, 7
                'goal_eef_xyz': goal_eef_xyz, # T, 3
            },
            'action': sample['action'][:, :7].astype(np.float32) # T, 7
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample,idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_episode_eval_setup(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """Return stored start/goal setup for exact evaluation reset."""
        start_cfg = self.start_configuration[episode_idx].astype(np.float32)
        end_cfg = self.end_configuration[episode_idx].astype(np.float32)
        return {
            'start_joint': start_cfg[:6],
            'start_eef_xyz': start_cfg[6:9],
            'start_gripper': start_cfg[13],
            'goal_eef_xyz': end_cfg[6:9],
        }
