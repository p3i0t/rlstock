import numpy as np
import torch
import torch.nn as nn
import gym
from gym.spaces import Discrete, Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType

class MyFCNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        assert len(obs_space.shape) == 1, 'observation should be 1d Tensor.'
        d_obs = obs_space.shape[0]
        self.shared_fc = nn.Sequential(
            nn.Linear(d_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        # if isinstance(action_space, Discrete):
        #     d_action = action_space.n
        assert isinstance(action_space, Box)
        d_action = action_space.shape[0]
        # else:
        #     raise ValueError(f'{action_space=} not supported.')

        self.actor = nn.Sequential(
            nn.Linear(64, d_action*2),
            # nn.Tanh()
        ) # mean and log_std
        self.critic = nn.Linear(64, 1)

        self._feature = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        # print(f"{obs.shape=} {obs.dtype}")
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs).float()

        self._feature = self.shared_fc(obs)
        action_logits = self.actor(self._feature)
        # print(f'action_logits shape: {action_logits.shape}')
        return action_logits, state


    def value_function(self) -> TensorType:
        return self.critic(self._feature).squeeze(dim=1)