from typing import Union, Callable, Tuple

import gym
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory


class PolicyGradientAgent(Agent):
    def __init__(
        self,
        policy: torch.nn.Module,
        action_std: Union[float, Callable] = 0.01,
        optimizer: torch.optim.Optimizer = None,
        env: gym.Env = None,
        lr_scheduler: _LRScheduler = None,
    ):
        self.env = env or TradingEnvironment()
        self.input_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        assert self.input_size == policy[0].in_features
        self.policy_net = policy
        self.action_std = action_std
        self.optimizer = optimizer or torch.optim.SGD(self.policy_net.parameters(), lr=1e-1)
        self.lr_scheduler = lr_scheduler or StepLR(self.optimizer, step_size=1, gamma=0.995)
        self.noise_dist = torch.distributions.Normal
        self.proportion_completed: float = 0.0

    def get_action(
        self, state: np.ndarray, deterministic: bool = False, include_log_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.tensor]]:
        assert not (deterministic and include_log_probs), "include_log_probs is only an option for deterministic output"
        mean_value = self.policy_net(torch.tensor(state, dtype=torch.float, requires_grad=False))
        std = self.action_std(self.proportion_completed) if isinstance(self.action_std, Callable) else self.action_std
        if deterministic:
            return mean_value.detach().numpy()
        action_dist = torch.distributions.Normal(loc=mean_value, scale=std * torch.ones_like(mean_value))
        action = action_dist.sample()
        if include_log_probs:
            log_probs = action_dist.log_prob(action)
            return action.detach().numpy(), log_probs
        return action.detach().numpy()

    def train(self, num_epochs: int = 1, reporting_freq: int = 100):
        learning_losses = []
        learning_rewards = []
        self.proportion_completed = 0.0
        for epoch in tqdm(range(num_epochs)):
            observations, actions, rewards, log_probs = generate_trajectory(self.env, self, include_log_probs=True)
            learning_rewards.append(rewards.mean())
            rewards = torch.tensor(rewards)
            future_rewards = self._calculate_future_rewards(rewards)
            loss = -torch.mean(log_probs * future_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % reporting_freq == 0:
                tqdm.write(str(loss.item()))
            learning_losses.append(loss.item())
            self.proportion_completed += 1 / (num_epochs - 1)
            self.lr_scheduler.step()
        return learning_losses, learning_rewards

    @staticmethod
    def _calculate_future_rewards(rewards: torch.tensor):
        flipped_rewards = torch.flip(rewards, dims=(-1,))
        cumulative_flipped = torch.cumsum(flipped_rewards, dim=-1)
        return torch.flip(cumulative_flipped, dims=(-1,))
