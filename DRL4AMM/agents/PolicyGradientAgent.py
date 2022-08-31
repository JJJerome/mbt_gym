from typing import Union, Callable, Tuple

import gym
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm

from DRL4AMM.agents.Agent import Agent
from DRL4AMM.gym.TradingEnvironment import TradingEnvironment
from DRL4AMM.gym.helpers.generate_trajectory import generate_trajectory


class PolicyGradientAgent(Agent):
    def __init__(
        self,
        policy: torch.nn.Module,
        action_stdev: Union[float, Callable] = 0.01,
        optimizer: torch.optim.Optimizer = None,
        env: gym.spaces.Space = None,
        scheduler: _LRScheduler = None,
    ):
        self.env = env or TradingEnvironment()
        self.input_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.num_trajectories = env.num_trajectories
        assert self.input_size == policy[0].in_features
        self.policy_net = policy
        self.action_stdev = action_stdev
        self.optimizer = optimizer or torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.scheduler = scheduler or StepLR(self.optimizer, step_size=1, gamma=0.995)
        self.noise_dist = torch.distributions.Normal
        self.proportion_completed: float = 0.0

    def get_action(
        self, state: np.ndarray, deterministic: bool = False, include_log_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.tensor]]:
        assert not (deterministic and include_log_probs), "Include log probs is only an option for deterministic output"
        mean_value = self.policy_net(torch.tensor(state, dtype=torch.float, requires_grad=False))
        sd = (
            self.action_stdev(self.proportion_completed)
            if isinstance(self.action_stdev, Callable)
            else self.action_stdev
        )
        if deterministic:
            return mean_value.detach().numpy()
        action_dist = torch.distributions.Normal(loc=mean_value, scale=sd * torch.ones_like(mean_value))
        action = action_dist.sample()
        if include_log_probs:
            log_probs = action_dist.log_prob(action)
            return action.detach().numpy(), log_probs
        return action.detach().numpy()

    def train(self, num_epochs: int = 1, reporting_freq: int = 100):
        learning_losses = []
        learning_rewards = []
        learning_actions = []
        self.proportion_completed = 0.0
        for epoch in tqdm(range(num_epochs)):
            observations, actions, rewards, log_probs = generate_trajectory(self.env, self, include_log_probs=True)
            learning_rewards.append(rewards.sum())
            rewards = torch.tensor(rewards)
            flipped_rewards = torch.flip(rewards, dims=(-1,))
            future_flipped = torch.cumsum(flipped_rewards, dim=-1)
            future_rewards = torch.flip(future_flipped, dims=(-1,))
            loss = -torch.mean(log_probs * future_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % reporting_freq == 0:
                tqdm.write(str(loss.item()))
                tqdm.write(
                    "Action at t=0.5*T & inventory 0 is"
                    + str(self.get_action(np.array([[self.env.terminal_time, 0.5]]), deterministic=True))
                )
            learning_losses.append(loss.item())
            learning_actions.append(self.get_action(np.array([[self.env.terminal_time, 0.5]]), deterministic=True))
            self.proportion_completed += 1 / (num_epochs - 1)
            self.scheduler.step()
        return learning_actions, learning_losses, learning_rewards
