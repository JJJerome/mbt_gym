from typing import Union, Callable

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from DRL4AMM.agents.Agent import Agent
from DRL4AMM.gym.VectorizedMarketMakingEnvironment import VectorizedMarketMakingEnvironment
from DRL4AMM.gym.helpers.generate_trajectory import generate_trajectory


class PolicyGradientAgent(Agent):
    def __init__(
        self,
        policy: torch.nn.Module,
        action_stdev: Union[float, Callable] = 0.01,
        optimizer: torch.optim.Optimizer = None,
        env: gym.Env = None,
    ):
        self.env = env or VectorizedMarketMakingEnvironment()
        self.input_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.num_trajectories = env.num_trajectories
        assert self.input_size == policy[0].in_features
        self.net = policy
        self.action_stdev = action_stdev
        self.optimizer = optimizer or torch.optim.Adam(policy.parameters(), lr=1e-4)
        self.noise_dist = torch.distributions.Normal
        self.proportion_completed = 0.0

    def get_action(self, state: np.ndarray, deterministic: bool = False, include_log_probs: bool = False) -> np.ndarray:
        assert not (deterministic and include_log_probs), "Cannot include log probs for a deterministic output."
        mean_value = self.net(torch.tensor(state, dtype=torch.float, requires_grad=False))
        sd = (
            self.action_stdev(self.proportion_completed)
            if isinstance(self.action_stdev, Callable)
            else self.action_stdev
        )
        if not deterministic:
            action_dist = torch.distributions.Normal(loc=mean_value, scale=sd * torch.ones_like(mean_value))
            action = action_dist.sample()
            log_probs = action_dist.log_prob(action)
        return (action.detach().numpy(), log_probs) if include_log_probs else mean_value.detach().numpy()

    def train(self, num_epochs: int = 1):
        losses = []
        self.proportion_completed = 0.0
        for epoch in tqdm(range(num_epochs)):
            observations, actions, rewards, log_probs = generate_trajectory(self.env, self, include_log_probs=True)
            rewards = torch.tensor(rewards)
            future_rewards = rewards.fliplr().cumsum(dim=1).fliplr()
            loss = -torch.mean(log_probs * future_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                tqdm.write(str(loss.item()))
            losses.append(loss.item())
            self.proportion_completed += 1 / (num_epochs - 1)
