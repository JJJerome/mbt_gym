import gym

import numpy as np

from DRL4AMM.agents.Agent import Agent
from DRL4AMM.gym.models import Action


class RandomAgent(Agent):
    def __init__(self, action_space: gym.spaces.Space, seed: int = None):
        self.action_space = action_space
        self.action_space.seed(seed)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


class FixedActionAgent(Agent):
    def __init__(self, fixed_action: tuple):
        self.fixed_action = fixed_action

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.fixed_action


class FixedSpreadAgent(Agent):
    def __init__(self, half_spread: float = 1.0, offset: float = 0.0):
        self.half_spread = half_spread
        self.offset = offset

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.half_spread - self.offset, self.half_spread + self.offset])


class HumanAgent(Agent):
    def get_action(self, state: np.ndarray):
        bid = float(input(f"Current state is {state}. How large do you want to set midprice-bid half spread? "))
        ask = float(input(f"Current state is {state}. How large do you want to set ask-midprice half spread? "))
        return np.array([bid, ask])
