import abc
from typing import Union, List

import gym
import numpy as np


class InfoCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, done: bool) -> dict:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: np.ndarray):
        pass


class ActionInfoCalculator(InfoCalculator):
    """ActionInfoCalculator records the actions taken throughout the episode and then outputs the mean actions taken at
    the terminal step as an info dict. This is the Stable Baselines 3 convention. See for example, the VecMonitor class
    of SB3."""

    def __init__(self, action_space: gym.spaces.Box, n_steps: int = 10 * 10, num_trajectories: int = 1000):
        self.action_space = action_space
        self.n_steps = n_steps
        self.num_trajectories = num_trajectories
        self.nan_matrix = np.empty((self.num_trajectories, self.action_space.shape[0], self.n_steps))
        self.nan_matrix[:] = np.nan
        self.actions = self.nan_matrix.copy()
        self.empty_infos = [{} for _ in range(self.num_trajectories)] if self.num_trajectories > 1 else {}
        self.count = 0

    def calculate(
        self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, done: bool
    ) -> Union[dict, List[dict]]:
        if done:
            mean_actions = self._calculate_mean_actions()
            return [
                {f"action_{j}": mean_actions[i, j] for j in range(mean_actions.shape[1])}
                for i in range(mean_actions.shape[0])
            ]
        else:
            self.actions[:, :, self.count] = action
            self.count += 1
            return self.empty_infos

    def reset(self, initial_state: np.ndarray):
        self.count = 0
        self.actions = self.nan_matrix.copy()

    def _calculate_mean_actions(self):
        return self.actions.nanmean(axis=2)
