import abc

import numpy as np

from RL4MM.gym.models import Action


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> Action:
        pass

    def get_expected_action(self, state: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        return np.array([self.get_action(state) for _ in range(n_samples)]).mean(axis=0)
