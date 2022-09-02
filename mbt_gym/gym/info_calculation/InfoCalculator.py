import abc

import numpy as np


class InfoCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, state: np.ndarray, action: np.ndarray, reward: float) -> dict:
        pass


class ActionInfoCalculator(InfoCalculator):
    def calculate(self, state: np.ndarray, action: np.ndarray, reward: float) -> dict:
        return {"action": action}
