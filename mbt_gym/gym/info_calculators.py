import abc
from typing import Union

import numpy as np


class InfoCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, state: np.ndarray, action: np.ndarray, reward: float) -> dict:
        pass


class ActionInfoCalculator(InfoCalculator):
    def calculate(self, state: np.ndarray, action: np.ndarray, reward: float) -> Union[dict, list[dict]]:
        return [{"action": a} for a in action]
