import abc
from copy import copy

import numpy as np
from numpy.random import default_rng


class StochasticProcessModel(metaclass=abc.ABCMeta):
    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.terminal_time = terminal_time
        self.num_trajectories = num_trajectories
        self.initial_state = initial_state
        self._check_attribute_shapes()
        self.current_state = copy(self.initial_vector_state)
        self.rng = default_rng(seed)
        self.seed_ = seed

    def reset(self):
        self.current_state = self.initial_vector_state

    @abc.abstractmethod
    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, state: np.ndarray = None):
        pass

    def seed(self, seed: int = None):
        self.rng = default_rng(seed)
        self.seed_ = seed

    def _check_attribute_shapes(self):
        for name in ["initial_state", "min_value", "max_value"]:
            attribute = getattr(self, name)
            assert (
                len(attribute.shape) == 2 and attribute.shape[0] == 1
            ), f"Attribute {name} must be a vector of shape (1, state_size)."

    @property
    def initial_vector_state(self) -> np.ndarray:
        initial_state = self.initial_state
        if isinstance(initial_state, list):
            initial_state = np.array([self.initial_state])
        return np.repeat(initial_state, self.num_trajectories, axis=0)
