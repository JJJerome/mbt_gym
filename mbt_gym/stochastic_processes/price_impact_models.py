import abc
from typing import Optional

import numpy as np

from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class PriceImpactModel(StochasticProcessModel):
    """PriceImpactModel models the price impact of orders in the order book."""

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
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, num_trajectories, seed)

    @abc.abstractmethod
    def get_impact(self, action: np.ndarray) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def max_speed(self) -> float:
        pass


class TemporaryPowerPriceImpact(PriceImpactModel):
    def __init__(
        self,
        temporary_impact_coefficient: float = 0.01,
        temporary_impact_exponent: float = 1.0,
        num_trajectories: int = 1,
    ):
        self.temporary_impact_coefficient = temporary_impact_coefficient
        self.temporary_impact_exponent = temporary_impact_exponent
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=None,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=None,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass

    def get_impact(self, action) -> np.ndarray:
        return self.temporary_impact_coefficient * action**self.temporary_impact_exponent

    @property
    def max_speed(self) -> float:
        return 100.0  # TODO: link to asset price perhaps?
