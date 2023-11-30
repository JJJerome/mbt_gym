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

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_impact(self, action) -> np.ndarray:
        return self.temporary_impact_coefficient * action**self.temporary_impact_exponent

    @property
    def max_speed(self) -> float:
        return 100.0  # TODO: link to asset price perhaps?


class TemporaryAndPermanentPriceImpact(PriceImpactModel):
    def __init__(
        self,
        temporary_impact_coefficient: float = 0.01,
        permanent_impact_coefficient: float = 0.01,
        n_steps: int = 20 * 10,
        terminal_time: float = 1.0,
        num_trajectories: int = 1,
    ):
        self.temporary_impact_coefficient = temporary_impact_coefficient
        self.permanent_impact_coefficient = permanent_impact_coefficient
        self.n_steps = n_steps
        self.terminal_time = terminal_time
        self.step_size = self.terminal_time / self.n_steps
        super().__init__(
            min_value=np.array([[-self.max_speed * self.terminal_time * self.permanent_impact_coefficient]]),
            max_value=np.array([[self.max_speed * self.terminal_time * self.permanent_impact_coefficient]]),
            step_size=self.step_size,
            terminal_time=0.0,
            initial_state=np.array([[0]]),
            num_trajectories=num_trajectories,
            seed=None,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        self.current_state = self.current_state + self.permanent_impact_coefficient * actions * self.step_size

    def get_impact(self, action) -> np.ndarray:
        return self.temporary_impact_coefficient * action + self.current_state

    @property
    def max_speed(self) -> float:
        return 10.0  # TODO: link to asset price perhaps?


class TemporaryAndTransientPriceImpact(PriceImpactModel):
    def __init__(
        self,
        temporary_impact_coefficient: float = 0.01,
        transient_impact_coefficient: float = 0.01, # kappa in Neuman-Voß (2022)
        resilience_coefficient: float = 0.01, # rho in Neuman-Voß (2022)
        initial_transient_impact: float = 0.01, # y in Neuman-Voß (2022)
        linear_kernel_coefficient: float = 0.01, # gamma in Neuman-Voß (2022)
        n_steps: int = 20 * 10,
        terminal_time: float = 1.0,
        num_trajectories: int = 1,
    ):
        self.temporary_impact_coefficient = temporary_impact_coefficient
        self.transient_impact_coefficient = transient_impact_coefficient
        self.resilience_coefficient = resilience_coefficient
        self.initial_transient_impact = initial_transient_impact
        self.linear_kernel_coefficient = linear_kernel_coefficient
        self.n_steps = n_steps
        self.terminal_time = terminal_time
        self.step_size = self.terminal_time / self.n_steps
        super().__init__(
            min_value=np.array([[-self.max_speed * self.terminal_time * self.transient_impact_coefficient]]),
            max_value=np.array([[self.max_speed * self.terminal_time * self.transient_impact_coefficient]]),
            step_size=self.step_size,
            terminal_time=0.0,
            initial_state=np.array([[self.initial_transient_impact]]),
            num_trajectories=num_trajectories,
            seed=None,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        self.current_state = (self.current_state - self.resilience_coefficient * self.current_state * self.step_size
                              + self.linear_kernel_coefficient * actions * self.step_size)

    def get_impact(self, action) -> np.ndarray:
        return self.temporary_impact_coefficient * action + self.transient_impact_coefficient * self.current_state

    @property
    def max_speed(self) -> float:
        return 10.0  # TODO: link to asset price



class TransientPriceImpact(PriceImpactModel):
    def __init__(
        self,
        transient_impact_coefficient: float = 0.01, # kappa in Neuman-Voß (2022)
        resilience_coefficient: float = 0.01, # rho in Neuman-Voß (2022)
        initial_transient_impact: float = 0.01, # y in Neuman-Voß (2022)
        linear_kernel_coefficient: float = 0.01, # gamma in Neuman-Voß (2022)
        n_steps: int = 20 * 10,
        terminal_time: float = 1.0,
        num_trajectories: int = 1,
    ):
        self.transient_impact_coefficient = transient_impact_coefficient
        self.resilience_coefficient = resilience_coefficient
        self.initial_transient_impact = initial_transient_impact
        self.linear_kernel_coefficient = linear_kernel_coefficient
        self.n_steps = n_steps
        self.terminal_time = terminal_time
        self.step_size = self.terminal_time / self.n_steps
        super().__init__(
            min_value=np.array([[-self.max_speed * self.terminal_time * self.transient_impact_coefficient]]),
            max_value=np.array([[self.max_speed * self.terminal_time * self.transient_impact_coefficient]]),
            step_size=self.step_size,
            terminal_time=0.0,
            initial_state=np.array([[self.initial_transient_impact]]),
            num_trajectories=num_trajectories,
            seed=None,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        self.current_state = (self.current_state - self.resilience_coefficient * self.current_state * self.step_size
                              + self.linear_kernel_coefficient * actions * self.step_size)

    def get_impact(self, action) -> np.ndarray:
        return self.transient_impact_coefficient * self.current_state

    @property
    def max_speed(self) -> float:
        return 10.0  # TODO: link to asset price


