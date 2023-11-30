import abc
from typing import Optional

import numpy as np

from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class ArrivalModel(StochasticProcessModel):
    """ArrivalModel models the arrival of orders to the order book. The first entry of arrivals represents an arrival
    of an exogenous SELL order (arriving on the buy side of the book) and the second entry represents an arrival of an
    exogenous BUY order (arriving on the sell side of the book).
    """

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
    def get_arrivals(self) -> np.ndarray:
        pass


class PoissonArrivalModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.intensity * self.step_size


class PoissonArrivalNonLinearModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < 1. - np.exp(-self.intensity * self.step_size)


class HawkesArrivalModel(ArrivalModel):
    def __init__(
        self,
        baseline_arrival_rate: np.ndarray = np.array([[10.0, 10.0]]),
        step_size: float = 0.01,
        jump_size: float = 40.0,
        mean_reversion_speed: float = 60.0,
        terminal_time: float = 1,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.baseline_arrival_rate = baseline_arrival_rate
        self.jump_size = jump_size  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        self.mean_reversion_speed = mean_reversion_speed
        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self._get_max_arrival_rate(),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=baseline_arrival_rate,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.mean_reversion_speed
            * (np.ones((self.num_trajectories, 2)) * self.baseline_arrival_rate - self.current_state)
            * self.step_size
            * np.ones((self.num_trajectories, 2))
            + self.jump_size * arrivals
        )
        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.current_state * self.step_size

    def _get_max_arrival_rate(self):
        return self.baseline_arrival_rate * 10

    # TODO: Improve this with 4*std
    # See: https://math.stackexchange.com/questions/4047342/expectation-of-hawkes-process-with-exponential-kernel
